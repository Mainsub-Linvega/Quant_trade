from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_submission_zip import DECLARED_EXTRA_FILES, DECLARED_MODULES, REQUIRED, audit
from make_submission import (EXCLUDED_MODULES, IMPORT_TO_DISTRIBUTION, REQUIREMENTS_NAME,
                             SUBMISSION_EXTRA_FILES, SUBMISSION_MODULES, _version_from_url,
                             analyze_requirements, check_requirements, check_submission_modules,
                             check_v3_hybrid_meta, eval_environment_versions,
                             inspect_requirements,
                             resolve_local_modules, resolve_third_party_imports)
from promote_v3_candidate import PUBLIC_BASELINE

# 评测机实测环境（outputs/cloud/delivery_cloud_py311_4t.json 的 environment 块）。
# 用例里**不写死版本号** —— 那份 JSON 刷新后这些用例应当跟着走，而不是集体变红。
EVAL_VERSIONS = eval_environment_versions()


def baseline_requirements() -> str:
    """一份「完全合规」的 requirements.txt：版本取自评测机实测真值。"""
    lines = ["# 由主办方 JupyterHub base 环境 pip freeze 生成",
             "pandas==2.0.3"]        # 评测 harness 自己要用，与我们的 import 闭包无关
    lines += [f"{IMPORT_TO_DISTRIBUTION[root]}=={version}"
              for root, version in sorted(EVAL_VERSIONS.items())]
    return "\n".join(lines) + "\n"


def baseline_meta() -> dict:
    """由 PUBLIC_BASELINE 反推一份「完全合规」的 hybrid_meta.json。"""
    return {
        "blend_weight": PUBLIC_BASELINE["blend_weight"],
        "num_iteration": PUBLIC_BASELINE["num_iteration"],
        "history_window": PUBLIC_BASELINE["history_window"],
        "history_positions": list(range(PUBLIC_BASELINE["history_positions_count"])),
        "prediction_scale": PUBLIC_BASELINE["prediction_scale"],
        "lgbm_model_files": [f"s{i}.txt" for i in range(PUBLIC_BASELINE["n_seeds"])],
        "market_lambda": PUBLIC_BASELINE["market_lambda"],
        "market_model_files": [f"m{i}.txt" for i in range(PUBLIC_BASELINE["market_model_count"])],
        "cross_section_weighted": PUBLIC_BASELINE["cross_section_weighted"],
        "slow_fast_window": PUBLIC_BASELINE["slow_fast_window"],
        "slow_fast_slow_relative": PUBLIC_BASELINE["slow_fast_slow_relative"],
        "slow_fast_fast_relative": PUBLIC_BASELINE["slow_fast_fast_relative"],
        "long_window": PUBLIC_BASELINE["long_window"],
    }


class PackagingMetaGateTest(unittest.TestCase):
    """打包前那道 meta 闸门。

    ⚠️ `check_v3_hybrid_meta` 的取值表与 `PUBLIC_BASELINE` 是**两处派生同一份口径** ——
    2026-08-13 往常量里加了三个键却漏改取值表，打包当场 KeyError。
    这组用例把那个耦合钉住：常量里有的键，取值表必须都能取到。
    """

    def _write(self, directory: Path, meta: dict) -> Path:
        model_dir = directory / "model"
        model_dir.mkdir()
        (model_dir / "hybrid_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return model_dir

    def test_accepts_baseline_and_covers_every_constant_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self._write(Path(directory), baseline_meta())
            found = check_v3_hybrid_meta(model_dir, off_baseline=False)
        self.assertEqual(set(found), set(PUBLIC_BASELINE),
                         "取值表与 PUBLIC_BASELINE 的键集不一致")

    def test_rejects_each_structural_switch(self) -> None:
        for key, bad in (("blend_weight", 0.5), ("num_iteration", 160),
                         ("market_lambda", 0.0), ("cross_section_weighted", False),
                         ("prediction_scale", 0.856)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                meta = baseline_meta()
                meta[key] = bad
                model_dir = self._write(Path(directory), meta)
                with self.assertRaises(SystemExit):
                    check_v3_hybrid_meta(model_dir, off_baseline=False)
                # --off-baseline 是有意偏离的出口，必须仍然放行
                check_v3_hybrid_meta(model_dir, off_baseline=True)

    def test_rejects_missing_slow_fast_keys(self) -> None:
        """2026-08-18 的回归：slow/fast 三键是公榜 0.0041150085 与 0.0039977510 的全部差别。

        它们此前不在 PUBLIC_BASELINE 里 ⟹ 丢键或写错值不会被任何门禁发现。
        """
        for key in ("slow_fast_window", "slow_fast_slow_relative", "slow_fast_fast_relative"):
            for mutate in ("drop", "wrong"):
                with self.subTest(key=key, mutate=mutate), \
                        tempfile.TemporaryDirectory() as directory:
                    meta = baseline_meta()
                    if mutate == "drop":
                        meta.pop(key)
                    else:
                        meta[key] = float(meta[key]) * 2.0 + 1.0
                    model_dir = self._write(Path(directory), meta)
                    with self.assertRaises(SystemExit):
                        check_v3_hybrid_meta(model_dir, off_baseline=False)
                    check_v3_hybrid_meta(model_dir, off_baseline=True)

    def test_long_window_drift_is_caught_in_both_directions(self) -> None:
        """2026-08-21 的回归。⚠️ 该键的基线在同日**翻转过**：

        转正前榜上那份没有长窗（基线 None）⟹ 带长窗的候选是偏离；
        转正后（公榜 0.0041833953）榜上那份**就是** 512 ⟹ **缺键/None 才是偏离**，
        缺键时 `main.py` 会静默关掉长窗、交出低 1.66% 的旧模型。
        两个方向都必须被抓住 —— 这正是 08-18 slow/fast 丢键那类事故的镜像。
        """
        self.assertEqual(PUBLIC_BASELINE["long_window"], 512,
                         "基线值变了就要同步改本用例的语义，别让它静默失效")
        for bad in (None, 64, 4096, 256):
            with self.subTest(long_window=bad), tempfile.TemporaryDirectory() as directory:
                meta = baseline_meta()
                if bad is None:
                    meta.pop("long_window")          # 缺键 = 旧模型
                else:
                    meta["long_window"] = bad
                model_dir = self._write(Path(directory), meta)
                with self.assertRaises(SystemExit):
                    check_v3_hybrid_meta(model_dir, off_baseline=False)
                # --off-baseline 是有意偏离的出口，必须仍然放行
                check_v3_hybrid_meta(model_dir, off_baseline=True)

    def test_baseline_long_window_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self._write(Path(directory), baseline_meta())
            self.assertEqual(check_v3_hybrid_meta(model_dir, off_baseline=False)["long_window"],
                             PUBLIC_BASELINE["long_window"])

    def test_rejects_missing_market_forest(self) -> None:
        """市场森林文件为空时，λ 再对也没用 —— 必须拦住。"""
        with tempfile.TemporaryDirectory() as directory:
            meta = baseline_meta()
            meta["market_model_files"] = []
            model_dir = self._write(Path(directory), meta)
            with self.assertRaises(SystemExit):
                check_v3_hybrid_meta(model_dir, off_baseline=False)


FROZEN_RIDGE = ROOT / "strategies" / "v3_hybrid" / "model" / "baseline_model.json"


class SubmissionPackagingTest(unittest.TestCase):
    def test_audit_accepts_minimal_valid_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.zip"
            meta = {"prediction_scale": 1.16, "num_iteration": 480,
                    "history_window": 5, "history_positions": list(range(40)),
                    "lgbm_model_files": ["seed.txt"]}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("main.py", "class Model: pass")
                archive.writestr(REQUIREMENTS_NAME, baseline_requirements())
                # main.py 顶层无条件 import 这三个模块，少一个 Model 就装不起来
                for module in ("features.py", "lgbm_numpy.py", "history.py"):
                    archive.writestr(module, "")
                archive.writestr("model/baseline_model.json", "{}")
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
                archive.writestr("model/seed.txt", "model")
            result = audit(path, 1.16, 480, 1)
            self.assertTrue(result["passed"], result["checks"])

    def test_audit_rejects_absent_market_forest_files(self) -> None:
        """2026-08-18 的回归：原实现只核 `lgbm_model_files` 在不在包里，

        `market_model_files` 一个都不核 —— 市场森林是架构的一半（公榜 +21.99% 的来源），
        漏打包会让审计照样 PASS。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_market.zip"
            meta = {"prediction_scale": 1.16, "num_iteration": 480,
                    "history_window": 5, "history_positions": list(range(40)),
                    "lgbm_model_files": ["seed.txt"],
                    "market_model_files": ["market.txt"]}       # 声明了，但不入包
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("main.py", "class Model: pass")
                archive.writestr(REQUIREMENTS_NAME, baseline_requirements())
                for module in ("features.py", "lgbm_numpy.py", "history.py"):
                    archive.writestr(module, "")
                archive.writestr("model/baseline_model.json", "{}")
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
                archive.writestr("model/seed.txt", "model")
            result = audit(path)
            self.assertFalse(result["passed"])
            self.assertEqual(result["absent_declared_market_models"], ["market.txt"])

    def test_audit_rejects_missing_required_module(self) -> None:
        """`main.py` 顶层 import 的模块少一个，`Model` 就装不起来 ⟹ 整份提交判无效。"""
        for dropped in ("features.py", "lgbm_numpy.py", "history.py"):
            with self.subTest(dropped=dropped), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "missing_module.zip"
                meta = {"lgbm_model_files": []}
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("main.py", "class Model: pass")
                    archive.writestr(REQUIREMENTS_NAME, baseline_requirements())
                    for module in ("features.py", "lgbm_numpy.py", "history.py"):
                        if module != dropped:
                            archive.writestr(module, "")
                    archive.writestr("model/baseline_model.json", "{}")
                    archive.writestr("model/hybrid_meta.json", json.dumps(meta))
                result = audit(path)
                self.assertFalse(result["passed"])
                self.assertIn(dropped, result["missing"])

    def test_audit_public_baseline_catches_pre_slowfast_package(self) -> None:
        """2026-08-18 的回归：slow/fast 转正后，缺这三个键的包就是低 2.93% 的旧模型。

        `main.py` 是 `PredictionTrail(int(window)) if window else None` ——
        缺键时 slow/fast 被**静默关掉**、不抛错，所以只能靠审计拦。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre_slowfast.zip"
            meta = baseline_meta()
            for key in ("slow_fast_window", "slow_fast_slow_relative",
                        "slow_fast_fast_relative"):
                meta.pop(key)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("main.py", "class Model: pass")
                archive.writestr(REQUIREMENTS_NAME, baseline_requirements())
                for module in ("features.py", "lgbm_numpy.py", "history.py"):
                    archive.writestr(module, "")
                archive.writestr("model/baseline_model.json", "{}")
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
                for name in meta["lgbm_model_files"] + meta["market_model_files"]:
                    archive.writestr(f"model/{name}", "model")
            # 不开 --expect-public-baseline 时，旧行为一切正常 ⟹ 正是原来那个洞
            self.assertTrue(audit(path)["passed"])
            strict = audit(path, expect_public_baseline=True)
            self.assertFalse(strict["passed"])
            self.assertEqual(len(strict["public_baseline_drift"]), 3)

    def test_audit_rejects_unexpected_module(self) -> None:
        """2026-08-19 的回归：原实现只查**缺**文件，不查**多**文件。

        打包那边当时是「除 train.py 外全收 *.py」⟹ 纯研究模块 `temporal.py` 混进了
        私榜包；它不在 `main.py` 的 import 闭包里，却会因为研究改动改变提交包字节。
        这一条钉住反方向：包里多一个 .py 就必须 FAIL。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stowaway.zip"
            meta = baseline_meta()
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("main.py", "class Model: pass")
                archive.writestr(REQUIREMENTS_NAME, baseline_requirements())
                for module in sorted(DECLARED_MODULES - {"main.py"}):
                    archive.writestr(module, "")
                archive.writestr("temporal.py", "# 研究模块，不该在包里")
                # 这个用例开了 --expect-public-baseline ⟹ 也会核冻结岭回归的
                # 文件身份；本用例测的是「多带一个模块」，所以岭回归给真的。
                archive.writestr("model/baseline_model.json",
                                 FROZEN_RIDGE.read_bytes())
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
                for name in meta["lgbm_model_files"] + meta["market_model_files"]:
                    archive.writestr(f"model/{name}", "model")
            result = audit(path, expect_public_baseline=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["unexpected_modules"], ["temporal.py"])
            # 只有这一条该红：模型身份本身是干净的（正是 20260818.zip 的真实处境）
            self.assertEqual([name for name, ok in result["checks"].items() if not ok],
                             ["no_unexpected_modules"])

    def test_required_is_derived_from_declared_modules(self) -> None:
        """`REQUIRED` 必须由 `SUBMISSION_MODULES` 派生，不能是第二份手抄清单。"""
        self.assertEqual(DECLARED_MODULES, SUBMISSION_MODULES["v3_hybrid"])
        self.assertTrue(DECLARED_MODULES <= REQUIRED)

    def test_audit_rejects_train_py(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            meta = {"lgbm_model_files": []}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("main.py", "")
                archive.writestr(REQUIREMENTS_NAME, baseline_requirements())
                archive.writestr("train.py", "")
                archive.writestr("model/baseline_model.json", "{}")
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
            self.assertFalse(audit(path)["passed"])


class RequirementsGateTest(unittest.TestCase):
    """`requirements.txt` 这道闸门。

    ⚠️ 2026-08-25 的回归，属于一个新类别：**「该查的项本身漏了一条」**。
    主办方 08-23 新文档 `submission_and_evaluation.md:53` 的「最终交付要求」第 3 条
    明写 ZIP 必须包含 `requirements.txt`；打包端从不放它、审计端 `REQUIRED` 里也没有它
    ⟹ `20260824.zip` 只有 12 个文件，而审计 11/11 全过。

    这里钉住的核心不是「文件在不在」，而是 CLAUDE.md 伤疤规则 11 要的那种
    **能失败、且独立于被测量本身的归属检查**：拿本机 `.venv` 的 freeze 冒充评测环境
    freeze 时，必须当场炸。
    """

    def _strategy(self, directory: str, requirements: str | None,
                  main_body: str | None = None) -> Path:
        """一个最小的假 v3_hybrid：main.py 硬 import numpy、函数体里延迟 import lightgbm。"""
        path = Path(directory) / "v3_hybrid"
        path.mkdir()
        (path / "main.py").write_text(
            main_body or "import json\nimport numpy\n\n\ndef go():\n    import lightgbm\n",
            encoding="utf-8")
        if requirements is not None:
            (path / REQUIREMENTS_NAME).write_text(requirements, encoding="utf-8")
        return path

    def _rejects(self, requirements: str | None, *, main_body: str | None = None,
                 off_env_baseline: bool = False) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy(directory, requirements, main_body)
            with self.assertRaises(SystemExit) as caught:
                check_requirements("v3_hybrid", path, off_env_baseline=off_env_baseline)
        return str(caught.exception)

    def _accepts(self, requirements: str, *, off_env_baseline: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy(directory, requirements)
            return check_requirements("v3_hybrid", path, off_env_baseline=off_env_baseline)

    # ---------------------------------------------------------------- 真实策略目录

    def test_real_strategy_third_party_surface_is_exactly_numpy_and_lightgbm(self) -> None:
        """拿**真实**的 strategies/v3_hybrid 跑。以后引入新的第三方依赖会在这里红。

        ROADMAP P-REQ 记的「真实依赖面」就是这两个：numpy 硬依赖、lightgbm 延迟 import，
        其余 json / pathlib / re / __future__ 全是标准库。无 pandas / scipy / sklearn。
        """
        self.assertEqual(resolve_third_party_imports(ROOT / "strategies" / "v3_hybrid"),
                         {"numpy", "lightgbm"})

    def test_every_strategy_is_classified_for_extra_files(self) -> None:
        """每个策略都必须在 SUBMISSION_EXTRA_FILES 里被显式分类，空集也要写出来。"""
        self.assertEqual(set(SUBMISSION_EXTRA_FILES), set(SUBMISSION_MODULES))
        self.assertEqual(SUBMISSION_EXTRA_FILES["v3_hybrid"], frozenset({REQUIREMENTS_NAME}))

    def test_eval_env_truth_is_available_and_not_this_machine(self) -> None:
        """归属检查的已知真值必须存在，且必须**不是**本机的版本。

        它没了，下面那些「版本对不上就炸」的用例会全部退化成静默通过 ——
        那正是这次要修的那类洞的形状。
        """
        self.assertEqual(set(EVAL_VERSIONS), {"numpy", "lightgbm"},
                         "outputs/cloud/delivery_cloud_py311_4t.json 的 environment 块缺项")
        import numpy
        self.assertNotEqual(EVAL_VERSIONS["numpy"], numpy.__version__,
                            "评测机 numpy 版本与本机相同 ⟹ 这道归属检查测不出「本机 freeze」，"
                            "需要换一个能区分两台机器的判据")

    # ---------------------------------------------------------------- 硬失败

    def test_missing_file_names_the_jupyterhub_command(self) -> None:
        message = self._rejects(None)
        self.assertIn(REQUIREMENTS_NAME, message)
        self.assertIn("pip freeze", message)

    def test_empty_file_is_rejected(self) -> None:
        self.assertIn("空", self._rejects("\n\n# 只有注释\n"))

    def test_missing_each_required_distribution_is_rejected(self) -> None:
        """漏掉我们真正 import 的包 —— 评测端 `import numpy` 会直接崩。"""
        for dropped in sorted(IMPORT_TO_DISTRIBUTION.values()):
            with self.subTest(dropped=dropped):
                text = "".join(f"{IMPORT_TO_DISTRIBUTION[root]}=={version}\n"
                               for root, version in sorted(EVAL_VERSIONS.items())
                               if IMPORT_TO_DISTRIBUTION[root] != dropped)
                self.assertIn(dropped, self._rejects(text))

    def test_unregistered_third_party_import_is_rejected(self) -> None:
        """将来往策略里引入一个新第三方包，必须有人在 IMPORT_TO_DISTRIBUTION 里按一下。

        否则「它在评测环境里到底装没装」这件事没有任何人核过，而门禁会继续绿着。
        """
        message = self._rejects(baseline_requirements(),
                                main_body="import numpy\nimport scipy\n\n\n"
                                          "def go():\n    import lightgbm\n")
        self.assertIn("scipy", message)
        self.assertIn("IMPORT_TO_DISTRIBUTION", message)

    def test_team_absolute_path_is_rejected(self) -> None:
        """交付要求第 7 条：不得写死 /home/jovyan 或队伍专属绝对路径。"""
        message = self._rejects(baseline_requirements()
                                + "mypkg @ file:///home/jovyan/wheels/mypkg-1.0-py3-none-any.whl\n")
        self.assertIn("/home/jovyan/", message)

    def test_unparsable_line_is_rejected(self) -> None:
        self.assertIn("不是合法的依赖声明",
                      self._rejects(baseline_requirements() + "这不是一行依赖\n"))

    def test_unknown_strategy_must_be_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy(directory, baseline_requirements())
            with self.assertRaises(SystemExit) as caught:
                check_requirements("v9_unknown", path, off_env_baseline=False)
        self.assertIn("SUBMISSION_EXTRA_FILES", str(caught.exception))

    # ---------------------------------------------------------------- 归属：对已知真值

    def test_local_venv_freeze_is_caught(self) -> None:
        """⭐ 核心用例：把本机 `.venv` 的版本塞进去，必须当场炸。

        这是整套门禁**能失败**的那一格。P-REQ 待办 2 明写「文档要求在 JupyterHub 环境
        生成，本机 freeze 不算数」—— 而「不算数」必须有机械手段兑现，不能靠记性。
        """
        import numpy
        text = (f"numpy=={numpy.__version__}\n"
                f"lightgbm=={EVAL_VERSIONS['lightgbm']}\n")
        message = self._rejects(text)
        self.assertIn(numpy.__version__, message)
        self.assertIn(EVAL_VERSIONS["numpy"], message)
        # --off-env-baseline 是有意偏离的出口，必须仍然放行
        self._accepts(text, off_env_baseline=True)

    def test_conda_forge_build_root_is_not_a_team_path(self) -> None:
        """conda-forge 的构建根就长成 `/home/conda/feedstock_root/` —— 不是队伍路径。

        评测机 base 里的 numpy 正是这一形状；把它判成第 7 条违规，就等于让**合法的**
        评测机 freeze 过不了闸门。
        """
        text = ("lightgbm==4.3.0\n"
                "numpy @ file:///home/conda/feedstock_root/build_artifacts/"
                "numpy_1682210216651/work\n")
        report = inspect_requirements(text, {"numpy", "lightgbm"}, EVAL_VERSIONS)
        self.assertEqual(report["summary"]["team_path_lines"], [])
        self.assertEqual([p for p in report["problems"] if "绝对路径" in p], [])

    def test_jovyan_path_is_still_rejected_alongside_the_build_root(self) -> None:
        """豁免只针对构建根前缀；同一份文件里的 `/home/jovyan/` 必须照样拦住。"""
        text = ("lightgbm==4.3.0\n"
                "numpy @ file:///home/conda/feedstock_root/build_artifacts/"
                "numpy_1682210216651/work\n"
                "mypkg @ file:///home/jovyan/wheels/mypkg-1.0-py3-none-any.whl\n")
        report = inspect_requirements(text, {"numpy", "lightgbm"}, EVAL_VERSIONS)
        self.assertEqual(report["summary"]["team_path_lines"],
                         ["mypkg @ file:///home/jovyan/wheels/mypkg-1.0-py3-none-any.whl"])

    def test_conda_direct_reference_with_readable_version_passes(self) -> None:
        """conda 装的包在 freeze 里没有 `==`，版本只藏在 wheel 文件名里 —— 要能读出来。"""
        version = EVAL_VERSIONS["numpy"]
        text = (f"numpy @ file:///croot/numpy_and_numpy_base_1708638617955/work/dist/"
                f"numpy-{version}-cp311-cp311-linux_x86_64.whl\n"
                f"lightgbm=={EVAL_VERSIONS['lightgbm']}\n")
        report = self._accepts(text)
        self.assertEqual(report["summary"]["pins"]["numpy"], version)
        self.assertEqual(len(report["summary"]["direct_reference_lines"]), 1)

    def test_direct_reference_without_readable_version_is_not_silently_passed(self) -> None:
        """版本读不出来 ⟹ 判「无法核」而**不是**当成核过了。"""
        text = ("numpy @ file:///croot/work/dist/anonymous.whl\n"
                f"lightgbm=={EVAL_VERSIONS['lightgbm']}\n")
        message = self._rejects(text)
        self.assertIn("读不出版本", message)
        self.assertIn("numpy", message)

    def test_version_extraction_does_not_mistake_build_directory_for_version(self) -> None:
        """`numpy_and_numpy_base_1708638617955` 不是版本 `1708638617955`。"""
        url = ("file:///croot/numpy_and_numpy_base_1708638617955/work/dist/"
               "numpy-1.24.3-cp311-cp311-linux_x86_64.whl")
        self.assertEqual(_version_from_url("numpy", url), "1.24.3")
        self.assertIsNone(_version_from_url("scipy", url))

    def test_comments_and_option_lines_do_not_break_parsing(self) -> None:
        summary = analyze_requirements("# 注释\n\n--index-url https://example.invalid\n"
                                       "Foo_Bar.Baz==1.0\n")
        self.assertEqual(summary["pins"], {"foo-bar-baz": "1.0"})   # PEP 503 规范化
        self.assertEqual(summary["option_lines"], ["--index-url https://example.invalid"])
        self.assertEqual(summary["unparsable"], [])

    def test_strategy_without_declared_requirements_is_skipped(self) -> None:
        """v1_ridge 已退役、显式声明为空集 ⟹ 不该被这道闸门拦住。"""
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy(directory, None)
            self.assertEqual(check_requirements("v1_ridge", path, off_env_baseline=False), {})

    # ---------------------------------------------------------------- 审计端

    def test_audit_required_derives_extra_files_instead_of_copying(self) -> None:
        """`REQUIRED` 必须由 `SUBMISSION_EXTRA_FILES` 派生，不能是第二份手抄清单。"""
        self.assertEqual(DECLARED_EXTRA_FILES, SUBMISSION_EXTRA_FILES["v3_hybrid"])
        self.assertIn(REQUIREMENTS_NAME, REQUIRED)

    def _zip(self, directory: str, requirements: str | None) -> Path:
        path = Path(directory) / "package.zip"
        meta = baseline_meta()
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("main.py", "class Model: pass")
            for module in sorted(DECLARED_MODULES - {"main.py"}):
                archive.writestr(module, "")
            if requirements is not None:
                archive.writestr(REQUIREMENTS_NAME, requirements)
            archive.writestr("model/baseline_model.json", FROZEN_RIDGE.read_bytes())
            archive.writestr("model/hybrid_meta.json", json.dumps(meta))
            for name in meta["lgbm_model_files"] + meta["market_model_files"]:
                archive.writestr(f"model/{name}", "model")
        return path

    def test_audit_rejects_package_without_requirements(self) -> None:
        """⭐ 这一格就是 `20260824.zip` 的真实处境：模型身份干净，但缺一条硬要求。"""
        with tempfile.TemporaryDirectory() as directory:
            result = audit(self._zip(directory, None), expect_public_baseline=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing"], [REQUIREMENTS_NAME])
        self.assertEqual(sorted(name for name, ok in result["checks"].items() if not ok),
                         ["required_files_present", "requirements_covers_dependencies"])

    def test_audit_accepts_package_with_valid_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = audit(self._zip(directory, baseline_requirements()),
                           expect_public_baseline=True)
        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["requirements_summary"]["pinned_versions"],
                         {IMPORT_TO_DISTRIBUTION[root]: version
                          for root, version in EVAL_VERSIONS.items()})

    def test_audit_catches_local_venv_freeze_and_off_env_baseline_releases_it(self) -> None:
        import numpy
        text = f"numpy=={numpy.__version__}\nlightgbm=={EVAL_VERSIONS['lightgbm']}\n"
        with tempfile.TemporaryDirectory() as directory:
            path = self._zip(directory, text)
            strict = audit(path, expect_public_baseline=True)
            relaxed = audit(path, expect_public_baseline=True, off_env_baseline=True)
        self.assertFalse(strict["passed"])
        self.assertEqual([name for name, ok in strict["checks"].items() if not ok],
                         ["requirements_matches_eval_env"])
        self.assertTrue(strict["requirements_env_drift"])
        self.assertTrue(relaxed["passed"], relaxed["checks"])


class SubmissionModuleWhitelistTest(unittest.TestCase):
    """入包 .py 清单这道闸门。

    ⚠️ 旧口径是「除 `train.py` 外全收 `*.py`」，注释里写明理由是**写死清单曾漏过
    `lgbm_numpy.py`**。所以这里不能退回硬编码：声明集必须与 `main.py` 的 AST import
    闭包双向对拍 —— 漏模块和多模块都要当场炸。
    """

    def test_declared_modules_match_main_import_closure(self) -> None:
        """拿**真实**策略目录跑。以后再往 strategies/v3_hybrid/ 扔研究文件会在这里红。"""
        closure = resolve_local_modules(ROOT / "strategies" / "v3_hybrid")
        self.assertEqual(closure, set(SUBMISSION_MODULES["v3_hybrid"]))
        # 目录里的每个 .py 都必须被分类：入包，或显式排除并写明理由
        present = {path.name for path in (ROOT / "strategies" / "v3_hybrid").glob("*.py")}
        self.assertEqual(present - set(SUBMISSION_MODULES["v3_hybrid"]) - set(EXCLUDED_MODULES),
                         set(), "有未分类的模块")
        self.assertEqual(check_submission_modules("v3_hybrid", ROOT / "strategies" / "v3_hybrid"),
                         sorted(SUBMISSION_MODULES["v3_hybrid"]))

    def _strategy_dir(self, directory: str, files: dict[str, str]) -> Path:
        path = Path(directory) / "probe"
        path.mkdir()
        for name, body in files.items():
            (path / name).write_text(body, encoding="utf-8")
        return path

    def test_rejects_unclassified_module(self) -> None:
        """研究文件掉进策略目录 ⟹ 打包必须失败，而不是把它一起装进去。"""
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy_dir(directory, {
                "main.py": "from helper import thing\n",
                "helper.py": "thing = 1\n",
                "scratch.py": "# 研究草稿\n",
            })
            declared = {"probe": frozenset({"main.py", "helper.py"})}
            with unittest.mock.patch.dict(SUBMISSION_MODULES, declared, clear=False):
                with self.assertRaises(SystemExit) as caught:
                    check_submission_modules("probe", path)
        self.assertIn("scratch.py", str(caught.exception))

    def test_rejects_module_reachable_but_undeclared(self) -> None:
        """闭包里有、声明集里没有 —— 就是当年漏掉 `lgbm_numpy.py` 的那个方向。"""
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy_dir(directory, {
                "main.py": "import helper\n",
                "helper.py": "value = 1\n",
            })
            declared = {"probe": frozenset({"main.py"})}
            with unittest.mock.patch.dict(SUBMISSION_MODULES, declared, clear=False):
                with self.assertRaises(SystemExit) as caught:
                    check_submission_modules("probe", path)
        self.assertIn("helper.py", str(caught.exception))

    def test_deferred_third_party_import_is_not_local(self) -> None:
        """`main.py:266` 那句延迟的 `import lightgbm` 不能被当成本地模块。"""
        with tempfile.TemporaryDirectory() as directory:
            path = self._strategy_dir(directory, {
                "main.py": "def go():\n    import lightgbm\n    return lightgbm\n",
            })
            self.assertEqual(resolve_local_modules(path), {"main.py"})


if __name__ == "__main__":
    unittest.main()
