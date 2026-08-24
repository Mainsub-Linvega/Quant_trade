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

from audit_submission_zip import DECLARED_MODULES, REQUIRED, audit
from make_submission import (EXCLUDED_MODULES, SUBMISSION_MODULES, check_submission_modules,
                             check_v3_hybrid_meta, resolve_local_modules)
from promote_v3_candidate import PUBLIC_BASELINE


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
                archive.writestr("train.py", "")
                archive.writestr("model/baseline_model.json", "{}")
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
            self.assertFalse(audit(path)["passed"])


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
