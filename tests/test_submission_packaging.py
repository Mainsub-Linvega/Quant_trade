from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_submission_zip import audit
from make_submission import check_v3_hybrid_meta
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

    def test_rejects_missing_market_forest(self) -> None:
        """市场森林文件为空时，λ 再对也没用 —— 必须拦住。"""
        with tempfile.TemporaryDirectory() as directory:
            meta = baseline_meta()
            meta["market_model_files"] = []
            model_dir = self._write(Path(directory), meta)
            with self.assertRaises(SystemExit):
                check_v3_hybrid_meta(model_dir, off_baseline=False)


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


if __name__ == "__main__":
    unittest.main()
