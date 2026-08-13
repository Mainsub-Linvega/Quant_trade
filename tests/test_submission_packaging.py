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
                archive.writestr("model/baseline_model.json", "{}")
                archive.writestr("model/hybrid_meta.json", json.dumps(meta))
                archive.writestr("model/seed.txt", "model")
            result = audit(path, 1.16, 480, 1)
            self.assertTrue(result["passed"])

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
