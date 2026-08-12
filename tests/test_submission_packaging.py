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
