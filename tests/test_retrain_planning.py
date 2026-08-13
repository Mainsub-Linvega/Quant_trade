from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments")]

from joint_recalibration import matrix
from retrain_extended import command_plan, validate_audit


class RetrainPlanningTest(unittest.TestCase):
    def test_audit_gate_requires_train_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({"comparison": {"changed": True, "splits": {
                "train": {"added": ["new.parquet"], "removed": [], "modified": [], "row_delta": 5}}}}))
            self.assertTrue(validate_audit(path)["changed"])
            path.write_text(json.dumps({"comparison": {"changed": False, "splits": {"train": {}}}}))
            with self.assertRaises(SystemExit):
                validate_audit(path)

    def test_command_plan_is_candidate_only(self) -> None:
        args = argparse.Namespace(data_root="data", candidate_dir="outputs/candidates/probe",
                                  ridge_alpha=2e6, ridge_feature_count=200,
                                  lgbm_feature_count=200, sample_modulo=5,
                                  num_iteration=480, n_seeds=3, prediction_scale=1.16)
        commands = command_plan(args)
        flattened = " ".join(part for command in commands for part in command)
        self.assertIn("outputs/candidates/probe", flattened)
        self.assertNotIn("strategies/v3_hybrid/model", flattened)

    def test_joint_matrix_is_finite_and_joint(self) -> None:
        payload = matrix()
        self.assertEqual(len(payload["ridge"]), 12)
        self.assertEqual(len(payload["lgbm"]), 9)
        self.assertTrue(all({"ridge_alpha", "feature_count", "sample_modulo"} <= row.keys()
                            for row in payload["ridge"]))


if __name__ == "__main__":
    unittest.main()
