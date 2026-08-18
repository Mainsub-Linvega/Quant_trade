from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "experiments"),
    str(ROOT / "strategies" / "v1_ridge"),
]

from src.io import time_sample_mask
from src.validation import rolling_time_folds
from walk_forward_rolling import sign_test_p
from walk_forward_rolling import write_single_report
from train import fit_model


class RidgeInfrastructureTest(unittest.TestCase):
    def test_fit_model_honors_explicit_selected_indices(self) -> None:
        features = np.arange(36, dtype=np.float32).reshape(12, 3)
        target = np.linspace(-0.3, 0.4, 12, dtype=np.float64)
        weight = np.ones(12, dtype=np.float64)
        time_ids = np.repeat(np.arange(4), 3)

        artifact, selected = fit_model(
            features,
            target,
            weight,
            time_ids,
            feature_count=1,
            ridge_alpha=10.0,
            selected_indices=np.array([0, 2]),
        )

        np.testing.assert_array_equal(selected, [0, 2])
        self.assertEqual(artifact["selected_indices"], [0, 2])
        self.assertEqual(artifact["selected_features"], ["feature_000", "feature_002"])
        self.assertEqual(len(artifact["coef"]), 4)

    def test_sign_test_ignores_zero_deltas(self) -> None:
        self.assertEqual(sign_test_p(0, 0), 1.0)
        self.assertEqual(sign_test_p(10, 0), 2 / 2**10)
        self.assertEqual(sign_test_p(5, 5), 1.0)

    def test_fold_offset_preserves_chunk_size(self) -> None:
        ids = np.arange(1000)
        reserved_offset = 50
        base = rolling_time_folds(ids, 5, 400, embargo=6, reserved_offset=reserved_offset)
        shifted = rolling_time_folds(
            ids, 5, 400, embargo=6, offset=50, reserved_offset=reserved_offset
        )
        self.assertEqual(len(base), 5)
        self.assertEqual(len(shifted), 5)
        self.assertEqual({len(valid) for _, valid in base}, {108})
        self.assertEqual({len(valid) for _, valid in shifted}, {108})
        self.assertEqual(shifted[0][1][0] - base[0][1][0], 50)
        for (base_train, base_valid), (shifted_train, shifted_valid) in zip(base, shifted):
            np.testing.assert_array_equal(shifted_train, base_train + 50)
            np.testing.assert_array_equal(shifted_valid, base_valid + 50)

    def test_default_folds_keep_historical_tail_behavior(self) -> None:
        ids = np.arange(1003)
        folds = rolling_time_folds(ids, 5, 400, embargo=6)
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[-1][1][-1], ids[-1])

    def test_phase_balanced_sampling_keeps_density_and_all_phases(self) -> None:
        ids = np.arange(1000)
        periodic = time_sample_mask(ids, 5)
        balanced = time_sample_mask(ids, 5, sampling="phase_balanced", phase_period=10)
        np.testing.assert_array_equal(periodic, ids % 5 == 0)
        self.assertEqual(int(periodic.sum()), 200)
        self.assertEqual(int(balanced.sum()), 200)
        self.assertEqual(
            [int(balanced[ids % 10 == phase].sum()) for phase in range(10)],
            [20] * 10,
        )

    def test_single_report_writes_both_outputs(self) -> None:
        folds = [{
            "fold": 0,
            "train_time_range": [0, 9],
            "valid_time_range": [16, 19],
            "embargo_gap": 7,
            "train_rows": 10,
            "valid_rows": 4,
            "scores": {"arm": 0.25},
            "score_denominator": 2.0,
            "elapsed_seconds": 0.1,
        }]
        config = {
            "feature_count": 2,
            "ridge_tol": 1e-4,
            "ridge_max_iter": 100,
            "prediction_scale": 1.0,
            "prediction_clip": 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            write_single_report(
                Path(directory), folds, "arm", config, 1, 1, 10, 6, 5,
                100.0, 0, 0,
            )
            self.assertTrue((Path(directory) / "walk_forward_rolling.json").exists())
            self.assertTrue((Path(directory) / "walk_forward_rolling.md").exists())


if __name__ == "__main__":
    unittest.main()
