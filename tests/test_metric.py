from __future__ import annotations

import unittest

import numpy as np

from src.metric import scale_invariant_score, weighted_zero_mean_r2


class MetricTest(unittest.TestCase):
    def test_peak_matches_closed_form_and_scale_scan(self) -> None:
        target = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float64)
        prediction = np.array([0.2, -0.3, 0.8, -0.1], dtype=np.float64)
        weight = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)
        result = scale_invariant_score(target, prediction, weight)
        scale = result["optimal_scale"]
        self.assertAlmostEqual(
            result["peak"], weighted_zero_mean_r2(target, scale * prediction, weight), places=14
        )
        grid = np.linspace(scale - 1.0, scale + 1.0, 2001)
        scanned = max(weighted_zero_mean_r2(target, a * prediction, weight) for a in grid)
        self.assertLessEqual(result["peak"] - scanned, 1e-6)

    def test_target_and_prediction_scaling_do_not_change_peak(self) -> None:
        target = np.array([1.0, -2.0, 0.5])
        prediction = np.array([0.4, -0.2, 0.1])
        weight = np.array([1.0, 1.0, 4.0])
        base = scale_invariant_score(target, prediction, weight)
        scaled_target = scale_invariant_score(target / 7.0, prediction, weight)
        scaled_prediction = scale_invariant_score(target, prediction * 3.5, weight)
        self.assertAlmostEqual(base["peak"], scaled_target["peak"], places=14)
        self.assertAlmostEqual(base["peak"], scaled_prediction["peak"], places=14)
        self.assertAlmostEqual(base["optimal_scale"] / 3.5, scaled_prediction["optimal_scale"], places=14)

    def test_negative_weights_are_zeroed(self) -> None:
        target = np.array([1.0, 2.0, -1.0])
        prediction = np.array([0.5, 99.0, -0.25])
        weight = np.array([2.0, -5.0, 1.0])
        clipped = np.maximum(weight, 0.0)
        self.assertAlmostEqual(
            weighted_zero_mean_r2(target, prediction, weight),
            weighted_zero_mean_r2(target, prediction, clipped),
        )
        self.assertEqual(
            scale_invariant_score(target, prediction, weight),
            scale_invariant_score(target, prediction, clipped),
        )

    def test_degenerate_energy(self) -> None:
        result = scale_invariant_score(np.zeros(3), np.ones(3), np.ones(3))
        self.assertEqual(result["peak"], 0.0)
        self.assertTrue(np.isnan(result["optimal_scale"]))

    def test_shape_and_finiteness_validation(self) -> None:
        with self.assertRaises(ValueError):
            scale_invariant_score(np.ones(2), np.ones(3), np.ones(2))
        with self.assertRaises(ValueError):
            scale_invariant_score(np.array([1.0, np.nan]), np.ones(2), np.ones(2))


if __name__ == "__main__":
    unittest.main()
