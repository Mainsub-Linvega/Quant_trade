from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from responder_predictability import fit_multi_ridge, predict_multi
from responder_residual_increment import optimal_weights


class ResponderResearchTest(unittest.TestCase):
    def test_shared_multi_target_fit_matches_separate_fits(self) -> None:
        rng = np.random.default_rng(7)
        n = 240
        features = rng.normal(size=(n, 8)).astype(np.float32)
        time_ids = np.repeat(np.arange(n // 6), 6)
        weight = rng.uniform(0.2, 2.0, size=n)
        targets = np.column_stack([
            features[:, 0] - 0.4 * features[:, 2],
            -0.5 * features[:, 1] + features[:, 3],
        ])
        model = fit_multi_ridge(features, time_ids, targets[:, 0], targets, weight, 5, 10.0)
        shared = predict_multi(model, features, time_ids)
        for column in range(targets.shape[1]):
            single = fit_multi_ridge(features, time_ids, targets[:, 0], targets[:, [column]], weight, 5, 10.0)
            separate = predict_multi(single, features, time_ids)[:, 0]
            np.testing.assert_allclose(shared[:, column], separate, atol=1e-10, rtol=1e-10)

    def test_optimal_weights_recover_linear_residual(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(500, 3))
        beta = np.array([0.4, -0.2, 0.7])
        residual = x @ beta
        got = optimal_weights(residual, x, np.ones(len(x)))
        np.testing.assert_allclose(got, beta, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
