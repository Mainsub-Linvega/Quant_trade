from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategies" / "v3_hybrid"))

from temporal import (ARMS, MarketRegimeHistory, MultiScaleAssetHistory, temporal_arm_blocks,
                      temporal_arm_width, temporal_atoms_from_lags)


class TemporalHistoryTest(unittest.TestCase):
    def test_offline_and_online_time_batches_are_identical(self) -> None:
        rng = np.random.default_rng(2026)
        time_ids = np.repeat(np.arange(12), 3)
        asset_ids = np.tile(np.arange(3), 12)
        current = rng.normal(size=(len(time_ids), 4)).astype(np.float32)
        offline = MultiScaleAssetHistory(4).transform(current, asset_ids, time_ids)
        online_history = MultiScaleAssetHistory(4)
        pieces = {key: [] for key in offline}
        for time_id in np.unique(time_ids):
            mask = time_ids == time_id
            got = online_history.transform(current[mask], asset_ids[mask], time_ids[mask])
            for key, value in got.items():
                pieces[key].append(value)
        for key in offline:
            np.testing.assert_array_equal(offline[key], np.concatenate(pieces[key]))

    def test_strict_lags_gap_and_no_history_defaults(self) -> None:
        history = MultiScaleAssetHistory(1)
        current = np.array([[1.0], [3.0], [8.0]], dtype=np.float32)
        got = history.transform(current, np.array([0, 0, 0]), np.array([10, 12, 17]))
        np.testing.assert_array_equal(got["lag1"][:, 0], [0.0, 1.0, 3.0])
        np.testing.assert_array_equal(got["lag2"][:, 0], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(got["difference"][:, 0], [1.0, 2.0, 5.0])
        np.testing.assert_array_equal(got["observation_gap"][:, 0], [0.0, 2.0, 5.0])
        self.assertEqual(float(got["ema3"][0, 0]), 0.0)
        self.assertEqual(float(got["ema3"][1, 0]), 1.0)
        self.assertAlmostEqual(float(got["ema3"][2, 0]), 7.0 / 3.0, places=6)

    def test_slope_uses_only_prior_observations(self) -> None:
        history = MultiScaleAssetHistory(1)
        current = np.arange(1, 7, dtype=np.float32)[:, None]
        got = history.transform(current, np.zeros(6), np.arange(6))
        self.assertEqual(float(got["slope5"][0, 0]), 0.0)
        self.assertEqual(float(got["slope5"][1, 0]), 0.0)
        np.testing.assert_allclose(got["slope5"][2:, 0], 1.0, atol=1e-7)

    def test_arm_widths_and_prefix(self) -> None:
        expected = {"baseline": 16, "t1_lags": 24, "t2_state": 40, "t3_full": 49,
                    "t4_regime": 76, "t5_zscore": 20}
        atoms = {key: np.zeros((2, (1 if key == "observation_gap" else
                                      20 if key.startswith("regime_") else 4)), np.float32)
                 for key in ("lag1", "difference", "mean5", "deviation5", "lag2", "lag5",
                             "ema3", "ema10", "std5", "std20", "slope5", "slope20",
                             "zscore5", "observation_gap", "regime_current", "regime_lag1",
                             "regime_difference")}
        for arm in ARMS:
            self.assertEqual(sum(x.shape[1] for x in temporal_arm_blocks(atoms, arm)), expected[arm])
            self.assertEqual(temporal_arm_width(4, arm), expected[arm])

    def test_lag_cache_reconstruction_matches_online(self) -> None:
        rng = np.random.default_rng(99)
        n, width = 30, 3
        current = rng.normal(size=(n, width)).astype(np.float32)
        asset = np.zeros(n, dtype=np.int64)
        time_ids = np.arange(n, dtype=np.int64) * 2
        online = MultiScaleAssetHistory(width).transform(current, asset, time_ids)
        lags = np.zeros((n, 20, width), dtype=np.float32)
        counts = np.minimum(np.arange(n), 20)
        for row in range(n):
            for j in range(min(row, 20)):
                lags[row, j] = current[row - j - 1]
        gap = np.r_[0, np.diff(time_ids)].astype(np.float32)
        rebuilt = temporal_atoms_from_lags(current, lags, counts, gap)
        for key in online:
            np.testing.assert_allclose(online[key], rebuilt[key], atol=2e-6, rtol=1e-6, err_msg=key)

    def test_market_regime_offline_online_equivalence(self) -> None:
        rng = np.random.default_rng(12)
        time_ids = np.repeat(np.arange(8), 4)
        current = rng.normal(size=(32, 40)).astype(np.float32)
        offline = MarketRegimeHistory(40).transform(current, time_ids)
        online_model = MarketRegimeHistory(40)
        parts = {key: [] for key in offline}
        for time_id in np.unique(time_ids):
            mask = time_ids == time_id
            got = online_model.transform(current[mask], time_ids[mask])
            for key in parts:
                parts[key].append(got[key])
        for key in parts:
            np.testing.assert_array_equal(offline[key], np.concatenate(parts[key]))
        self.assertEqual(offline["regime_current"].shape[1], 20)

    def test_non_increasing_asset_time_rejected(self) -> None:
        history = MultiScaleAssetHistory(1)
        with self.assertRaises(ValueError):
            history.transform(np.ones((2, 1), np.float32), np.array([0, 0]), np.array([2, 2]))


if __name__ == "__main__":
    unittest.main()
