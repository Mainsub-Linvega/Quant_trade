"""`AssetLongWindow` 的逐位一致性 —— 这是长窗块能不能进生产的地基。

要害只有一条：**离线整块 / 离线分批 / 在线逐 time_id 三条路径必须逐位相同**。
不成立的话 `scripts/check_consistency.py` 会当场报红，训练出来的模型也不是榜上那个。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "strategies" / "v3_hybrid") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "strategies" / "v3_hybrid"))

from history import AssetHistory, AssetLongWindow  # noqa: E402


def _stream(n_time: int, n_assets: int, width: int, seed: int = 0):
    """按 time_id 升序、每个 time_id 各资产一行 —— 与官方 runner 的喂法一致。"""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_time * n_assets, width)).astype(np.float32) * 2.0
    assets = np.tile(np.arange(n_assets, dtype=np.int64), n_time)
    return values, assets


def _naive(values, assets, window, n_assets):
    """最朴素的参考实现：逐 asset、逐行、对前 min(n,W) 个观测取均值。"""
    mean = np.zeros_like(values, dtype=np.float32)
    for a in range(n_assets):
        rows = np.flatnonzero(assets == a)
        series = values[rows].astype(np.float64)
        for i, row in enumerate(rows):
            if i == 0:
                continue
            mean[row] = series[max(0, i - window):i].mean(axis=0).astype(np.float32)
    return mean


class BitIdenticalTest(unittest.TestCase):
    WINDOW = 7
    WIDTH = 4
    ASSETS = 5
    TIMES = 120

    def setUp(self):
        self.values, self.assets = _stream(self.TIMES, self.ASSETS, self.WIDTH, seed=3)

    def _offline_whole(self):
        state = AssetLongWindow(feature_count=self.WIDTH, window=self.WINDOW)
        return state.transform(self.values, self.assets)

    def _offline_batched(self, batch_times: int):
        """按 parquet 批次切 —— 与 stream_long_window_blocks 的喂法一致。"""
        state = AssetLongWindow(feature_count=self.WIDTH, window=self.WINDOW)
        means, devs = [], []
        step = batch_times * self.ASSETS
        for start in range(0, len(self.values), step):
            m, d = state.transform(self.values[start:start + step],
                                   self.assets[start:start + step])
            means.append(m); devs.append(d)
        return np.vstack(means), np.vstack(devs)

    def _online(self):
        state = AssetLongWindow(feature_count=self.WIDTH, window=self.WINDOW)
        means, devs = [], []
        for start in range(0, len(self.values), self.ASSETS):
            m, d = state.transform_online(self.values[start:start + self.ASSETS],
                                          self.assets[start:start + self.ASSETS])
            means.append(m); devs.append(d)
        return np.vstack(means), np.vstack(devs)

    def test_online_matches_offline_bit_for_bit(self):
        """⭐ 最要紧的一条。差一个 ulp 就会被树的阶跃放大到 1e-3 量级。"""
        off_m, off_d = self._offline_whole()
        on_m, on_d = self._online()
        self.assertTrue(np.array_equal(off_m, on_m), "rolling_mean 离线/在线不逐位相同")
        self.assertTrue(np.array_equal(off_d, on_d), "deviation 离线/在线不逐位相同")

    def test_batched_offline_matches_whole_offline_bit_for_bit(self):
        """离线是逐 parquet 批次喂的，批边界不得改变结果。"""
        whole_m, whole_d = self._offline_whole()
        for batch_times in (1, 3, 17, 119):
            got_m, got_d = self._offline_batched(batch_times)
            self.assertTrue(np.array_equal(whole_m, got_m), f"batch={batch_times} 均值不一致")
            self.assertTrue(np.array_equal(whole_d, got_d), f"batch={batch_times} 偏离不一致")

    def test_matches_naive_reference(self):
        mean, dev = self._offline_whole()
        np.testing.assert_allclose(mean, _naive(self.values, self.assets, self.WINDOW,
                                                self.ASSETS), atol=2e-6)
        np.testing.assert_allclose(dev, self.values - mean, atol=0)

    def test_window_longer_than_the_stream(self):
        """W 大于总行数 ⟹ 退化成「到目前为止的全部历史」，不得越界。"""
        state = AssetLongWindow(feature_count=self.WIDTH, window=10_000)
        mean, _ = state.transform(self.values, self.assets)
        np.testing.assert_allclose(mean, _naive(self.values, self.assets, 10_000,
                                                self.ASSETS), atol=2e-6)


class BoundaryTest(unittest.TestCase):
    def test_first_observation_has_zero_mean_and_deviation_equals_current(self):
        """与 AssetHistory 的「无历史即 0」同语义。"""
        values = np.arange(6, dtype=np.float32).reshape(3, 2)
        assets = np.array([0, 1, 0])
        mean, dev = AssetLongWindow(feature_count=2, window=4).transform(values, assets)
        np.testing.assert_array_equal(mean[0], [0.0, 0.0])
        np.testing.assert_array_equal(mean[1], [0.0, 0.0])
        np.testing.assert_array_equal(dev[0], values[0])
        np.testing.assert_array_equal(mean[2], values[0])          # asset 0 的上一次观测

    def test_assets_do_not_leak(self):
        values = np.array([[1.0], [100.0], [2.0], [200.0]], dtype=np.float32)
        assets = np.array([0, 1, 0, 1])
        mean, _ = AssetLongWindow(feature_count=1, window=8).transform(values, assets)
        self.assertEqual(mean[2, 0], 1.0)
        self.assertEqual(mean[3, 0], 100.0)

    def test_window_one_equals_previous_observation(self):
        values = np.arange(5, dtype=np.float32).reshape(5, 1)
        assets = np.zeros(5, dtype=np.int64)
        mean, _ = AssetLongWindow(feature_count=1, window=1).transform(values, assets)
        np.testing.assert_array_equal(mean[:, 0], [0.0, 0.0, 1.0, 2.0, 3.0])

    def test_online_rejects_duplicate_assets_in_one_batch(self):
        state = AssetLongWindow(feature_count=1, window=4)
        with self.assertRaises(ValueError):
            state.transform_online(np.zeros((2, 1), dtype=np.float32), np.array([0, 0]))


class AgreesWithAssetHistoryTest(unittest.TestCase):
    def test_same_rolling_mean_as_AssetHistory_for_a_short_window(self):
        """短窗上两套实现必须给出同一个 rolling_mean —— 否则说明语义定义就不一致。

        `AssetHistory` 是 O(W) 定序求和、`AssetLongWindow` 是累积和相减，
        浮点路径不同，故用 atol 而非逐位。
        """
        values, assets = _stream(60, 4, 3, seed=11)
        _, _, hist_mean, hist_dev = AssetHistory(feature_count=3, window_size=5).transform(
            values, assets)
        long_mean, long_dev = AssetLongWindow(feature_count=3, window=5).transform(values, assets)
        np.testing.assert_allclose(long_mean, hist_mean, atol=2e-6)
        np.testing.assert_allclose(long_dev, hist_dev, atol=2e-6)


class LargeWindowTest(unittest.TestCase):
    def test_production_sized_window_stays_bit_identical(self):
        """W=512、40 列、15 资产 —— 生产规格。"""
        values, assets = _stream(400, 15, 40, seed=7)
        off = AssetLongWindow(feature_count=40, window=512).transform(values, assets)
        state = AssetLongWindow(feature_count=40, window=512)
        means, devs = [], []
        for start in range(0, len(values), 15):
            m, d = state.transform_online(values[start:start + 15], assets[start:start + 15])
            means.append(m); devs.append(d)
        self.assertTrue(np.array_equal(off[0], np.vstack(means)))
        self.assertTrue(np.array_equal(off[1], np.vstack(devs)))

    def test_ring_wraps_correctly_past_the_window(self):
        """跑满 3 圈以上，确认环形下标没有提前覆盖。"""
        values, assets = _stream(200, 3, 2, seed=9)
        mean, _ = AssetLongWindow(feature_count=2, window=13).transform(values, assets)
        np.testing.assert_allclose(mean, _naive(values, assets, 13, 3), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
