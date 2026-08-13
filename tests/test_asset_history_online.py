"""`AssetHistory.transform_online` 必须与离线的 `transform` **逐位相同**。

在线快路径是 2026-08-13 为压推理耗时加的（0.338 → 0.019 ms/次，占整次 `predict` 16%）。
它和离线路径共用 `_blocks`，差别只在 `lags` 怎么取 —— 但「只在取法上不同」正是
工程坑第 7 条的形状（`cross_sectional_deviation` 的 1-ulp 被树放大到 2.85e-03）。
所以这里要求的是 `assert_array_equal`（逐位），不是 `allclose`。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategies" / "v3_hybrid"))

from history import AssetHistory


BLOCK_NAMES = ("previous", "difference", "rolling_mean", "rolling_deviation")


def _online_blocks(feature_count: int, window: int, batches):
    history = AssetHistory(feature_count=feature_count, window_size=window)
    parts: list[list[np.ndarray]] = [[], [], [], []]
    for current, asset_ids in batches:
        for slot, block in zip(parts, history.transform_online(current, asset_ids)):
            slot.append(block)
    return [np.concatenate(slot) for slot in parts]


class AssetHistoryOnlineTest(unittest.TestCase):
    def test_online_matches_offline_on_full_cross_sections(self) -> None:
        """满截面、时长跨过 window 好几轮 —— 覆盖冷启动与窗口填满两段。"""
        rng = np.random.default_rng(20260813)
        n_assets, n_times, feature_count, window = 15, 40, 6, 5
        asset_ids = np.tile(np.arange(n_assets), n_times)
        current = rng.normal(size=(len(asset_ids), feature_count)).astype(np.float32)

        offline = AssetHistory(feature_count, window).transform(current, asset_ids)
        batches = [(current[i * n_assets:(i + 1) * n_assets],
                    asset_ids[i * n_assets:(i + 1) * n_assets]) for i in range(n_times)]
        online = _online_blocks(feature_count, window, batches)
        for name, expected, found in zip(BLOCK_NAMES, offline, online):
            np.testing.assert_array_equal(expected, found, err_msg=f"{name} 不逐位相同")

    def test_online_matches_offline_with_missing_and_reordered_assets(self) -> None:
        """截面不满、资产乱序 —— 生产数据里每 time_id 的行数并不恒为 15。"""
        rng = np.random.default_rng(99)
        feature_count, window, n_assets = 4, 3, 15
        batches = []
        for _ in range(30):
            size = int(rng.integers(2, n_assets + 1))
            assets = rng.choice(n_assets, size=size, replace=False)
            batches.append((rng.normal(size=(size, feature_count)).astype(np.float32), assets))

        offline = AssetHistory(feature_count, window).transform(
            np.concatenate([current for current, _ in batches]),
            np.concatenate([assets for _, assets in batches]))
        online = _online_blocks(feature_count, window, batches)
        for name, expected, found in zip(BLOCK_NAMES, offline, online):
            np.testing.assert_array_equal(expected, found, err_msg=f"{name} 不逐位相同")

    def test_cold_start_is_zero_not_nan(self) -> None:
        """首个 time_id 无历史 ⟹ previous / rolling_mean 为 0（模型训练时见过这种输入）。"""
        history = AssetHistory(feature_count=2, window_size=5)
        current = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        previous, difference, rolling_mean, rolling_deviation = history.transform_online(
            current, np.array([0, 1]))
        np.testing.assert_array_equal(previous, np.zeros_like(current))
        np.testing.assert_array_equal(rolling_mean, np.zeros_like(current))
        np.testing.assert_array_equal(difference, current)
        np.testing.assert_array_equal(rolling_deviation, current)

    def test_two_paths_cannot_be_mixed(self) -> None:
        """两条路径各有各的状态；混用会静默丢历史，所以必须拒绝。"""
        current = np.ones((2, 2), dtype=np.float32)
        assets = np.array([0, 1])

        online_first = AssetHistory(feature_count=2, window_size=3)
        online_first.transform_online(current, assets)
        with self.assertRaises(RuntimeError):
            online_first.transform(current, assets)

        offline_first = AssetHistory(feature_count=2, window_size=3)
        offline_first.transform(current, assets)
        with self.assertRaises(RuntimeError):
            offline_first.transform_online(current, assets)

    def test_repeated_asset_in_one_batch_is_rejected(self) -> None:
        """一次一个 time_id 是前提；重复 asset 说明调用方口径错了，不能静默算错。"""
        history = AssetHistory(feature_count=2, window_size=3)
        with self.assertRaises(ValueError):
            history.transform_online(np.ones((2, 2), dtype=np.float32), np.array([3, 3]))


if __name__ == "__main__":
    unittest.main()
