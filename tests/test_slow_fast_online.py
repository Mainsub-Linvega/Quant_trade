"""slow/fast 的在线状态必须与离线实现逐位一致。

离线口径是 `experiments/slow_fast_csv.py:causal_trailing_mean`（公榜 +2.93% 那份 CSV
就是它算的）；线上是 `strategies/v3_hybrid/main.py:PredictionTrail`。两端只要差一点，
公榜验过的东西就不等于私榜跑的东西 —— 这正是本项目 08-13 那次事故的形状。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "strategies" / "v3_hybrid"), str(_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, relative: str):
    """按**文件路径**加载，不依赖 sys.path 顺序。

    ⚠️ `main.py` 在 `v1_ridge` 与 `v3_hybrid` 里同名：用 `discover` 跑全套时，
    别的测试可能已经把 v1_ridge 排到前面，`from main import ...` 就会解析错文件
    （CLAUDE.md 长期伤疤 §4）。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PredictionTrail = _load("v3_hybrid_main", "strategies/v3_hybrid/main.py").PredictionTrail
causal_trailing_mean = _load("slow_fast_csv_mod",
                             "experiments/slow_fast_csv.py").causal_trailing_mean


class SlowFastOnlineTest(unittest.TestCase):
    def _compare(self, time_ids, asset_ids, values, window):
        offline = causal_trailing_mean(values, time_ids, asset_ids, window)
        trail = PredictionTrail(window)
        online = np.empty_like(offline)
        for t in np.unique(time_ids):          # runner 每次恰好喂一个 time_id
            m = time_ids == t
            online[m] = trail.transform_online(values[m], asset_ids[m], int(t))
        np.testing.assert_allclose(online, offline, rtol=0, atol=0)
        return offline

    def test_matches_offline_on_dense_stream(self):
        rng = np.random.default_rng(7)
        n_t, assets = 400, 15
        time_ids = np.repeat(np.arange(n_t, dtype=np.int64), assets)
        asset_ids = np.tile(np.arange(assets, dtype=np.int64), n_t)
        values = rng.normal(size=len(time_ids))
        self._compare(time_ids, asset_ids, values, window=50)

    def test_matches_offline_with_time_id_gaps(self):
        """测试集 time_id 有跳号（实测两处 7、两处 1446）⟹ 窗口按真实步长必须对得上。"""
        rng = np.random.default_rng(11)
        stamps = np.cumsum(rng.choice([1, 1, 1, 7, 40], size=200)).astype(np.int64)
        assets = 5
        time_ids = np.repeat(stamps, assets)
        asset_ids = np.tile(np.arange(assets, dtype=np.int64), len(stamps))
        values = rng.normal(size=len(time_ids))
        self._compare(time_ids, asset_ids, values, window=30)

    def test_matches_offline_when_assets_appear_late_or_intermittently(self):
        """资产不是每个 time_id 都出现（实测每 time_id 有 2~15 行）。"""
        rng = np.random.default_rng(13)
        rows_t, rows_a, rows_v = [], [], []
        for t in range(300):
            present = rng.choice(8, size=rng.integers(2, 9), replace=False)
            for a in present:
                rows_t.append(t); rows_a.append(a); rows_v.append(rng.normal())
        order = np.lexsort((np.array(rows_a), np.array(rows_t)))
        self._compare(np.array(rows_t)[order].astype(np.int64),
                      np.array(rows_a)[order].astype(np.int64),
                      np.array(rows_v)[order], window=25)

    def test_first_observation_gives_zero_fast(self):
        """规则 1：资产首次出现 ⟹ slow = 当期值 ⟹ fast = 0，不制造假信号。"""
        trail = PredictionTrail(10)
        slow = trail.transform_online(np.array([0.5, -0.25]), np.array([0, 1]), 100)
        np.testing.assert_allclose(slow, [0.5, -0.25], rtol=0, atol=0)

    def test_gap_larger_than_window_gives_zero_slow(self):
        """规则 2：见过但窗口内无历史 ⟹ slow = 0.0（照搬离线语义，不「修正」）。"""
        window = 10
        time_ids = np.array([0, 100], dtype=np.int64)
        asset_ids = np.array([0, 0], dtype=np.int64)
        values = np.array([1.0, 2.0])
        offline = self._compare(time_ids, asset_ids, values, window)
        self.assertEqual(offline[0], 1.0)      # 首次 ⟹ 自身
        self.assertEqual(offline[1], 0.0)      # 跳号超窗 ⟹ 0

    def test_window_boundary_is_inclusive(self):
        """窗口左端是闭的：t_prev == t − window 仍算在内（离线用 side='left'）。"""
        window = 10
        time_ids = np.array([0, 10, 11], dtype=np.int64)
        asset_ids = np.zeros(3, dtype=np.int64)
        values = np.array([4.0, 0.0, 0.0])
        offline = self._compare(time_ids, asset_ids, values, window)
        self.assertEqual(offline[1], 4.0)      # t=10, cutoff=0 ⟹ 含 t=0
        self.assertEqual(offline[2], 0.0)      # t=11, cutoff=1 ⟹ t=0 被排除，只含 t=10（值 0.0）


if __name__ == "__main__":
    unittest.main()
