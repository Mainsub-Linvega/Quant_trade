"""peer 可部署探针的构造正确性。

这个探针要回答的是「把 oracle 的 `peer_e_lag1` 换成模型自身预测之后还剩多少」，
所以**特征构造必须与原探针逐位同语义** —— 否则测出来的差异分不清是「换了量」
还是「换了实现」。本文件钉住四件事：

1. 6 列之和 ≡ `xs_peer_pair_probe.build_peer_feature` 的单列（只差 float32 舍入）；
2. 无搭档的 9 个资产该列恒为 0；
3. `shift(1)` 是**严格滞后** —— 用一个只有当期有值的构造去反证；
4. 阴性对照的搭档与真对子无交集，且**目标**资产集合相同（否则非零行都不一样，
   「多一列就变好」这个混淆就没被对照掉）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xs_peer_deployable_probe import (NEG_PAIRS, cross_sectional_residual,  # noqa: E402
                                      peer_columns)
from xs_peer_pair_probe import PAIRS, build_peer_feature  # noqa: E402

N_ASSETS = 15


def panel(n_times: int = 8, seed: int = 0):
    """合成面板：连续 time_id × 15 资产，行按 time_id 升序。"""
    rng = np.random.default_rng(seed)
    time_id = np.repeat(np.arange(100, 100 + n_times), N_ASSETS)
    asset_id = np.tile(np.arange(N_ASSETS), n_times)
    target = rng.normal(size=len(time_id))
    return time_id, asset_id, target


class PeerColumnConstructionTest(unittest.TestCase):
    def test_six_columns_sum_to_the_original_single_column(self) -> None:
        t, a, y = panel()
        reference = build_peer_feature(t, a, y).astype(np.float64)
        cols = peer_columns(cross_sectional_residual(y, t), t, a,
                            np.ones(len(t), bool), PAIRS, lag=1)
        # 原函数返回 float32 ⟹ 只能到 float32 精度；这里要的是「同语义」不是「同 dtype」
        np.testing.assert_allclose(np.sum(np.column_stack(cols), axis=1),
                                   reference, rtol=0, atol=1e-6)

    def test_each_row_is_nonzero_in_at_most_one_column(self) -> None:
        # 每个资产在 PAIRS 里至多做一次 key ⟹ 6 列在行上互斥。
        # 不互斥的话「6 列之和 ≡ 单列」这个等式就不成立，上一条会假通过。
        t, a, y = panel()
        cols = peer_columns(cross_sectional_residual(y, t), t, a,
                            np.ones(len(t), bool), PAIRS, lag=1)
        per_row = np.count_nonzero(np.column_stack(cols), axis=1)
        self.assertLessEqual(int(per_row.max()), 1)

    def test_assets_without_a_partner_stay_zero(self) -> None:
        t, a, y = panel()
        cols = peer_columns(cross_sectional_residual(y, t), t, a,
                            np.ones(len(t), bool), PAIRS, lag=1)
        merged = np.sum(np.column_stack(cols), axis=1)
        orphans = [x for x in range(N_ASSETS) if x not in PAIRS]
        self.assertEqual(len(orphans), 9)
        self.assertTrue(np.all(merged[np.isin(a, orphans)] == 0.0))

    def test_lag_one_is_strictly_causal(self) -> None:
        """只在**最后**一个 time_id 给搭档非零值；lag=1 时它不该出现在任何评估行上。

        这是对「当期泄漏」的反证：若实现误用了 shift(0)，最后一期的值会落回同一期，
        断言当场失败。
        """
        t, a, _ = panel(n_times=5)
        source = np.zeros(len(t), dtype=np.float64)
        last = t == t.max()
        source[last & (a == PAIRS[0])] = 7.0          # 只给 asset 0 的搭档、只给最后一期

        lagged = np.sum(np.column_stack(
            peer_columns(source, t, a, np.ones(len(t), bool), PAIRS, lag=1)), axis=1)
        self.assertTrue(np.all(lagged == 0.0), "lag=1 不该看到最后一期的值")

        now = np.sum(np.column_stack(
            peer_columns(source, t, a, np.ones(len(t), bool), PAIRS, lag=0)), axis=1)
        self.assertEqual(float(now[last & (a == 0)][0]), 7.0, "lag=0 应当看到当期值")

    def test_lag_one_reads_exactly_the_previous_sampled_time_id(self) -> None:
        t, a, _ = panel(n_times=5)
        times = np.unique(t)
        source = np.zeros(len(t), dtype=np.float64)
        source[(t == times[2]) & (a == PAIRS[0])] = 3.5
        merged = np.sum(np.column_stack(
            peer_columns(source, t, a, np.ones(len(t), bool), PAIRS, lag=1)), axis=1)
        hit = merged[(t == times[3]) & (a == 0)]
        self.assertEqual(float(hit[0]), 3.5)
        # 除那一行外全为 0 —— 值不该扩散到别的期或别的资产
        self.assertEqual(int(np.count_nonzero(merged)), 1)

    def test_eval_mask_selects_rows_after_the_shift_not_before(self) -> None:
        """先在全网格 shift、再切评估行；反过来做会让折边界的「上一期」跨过整段训练区。"""
        t, a, y = panel(n_times=6)
        e = cross_sectional_residual(y, t)
        times = np.unique(t)
        mask = np.isin(t, times[3:])                       # 只评估后 3 期
        full = np.sum(np.column_stack(
            peer_columns(e, t, a, np.ones(len(t), bool), PAIRS, lag=1)), axis=1)
        sub = np.sum(np.column_stack(
            peer_columns(e, t, a, mask, PAIRS, lag=1)), axis=1)
        np.testing.assert_allclose(sub, full[mask], rtol=0, atol=0)


class NegativeControlTest(unittest.TestCase):
    def test_targets_match_but_partners_are_disjoint(self) -> None:
        self.assertEqual(set(NEG_PAIRS), set(PAIRS),
                         "目标资产集合必须相同，否则非零行不同、对照不干净")
        self.assertFalse(set(NEG_PAIRS.values()) & set(PAIRS),
                         "阴性搭档不得是任何真对子的成员")
        for asset in PAIRS:
            self.assertNotEqual(NEG_PAIRS[asset], PAIRS[asset])

    def test_negative_control_produces_the_same_nonzero_rows(self) -> None:
        t, a, y = panel()
        e = cross_sectional_residual(y, t)
        ones = np.ones(len(t), bool)
        real = np.column_stack(peer_columns(e, t, a, ones, PAIRS, lag=1))
        neg = np.column_stack(peer_columns(e, t, a, ones, NEG_PAIRS, lag=1))
        # 「哪些行有值」必须一致；只有「值是谁的」不同 ⟹ 对照掉了「多一列」的效应
        np.testing.assert_array_equal(real != 0.0, neg != 0.0)


class CrossSectionalResidualTest(unittest.TestCase):
    def test_residual_is_zero_mean_within_each_time_id(self) -> None:
        t, _, y = panel()
        e = cross_sectional_residual(y, t)
        for tid in np.unique(t):
            self.assertAlmostEqual(float(e[t == tid].mean()), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
