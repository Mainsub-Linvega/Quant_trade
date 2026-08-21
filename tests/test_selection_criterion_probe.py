"""选列准则探针的数学与判定逻辑。

只测**不依赖数据**的部分。最要紧的是 `test_union_then_slice_matches_per_arm` ——
探针为了省 4 倍的流式扫描，把四个臂的 history 列取并集只算一次再切片，
这条等价性必须被钉死，否则四个臂比的就不是同一个东西。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
if str(_REPO_ROOT / "strategies" / "v3_hybrid") not in sys.path:
    sys.path.append(str(_REPO_ROOT / "strategies" / "v3_hybrid"))

import selection_criterion_probe as probe  # noqa: E402


class LaggedAndRollingTest(unittest.TestCase):
    def test_lag1_is_strictly_lagged_per_asset_with_zero_at_first_row(self):
        values = np.arange(12, dtype=np.float32).reshape(6, 2)
        assets = np.array([0, 1, 0, 1, 0, 1])
        lag1, _ = probe.lagged_and_rolling(values, assets)
        np.testing.assert_array_equal(lag1[0], [0.0, 0.0])      # asset 0 首行无历史
        np.testing.assert_array_equal(lag1[1], [0.0, 0.0])      # asset 1 首行无历史
        np.testing.assert_array_equal(lag1[2], values[0])       # asset 0 的上一次观测
        np.testing.assert_array_equal(lag1[3], values[1])
        np.testing.assert_array_equal(lag1[4], values[2])

    def test_rolling_mean_excludes_the_current_row(self):
        """严格滞后：含当前行就是泄漏。"""
        values = np.arange(8, dtype=np.float32).reshape(8, 1)
        assets = np.zeros(8, dtype=np.int64)
        _, roll = probe.lagged_and_rolling(values, assets)
        self.assertEqual(roll[0, 0], 0.0)                        # 无历史 ⟹ 0
        self.assertAlmostEqual(roll[1, 0], 0.0, places=6)        # 只有 v[0]=0
        self.assertAlmostEqual(roll[3, 0], np.mean([0, 1, 2]), places=6)
        self.assertAlmostEqual(roll[7, 0], np.mean([2, 3, 4, 5, 6]), places=6)  # 窗口 5

    def test_assets_do_not_leak_into_each_other(self):
        values = np.array([[1.0], [100.0], [2.0], [200.0]], dtype=np.float32)
        assets = np.array([0, 1, 0, 1])
        lag1, _ = probe.lagged_and_rolling(values, assets)
        self.assertEqual(lag1[2, 0], 1.0)                        # asset 0 只看 asset 0
        self.assertEqual(lag1[3, 0], 100.0)


class UnionSliceEquivalenceTest(unittest.TestCase):
    def test_union_then_slice_matches_per_arm(self):
        """AssetHistory 的四个块逐列独立 ⟹ 算并集再切片必须与单独算逐位相同。

        探针靠这条把每折的流式扫描从 4 次降到 1 次。不成立的话，四个臂比的
        就不是同一个东西。
        """
        from history import AssetHistory

        rng = np.random.default_rng(7)
        rows, width, window = 400, 9, 5
        data = rng.standard_normal((rows, width)).astype(np.float32)
        assets = rng.integers(0, 4, size=rows)

        union_blocks = AssetHistory(feature_count=width, window_size=window).transform(data, assets)
        subset = np.array([1, 4, 7])
        sliced = [block[:, subset] for block in union_blocks]

        direct = AssetHistory(feature_count=len(subset), window_size=window).transform(
            np.ascontiguousarray(data[:, subset]), assets)
        for got, want in zip(sliced, direct):
            np.testing.assert_array_equal(got, want)


class LassoSelectTest(unittest.TestCase):
    def test_returns_exactly_count_sorted_indices(self):
        rng = np.random.default_rng(3)
        n, p = 6000, 40
        features = rng.standard_normal((n, p)).astype(np.float32)
        label = features[:, [2, 9, 17]] @ np.array([1.0, -0.8, 0.6]) + rng.standard_normal(n)
        selected = probe.lasso_select(features, label, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(list(selected), sorted(selected))
        self.assertEqual(len(set(selected.tolist())), 8)

    def test_recovers_the_true_support_first(self):
        rng = np.random.default_rng(4)
        n, p = 8000, 30
        features = rng.standard_normal((n, p)).astype(np.float32)
        label = features[:, [5, 11]] @ np.array([2.0, -2.0]) + 0.2 * rng.standard_normal(n)
        self.assertEqual(set(probe.lasso_select(features, label, 2).tolist()), {5, 11})

    def test_drops_a_redundant_collinear_copy(self):
        """这正是用户指出的单变量筛子的失败模式：共线的副本会被重复选进来。"""
        rng = np.random.default_rng(6)
        n = 8000
        signal = rng.standard_normal(n)
        noise_cols = rng.standard_normal((n, 6))
        features = np.column_stack([signal, signal + 0.01 * rng.standard_normal(n),
                                    noise_cols]).astype(np.float32)
        label = 2.0 * signal + 0.3 * noise_cols[:, 0] + 0.5 * rng.standard_normal(n)
        selected = set(probe.lasso_select(features, label, 2).tolist())
        self.assertNotEqual(selected, {0, 1}, "LASSO 不该同时选中两个几乎相同的列")
        self.assertIn(2, selected, "应当选中那个独立贡献的列（noise_cols[:,0]）")


class PeakTest(unittest.TestCase):
    def test_peak_is_invariant_to_prediction_scaling(self):
        """整个探针只读 peak，就是因为它对缩放严格不变（ΔA/ΔB 不是）。"""
        rng = np.random.default_rng(8)
        n = 5000
        label = rng.standard_normal(n)
        pred = 0.05 * label + rng.standard_normal(n)
        weight = rng.uniform(0.5, 2.0, n)
        base, _, _ = probe.peak_of(label, pred, weight)
        for scale in (0.1, 3.7, 1000.0):
            scaled, a, b = probe.peak_of(label, scale * pred, weight)
            self.assertAlmostEqual(scaled, base, places=12)
        # 而 A、B 都随缩放变 —— 这就是 2ΔA>ΔB 被剔除的原因
        _, a1, b1 = probe.peak_of(label, pred, weight)
        _, a2, b2 = probe.peak_of(label, 2.0 * pred, weight)
        self.assertAlmostEqual(a2 / a1, 2.0, places=9)
        self.assertAlmostEqual(b2 / b1, 4.0, places=9)


class GateTest(unittest.TestCase):
    @staticmethod
    def _rows(peaks_by_arm):
        return [{"fold": i, "n_valid": 1000, "xs_overlap_uni_lasso": 150,
                 "arms": {n: {"peak": p[i], "A": p[i], "B": 1.0,
                              "history_overlap_with_base": 40}
                          for n, p in peaks_by_arm.items()}}
                for i in range(5)]

    def test_clear_winner_passes_and_is_flagged_above_floor(self):
        rows = self._rows({"base": [1.0] * 5, "hist_lag1": [1.10] * 5,
                           "hist_roll5": [1.0] * 5, "lasso200": [0.9] * 5})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertTrue(report["summary"]["hist_lag1"]["passed"])
        self.assertTrue(report["summary"]["hist_lag1"]["exceeds_detection_floor"])
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["passing_arms"], ["hist_lag1"])

    def test_gain_between_three_and_six_percent_passes_gates_but_not_the_floor(self):
        """3%~6.1% 这段必须能被区分出来 —— 它只够花公榜额度，不够直接晋级。"""
        rows = self._rows({"base": [1.0] * 5, "hist_lag1": [1.04] * 5,
                           "hist_roll5": [1.0] * 5, "lasso200": [1.0] * 5})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        s = report["summary"]["hist_lag1"]
        self.assertTrue(s["passed"])
        self.assertFalse(s["exceeds_detection_floor"])

    def test_one_lucky_fold_fails_drop_best(self):
        rows = self._rows({"base": [1.0] * 5, "hist_lag1": [1.6, 0.99, 0.99, 0.99, 0.99],
                           "hist_roll5": [1.0] * 5, "lasso200": [1.0] * 5})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertFalse(report["summary"]["hist_lag1"]["gates"]["3_survives_drop_best_fold"])
        self.assertEqual(report["verdict"], "REJECTED")

    def test_delta_a_and_b_are_reported_but_never_gate(self):
        rows = self._rows({"base": [1.0] * 5, "hist_lag1": [1.10] * 5,
                           "hist_roll5": [1.0] * 5, "lasso200": [1.0] * 5})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        gates = report["summary"]["hist_lag1"]["gates"]
        self.assertNotIn("2_two_delta_A_exceeds_delta_B", gates)
        # 判据里允许出现 Δ**peak**（尺度不变），但绝不能出现 ΔA / ΔB
        self.assertTrue(all("delta_A" not in key and "delta_B" not in key
                            and "_A_" not in key and "_B_" not in key for key in gates),
                        f"判据里混进了尺度相关量：{list(gates)}")
        self.assertIn("delta_A_not_a_gate", report["summary"]["hist_lag1"])


if __name__ == "__main__":
    unittest.main()
