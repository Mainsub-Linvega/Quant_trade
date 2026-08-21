"""长窗列数阶梯的数学与判定逻辑（不依赖数据的部分）。

要害是 `trailing_mean_and_deviation` 的**严格滞后**：含当前行就是泄漏，
整个实验的结论会反过来。
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

import long_window_ladder as ladder  # noqa: E402


def _naive_trailing(series: np.ndarray, window: int) -> np.ndarray:
    """最朴素的逐行实现，只用来对拍向量化版本。"""
    out = np.zeros_like(series, dtype=np.float64)
    for t in range(len(series)):
        if t == 0:
            continue
        out[t] = series[max(0, t - window):t].mean(axis=0)
    return out


class TrailingMeanTest(unittest.TestCase):
    def test_matches_naive_loop(self):
        rng = np.random.default_rng(1)
        series = rng.standard_normal((300, 4))
        for window in (1, 3, 64, 512):
            mean, deviation = ladder.trailing_mean_and_deviation(series, window)
            np.testing.assert_allclose(mean, _naive_trailing(series, window), atol=1e-10)
            np.testing.assert_allclose(deviation, series - mean, atol=1e-12)

    def test_current_row_is_excluded(self):
        """含当前行就是泄漏 —— 这一条错了整个实验的结论会反过来。"""
        series = np.arange(10, dtype=np.float64).reshape(10, 1)
        mean, _ = ladder.trailing_mean_and_deviation(series, 3)
        self.assertEqual(mean[0, 0], 0.0)                     # 无历史 ⟹ 0
        self.assertAlmostEqual(mean[1, 0], 0.0)               # 只有 v[0]=0
        self.assertAlmostEqual(mean[5, 0], np.mean([2, 3, 4]))
        self.assertAlmostEqual(mean[9, 0], np.mean([6, 7, 8]))

    def test_window_larger_than_series_uses_all_history(self):
        series = np.arange(5, dtype=np.float64).reshape(5, 1)
        mean, _ = ladder.trailing_mean_and_deviation(series, 10_000)
        self.assertAlmostEqual(mean[4, 0], np.mean([0, 1, 2, 3]))

    def test_columns_are_independent(self):
        rng = np.random.default_rng(2)
        series = rng.standard_normal((80, 3))
        joint, _ = ladder.trailing_mean_and_deviation(series, 7)
        for col in range(3):
            single, _ = ladder.trailing_mean_and_deviation(series[:, [col]], 7)
            np.testing.assert_allclose(joint[:, [col]], single, atol=1e-12)


class WideLadderTest(unittest.TestCase):
    def test_ladder_is_independent_of_the_frozen_one(self):
        """⚠️ 绝不能就地改 function_class_probe.ALPHA_LADDER —— 它被已结案实验 import。"""
        import function_class_probe as frozen
        self.assertEqual(frozen.ALPHA_LADDER, (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
                         "旧梯子被改动了；两个已结案实验的复跑结果会与落盘产物对不上")
        self.assertGreater(len(ladder.WIDE_ALPHA_LADDER), len(frozen.ALPHA_LADDER))
        self.assertLess(ladder.WIDE_ALPHA_LADDER[0], frozen.ALPHA_LADDER[0])
        self.assertGreater(ladder.WIDE_ALPHA_LADDER[-1], frozen.ALPHA_LADDER[-1])

    def test_boundary_flag_is_consistent_with_the_chosen_alpha(self):
        rng = np.random.default_rng(9)
        n, p = 4000, 20
        design = rng.standard_normal((n, p)).astype(np.float32)
        label = design @ rng.standard_normal(p) + 0.05 * rng.standard_normal(n)
        weight = np.ones(n)
        tid = np.repeat(np.arange(n // 4), 4)
        _, info = ladder.fit_linear_wide(design, design[:100], label, weight, tid)
        self.assertIn(info["alpha_relative"], ladder.WIDE_ALPHA_LADDER)
        expected = info["alpha_relative"] in (ladder.WIDE_ALPHA_LADDER[0],
                                              ladder.WIDE_ALPHA_LADDER[-1])
        self.assertEqual(info["alpha_at_boundary"], expected)
        self.assertEqual(len(info["alpha_trace"]), len(ladder.WIDE_ALPHA_LADDER))

    def test_high_snr_prefers_light_regularisation(self):
        rng = np.random.default_rng(10)
        n, p = 20_000, 10
        design = rng.standard_normal((n, p)).astype(np.float32)
        label = design @ rng.standard_normal(p) + 0.01 * rng.standard_normal(n)
        tid = np.repeat(np.arange(n // 4), 4)
        _, info = ladder.fit_linear_wide(design, design[:50], label, np.ones(n), tid)
        self.assertLess(info["alpha_relative"], 1e-2, "高信噪下不该选重正则")


class GateTest(unittest.TestCase):
    @staticmethod
    def _rows(tree, linear=None, boundary=False):
        linear = linear or tree
        return [{"fold": i, "n_valid": 1000,
                 "arms": {n: {"peak": tree[n][i], "A": tree[n][i], "B": 1.0,
                              "n_columns": 361 if n == "base" else 441,
                              "linear_peak": linear[n][i],
                              "linear_alpha_relative": 1e-3,
                              "linear_alpha_at_boundary": boundary}
                          for n in ladder.ARMS}}
                for i in range(5)]

    def test_clear_winner_passes(self):
        rows = self._rows({"base": [1.0] * 5, "w64": [1.0] * 5,
                           "w512": [1.10] * 5, "w4096": [0.95] * 5})
        report = ladder.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertEqual(report["passing_arms"], ["w512"])
        self.assertEqual(report["verdict"], "PASS")
        self.assertTrue(report["summary"]["tree"]["w512"]["exceeds_detection_floor"])

    def test_all_negative_is_rejected(self):
        rows = self._rows({"base": [1.0] * 5, "w64": [0.98] * 5,
                           "w512": [0.97] * 5, "w4096": [0.90] * 5})
        report = ladder.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertEqual(report["passing_arms"], [])

    def test_linear_reading_is_invalidated_when_alpha_hits_a_boundary(self):
        rows = self._rows({"base": [1.0] * 5, "w64": [1.2] * 5,
                           "w512": [1.0] * 5, "w4096": [1.0] * 5}, boundary=True)
        report = ladder.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertFalse(report["summary"]["linear"]["w64"]["reading_valid"])
        # 树是主判据，不受线性撞端影响
        self.assertTrue(report["summary"]["tree"]["w64"]["passed"])

    def test_tree_and_linear_are_reported_separately(self):
        rows = self._rows({"base": [1.0] * 5, "w64": [1.2] * 5, "w512": [1.0] * 5,
                           "w4096": [1.0] * 5},
                          linear={"base": [1.0] * 5, "w64": [0.8] * 5, "w512": [1.0] * 5,
                                  "w4096": [1.0] * 5})
        report = ladder.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertGreater(report["summary"]["tree"]["w64"]["pooled_gain"], 0)
        self.assertLess(report["summary"]["linear"]["w64"]["pooled_gain"], 0)
        self.assertEqual(report["passing_arms"], ["w64"], "裁决只看树")


if __name__ == "__main__":
    unittest.main()
