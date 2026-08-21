"""确认档的裁决分支 —— 尤其是「五道全过但低于检出下限」这一档。

筛选档给的是 +6.80%，而 3s480 的检出下限是 **8.7%**（比筛选档的 6.1% 更高）。
⟹ 最可能出现的就是 `PASS_BUT_BELOW_DETECTION_FLOOR`，这一档必须能被干净地区分出来，
否则很容易被误读成「过了就能晋级」。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import long_window_confirm as confirm  # noqa: E402


def _rows(base, arm):
    return [{"fold": i, "n_valid": 1000,
             "arms": {"base": {"peak": base[i], "A": base[i], "B": 1.0, "n_columns": 361,
                               "linear_peak": 1.0, "linear_matches_ladder": True,
                               "linear_alpha_relative": 1e-3, "linear_alpha_at_boundary": False},
                      "w512": {"peak": arm[i], "A": arm[i], "B": 1.0, "n_columns": 441,
                               "linear_peak": 1.0, "linear_matches_ladder": True,
                               "linear_alpha_relative": 1e-3, "linear_alpha_at_boundary": False}}}
            for i in range(5)]


_ARGS = type("A", (), {"stage1": False})()


class VerdictTest(unittest.TestCase):
    def test_detection_floor_is_the_3s480_value_not_the_1s160_one(self):
        """8.7% 而不是 6.1% —— 这一条防的是从筛选档脚本复制粘贴时漏改。"""
        self.assertAlmostEqual(confirm.DETECTION_FLOOR, 0.087)
        self.assertEqual(confirm.NUM_ITERATION, 480)
        self.assertEqual(confirm.N_SEEDS, 3)

    def test_pass_requires_both_gates_and_floor(self):
        report = confirm.build_report(_rows([1.0] * 5, [1.12] * 5), "sha", _ARGS, [], 1.0)
        self.assertTrue(report["summary"]["passed"])
        self.assertTrue(report["summary"]["exceeds_detection_floor"])
        self.assertEqual(report["verdict"], "PASS")

    def test_gates_pass_but_below_floor_is_its_own_verdict(self):
        """+5% ⟹ 五道过、但 < 8.7% ⟹ 方向可信、幅度测不出，不构成晋级依据。"""
        report = confirm.build_report(_rows([1.0] * 5, [1.05] * 5), "sha", _ARGS, [], 1.0)
        self.assertTrue(report["summary"]["passed"])
        self.assertFalse(report["summary"]["exceeds_detection_floor"])
        self.assertEqual(report["verdict"], "PASS_BUT_BELOW_DETECTION_FLOOR")
        self.assertLess(report["summary"]["floor_multiple"], 1.0)

    def test_failing_gates_is_rejected(self):
        report = confirm.build_report(_rows([1.0] * 5, [1.01] * 5), "sha", _ARGS, [], 1.0)
        self.assertFalse(report["summary"]["passed"])
        self.assertEqual(report["verdict"], "REJECTED")

    def test_one_lucky_fold_fails_drop_best(self):
        report = confirm.build_report(
            _rows([1.0] * 5, [1.9, 0.99, 0.99, 0.99, 0.99]), "sha", _ARGS, [], 1.0)
        self.assertFalse(report["summary"]["gates"]["3_survives_drop_best_fold"])
        self.assertEqual(report["verdict"], "REJECTED")

    def test_linear_mismatch_invalidates_everything(self):
        """线性不依赖树超参；对不上说明数据路径变了，树的读数也不可信。"""
        report = confirm.build_report(_rows([1.0] * 5, [1.20] * 5), "sha", _ARGS,
                                      ["fold 2/w512: 线性 peak 不一致"], 1.0)
        self.assertTrue(report["summary"]["passed"])          # 门槛本身是过的
        self.assertEqual(report["verdict"], "INVALID_LINEAR_CROSSCHECK_FAILED")

    def test_only_two_arms_are_declared(self):
        """w64 / w4096 两个评价器不一致，不得带上确认档（多重比较捞鱼）。"""
        self.assertEqual(confirm.ARMS, ("base", "w512"))
        self.assertEqual(confirm.WINDOW, 512)


if __name__ == "__main__":
    unittest.main()
