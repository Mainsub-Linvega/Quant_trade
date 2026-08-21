"""钉住 `select_features` 的行为 —— 它有 77 个调用点却一个测试都没有。

2026-08-21 查出：`strategies/v1_ridge/train.py:select_features` 是研究代码里被依赖
最多的函数（实验脚本 + 生产训练共 77 处调用），但没有任何用例钉住它的口径。
这些用例只覆盖**契约**（返回有序、count 截断、权重语义、|corr| 双向、零方差列），
不覆盖「它是不是好的选择器」—— 那是 `selection_criterion_probe` 的事。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train import select_features  # noqa: E402


def _frame(n: int = 4000, p: int = 12, seed: int = 0):
    """列 3 正相关、列 7 负相关，其余是噪声。`p` 必须 >= 8。"""
    if p < 8:
        raise ValueError("_frame 的信号列写在 3 与 7 上，p 必须 >= 8")
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, p)).astype(np.float32)
    target = features[:, 3] * 0.5 - features[:, 7] * 0.3 + rng.standard_normal(n)
    return features, target.astype(np.float64), np.ones(n)


class SelectFeaturesContractTest(unittest.TestCase):
    def test_returns_sorted_original_column_order(self):
        """返回值必须升序 —— 下游用 np.searchsorted 定位，乱序会静默错位。"""
        features, target, weight = _frame()
        selected = select_features(features, target, weight, 5)
        self.assertEqual(list(selected), sorted(selected))
        self.assertEqual(len(set(selected.tolist())), 5, "不得有重复列")

    def test_count_is_clamped_to_available_columns(self):
        features, target, weight = _frame(p=8)
        self.assertEqual(len(select_features(features, target, weight, 99)), 8)

    def test_count_below_one_is_clamped_to_one(self):
        features, target, weight = _frame()
        self.assertEqual(len(select_features(features, target, weight, 0)), 1)
        self.assertEqual(len(select_features(features, target, weight, -3)), 1)

    def test_picks_the_signal_carrying_columns(self):
        features, target, weight = _frame()
        selected = set(select_features(features, target, weight, 2).tolist())
        self.assertEqual(selected, {3, 7}, "两个真信号列必须排在最前")

    def test_ranks_by_absolute_correlation_so_sign_does_not_matter(self):
        """列 7 是**负**相关。若判据漏了 abs，它会掉到最后。"""
        features, target, weight = _frame()
        top = select_features(features, target, weight, 2)
        self.assertIn(7, top.tolist())

    def test_zero_weight_rows_are_equivalent_to_dropping_them(self):
        """权重是加权矩，给 0 权必须与整行删掉等价 —— 生产 ridge 选列就靠这条。"""
        features, target, _ = _frame(n=3000, seed=5)
        keep = np.zeros(len(target), dtype=bool)
        keep[:1500] = True
        weight = keep.astype(np.float64)
        with_zero = select_features(features, target, weight, 4)
        dropped = select_features(features[keep], target[keep], np.ones(int(keep.sum())), 4)
        np.testing.assert_array_equal(with_zero, dropped)

    def test_constant_column_does_not_crash_and_ranks_last(self):
        features, target, weight = _frame(p=8)
        features[:, 2] = 3.14                       # 零方差
        selected = select_features(features, target, weight, 7).tolist()
        self.assertNotIn(2, selected, "零方差列不该进 top-7")
        self.assertTrue(np.all(np.isfinite(features[:, 2])))

    def test_is_deterministic(self):
        features, target, weight = _frame()
        first = select_features(features, target, weight, 6)
        second = select_features(features, target, weight, 6)
        np.testing.assert_array_equal(first, second)

    def test_does_not_mutate_inputs(self):
        features, target, weight = _frame()
        f0, t0, w0 = features.copy(), target.copy(), weight.copy()
        select_features(features, target, weight, 5)
        np.testing.assert_array_equal(features, f0)
        np.testing.assert_array_equal(target, t0)
        np.testing.assert_array_equal(weight, w0)


if __name__ == "__main__":
    unittest.main()
