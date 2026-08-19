from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "strategies" / "v4_mlp")]

from mlp_numpy import NumpyMLP
from multitask_mlp import AUX_LADDER, AUX_LAMBDA, build_multitask_targets, standardize
from target_mlp_oracle_blend import oracle_two_component_peak


class MultitaskTargetLayoutTest(unittest.TestCase):
    """辅助目标矩阵的排布。第 0 列的身份是**推理契约**：只有它会被拿去当预测。"""

    def _fixture(self, rows: int = 512, aux: int = 3, seed: int = 7):
        rng = np.random.default_rng(seed)
        e = rng.normal(0.0, 0.02, rows)
        responders = rng.normal(0.5, 0.3, (rows, aux))
        return e, responders, np.ones(rows)

    def test_target_is_column_zero_and_standardised(self) -> None:
        e, responders, weight = self._fixture()
        targets, mean, std = build_multitask_targets(e, responders, weight, AUX_LAMBDA)
        self.assertEqual(targets.shape, (len(e), 1 + responders.shape[1]))
        expected, expected_mean, expected_std = standardize(e, weight)
        np.testing.assert_allclose(targets[:, 0], expected, rtol=0, atol=0)
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(std, expected_std)
        # 反标准化必须能还原 e —— 推理端就是这么取回预测的
        np.testing.assert_allclose(targets[:, 0] * std + mean, e, rtol=1e-12, atol=1e-12)

    def test_auxiliary_columns_carry_sqrt_lambda(self) -> None:
        """辅助列的尺度必须恰好是 √λ —— λ 就是靠这个实现的，不是靠别的地方。"""
        e, responders, weight = self._fixture()
        targets, _, _ = build_multitask_targets(e, responders, weight, AUX_LAMBDA)
        for index in range(responders.shape[1]):
            standardised, _, _ = standardize(responders[:, index].astype(np.float64), weight)
            np.testing.assert_allclose(targets[:, 1 + index],
                                       np.sqrt(AUX_LAMBDA) * standardised, rtol=1e-12, atol=1e-12)
            # 标准化后方差为 1 ⟹ 该列的能量恰好是 λ
            self.assertAlmostEqual(float(np.mean(targets[:, 1 + index] ** 2)), AUX_LAMBDA, places=6)

    def test_ladder_is_the_pre_registered_five(self) -> None:
        """辅助目标集是预注册的，改动必须是有意的，不能顺手加一个。"""
        self.assertEqual(AUX_LADDER,
                         ("responder_00", "responder_02", "responder_03",
                          "responder_04", "responder_05"))
        self.assertEqual(AUX_LAMBDA, 0.3)


class SqrtLambdaEquivalenceTest(unittest.TestCase):
    """`√λ 缩放目标列 ⟺ 给该头损失权重 λ`。

    这是整条 B 线成立的前提：不自写训练循环，靠 sklearn 的多输出 MSE 实现加权多任务。
    严格等价只在**无 L2 罚**（alpha=0）时成立 —— alpha>0 会因输出层权重被缩放而引入
    二阶差异（脚本 docstring 里写明了，两个臂用同一个 alpha 所以比较仍然干净）。
    这里在闭式可解的线性最小二乘上验证，不依赖 SGD 的随机性。
    """

    def test_scaling_target_scales_loss_weight(self) -> None:
        rng = np.random.default_rng(11)
        rows, features = 400, 6
        design = rng.normal(size=(rows, features))
        y_main = design @ rng.normal(size=features) + 0.1 * rng.normal(size=rows)
        y_aux = design @ rng.normal(size=features) + 0.1 * rng.normal(size=rows)
        lam = AUX_LAMBDA
        root = np.sqrt(lam)

        # (a) 多输出最小二乘，辅助列乘 √λ：各输出独立求解 ⟹ 系数应恰好是 √λ 倍
        stacked = np.column_stack([y_main, root * y_aux])
        coef_stacked, *_ = np.linalg.lstsq(design, stacked, rcond=None)
        coef_aux_plain, *_ = np.linalg.lstsq(design, y_aux, rcond=None)
        np.testing.assert_allclose(coef_stacked[:, 1], root * coef_aux_plain, rtol=1e-9, atol=1e-12)

        # (b) 该列对总损失的贡献 = λ × 该头自身的 MSE
        residual_scaled = root * y_aux - design @ coef_stacked[:, 1]
        residual_plain = y_aux - design @ coef_aux_plain
        np.testing.assert_allclose(float(np.sum(residual_scaled ** 2)),
                                   lam * float(np.sum(residual_plain ** 2)),
                                   rtol=1e-9, atol=1e-12)

        # (c) 主头不受影响（输出层各列独立）
        coef_main_plain, *_ = np.linalg.lstsq(design, y_main, rcond=None)
        np.testing.assert_allclose(coef_stacked[:, 0], coef_main_plain, rtol=1e-9, atol=1e-12)


class MultiOutputNumpyParityTest(unittest.TestCase):
    """多输出 `NumpyMLP` 与 sklearn 的前向对拍 —— 推理侧只取第 0 列，必须逐位可信。"""

    def test_multi_output_forward_matches(self) -> None:
        rng = np.random.default_rng(3)
        coefs = [rng.normal(size=(8, 5)), rng.normal(size=(5, 1 + len(AUX_LADDER)))]
        intercepts = [rng.normal(size=5), rng.normal(size=1 + len(AUX_LADDER))]
        model = NumpyMLP(coefs, intercepts)
        design = rng.normal(size=(64, 8))
        expected = np.maximum(design @ coefs[0] + intercepts[0], 0.0) @ coefs[1] + intercepts[1]
        np.testing.assert_allclose(model.predict(design), expected, rtol=0, atol=0)
        self.assertEqual(model.predict(design).shape, (64, 1 + len(AUX_LADDER)))

    def test_single_output_still_flattens(self) -> None:
        """单输出（target_only 对照臂）必须仍然返回一维 —— 两个臂共用同一段取值代码。"""
        rng = np.random.default_rng(4)
        model = NumpyMLP([rng.normal(size=(8, 5)), rng.normal(size=(5, 1))],
                         [rng.normal(size=5), rng.normal(size=1)])
        self.assertEqual(model.predict(rng.normal(size=(64, 8))).shape, (64,))


class OracleBlendClosedFormTest(unittest.TestCase):
    """两分量最优配比的闭式解 —— Stage 1 的判据直接建在它上面。"""

    def test_matches_brute_force_and_dominates_single_components(self) -> None:
        rng = np.random.default_rng(5)
        rows = 2000
        y = rng.normal(size=rows)
        base = 0.3 * y + rng.normal(scale=0.5, size=rows)
        extra = 0.1 * y + rng.normal(scale=0.5, size=rows)
        weight = rng.uniform(0.5, 1.5, rows)
        denominator = float(np.dot(weight, y * y))

        def moments(vector):
            return (float(np.dot(weight, y * vector)) / denominator,
                    float(np.dot(weight, vector * vector)) / denominator)

        a_b, b_b = moments(base)
        a_m, b_m = moments(extra)
        cross = float(np.dot(weight * base, extra)) / denominator
        closed = oracle_two_component_peak(a_b, b_b, a_m, b_m, cross)

        # 直接解 (c1,c2) 再算 peak，应与闭式一致
        gram = np.array([[np.dot(weight * base, base), np.dot(weight * base, extra)],
                         [np.dot(weight * base, extra), np.dot(weight * extra, extra)]])
        rhs = np.array([np.dot(weight * y, base), np.dot(weight * y, extra)])
        c1, c2 = np.linalg.solve(gram, rhs)
        blended = c1 * base + c2 * extra
        a_e = float(np.dot(weight, y * blended)) / denominator
        b_e = float(np.dot(weight, blended * blended)) / denominator
        self.assertAlmostEqual(closed, a_e * a_e / b_e, places=12)
        # 两分量至少能退化成单分量 ⟹ 必须不低于两者
        self.assertGreaterEqual(closed, max(a_b * a_b / b_b, a_m * a_m / b_m) - 1e-15)

    def test_rejects_degenerate_gram(self) -> None:
        with self.assertRaises(ValueError):
            oracle_two_component_peak(1.0, 1.0, 1.0, 1.0, 1.0)   # 完全共线 ⟹ det=0


if __name__ == "__main__":
    unittest.main()
