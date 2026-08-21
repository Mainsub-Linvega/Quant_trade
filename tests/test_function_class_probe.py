"""函数类探针的数学与判定逻辑。

这些用例只测**不依赖数据**的部分：分块累加是否与整块一致、集成增益闭式是否
与暴力最优配比一致、零增益边界是否恰好落在 r=ρ、以及五道门槛的机判。
数据侧的对齐由脚本运行时的断言负责（自算 e_va 必须与 cache 的 e_target 一致）。
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

import function_class_probe as probe  # noqa: E402


class TestBlendAlgebra(unittest.TestCase):
    def test_zero_gain_boundary_is_exactly_r_equals_rho(self):
        """CLAUDE.md §8.6「低相关不代表适合集成」的精确形式。"""
        for rho in (0.0, 0.3, 0.6, 0.9):
            self.assertAlmostEqual(probe.blend_gain_ic(rho, rho), 0.0, places=12,
                                   msg=f"r=rho={rho} 应该恰好零增益")

    def test_matches_brute_force_optimal_blend(self):
        """闭式增益必须等于暴力搜出来的两分量最优配比增益。"""
        rng = np.random.default_rng(11)
        n = 40_000
        y = rng.standard_normal(n)
        p1 = 0.06 * y + rng.standard_normal(n)
        p2 = 0.04 * y + 0.7 * p1 + rng.standard_normal(n)
        w = rng.uniform(0.5, 2.0, n)
        ic1, *_ = probe.weighted_ic(y, p1, w)
        ic2, *_ = probe.weighted_ic(y, p2, w)
        rho = probe.weighted_corr(p1, p2, w)
        closed = probe.blend_gain_ic(ic2 / ic1, rho)
        best = max(probe.weighted_ic(y, p1 + t * p2, w)[0]
                   for t in np.linspace(-4.0, 4.0, 20_001))
        self.assertAlmostEqual(closed, best / ic1 - 1.0, places=4)

    def test_gain_is_non_negative_and_symmetric_about_r_equals_rho(self):
        """恒等式 (1+r²−2ρr)/(1−ρ²) = 1 + (r−ρ)²/(1−ρ²)。

        ⟹ oracle 最优配比**永远不会比单分量差**（第二个权重取 0 即可），
        增益只取决于 |r−ρ|。这一条推翻了「r<ρ 会掉分」的直觉，也正是判据 1
        必须**单边**的原因：两侧都为正，但 r<ρ 那侧靠的是负权重减方差（ΔB），
        本项目已多次证明该机制不迁移。
        """
        for r, rho in ((0.3, 0.7), (0.5, 0.7), (0.9, 0.7), (1.2, 0.7)):
            self.assertGreaterEqual(probe.blend_gain_ic(r, rho), 0.0)
        for delta in (0.1, 0.25, 0.4):
            self.assertAlmostEqual(probe.blend_gain_ic(0.6 + delta, 0.6),
                                   probe.blend_gain_ic(0.6 - delta, 0.6), places=12)
        self.assertGreater(probe.blend_gain_ic(1.0, 0.6), 0.1)


class TestRffGram(unittest.TestCase):
    def test_chunked_accumulation_matches_whole_array(self):
        rng = np.random.default_rng(3)
        n, d, dim = 5_000, 12, 64
        design = rng.standard_normal((n, d)).astype(np.float32)
        weight = rng.uniform(0.5, 2.0, n)
        label = rng.standard_normal(n)
        proj = rng.standard_normal((d, dim)).astype(np.float32)
        phase = rng.uniform(0, 2 * np.pi, dim).astype(np.float32)

        original = probe.GRAM_CHUNK
        try:
            probe.GRAM_CHUNK = 137          # 不整除 n，逼出边界 bug
            gram, rhs, dss = probe.rff_gram(design, weight, label, proj, phase)
        finally:
            probe.GRAM_CHUNK = original
        z = np.cos(design @ proj + phase, dtype=np.float32).astype(np.float64) * np.sqrt(2.0 / dim)
        # 容差按**相对**量给：设计矩阵是 float32，分块与整块的 BLAS 分块方式不同，
        # `design @ proj` 会差最后一个 ulp，cos 之后放大到约 1e-7 绝对。下游全是
        # 岭回归（光滑），不像树那样一个 ulp 会翻叶子 ⟹ 1e-6 相对足够严。
        reference = (z * weight[:, None]).T @ z
        self.assertLess(np.abs(gram - reference).max() / np.abs(reference).max(), 1e-6)
        rhs_reference = (z * weight[:, None]).T @ label
        self.assertLess(np.abs(rhs - rhs_reference).max() / np.abs(rhs_reference).max(), 1e-6)
        self.assertAlmostEqual(dss, float(np.dot(weight * label, label)), places=9)

    def test_predict_matches_direct_evaluation(self):
        rng = np.random.default_rng(4)
        n, d, dim = 2_000, 8, 32
        design = rng.standard_normal((n, d)).astype(np.float32)
        proj = rng.standard_normal((d, dim)).astype(np.float32)
        phase = rng.uniform(0, 2 * np.pi, dim).astype(np.float32)
        beta = rng.standard_normal(dim)
        original = probe.GRAM_CHUNK
        try:
            probe.GRAM_CHUNK = 91
            got = probe.rff_predict(design, proj, phase, beta)
        finally:
            probe.GRAM_CHUNK = original
        z = np.cos(design @ proj + phase, dtype=np.float32).astype(np.float64) * np.sqrt(2.0 / dim)
        reference = z @ beta
        self.assertLess(np.abs(got - reference).max() / np.abs(reference).std(), 1e-5)


class TestRidge(unittest.TestCase):
    def test_alpha_is_scale_free_in_row_count(self):
        """alpha 用 trace/d 归一化 ⟹ 复制一份数据不该改变解。"""
        rng = np.random.default_rng(5)
        n, d = 3_000, 10
        z = rng.standard_normal((n, d))
        y = z @ rng.standard_normal(d) + rng.standard_normal(n)
        gram, rhs = z.T @ z, z.T @ y
        one = probe.solve_ridge(gram, rhs, 1e-3)
        two = probe.solve_ridge(2 * gram, 2 * rhs, 1e-3)
        self.assertLess(np.abs(one - two).max(), 1e-9)

    def test_pick_alpha_selects_inner_best(self):
        rng = np.random.default_rng(6)
        n, d = 4_000, 15
        z = rng.standard_normal((n, d))
        y = z @ rng.standard_normal(d) * 0.05 + rng.standard_normal(n)
        half = n // 2
        ga, ra = z[:half].T @ z[:half], z[:half].T @ y[:half]
        gb, rb = z[half:].T @ z[half:], z[half:].T @ y[half:]
        dss = float(np.dot(y[half:], y[half:]))
        alpha, trace = probe.pick_alpha(ga, ra, gb, rb, dss)
        self.assertEqual(len(trace), len(probe.ALPHA_LADDER))
        self.assertAlmostEqual(alpha, max(trace, key=lambda t: t["inner_peak"])["alpha_relative"])


class TestGates(unittest.TestCase):
    @staticmethod
    def _rows(r_by_arm):
        return [{"fold": i, "ic_e_lgbm": 0.04,
                 "arms": {name: {"r": r[i], "rho": rho[i],
                                 "blend_gain_ic": probe.blend_gain_ic(r[i], rho[i])}
                          for name, (r, rho) in r_by_arm.items()}}
                for i in range(5)]

    def test_strong_independent_arm_passes(self):
        rows = self._rows({
            "linear": ([0.5] * 5, [0.9] * 5),
            "rff_full": ([1.0] * 5, [0.6] * 5),
            "rff_pca64": ([0.5] * 5, [0.7] * 5)})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertTrue(report["summary"]["rff_full"]["passed"])
        self.assertFalse(report["summary"]["rff_pca64"]["passed"])
        self.assertEqual(report["verdict"], "PASS")

    def test_linear_control_passing_invalidates_the_experiment(self):
        rows = self._rows({
            "linear": ([1.0] * 5, [0.6] * 5),
            "rff_full": ([1.0] * 5, [0.6] * 5),
            "rff_pca64": ([1.0] * 5, [0.6] * 5)})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertEqual(report["verdict"], "INVALID_LINEAR_CONTROL_PASSED")
        self.assertFalse(report["summary"]["rff_full"]["gates"][
            "5_linear_control_does_not_pass_gate_1"])

    def test_one_lucky_fold_does_not_survive_drop_best(self):
        rows = self._rows({
            "linear": ([0.4] * 5, [0.9] * 5),
            "rff_full": ([1.6, 0.6, 0.6, 0.6, 0.6], [0.6] * 5),
            "rff_pca64": ([0.6] * 5, [0.6] * 5)})
        report = probe.build_report(rows, "sha", type("A", (), {"stage1": False})(), 1.0)
        self.assertFalse(report["summary"]["rff_full"]["gates"]["4_survives_drop_best_fold"])
        self.assertEqual(report["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
