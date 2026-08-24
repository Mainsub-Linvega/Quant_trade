from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from promote_v3_candidate import PUBLIC_BASELINE, slow_fast_defaults, stage_candidate

# ⚠️ 2026-08-24：候选目录必须带**真的**冻结岭回归。此前这里写的是占位 `"{}"`，
# 而 `stage_candidate` 从 08-24 起会核 `baseline_model.json` 的 sha256
# （换岭回归 = 换市场块，见 promote_v3_candidate.PRODUCTION_RIDGE_SHA256）。
# 用占位值等于让这些用例全部撞在 ridge 门上 —— 它们测的是 meta / 森林 / 种子数，
# 不该被无关的门挡住，所以夹具带一份真的。
FROZEN_RIDGE = ROOT / "strategies" / "v3_hybrid" / "model" / "baseline_model.json"
needs_frozen_ridge = unittest.skipUnless(FROZEN_RIDGE.is_file(), "生产冻结岭回归不在盘上")


def make_candidate(root: Path, **overrides) -> tuple[Path, list[str], list[str]]:
    """造一个最小候选目录。默认与公榜基线一致，`overrides` 用来制造偏离。"""
    source = root / "candidate"
    source.mkdir()
    (source / "baseline_model.json").write_bytes(FROZEN_RIDGE.read_bytes())
    models = [f"seed{i}.txt" for i in range(3)]
    market = [f"market{i}.txt" for i in range(3)]
    for name in models + market:
        (source / name).write_text(name)
    meta = {
        "num_iteration": PUBLIC_BASELINE["num_iteration"],
        "history_window": PUBLIC_BASELINE["history_window"],
        "history_positions": list(range(PUBLIC_BASELINE["history_positions_count"])),
        "prediction_scale": 0.856,          # train.py 的本地占位值，转正时必须被覆写
        "blend_weight": 0.5,                # 同上
        "market_lambda": PUBLIC_BASELINE["market_lambda"],
        "cross_section_weighted": PUBLIC_BASELINE["cross_section_weighted"],
        "lgbm_model_files": models,
        "market_model_files": market,
        # slow/fast 三键属模型身份（2026-08-18 转正）。同样从 PUBLIC_BASELINE 派生，
        # 这样以后往那张表加键时，夹具不会悄悄落后。
        "slow_fast_window": PUBLIC_BASELINE["slow_fast_window"],
        "slow_fast_slow_relative": PUBLIC_BASELINE["slow_fast_slow_relative"],
        "slow_fast_fast_relative": PUBLIC_BASELINE["slow_fast_fast_relative"],
        # ⚠️ 2026-08-23：`long_window` 与上面三个键**不同类**，别照抄它们的心智模型。
        # slow/fast 是纯后处理，train.py 不产出、由 staging 补写；
        # 而 long_window 决定截面设计矩阵的宽度（441 vs 361 列）—— 它是**训练进森林里的**，
        # train.py:593 会写它，staging **绝不覆写**。夹具里给它基线值，是因为夹具扮演的是
        # 「一个用 --long-window 512 训出来的合格候选」。缺键的情形由下面的回归用例单独覆盖。
        "long_window": PUBLIC_BASELINE["long_window"],
    }
    meta.update(overrides)
    (source / "hybrid_meta.json").write_text(json.dumps(meta))
    return source, models, market


@needs_frozen_ridge
class LongWindowIdentityTest(unittest.TestCase):
    """2026-08-23 的回归：`long_window` 是第 14 个身份键，此前 validate_meta 一条都没查它。

    实测过的后果：脚本**默认**候选 `outputs/candidates/v3_hybrid_mkt_shrunk`
    （long_window=None、截面森林 361 列、公榜低 1.662%）零参数就能把
    baseline drift / validate_meta / 双后端烟测三道门全部走过。
    """

    def test_missing_long_window_is_rejected(self) -> None:
        """缺键 = 训练时没传 `--long-window`（train.py:335 默认 0 ⟹ meta 写 None）。

        而 main.py:207 是 `if long_window and ...` ⟹ 缺键**静默关掉长窗、不报错**。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_candidate(root, long_window=None)
            with self.assertRaises(ValueError) as caught:
                stage_candidate(source, root / "stage", scale=1.16, n_seeds=3, blend_weight=1.0)
            self.assertIn("long_window_matches", str(caught.exception))

    def test_wrong_long_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_candidate(root, long_window=64)   # long_window_ladder 的另一档
            with self.assertRaises(ValueError) as caught:
                stage_candidate(source, root / "stage", scale=1.16, n_seeds=3, blend_weight=1.0)
            self.assertIn("long_window_matches", str(caught.exception))

    def test_off_baseline_is_the_only_way_past(self) -> None:
        """偏离必须显式按下去 —— 留给 8/23 回补数据后有意重训不带长窗的情形。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_candidate(root, long_window=None)
            stage_candidate(source, root / "stage", scale=1.16, n_seeds=3,
                            blend_weight=1.0, off_baseline=True)

    def test_staging_never_invents_long_window(self) -> None:
        """⭐ 与 slow/fast 相反的方向：staging **不得**替候选补上这个键。

        补上等于给一个 361 列的森林盖「有长窗」的章 —— 推理期好的情况是撞上
        `lgbm_numpy.py:283` 的列宽 ValueError，坏的情况是交出一个错模型。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_candidate(root, long_window=None)
            stage = root / "stage"
            stage_candidate(source, stage, scale=1.16, n_seeds=3,
                            blend_weight=1.0, off_baseline=True)
            staged = json.loads((stage / "hybrid_meta.json").read_text())
            self.assertIsNone(staged["long_window"],
                              "staging 把候选的 long_window 改掉了 —— meta 会与森林宽度打架")


@needs_frozen_ridge
class PromoteV3CandidateTest(unittest.TestCase):
    def test_stage_overrides_scale_and_blend_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, models, market = make_candidate(root)
            before = (source / "hybrid_meta.json").read_bytes()
            stage = root / "stage"
            stage_candidate(source, stage, scale=1.16, n_seeds=3)
            staged = json.loads((stage / "hybrid_meta.json").read_text())
            # 两个本地占位值必须被覆写成公榜那份的口径
            self.assertEqual(staged["prediction_scale"], 1.16)
            self.assertEqual(staged["blend_weight"], 1.0)
            self.assertEqual((source / "hybrid_meta.json").read_bytes(), before)

    def test_market_forest_files_are_staged(self) -> None:
        """2026-08-13 的回归：原实现只搬截面块的森林，市场森林一个都没复制。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, models, market = make_candidate(root)
            stage = root / "stage"
            stage_candidate(source, stage, scale=1.16, n_seeds=3)
            for name in models + market:
                self.assertTrue((stage / name).is_file(), f"{name} 没有进 staging 目录")

    def test_seed_count_trims_both_forests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, models, market = make_candidate(root)
            stage = root / "stage"
            stage_candidate(source, stage, scale=1.16, n_seeds=2)
            staged = json.loads((stage / "hybrid_meta.json").read_text())
            self.assertEqual(staged["lgbm_model_files"], models[:2])
            self.assertEqual(staged["market_model_files"], market[:2])
            self.assertFalse((stage / models[2]).exists())
            self.assertFalse((stage / market[2]).exists())

    def test_staging_writes_slow_fast_keys_absent_from_candidate(self) -> None:
        """2026-08-19 的回归：`train.py` 的 CLI 里没有 slow/fast 概念。

        ⟹ 8/23 重训出来的候选 meta 一定缺这三个键，而 `main.py:222` 是
        `PredictionTrail(int(window)) if window else None` —— 缺键会**静默关掉**
        slow/fast、退回单一 scale 的旧模型（公榜低 2.93%）。
        staging 必须像补 scale/blend_weight 一样把它们补上。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_candidate(root)
            # 模拟重训候选：把三个键从候选 meta 里彻底拿掉
            meta_path = source / "hybrid_meta.json"
            meta = json.loads(meta_path.read_text())
            for key in slow_fast_defaults():
                meta.pop(key)
            meta_path.write_text(json.dumps(meta))

            stage = root / "stage"
            manifest = stage_candidate(source, stage, scale=1.16, n_seeds=3)
            staged = json.loads((stage / "hybrid_meta.json").read_text())
            for key, expected in slow_fast_defaults().items():
                self.assertEqual(staged[key], expected, key)
                self.assertEqual(manifest["configuration"][key], expected, key)

    def test_explicit_slow_fast_recalibration_needs_off_baseline(self) -> None:
        """RUNBOOK D1 的 (a) 路：用新 OOF 重标定两个 relative。

        写进 staged meta，但因为偏离公榜基线，必须显式按下 `--off-baseline` 才放行 ——
        偏离要是按下去的，不是漏掉的。
        """
        recalibrated = dict(slow_fast_defaults())
        recalibrated["slow_fast_slow_relative"] = 0.42
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_candidate(root)
            with self.assertRaises(ValueError):
                stage_candidate(source, root / "stage", scale=1.16, n_seeds=3,
                                slow_fast=recalibrated)
            staged_dir = root / "stage_off"
            stage_candidate(source, staged_dir, scale=1.16, n_seeds=3,
                            slow_fast=recalibrated, off_baseline=True)
            staged = json.loads((staged_dir / "hybrid_meta.json").read_text())
            self.assertEqual(staged["slow_fast_slow_relative"], 0.42)

    def test_structural_switches_are_gated(self) -> None:
        """λ=0 会让整片市场森林白跑、不带权会换掉截面块的损失 —— 都必须被拦住。"""
        for overrides, reason in (
            ({"market_lambda": 0.0}, "market_lambda"),
            ({"market_model_files": []}, "market_model_count"),
            ({"cross_section_weighted": False}, "cross_section_weighted"),
            ({"num_iteration": 960}, "num_iteration"),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _, _ = make_candidate(root, **overrides)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    stage_candidate(source, root / "stage", scale=1.16, n_seeds=3)


if __name__ == "__main__":
    unittest.main()
