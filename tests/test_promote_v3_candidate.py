from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from promote_v3_candidate import PUBLIC_BASELINE, slow_fast_defaults, stage_candidate


def make_candidate(root: Path, **overrides) -> tuple[Path, list[str], list[str]]:
    """造一个最小候选目录。默认与公榜基线一致，`overrides` 用来制造偏离。"""
    source = root / "candidate"
    source.mkdir()
    (source / "baseline_model.json").write_text("{}")
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
    }
    meta.update(overrides)
    (source / "hybrid_meta.json").write_text(json.dumps(meta))
    return source, models, market


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
