from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from promote_v3_candidate import PUBLIC_BASELINE, stage_candidate


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
