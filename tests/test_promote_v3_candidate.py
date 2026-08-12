from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from promote_v3_candidate import stage_candidate


class PromoteV3CandidateTest(unittest.TestCase):
    def test_stage_overrides_scale_and_seed_count_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate"; source.mkdir()
            (source / "baseline_model.json").write_text("{}")
            models = [f"seed{i}.txt" for i in range(3)]
            for name in models:
                (source / name).write_text(name)
            meta = {"num_iteration": 480, "history_window": 5,
                    "history_positions": list(range(40)), "prediction_scale": 0.856,
                    "lgbm_model_files": models}
            (source / "hybrid_meta.json").write_text(json.dumps(meta))
            before = (source / "hybrid_meta.json").read_bytes()
            stage = root / "stage"
            stage_candidate(source, stage, scale=1.16, n_seeds=2)
            staged = json.loads((stage / "hybrid_meta.json").read_text())
            self.assertEqual(staged["prediction_scale"], 1.16)
            self.assertEqual(staged["lgbm_model_files"], models[:2])
            self.assertFalse((stage / models[2]).exists())
            self.assertEqual((source / "hybrid_meta.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
