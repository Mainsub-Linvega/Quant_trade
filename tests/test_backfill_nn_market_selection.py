from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_load_frozen_selection_reads_nested_selection_object(tmp_path) -> None:
    from backfill_nn_market_residual_search import load_frozen_selection

    path = tmp_path / "selection.json"
    path.write_text(json.dumps({
        "selection": {
            "gamma": 0.9,
            "normalization": 0.8,
            "scale": 1.0,
            "beta": 0.2,
        }
    }), encoding="utf-8")

    assert load_frozen_selection(path) == {
        "gamma": 0.9,
        "normalization": 0.8,
        "scale": 1.0,
        "beta": 0.2,
    }
