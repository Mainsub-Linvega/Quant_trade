from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3_market_task_aligned_diagnostic import (
    attribute_peak_delta,
    market_block_stability,
    market_overlap_summary,
    write_diagnostic_bundle,
)

from experiments.run_market_task_aligned_diagnostic import _row_slice

def test_market_overlap_reports_added_removed_and_jaccard():
    summary = market_overlap_summary(np.array([0, 1, 2]), np.array([1, 2, 3]))

    assert summary["overlap_count"] == 2
    assert summary["jaccard"] == pytest.approx(0.5)
    assert summary["added"] == [3]
    assert summary["removed"] == [0]


def test_peak_attribution_labels_alignment_loss():
    result = attribute_peak_delta(
        {"A": 1.0, "B": 1.0, "peak": 1.0},
        {"A": 0.8, "B": 0.9, "peak": 0.7},
        scale=1.16,
    )

    assert result["primary_cause"] == "alignment_loss"
    assert result["delta_A"] < 0.0
    assert result["delta_B"] < 0.0


def test_market_block_stability_marks_sign_flip_as_unstable():
    time_ids = np.repeat(np.arange(8), 2)
    target = np.repeat(np.arange(8, dtype=float), 2)
    feature = target.copy()
    feature[time_ids >= 4] *= -1.0

    stability = market_block_stability(
        np.column_stack([feature, target]), target, time_ids, n_blocks=4, top_count=2,
    )

    assert stability["sign_consistency"][0] < 1.0
    assert stability["block_ranks"].shape == (4, 2)


def test_write_diagnostic_bundle_is_atomic_and_contains_no_submission(tmp_path):
    paths = write_diagnostic_bundle({"folds": [{"fold": 0}]}, tmp_path, "market_diag")

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert (paths["fold_dir"] / "fold_0.json").exists()
    assert not any("submission" in path.name.lower() for path in tmp_path.iterdir())


def test_row_slice_returns_complete_contiguous_time_range():
    time_ids = np.repeat(np.arange(4), 2)

    result = _row_slice(time_ids, np.array([1, 2]))

    assert result == slice(2, 6)
