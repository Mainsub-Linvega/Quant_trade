from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3_task_aligned_reselection import (
    P4_COUNTS,
    select_market_task_aligned,
    select_xs_time_stable,
    resolve_p4_arm,
)
from experiments.v3_task_aligned_reselection import select_history_lag_aligned


def test_market_selector_uses_per_time_unweighted_means_and_index_ties():
    time_ids = np.repeat(np.arange(4), 3)
    target = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
    features = np.column_stack([
        np.repeat([1.0, 2.0, 3.0, 4.0], 3),
        np.repeat([4.0, 3.0, 2.0, 1.0], 3),
        np.tile([0.0, 1.0, 2.0], 4),
    ])

    selected = select_market_task_aligned(features, target, time_ids, count=2)

    np.testing.assert_array_equal(selected, [0, 1])


def test_xs_stable_selector_prefers_sign_consistent_signal():
    time_ids = np.repeat(np.arange(16), 2)
    cross_target = np.tile([-1.0, 1.0], 16)
    stable = np.tile([-1.0, 1.0], 16)
    unstable = stable.copy()
    unstable[time_ids >= 8] *= -1.0

    selected = select_xs_time_stable(
        np.column_stack([stable, unstable]), cross_target, time_ids,
        count=1,
    )

    np.testing.assert_array_equal(selected, [0])


def test_selectors_reject_unsorted_time_ids():
    with pytest.raises(ValueError, match="sorted"):
        select_market_task_aligned(
            np.ones((4, 2)), np.ones(4), np.array([1, 0, 1, 2]), count=1,
        )


def test_p4_arm_contract_keeps_single_stage_changes():
    baseline = {
        name: np.arange(count, dtype=np.int64)
        for name, count in P4_COUNTS.items()
    }
    candidates = {
        "market_task_aligned": {"market": np.arange(200, 400, dtype=np.int64)},
        "xs_time_stable": {"xs": np.arange(123, 323, dtype=np.int64)},
        "history_lag_aligned": {"history": np.arange(40, dtype=np.int64)},
    }

    result = resolve_p4_arm(
        "market_task_aligned", baseline, candidates, P4_COUNTS,
    )

    np.testing.assert_array_equal(result["ridge"], baseline["ridge"])
    np.testing.assert_array_equal(result["xs"], baseline["xs"])
    np.testing.assert_array_equal(
        result["market"], candidates["market_task_aligned"]["market"]
    )
    np.testing.assert_array_equal(result["history"], baseline["history"])


def test_history_selector_uses_causal_asset_lags_and_is_deterministic():
    time_ids = np.repeat(np.arange(12), 2)
    asset_ids = np.tile(np.arange(2), 12)
    current = np.column_stack([
        np.arange(len(time_ids), dtype=float),
        np.sin(np.arange(len(time_ids), dtype=float)),
    ])
    target = np.zeros(len(time_ids), dtype=float)
    target[2:] = current[:-2, 0]

    selected = select_history_lag_aligned(
        current, target, time_ids, asset_ids, np.array([0, 1], dtype=np.int64),
        count=1, window=2, n_blocks=4,
    )
    repeated = select_history_lag_aligned(
        current, target, time_ids, asset_ids, np.array([0, 1], dtype=np.int64),
        count=1, window=2, n_blocks=4,
    )

    np.testing.assert_array_equal(
        selected["selected_indices"], repeated["selected_indices"]
    )
    assert selected["selected_count"] == 1
    assert set(selected["families"]) <= {
        "previous", "difference", "rolling_mean", "rolling_deviation",
    }


def test_history_selector_rejects_missing_asset_ids():
    with pytest.raises(ValueError, match="asset_ids"):
        select_history_lag_aligned(
            np.ones((8, 2)), np.ones(8), np.repeat(np.arange(4), 2),
            None, np.array([0, 1]), count=1,
        )


from experiments.v3_task_aligned_reselection import paired_gate


def test_p4_arm_contract_rejects_history_outside_xs():
    baseline = {
        "ridge": np.arange(200),
        "xs": np.arange(200),
        "market": np.arange(200),
        "history": np.arange(40),
    }
    candidates = {
        "history_lag_aligned": {"history": np.arange(40) + 200},
    }
    with pytest.raises(ValueError, match="subset"):
        resolve_p4_arm("history_lag_aligned", baseline, candidates, P4_COUNTS)


def _paired_row(fold: int, candidate_peak: float) -> dict[str, object]:
    return {
        "fold": fold,
        "baseline": {"peak": 1.0, "A": 2.0, "B": 4.0},
        "candidate": {"peak": candidate_peak, "A": 2.2, "B": 4.0},
    }


def test_paired_gate_requires_all_screen_conditions():
    gate = paired_gate([_paired_row(fold, 1.1) for fold in range(5)])
    assert gate["passed"] is True
    assert gate["positive_folds"] == 5
    assert gate["drop_best_mean_peak_delta"] > 0.0
    assert gate["alignment_energy_passed"] is True


def test_paired_gate_rejects_candidate_that_improves_only_one_fold():
    rows = [_paired_row(fold, 2.0 if fold == 0 else 0.99) for fold in range(5)]
    gate = paired_gate(rows)
    assert gate["passed"] is False
    assert gate["positive_folds"] == 1
    assert gate["drop_best_mean_peak_delta"] < 0.0


def test_paired_gate_rejects_non_finite_metrics():
    rows = [_paired_row(fold, 1.1) for fold in range(5)]
    rows[2]["candidate"]["A"] = np.nan
    gate = paired_gate(rows)
    assert gate["passed"] is False
    assert gate["all_finite"] is False
