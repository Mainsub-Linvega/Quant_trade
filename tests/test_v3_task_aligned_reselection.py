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
from experiments.v3_task_aligned_reselection import derive_p4_selections
from experiments.v3_task_aligned_reselection import parse_p4_args
from experiments.v3_task_aligned_reselection import runner_import_paths
from experiments.v3_task_aligned_reselection import spill_p4_features
from experiments.v3_task_aligned_reselection import allocate_p4_arrays


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


from experiments.v3_task_aligned_reselection import (
    candidate_gate_impossible,
    paired_gate,
    validate_frozen_protocol,
    write_p4_bundle,
)


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


def test_xs_arm_keeps_frozen_history_outside_candidate_xs():
    baseline = {
        "ridge": np.arange(200),
        "xs": np.arange(200),
        "market": np.arange(200),
        "history": np.arange(40),
    }
    candidates = {
        "xs_time_stable": {"xs": np.arange(123, 323)},
    }

    resolved = resolve_p4_arm("xs_time_stable", baseline, candidates, P4_COUNTS)

    np.testing.assert_array_equal(resolved["history"], baseline["history"])
    assert not set(resolved["history"]).issubset(set(resolved["xs"]))


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


def test_candidate_gate_impossible_uses_remaining_positive_folds():
    assert candidate_gate_impossible([-0.1, -0.2], total_folds=5) is True
    assert candidate_gate_impossible([0.1, -0.2], total_folds=5) is False


def test_validate_frozen_protocol_accepts_only_registered_modes():
    screen = {
        "n_folds": 5, "train_window": 78960, "embargo": 6,
        "sample_modulo": 5, "sampling": "phase_balanced",
        "n_seeds": 1, "num_iteration": 160, "market_lambda": 0.7,
        "blend_weight": 1.17, "prediction_scale": 1.16,
        "prediction_clip": 0.5, "history_window": 5,
    }
    assert validate_frozen_protocol("screen", screen)["n_seeds"] == 1
    with pytest.raises(ValueError, match="num_iteration"):
        validate_frozen_protocol("screen", {**screen, "num_iteration": 480})
    confirmation = {**screen, "n_seeds": 3, "num_iteration": 480}
    assert validate_frozen_protocol("confirmation", confirmation)["n_seeds"] == 3


def test_atomic_bundle_contains_fold_manifests_and_no_submission(tmp_path):
    payload = {
        "experiment": "p4",
        "status": "running",
        "protocol": {"n_folds": 1},
        "arms": {"baseline_corr": {}},
        "folds": [{"fold": 0, "arms": {"baseline_corr": {"peak": 1.0}}}],
        "gates": {},
        "submission_generated": False,
    }

    paths = write_p4_bundle(payload, tmp_path, "p4_test")

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert (paths["fold_dir"] / "fold_0.json").exists()
    assert not any("submission" in path.name.lower() for path in tmp_path.iterdir())


def test_synthetic_selection_pipeline_is_deterministic_and_keeps_fixed_contract():
    time_ids = np.repeat(np.arange(12), 4)
    asset_ids = np.tile(np.arange(4), 12)
    row = np.arange(len(time_ids), dtype=np.float64)
    cross_target = np.tile([-1.5, -0.5, 0.5, 1.5], 12)
    target = 0.2 * np.repeat(np.arange(12, dtype=np.float64), 4) + cross_target
    features = np.column_stack([
        np.repeat(np.arange(12, dtype=np.float64), 4),
        np.tile([-1.5, -0.5, 0.5, 1.5], 12),
        np.roll(np.tile([-1.5, -0.5, 0.5, 1.5], 12), 4),
        np.sin(row / 3.0),
        np.cos(row / 5.0),
        np.zeros(len(row)),
    ])
    counts = {"ridge": 3, "xs": 3, "market": 3, "history": 2}

    first = derive_p4_selections(
        features, target, cross_target, time_ids, asset_ids, counts=counts,
    )
    second = derive_p4_selections(
        features, target, cross_target, time_ids, asset_ids, counts=counts,
    )

    assert set(first) == {"baseline", "candidates", "arms"}
    assert set(first["arms"]) == {
        "baseline_corr", "market_task_aligned", "xs_time_stable", "history_lag_aligned",
    }
    for name, arm in first["arms"].items():
        assert set(arm) == set(counts)
        assert all(len(arm[task]) == counts[task] for task in counts)
        if name != "xs_time_stable":
            assert set(arm["history"]).issubset(set(arm["xs"]))
    for name in first["arms"]:
        for task in counts:
            np.testing.assert_array_equal(first["arms"][name][task], second["arms"][name][task])


def test_p4_cli_rejects_screen_round_drift_and_parses_arm_set():
    with pytest.raises(SystemExit):
        parse_p4_args(["--mode", "screen", "--num-iteration", "480"])
    args = parse_p4_args([
        "--mode", "confirmation", "--arm-set", "market_task_aligned,xs_time_stable",
    ])
    assert args.mode == "confirmation"
    assert args.arm_set == ["market_task_aligned", "xs_time_stable"]
    assert args.n_seeds == 3
    assert args.num_iteration == 480


def test_runner_import_paths_keep_legacy_train_before_history_package(tmp_path):
    paths = runner_import_paths(tmp_path)

    assert paths == (
        str(tmp_path / "strategies" / "v1_ridge"),
        str(tmp_path / "experiments"),
    )
    assert all("v3_hybrid" not in path for path in paths)


def test_spill_p4_features_moves_matrix_to_memmap_and_removes_source(tmp_path):
    data = {
        "features": np.arange(24, dtype=np.float32).reshape(6, 4),
        "target": np.arange(6, dtype=np.float64),
    }
    mapped = spill_p4_features(data, tmp_path / "features.npy")

    assert "features" not in data
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, np.arange(24, dtype=np.float32).reshape(6, 4))


def test_allocate_p4_arrays_uses_feature_memmap_and_aligned_metadata(tmp_path):
    arrays = allocate_p4_arrays(5, 3, tmp_path / "features.npy")

    assert isinstance(arrays["features"], np.memmap)
    assert arrays["features"].shape == (5, 3)
    for name in ("target", "weight", "time_id", "asset_id"):
        assert arrays[name].shape == (5,)
    arrays["features"][:] = 2.5
    arrays["features"].flush()
    reopened = np.load(tmp_path / "features.npy", mmap_mode="r")
    np.testing.assert_array_equal(reopened, np.full((5, 3), 2.5, dtype=np.float32))
