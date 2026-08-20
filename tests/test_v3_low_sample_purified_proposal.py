from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np

import pytest

from experiments.v3_low_sample_purified_proposal import (
    chronological_time_blocks,
    aggregate_pair_scores,
    enumerate_lexical_pairs,
    load_baseline_indices,
    proposal_eligible,
    proposal_split_rows,
    scan_prebinned_pairs,
    sample_complete_time_groups,
    split_proposal_gate_rows,
    select_proposal_candidates,
    validate_proposal_arrays,
    default_proposal_protocol,
    validate_proposal_protocol,
)


def test_default_proposal_protocol_matches_frozen_design() -> None:
    protocol = default_proposal_protocol()

    assert protocol["schema_version"] == 1
    assert protocol["source_feature_count"] == 323
    assert protocol["source_pair_count"] == 52_003
    assert protocol["proposal_folds"] == [0, 1, 2]
    assert protocol["gate_folds"] == [3, 4]
    assert protocol["proposal_blocks"] == 4
    assert protocol["proposal_bins"] == 4
    assert protocol["sampling"]["ridge_xs"] == {
        "row_cap_per_block": 40_000,
        "fallback_row_cap_per_block": 20_000,
        "min_cell_weight": 32.0,
    }
    assert protocol["sampling"]["market"] == {
        "time_cap_per_block": 20_000,
        "min_cell_weight": 8.0,
    }
    assert protocol["benchmark"] == {
        "lexical_pair_count": 1_024,
        "runtime_ceiling_seconds": 1_800.0,
        "peak_rss_ceiling_bytes": 4 * 1024**3,
    }
    assert protocol["eligibility"] == {
        "minimum_positive_blocks": 2,
        "positive_drop_best_mean_gain": True,
        "minimum_coverage": 0.80,
        "maximum_dominant_cell_gain_share": 0.50,
    }
    assert protocol["candidate_budget"] == {
        "core": 192,
        "diversity": 64,
        "maximum": 256,
        "diversity_parent_cap": 4,
    }
    assert protocol["fusion"] == {
        "market_lambda": 0.7,
        "blend_weight": 1.17,
        "prediction_scale": 1.16,
    }
    validate_proposal_protocol(protocol)


def test_default_proposal_protocol_is_an_independent_copy() -> None:
    first = default_proposal_protocol()
    first["proposal_folds"][0] = 4

    assert default_proposal_protocol()["proposal_folds"] == [0, 1, 2]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda p: p.update({"gate_folds": [2, 3, 4]}),
            "fold",
        ),
        (
            lambda p: p["fusion"].update({"market_lambda": 0.8}),
            "fusion",
        ),
        (
            lambda p: p["candidate_budget"].update(
                {"core": 193, "diversity": 64}
            ),
            "candidate budget",
        ),
        (
            lambda p: p["eligibility"].update({"minimum_coverage": 0.79}),
            "eligibility",
        ),
    ],
)
def test_proposal_protocol_rejects_result_dependent_mutations(
    mutation,
    message: str,
) -> None:
    protocol = copy.deepcopy(default_proposal_protocol())
    mutation(protocol)

    with pytest.raises(ValueError, match=message):
        validate_proposal_protocol(protocol)


def _proposal_arrays() -> dict[str, np.ndarray]:
    time_id = np.repeat(np.arange(10), 2)
    return {
        "features": np.arange(20 * 3, dtype=np.float32).reshape(20, 3),
        "residual": np.linspace(-1.0, 1.0, 20),
        "weight": np.ones(20),
        "time_id": time_id,
        "fold": np.repeat([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], 2),
        "feature_indices": np.arange(3),
    }


def test_validate_proposal_arrays_requires_explicit_aligned_fold() -> None:
    arrays = _proposal_arrays()
    del arrays["fold"]
    with pytest.raises(ValueError, match="missing.*fold"):
        validate_proposal_arrays(arrays)

    arrays = _proposal_arrays()
    arrays["fold"] = arrays["fold"][:-1]
    with pytest.raises(ValueError, match="row-aligned"):
        validate_proposal_arrays(arrays)


def test_validate_proposal_arrays_rejects_mixed_fold_time_group() -> None:
    arrays = _proposal_arrays()
    arrays["fold"][3] = 1

    with pytest.raises(ValueError, match="one fold"):
        validate_proposal_arrays(arrays)


def test_validate_proposal_arrays_rejects_unordered_time() -> None:
    arrays = _proposal_arrays()
    arrays["time_id"] = arrays["time_id"][::-1]

    with pytest.raises(ValueError, match="nondecreasing"):
        validate_proposal_arrays(arrays)


def test_proposal_and_gate_rows_are_disjoint_and_time_isolated() -> None:
    arrays = validate_proposal_arrays(_proposal_arrays())

    proposal, gate = split_proposal_gate_rows(
        arrays["fold"], default_proposal_protocol()
    )

    np.testing.assert_array_equal(np.unique(arrays["fold"][proposal]), [0, 1, 2])
    np.testing.assert_array_equal(np.unique(arrays["fold"][gate]), [3, 4])
    assert not set(proposal).intersection(gate)
    assert not set(arrays["time_id"][proposal]).intersection(arrays["time_id"][gate])


def test_chronological_blocks_keep_complete_ordered_time_groups() -> None:
    arrays = validate_proposal_arrays(_proposal_arrays())
    proposal, _ = split_proposal_gate_rows(
        arrays["fold"], default_proposal_protocol()
    )

    blocks = chronological_time_blocks(
        arrays["time_id"], proposal, n_blocks=4
    )

    assert len(blocks) == 4
    assert np.array_equal(np.concatenate(blocks), proposal)
    for left, right in zip(blocks, blocks[1:]):
        assert arrays["time_id"][left[-1]] < arrays["time_id"][right[0]]
    for block in blocks:
        for value in np.unique(arrays["time_id"][block]):
            np.testing.assert_array_equal(
                block[arrays["time_id"][block] == value],
                np.flatnonzero(arrays["time_id"] == value),
            )


def test_complete_time_sampling_is_even_deterministic_and_bounded() -> None:
    time_id = np.repeat(np.arange(10), 3)
    candidate_rows = np.arange(30)

    first = sample_complete_time_groups(time_id, candidate_rows, row_cap=12)
    second = sample_complete_time_groups(time_id, candidate_rows, row_cap=12)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 12
    selected_times = np.unique(time_id[first])
    assert len(selected_times) == 4
    assert selected_times[0] == 0
    assert selected_times[-1] == 9
    assert np.max(np.diff(selected_times)) - np.min(np.diff(selected_times)) <= 1
    for value in selected_times:
        np.testing.assert_array_equal(
            first[time_id[first] == value], np.flatnonzero(time_id == value)
        )


def test_complete_time_sampling_never_splits_or_exceeds_variable_groups() -> None:
    time_id = np.repeat(np.arange(6), [2, 4, 3, 5, 2, 4])
    candidate_rows = np.arange(len(time_id))

    got = sample_complete_time_groups(time_id, candidate_rows, row_cap=9)

    assert 0 < len(got) <= 9
    for value in np.unique(time_id[got]):
        np.testing.assert_array_equal(
            got[time_id[got] == value], np.flatnonzero(time_id == value)
        )
    remaining_group_sizes = [
        int(np.sum(time_id == value))
        for value in np.unique(time_id)
        if value not in set(time_id[got])
    ]
    assert all(len(got) + size > 9 for size in remaining_group_sizes)


def test_enumerates_all_323_pairs_once_in_lexical_order() -> None:
    pairs = enumerate_lexical_pairs(323)

    assert len(pairs) == 52_003
    assert pairs[:3] == [(0, 1), (0, 2), (0, 3)]
    assert pairs[-1] == (321, 322)
    assert len(set(pairs)) == len(pairs)
    assert all(left < right for left, right in pairs)


def test_proposal_splits_expand_train_into_next_block() -> None:
    blocks = [np.array([0, 1]), np.array([2]), np.array([3, 4]), np.array([5])]

    splits = proposal_split_rows(blocks)

    expected = [
        (np.array([0, 1]), np.array([2])),
        (np.array([0, 1, 2]), np.array([3, 4])),
        (np.array([0, 1, 2, 3, 4]), np.array([5])),
    ]
    for got, want in zip(splits, expected):
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_aggregate_pair_scores_computes_frozen_statistics() -> None:
    summary = aggregate_pair_scores([
        {"gain": 0.03, "coverage": 0.91, "dominant_cell_gain_share": 0.20, "finite": True},
        {"gain": 0.01, "coverage": 0.85, "dominant_cell_gain_share": 0.30, "finite": True},
        {"gain": -0.01, "coverage": 0.82, "dominant_cell_gain_share": 0.45, "finite": True},
    ])

    assert summary["median_gain"] == pytest.approx(0.01)
    assert summary["mean_gain"] == pytest.approx(0.01)
    assert summary["drop_best_mean_gain"] == pytest.approx(0.0)
    assert summary["positive_blocks"] == 2
    assert summary["minimum_coverage"] == pytest.approx(0.82)
    assert summary["maximum_dominant_cell_gain_share"] == pytest.approx(0.45)
    assert summary["all_finite"] is True


def test_proposal_eligibility_uses_exact_frozen_boundaries() -> None:
    protocol = default_proposal_protocol()
    passing = {
        "positive_blocks": 2,
        "drop_best_mean_gain": 1e-12,
        "minimum_coverage": 0.80,
        "maximum_dominant_cell_gain_share": 0.50,
        "all_finite": True,
    }

    assert proposal_eligible(passing, protocol)
    for name, value in (
        ("positive_blocks", 1),
        ("drop_best_mean_gain", 0.0),
        ("minimum_coverage", 0.7999),
        ("maximum_dominant_cell_gain_share", 0.5001),
        ("all_finite", False),
    ):
        failed = dict(passing)
        failed[name] = value
        assert not proposal_eligible(failed, protocol)


def test_small_scan_scores_each_pair_once_per_split() -> None:
    rng = np.random.default_rng(44)
    features = rng.normal(size=(160, 4)).astype(np.float32)
    residual = ((features[:, 0] > 0) ^ (features[:, 1] > 0)).astype(float)
    weight = np.ones(160)
    blocks = [np.arange(left, left + 40) for left in range(0, 160, 40)]

    scores = scan_prebinned_pairs(
        features,
        residual,
        weight,
        blocks,
        bins=4,
        min_cell_weight=2.0,
        max_surface_cells=16,
    )

    assert scores["pair_indices"].shape == (6, 2)
    assert scores["split_gain"].shape == (6, 3)
    assert scores["split_coverage"].shape == (6, 3)
    assert scores["surface_checksum"].shape == (6, 3)
    np.testing.assert_array_equal(
        scores["pair_indices"], enumerate_lexical_pairs(4)
    )
    assert scores["scored_pair_split_count"] == 18
    assert scores["eligible"].dtype == np.bool_


def test_load_baseline_indices_uses_task_specific_frozen_file(
    tmp_path: Path,
) -> None:
    ridge = {"selected_indices": list(range(200))}
    xs_indices = list(range(123, 323))
    hybrid = {"lgbm_features": [f"feature_{value:03d}" for value in xs_indices[:200]]}
    (tmp_path / "baseline_model.json").write_text(
        json.dumps(ridge), encoding="utf-8"
    )
    (tmp_path / "hybrid_meta.json").write_text(
        json.dumps(hybrid), encoding="utf-8"
    )

    ridge_ref = load_baseline_indices("ridge", tmp_path)
    xs_ref = load_baseline_indices("xs", tmp_path)
    market_ref = load_baseline_indices("market", tmp_path)

    assert ridge_ref["indices"] == ridge["selected_indices"]
    assert xs_ref["indices"] == xs_indices[:200]
    assert market_ref["indices"] == xs_indices[:200]
    assert ridge_ref["source_path"].endswith("baseline_model.json")
    assert xs_ref["source_path"].endswith("hybrid_meta.json")
    assert len(ridge_ref["source_sha256"]) == 64


def _selection_scores() -> dict[str, np.ndarray]:
    pairs = enumerate_lexical_pairs(30)[:270]
    strength = np.arange(len(pairs), 0, -1, dtype=np.float64)
    return {
        "pair_indices": np.asarray(pairs, dtype=np.int16),
        "drop_best_mean_gain": strength,
        "median_gain": strength / 2,
        "mean_gain": strength / 3,
        "eligible": np.ones(len(pairs), dtype=bool),
    }


def test_selection_preserves_unrestricted_core_ranking() -> None:
    scores = _selection_scores()

    got = select_proposal_candidates(
        scores,
        baseline_indices=set(range(20)),
        protocol=default_proposal_protocol(),
    )

    expected_core = [tuple(pair) for pair in scores["pair_indices"][:192]]
    assert got["core_ranked"] == expected_core
    assert len(got["core_ranked"]) == 192
    assert not set(got["core_ranked"]).intersection(got["diversity_ranked"])


def test_diversity_prefers_outside_top200_and_caps_each_parent() -> None:
    scores = _selection_scores()

    got = select_proposal_candidates(
        scores,
        baseline_indices=set(range(20)),
        protocol=default_proposal_protocol(),
    )

    diversity = got["diversity_ranked"]
    counts: dict[int, int] = {}
    for left, right in diversity:
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
    assert diversity
    assert max(counts.values()) <= 4
    assert any(left >= 20 or right >= 20 for left, right in diversity)


def test_selection_never_fills_budget_from_ineligible_pairs() -> None:
    scores = _selection_scores()
    scores["eligible"][5:] = False

    got = select_proposal_candidates(
        scores,
        baseline_indices=set(range(20)),
        protocol=default_proposal_protocol(),
    )

    assert len(got["pairs"]) == 5
    assert got["diversity_ranked"] == []


def test_selection_manifest_is_lexical_and_deterministic() -> None:
    scores = _selection_scores()

    first = select_proposal_candidates(
        scores, baseline_indices=set(range(20)), protocol=default_proposal_protocol()
    )
    second = select_proposal_candidates(
        scores, baseline_indices=set(range(20)), protocol=default_proposal_protocol()
    )

    assert first["pairs"] == sorted(first["pairs"])
    assert first["manifest_json"] == second["manifest_json"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["manifest_sha256"]) == 64
