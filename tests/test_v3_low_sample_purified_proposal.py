from __future__ import annotations

import copy
import numpy as np

import pytest

from experiments.v3_low_sample_purified_proposal import (
    chronological_time_blocks,
    sample_complete_time_groups,
    split_proposal_gate_rows,
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
