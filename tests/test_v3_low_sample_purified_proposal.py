from __future__ import annotations

import copy

import pytest

from experiments.v3_low_sample_purified_proposal import (
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
