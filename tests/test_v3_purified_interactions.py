from __future__ import annotations

import copy

import pytest

from experiments.v3_purified_interactions import (
    default_purified_protocol,
    validate_purified_protocol,
)


def test_default_protocol_matches_frozen_design() -> None:
    protocol = default_purified_protocol()

    assert protocol["schema_version"] == 1
    assert protocol["outer"] == {
        "n_folds": 5,
        "train_window": 78_960,
        "embargo": 6,
        "sample_modulo": 5,
        "sampling": "phase_balanced",
    }
    assert protocol["tasks"]["ridge"]["bins"] == 8
    assert protocol["tasks"]["xs"]["bins"] == 8
    assert protocol["tasks"]["market"]["bins"] == 4
    assert protocol["fusion"] == {
        "market_lambda": 0.7,
        "blend_weight": 1.17,
        "prediction_scale": 1.16,
    }
    validate_purified_protocol(protocol)


def test_default_protocol_returns_an_independent_copy() -> None:
    first = default_purified_protocol()
    first["tasks"]["ridge"]["bins"] = 99

    second = default_purified_protocol()

    assert second["tasks"]["ridge"]["bins"] == 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda protocol: protocol["tasks"]["xs"].update(
                {"choose_best_bins": [8, 16]}
            ),
            "one primary bin count",
        ),
        (
            lambda protocol: protocol["null"].update({"quantile": 0.5}),
            "null quantile",
        ),
        (
            lambda protocol: protocol["null"].update(
                {"seeds": [2026, 2026]}
            ),
            "null seeds",
        ),
        (
            lambda protocol: protocol.update({"history_enabled": True}),
            "history",
        ),
        (
            lambda protocol: protocol["fusion"].update(
                {"market_lambda": 0.71}
            ),
            "frozen fusion",
        ),
    ],
)
def test_protocol_rejects_unfrozen_or_result_dependent_settings(
    mutation,
    message: str,
) -> None:
    protocol = copy.deepcopy(default_purified_protocol())
    mutation(protocol)

    with pytest.raises(ValueError, match=message):
        validate_purified_protocol(protocol)


@pytest.mark.parametrize(
    "path",
    [
        ("inner_blocks",),
        ("tasks", "ridge", "min_cell_weight"),
        ("budgets", "max_pairs"),
        ("budgets", "max_surface_cells"),
    ],
)
def test_protocol_rejects_nonpositive_budgets(path: tuple[str, ...]) -> None:
    protocol = copy.deepcopy(default_purified_protocol())
    node = protocol
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = 0

    with pytest.raises(ValueError, match="positive"):
        validate_purified_protocol(protocol)
