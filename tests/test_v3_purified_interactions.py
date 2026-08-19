from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.v3_purified_interactions import (
    PurifiedPairSurface,
    assign_quantile_bins,
    default_purified_protocol,
    fit_quantile_edges,
    fit_weighted_residual_surface,
    purify_pair_surface,
    transform_purified_surface,
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


def test_purification_removes_both_parent_main_effects() -> None:
    scores = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]])
    weights = np.array([[2.0, 1.0, 3.0], [1.0, 4.0, 2.0]])

    pure, impurities, intercept = purify_pair_surface(scores, weights)

    np.testing.assert_allclose(
        np.sum(pure * weights, axis=0), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.sum(pure * weights, axis=1), 0.0, atol=1e-10
    )
    reconstructed = (
        pure + impurities[0][:, None] + impurities[1][None, :] + intercept
    )
    np.testing.assert_allclose(reconstructed, scores, atol=1e-10)


def test_additive_surface_purifies_to_zero() -> None:
    left = np.array([-2.0, 1.0, 5.0])
    right = np.array([3.0, -1.0])
    scores = left[:, None] + right[None, :]

    pure, _, _ = purify_pair_surface(scores, np.ones_like(scores))

    np.testing.assert_allclose(pure, 0.0, atol=1e-10)


def test_quantile_edges_are_fitted_on_training_values_only() -> None:
    edges = fit_quantile_edges(np.arange(8.0), bins=4)

    np.testing.assert_allclose(edges, [0.0, 1.75, 3.5, 5.25, 7.0])
    got = assign_quantile_bins(
        np.array([-10.0, 0.0, 1.75, 7.0, 100.0, np.nan]), edges
    )
    np.testing.assert_array_equal(got, [0, 0, 1, 3, 3, -1])


def test_repeated_quantile_edges_still_produce_valid_bins() -> None:
    edges = fit_quantile_edges(np.array([0.0, 0.0, 0.0, 1.0]), bins=4)

    got = assign_quantile_bins(np.array([0.0, 0.5, 1.0]), edges)

    assert len(edges) == 5
    assert np.all((got >= 0) & (got < 4))


def test_surface_maps_missing_and_unseen_cells_to_zero() -> None:
    surface = PurifiedPairSurface(
        left_feature=3,
        right_feature=9,
        edges_left=np.array([0.0, 1.0, 2.0]),
        edges_right=np.array([0.0, 1.0, 2.0]),
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        cell_weights=np.array([[1.0, 0.0], [1.0, 1.0]]),
        coverage=0.75,
    )

    got = transform_purified_surface(
        surface,
        np.array([0.5, 0.5, np.nan]),
        np.array([0.5, 1.5, 0.5]),
    )

    assert got[0] != 0.0
    assert got[1] == 0.0
    assert got[2] == 0.0
    assert np.all(np.isfinite(got))


def test_low_support_cells_are_shrunk_to_zero() -> None:
    surface = fit_weighted_residual_surface(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([-2.0, 1.0, -1.0, 2.0]),
        np.array([1.0, -1.0, -1.0, 1.0]),
        np.ones(4),
        bins=2,
        min_cell_weight=2.0,
        max_surface_cells=4,
        left_feature=0,
        right_feature=1,
    )

    np.testing.assert_array_equal(surface.values, np.zeros((2, 2)))
    assert surface.coverage == 0.0


def test_surface_rejects_budget_before_allocation() -> None:
    with pytest.raises(MemoryError, match="surface"):
        fit_weighted_residual_surface(
            np.arange(8.0),
            np.arange(8.0),
            np.arange(8.0),
            np.ones(8),
            bins=4,
            min_cell_weight=1.0,
            max_surface_cells=15,
            left_feature=0,
            right_feature=1,
        )
