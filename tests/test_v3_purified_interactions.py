from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import experiments.v3_purified_interaction_diagnostic as diagnostic_module
from experiments.v3_purified_interaction_diagnostic import (
    _synthetic_arrays,
    deterministic_pairs,
    load_pair_manifest,
    parse_args as parse_diagnostic_args,
    run_diagnostic,
    validate_pair_manifest,
    validate_diagnostic_arrays,
    write_diagnostic_report,
)
from experiments.v3_purified_interactions import (
    PurifiedPairSurface,
    assign_quantile_bins,
    default_purified_protocol,
    fit_quantile_edges,
    fit_weighted_residual_surface,
    empirical_null_threshold,
    interaction_stability_gate,
    make_split_task_nulls,
    make_task_null,
    purify_pair_surface,
    score_pair_split,
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


def test_pair_score_detects_nonadditive_signal_not_additive_signal() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(800, 2))
    weight = np.ones(800)
    joint = (
        (features[:, 0] > 0.0) & (features[:, 1] > 0.0)
    ).astype(np.float64)
    nonlinear = score_pair_split(
        features[:600],
        features[600:],
        joint[:600],
        joint[600:],
        weight[:600],
        weight[600:],
        pair=(0, 1),
        bins=4,
        min_cell_weight=5.0,
        max_surface_cells=16,
    )
    additive_y = features[:, 0] + 2.0 * features[:, 1]
    additive = score_pair_split(
        features[:600],
        features[600:],
        additive_y[:600],
        additive_y[600:],
        weight[:600],
        weight[600:],
        pair=(0, 1),
        bins=4,
        min_cell_weight=5.0,
        max_surface_cells=16,
    )

    assert nonlinear["gain"] > 0.05
    assert abs(additive["gain"]) < nonlinear["gain"]
    assert nonlinear["surface_checksum"] != additive["surface_checksum"]


def test_xs_null_preserves_time_groups_but_breaks_asset_alignment() -> None:
    residual = np.arange(12.0)
    time_id = np.repeat(np.arange(4), 3)

    got = make_task_null(
        "xs", residual, time_id, seed=2026, embargo=6
    )

    for value in np.unique(time_id):
        rows = time_id == value
        np.testing.assert_array_equal(
            np.sort(got[rows]), np.sort(residual[rows])
        )
    assert not np.array_equal(got, residual)


def test_market_null_uses_an_embargo_safe_time_shift() -> None:
    residual = np.arange(12.0)
    time_id = np.arange(12)

    got = make_task_null(
        "market", residual, time_id, seed=2026, embargo=3
    )

    np.testing.assert_array_equal(got, np.roll(residual, 4))


def test_ridge_null_breaks_both_asset_and_time_alignment() -> None:
    residual = np.arange(24.0)
    time_id = np.repeat(np.arange(8), 3)

    got = make_task_null(
        "ridge", residual, time_id, seed=2026, embargo=2
    )

    assert not np.array_equal(got, residual)
    assert set(got.tolist()) == set(residual.tolist())
    for rows in np.split(got, 8):
        assert len(rows) == 3


def test_ridge_null_shifts_group_means_when_panel_width_varies() -> None:
    counts = np.array([2, 3, 2, 3, 2, 3, 2, 3])
    time_id = np.repeat(np.arange(len(counts)), counts)
    residual = np.arange(len(time_id), dtype=np.float64)
    embargo = 2

    shuffled = make_task_null(
        "xs", residual, time_id, seed=2026, embargo=embargo
    )
    got = make_task_null(
        "ridge", residual, time_id, seed=2026, embargo=embargo
    )

    starts = np.r_[0, np.cumsum(counts)[:-1]]
    shuffled_means = np.add.reduceat(shuffled, starts) / counts
    got_means = np.add.reduceat(got, starts) / counts
    np.testing.assert_allclose(got_means, np.roll(shuffled_means, embargo + 1))
    np.testing.assert_allclose(
        got - np.repeat(got_means, counts),
        shuffled - np.repeat(shuffled_means, counts),
    )
    assert got.shape == residual.shape
    assert np.all(np.isfinite(got))


def test_task_null_is_deterministic_and_requires_ordered_time() -> None:
    residual = np.arange(12.0)
    time_id = np.repeat(np.arange(4), 3)

    first = make_task_null("xs", residual, time_id, seed=9, embargo=6)
    second = make_task_null("xs", residual, time_id, seed=9, embargo=6)

    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="nondecreasing"):
        make_task_null(
            "xs", residual, time_id[::-1], seed=9, embargo=6
        )


def test_split_nulls_never_move_validation_values_into_training() -> None:
    train_residual = np.arange(24.0)
    valid_residual = np.arange(100.0, 112.0)
    train_time = np.repeat(np.arange(8), 3)
    valid_time = np.repeat(np.arange(20, 24), 3)

    nulls = make_split_task_nulls(
        "ridge",
        train_residual,
        valid_residual,
        train_time,
        valid_time,
        seeds=[2026, 2027],
        embargo=2,
    )

    assert len(nulls) == 2
    for null_train, null_valid in nulls:
        assert set(null_train.tolist()) == set(train_residual.tolist())
        assert set(null_valid.tolist()) == set(valid_residual.tolist())
        assert max(null_train) < min(null_valid)


def test_empirical_null_threshold_uses_requested_quantile() -> None:
    got = empirical_null_threshold(
        np.array([-0.3, -0.1, 0.1, 0.4]), 0.75
    )

    assert got == pytest.approx(0.175)


def test_stability_gate_requires_null_coverage_and_drop_best_gain() -> None:
    passing = interaction_stability_gate(
        [
            {"gain": 0.08, "coverage": 0.9, "dominant_cell_gain_share": 0.3},
            {"gain": 0.04, "coverage": 0.85, "dominant_cell_gain_share": 0.4},
            {"gain": 0.02, "coverage": 0.95, "dominant_cell_gain_share": 0.2},
        ],
        null_threshold=0.01,
        minimum_positive_blocks=2,
        minimum_coverage=0.8,
        maximum_single_cell_gain_share=0.5,
    )
    assert passing["passed"] is True
    assert passing["drop_best_mean_gain"] > 0.0

    tail_only = interaction_stability_gate(
        [
            {"gain": 0.08, "coverage": 0.9, "dominant_cell_gain_share": 0.8},
            {"gain": 0.04, "coverage": 0.9, "dominant_cell_gain_share": 0.7},
            {"gain": 0.02, "coverage": 0.9, "dominant_cell_gain_share": 0.9},
        ],
        null_threshold=0.01,
        minimum_positive_blocks=2,
        minimum_coverage=0.8,
        maximum_single_cell_gain_share=0.5,
    )
    assert tail_only["passed"] is False
    assert tail_only["checks"]["tail_concentration"] is False


def test_diagnostic_cli_defaults_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["v3_purified_interaction_diagnostic.py"]
    )

    args = parse_diagnostic_args()

    assert args.task == "ridge"
    assert args.max_pairs == 256
    assert args.synthetic_smoke is False
    assert args.write_candidate is False


def test_deterministic_pairs_are_lexical_and_exclude_self_pairs() -> None:
    got = deterministic_pairs(4, max_pairs=5)

    assert got == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    assert all(left < right for left, right in got)


def test_pair_manifest_preserves_preregistered_order_and_validates_budget(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(
        '{"pairs": [[0, 1], [17, 322], [8, 144]]}\n',
        encoding="utf-8",
    )

    got = load_pair_manifest(path, n_features=323, max_pairs=3)

    assert got == [(0, 1), (17, 322), (8, 144)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_pair_manifest([(0, 1), (0, 1)], n_features=323, max_pairs=3)
    with pytest.raises(ValueError, match="ordered"):
        validate_pair_manifest([(3, 3)], n_features=323, max_pairs=3)
    with pytest.raises(ValueError, match="budget"):
        validate_pair_manifest(
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            n_features=323,
            max_pairs=3,
        )


def test_diagnostic_scores_explicit_pair_manifest() -> None:
    requested = [(0, 1), (17, 322)]

    result = run_diagnostic(
        _synthetic_arrays(),
        task="ridge",
        protocol=default_purified_protocol(),
        max_pairs=2,
        pair_manifest=requested,
    )

    assert {tuple(item["pair"]) for item in result["pairs"]} == set(requested)
    assert result["pair_source"] == "manifest"


def test_diagnostic_scores_only_the_current_pair_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = diagnostic_module.score_pair_split
    widths: list[tuple[int, int]] = []
    scored_pairs: list[tuple[int, int]] = []

    def capture_widths(
        train_features: np.ndarray,
        valid_features: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        widths.append((train_features.shape[1], valid_features.shape[1]))
        scored_pairs.append(kwargs["pair"])
        return original(train_features, valid_features, *args, **kwargs)

    monkeypatch.setattr(diagnostic_module, "score_pair_split", capture_widths)

    result = run_diagnostic(
        _synthetic_arrays(),
        task="ridge",
        protocol=default_purified_protocol(),
        max_pairs=2,
    )

    assert widths and set(widths) == {(2, 2)}
    assert set(scored_pairs) == {(0, 1)}
    assert {tuple(item["pair"]) for item in result["pairs"]} == {
        (0, 1),
        (0, 2),
    }


def test_diagnostic_arrays_require_exactly_323_features() -> None:
    arrays = {
        "features": np.zeros((8, 323), dtype=np.float32),
        "residual": np.zeros(8),
        "weight": np.ones(8),
        "time_id": np.arange(8),
    }

    validated = validate_diagnostic_arrays(arrays)

    assert validated["features"].shape == (8, 323)
    with pytest.raises(ValueError, match="323"):
        validate_diagnostic_arrays({
            **arrays,
            "features": np.zeros((8, 322), dtype=np.float32),
        })
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_diagnostic_arrays({
            **arrays,
            "time_id": np.arange(8)[::-1],
        })


def test_diagnostic_report_never_generates_candidate(tmp_path: Path) -> None:
    paths = write_diagnostic_report(
        tmp_path,
        "smoke",
        payload={
            "experiment": "v3_purified_interaction_p0",
            "status": "passed_p0",
            "task": "ridge",
            "pairs": [],
        },
    )

    assert set(paths) == {"json", "markdown"}
    assert all(path.exists() for path in paths.values())
    assert not list(tmp_path.glob("*candidate*"))
    assert not list(tmp_path.glob("*.csv"))
    with pytest.raises(FileExistsError, match="force"):
        write_diagnostic_report(
            tmp_path,
            "smoke",
            payload={"status": "again", "pairs": []},
        )


def test_purified_diagnostic_script_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "experiments/v3_purified_interaction_diagnostic.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--synthetic-smoke" in result.stdout
    assert "--max-pairs" in result.stdout
    assert "--pair-manifest" in result.stdout


def test_synthetic_smoke_accepts_the_planted_pair() -> None:
    result = run_diagnostic(
        _synthetic_arrays(),
        task="ridge",
        protocol=default_purified_protocol(),
        max_pairs=1,
    )

    assert result["status"] == "passed_p0"
    assert result["accepted_pairs"] == 1
    assert result["pairs"][0]["pair"] == [0, 1]
    assert result["pairs"][0]["gate"]["passed"] is True


def test_diagnostic_rejects_row_level_market_input() -> None:
    arrays = _synthetic_arrays()

    with pytest.raises(ValueError, match="one row per time_id"):
        run_diagnostic(
            arrays,
            task="market",
            protocol=default_purified_protocol(),
            max_pairs=1,
        )


def test_diagnostic_rejects_aggregated_xs_input() -> None:
    rng = np.random.default_rng(12)
    rows = 20
    arrays = {
        "features": rng.normal(size=(rows, 323)).astype(np.float32),
        "residual": rng.normal(size=rows),
        "weight": np.ones(rows),
        "time_id": np.arange(rows),
    }

    with pytest.raises(ValueError, match="multiple assets per time_id"):
        run_diagnostic(
            arrays,
            task="xs",
            protocol=default_purified_protocol(),
            max_pairs=1,
        )
