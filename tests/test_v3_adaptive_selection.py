from __future__ import annotations

import numpy as np
import pytest

import sys
from experiments import v3_adaptive_selection_manifest as manifest_module
from experiments.v3_adaptive_selection import (
    aggregate_path_support,
    circular_shift_shadows,
    extract_tree_paths,
    linear_evidence,
    select_from_clusters,
    select_task_features,
)
from experiments.v3_adaptive_selection_manifest import (
    assemble_manifest,
    chronological_inner_splits,
    make_shadow_columns,
    selection_task_views,
    validation_tree_evidence,
)


def test_circular_shift_shadows_are_deterministic_and_non_identity() -> None:
    target = np.arange(8, dtype=float)

    first = circular_shift_shadows(target, n_shadows=3)
    second = circular_shift_shadows(target, n_shadows=3)

    np.testing.assert_array_equal(first["shifts"], [2, 4, 6])
    np.testing.assert_array_equal(first["targets"], second["targets"])
    assert first["targets"].shape == (3, 8)
    assert all(not np.array_equal(row, target) for row in first["targets"])


def test_linear_evidence_requires_stability_above_shadow_floor() -> None:
    correlations = np.array(
        [
            [0.20, 0.20, 0.03],
            [0.18, -0.19, 0.02],
            [0.21, 0.18, -0.01],
            [0.19, -0.17, 0.01],
        ]
    )
    shadow_correlations = np.full((4, 4), 0.05)

    evidence = linear_evidence(
        correlations,
        shadow_correlations,
        min_direction_consistency=0.75,
        shadow_quantile=1.0,
    )

    np.testing.assert_array_equal(evidence["passed"], [True, False, False])
    assert evidence["shadow_floor"] == 0.05


def test_extract_tree_paths_keeps_three_to_six_distinct_features() -> None:
    tree = {
        "split_feature": 0,
        "left_child": {
            "split_feature": 1,
            "left_child": {
                "split_feature": 2,
                "left_child": {"leaf_value": 1.0},
                "right_child": {"leaf_value": -1.0},
            },
            "right_child": {"leaf_value": 0.0},
        },
        "right_child": {
            "split_feature": 0,
            "left_child": {"leaf_value": 0.0},
            "right_child": {"leaf_value": 0.0},
        },
    }

    paths = extract_tree_paths({"tree_info": [{"tree_structure": tree}]})

    assert paths == [(0, 1, 2)]


def test_aggregate_path_support_accepts_repeated_three_feature_path() -> None:
    support = aggregate_path_support(
        [
            [(0, 1, 2), (0, 1, 2)],
            [(2, 1, 0), (1, 3, 4)],
            [(0, 2, 1)],
        ],
        min_blocks=2,
    )

    assert support == [
        {"features": [0, 1, 2], "support": 3, "blocks": [0, 1, 2]}
    ]


def test_select_from_clusters_keeps_best_representative_and_path_alternate() -> None:
    result = select_from_clusters(
        passed=np.array([True, True, True, True]),
        cluster_labels=np.array([10, 10, 20, 30]),
        scores=np.array([0.8, 0.6, 0.7, 0.5]),
        path_hyperedges=[
            {"features": [1, 2, 3], "support": 3, "blocks": [0, 1, 2]}
        ],
    )

    assert result["representatives"] == [0, 2, 3]
    assert result["alternates"] == [1]
    assert result["selected_indices"] == [0, 1, 2, 3]


def test_select_task_features_uses_path_gate_and_has_variable_stopping_count() -> None:
    weak = select_task_features(
        block_correlations=np.full((4, 4), 0.01),
        shadow_block_correlations=np.full((4, 3), 0.05),
        cluster_labels=np.arange(4),
        block_tree_gains=np.zeros((4, 4)),
        shadow_block_tree_gains=np.full((4, 3), 0.1),
        paths_by_block=[],
        shadow_quantile=1.0,
    )
    path_supported = select_task_features(
        block_correlations=np.full((4, 4), 0.01),
        shadow_block_correlations=np.full((4, 3), 0.05),
        cluster_labels=np.arange(4),
        block_tree_gains=np.zeros((4, 4)),
        shadow_block_tree_gains=np.full((4, 3), 0.1),
        paths_by_block=[[(0, 1, 2)], [(2, 0, 1)], [(0, 1, 2)]],
        min_path_blocks=2,
        shadow_quantile=1.0,
    )

    assert weak["selected_indices"] == []
    assert path_supported["selected_indices"] == [0, 1, 2]
    assert path_supported["reasons"] == {
        "0": ["path"],
        "1": ["path"],
        "2": ["path"],
    }
    assert path_supported["path_hyperedges"][0]["features"] == [0, 1, 2]
    assert len(weak["selected_indices"]) != len(path_supported["selected_indices"])


def test_select_task_features_accepts_stable_tree_gain_above_shadow() -> None:
    result = select_task_features(
        block_correlations=np.full((4, 3), 0.01),
        shadow_block_correlations=np.full((4, 2), 0.05),
        cluster_labels=np.arange(3),
        block_tree_gains=np.array(
            [
                [0.0, 0.4, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.4, 0.0],
            ]
        ),
        shadow_block_tree_gains=np.full((4, 2), 0.2),
        paths_by_block=[],
        shadow_quantile=1.0,
    )

    assert result["selected_indices"] == [1]
    assert result["reasons"] == {"1": ["tree_gain"]}


def test_lower_index_stable_passer_beats_path_only_cluster_member() -> None:
    result = select_task_features(
        block_correlations=np.array(
            [
                [0.20, 100.0, 0.01, 0.01],
                [0.18, -100.0, 0.01, 0.01],
                [0.21, 100.0, 0.01, 0.01],
                [0.19, -100.0, 0.01, 0.01],
            ]
        ),
        shadow_block_correlations=np.full((4, 2), 0.05),
        cluster_labels=np.array([10, 10, 20, 30]),
        block_tree_gains=np.zeros((4, 4)),
        shadow_block_tree_gains=np.full((4, 2), 0.1),
        paths_by_block=[
            [(1, 2, 3)],
            [(3, 1, 2)],
            [(1, 2, 3)],
        ],
        shadow_quantile=1.0,
    )

    assert result["representatives"] == [0, 2, 3]
    assert result["alternates"] == [1]
    assert result["selected_indices"] == [0, 1, 2, 3]
    assert result["evidence"][1]["linear_passed"] is False


def test_higher_index_stable_passer_beats_path_only_cluster_member() -> None:
    result = select_task_features(
        block_correlations=np.array(
            [
                [100.0, 0.20, 0.01, 0.01],
                [-100.0, 0.18, 0.01, 0.01],
                [100.0, 0.21, 0.01, 0.01],
                [-100.0, 0.19, 0.01, 0.01],
            ]
        ),
        shadow_block_correlations=np.full((4, 2), 0.05),
        cluster_labels=np.array([10, 10, 20, 30]),
        block_tree_gains=np.zeros((4, 4)),
        shadow_block_tree_gains=np.full((4, 2), 0.1),
        paths_by_block=[
            [(0, 2, 3)],
            [(3, 0, 2)],
            [(0, 2, 3)],
        ],
        shadow_quantile=1.0,
    )

    assert result["representatives"] == [1, 2, 3]
    assert result["alternates"] == [0]
    assert result["selected_indices"] == [0, 1, 2, 3]
    assert result["evidence"][0]["linear_passed"] is False


def test_zero_shadow_floor_representative_is_invariant_to_evidence_scaling() -> None:
    def select(linear_scale: float) -> dict[str, object]:
        return select_task_features(
            block_correlations=np.array(
                [
                    [0.20 * linear_scale, 0.0],
                    [0.18 * linear_scale, 0.0],
                    [0.21 * linear_scale, 0.0],
                    [0.19 * linear_scale, 0.0],
                ]
            ),
            shadow_block_correlations=np.zeros((4, 2)),
            cluster_labels=np.array([10, 10]),
            block_tree_gains=np.array(
                [
                    [0.0, 0.30],
                    [0.0, 0.35],
                    [0.0, 0.32],
                    [0.0, 0.31],
                ]
            ),
            shadow_block_tree_gains=np.zeros((4, 2)),
            paths_by_block=[],
            shadow_quantile=1.0,
        )

    baseline = select(1.0)
    scaled = select(10.0)

    assert baseline["representatives"] == [0]
    assert scaled["representatives"] == baseline["representatives"]
    assert scaled["selected_indices"] == baseline["selected_indices"]


def test_linear_evidence_rejects_nonfinite_shadow_measurements() -> None:
    feature_correlations = np.array(
        [
            [0.2, np.nan],
            [0.2, np.nan],
            [0.2, np.nan],
            [0.2, np.nan],
        ]
    )
    finite_shadows = np.full((4, 2), 0.05)

    evidence = linear_evidence(
        feature_correlations,
        finite_shadows,
        shadow_quantile=1.0,
    )

    np.testing.assert_array_equal(evidence["passed"], [True, False])
    with pytest.raises(ValueError, match="shadow_block_correlations.*finite"):
        linear_evidence(
            feature_correlations,
            np.array(
                [
                    [0.05, np.nan],
                    [0.05, 0.05],
                    [0.05, 0.05],
                    [0.05, 0.05],
                ]
            ),
        )


def test_selector_rejects_nonfinite_shadow_tree_gains() -> None:
    with pytest.raises(ValueError, match="shadow_block_tree_gains.*finite"):
        select_task_features(
            block_correlations=np.full((4, 2), 0.2),
            shadow_block_correlations=np.full((4, 2), 0.05),
            cluster_labels=np.arange(2),
            block_tree_gains=np.full((4, 2), 0.3),
            shadow_block_tree_gains=np.array(
                [
                    [0.1, np.inf],
                    [0.1, 0.1],
                    [0.1, 0.1],
                    [0.1, 0.1],
                ]
            ),
        )


def test_unrelated_cluster_rank_changes_do_not_flip_representative() -> None:
    cluster_labels = np.array([10, 10, 20, 20, 20, 20, 20])

    def select(
        unrelated_linear: list[float],
        unrelated_tree: list[float],
        paths_by_block: list[list[tuple[int, ...]]],
    ) -> dict[str, object]:
        correlations = np.array([0.20, 0.0, *unrelated_linear])
        tree_gains = np.array([0.0, 0.60, *unrelated_tree])
        return select_task_features(
            block_correlations=np.tile(correlations, (4, 1)),
            shadow_block_correlations=np.zeros((4, 2)),
            cluster_labels=cluster_labels,
            block_tree_gains=np.tile(tree_gains, (4, 1)),
            shadow_block_tree_gains=np.zeros((4, 2)),
            paths_by_block=paths_by_block,
            shadow_quantile=1.0,
        )

    baseline = select(
        unrelated_linear=[0.40, 0.40, 0.40, 0.40, 0.40],
        unrelated_tree=[0.80, 0.80, 0.80, 0.80, 0.80],
        paths_by_block=[
            [(2, 3, 4, 5, 6)],
            [(2, 3, 4, 5, 6)],
        ],
    )
    changed = select(
        unrelated_linear=[0.40, 0.50, 1e12, 0.50, 0.40],
        unrelated_tree=[0.10, 0.20, 0.20, 0.30, 0.10],
        paths_by_block=[
            [(2, 3, 4), (4, 5, 6)],
            [(2, 3, 4), (4, 5, 6)],
            [(2, 3, 4)],
        ],
    )

    def representative_for(
        result: dict[str, object],
        cluster: int,
    ) -> int:
        representatives = result["representatives"]
        assert isinstance(representatives, list)
        return next(
            feature
            for feature in representatives
            if cluster_labels[feature] == cluster
        )

    assert representative_for(baseline, 10) == 0
    assert representative_for(changed, 10) == 0


def test_chronological_inner_splits_expand_without_splitting_time_ids() -> None:
    time_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])

    splits = chronological_inner_splits(time_ids, n_blocks=4)

    assert len(splits) == 3
    expected = [
        (np.array([0, 1, 2, 3]), np.array([4])),
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6])),
        (np.array([0, 1, 2, 3, 4, 5, 6]), np.array([7])),
    ]
    for (train_rows, valid_rows), (expected_train, expected_valid) in zip(
        splits, expected, strict=True
    ):
        np.testing.assert_array_equal(train_rows, expected_train)
        np.testing.assert_array_equal(valid_rows, expected_valid)
        assert time_ids[train_rows[-1]] < time_ids[valid_rows[0]]


def test_make_shadow_columns_is_deterministic_and_non_identity() -> None:
    features = np.arange(24, dtype=np.float64).reshape(8, 3)

    first = make_shadow_columns(features, n_shadows=5)
    second = make_shadow_columns(features, n_shadows=5)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 5)
    for shadow in first.T:
        assert all(
            not np.array_equal(shadow, original) for original in features.T
        )


def test_validation_tree_evidence_uses_oos_contributions_and_excludes_shadows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_sizes: list[int] = []
    valid_sizes: list[int] = []

    class FakeBooster:
        def __init__(self, n_columns: int) -> None:
            self.n_columns = n_columns

        def predict(
            self, values: np.ndarray, *, pred_contrib: bool
        ) -> np.ndarray:
            assert pred_contrib is True
            valid_sizes.append(len(values))
            contribution = np.arange(
                1, self.n_columns + 2, dtype=np.float64
            )
            return np.tile(contribution, (len(values), 1))

        def dump_model(self) -> dict[str, object]:
            return {
                "tree_info": [
                    {
                        "tree_structure": {
                            "split_feature": 0,
                            "left_child": {
                                "split_feature": 1,
                                "left_child": {
                                    "split_feature": 2,
                                    "left_child": {
                                        "split_feature": 3,
                                        "left_child": {"leaf_value": 1.0},
                                        "right_child": {"leaf_value": -1.0},
                                    },
                                    "right_child": {"leaf_value": 0.0},
                                },
                                "right_child": {"leaf_value": 0.0},
                            },
                            "right_child": {"leaf_value": 0.0},
                        }
                    }
                ]
            }

    def fake_train(
        features: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray,
        *,
        params: dict[str, object],
        num_boost_round: int,
    ) -> FakeBooster:
        assert len(features) == len(target) == len(weight)
        assert params["max_depth"] == 4
        assert params["num_leaves"] == 15
        assert num_boost_round == 80
        train_sizes.append(len(features))
        return FakeBooster(features.shape[1])

    monkeypatch.setattr(
        manifest_module, "_train_lightgbm_booster", fake_train
    )
    features = np.arange(24, dtype=np.float64).reshape(8, 3)
    target = np.linspace(-1.0, 1.0, 8)
    weight = np.arange(1.0, 9.0)
    time_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])

    result = validation_tree_evidence(
        features,
        target,
        weight,
        time_ids,
        n_shadows=2,
    )

    np.testing.assert_array_equal(
        result["block_feature_evidence"],
        np.tile([1.0, 2.0, 3.0], (3, 1)),
    )
    np.testing.assert_array_equal(
        result["block_shadow_evidence"],
        np.tile([4.0, 5.0], (3, 1)),
    )
    assert result["paths_by_block"] == [[(0, 1, 2)]] * 3
    assert train_sizes == [4, 4, 4, 5, 5, 5, 7, 7, 7]
    assert valid_sizes == [1, 1, 1, 2, 2, 2, 1, 1, 1]


def test_validation_tree_evidence_fits_preprocessing_on_each_expanding_train_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_train_features: list[np.ndarray] = []

    class FakeBooster:
        def __init__(self, n_columns: int) -> None:
            self.n_columns = n_columns

        def predict(
            self, values: np.ndarray, *, pred_contrib: bool
        ) -> np.ndarray:
            return np.zeros((len(values), self.n_columns + 1), dtype=np.float64)

        def dump_model(self) -> dict[str, object]:
            return {"tree_info": []}

    def fake_train(
        features: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray,
        *,
        params: dict[str, object],
        num_boost_round: int,
    ) -> FakeBooster:
        captured_train_features.append(features[:, :1].copy())
        return FakeBooster(features.shape[1])

    monkeypatch.setattr(manifest_module, "_train_lightgbm_booster", fake_train)
    raw = np.arange(8, dtype=np.float64).reshape(-1, 1)
    target = np.linspace(-1.0, 1.0, 8)
    weight = np.ones(8)
    time_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4])

    manifest_module.validation_tree_evidence(
        raw,
        target,
        weight,
        time_ids,
        n_shadows=1,
        seeds=(2026,),
    )

    expected, _ = manifest_module.robust_transform_fit(raw[:4].copy())
    np.testing.assert_allclose(captured_train_features[0], expected)


def test_validation_tree_evidence_preprocesses_nonfinite_raw_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBooster:
        def __init__(self, n_columns: int) -> None:
            self.n_columns = n_columns

        def predict(
            self, values: np.ndarray, *, pred_contrib: bool
        ) -> np.ndarray:
            return np.zeros((len(values), self.n_columns + 1), dtype=np.float64)

        def dump_model(self) -> dict[str, object]:
            return {"tree_info": []}

    monkeypatch.setattr(
        manifest_module,
        "_train_lightgbm_booster",
        lambda features, target, weight, **kwargs: FakeBooster(features.shape[1]),
    )
    raw = np.arange(8, dtype=np.float64).reshape(-1, 1)
    raw[1, 0] = np.nan
    raw[5, 0] = np.inf

    result = manifest_module.validation_tree_evidence(
        raw,
        np.linspace(-1.0, 1.0, 8),
        np.ones(8),
        np.array([0, 0, 1, 1, 2, 3, 3, 4]),
        n_shadows=1,
        seeds=(2026,),
    )

    assert result["block_feature_evidence"].shape == (3, 1)


def test_market_tree_evidence_uses_aggregated_validation_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBooster:
        def __init__(self, n_columns: int) -> None:
            self.n_columns = n_columns

        def predict(
            self, values: np.ndarray, *, pred_contrib: bool
        ) -> np.ndarray:
            return np.zeros((len(values), self.n_columns + 1), dtype=np.float64)

        def dump_model(self) -> dict[str, object]:
            return {"tree_info": []}

    monkeypatch.setattr(
        manifest_module,
        "_train_lightgbm_booster",
        lambda features, target, weight, **kwargs: FakeBooster(features.shape[1]),
    )
    raw = np.arange(10, dtype=np.float64).reshape(-1, 1)

    result = manifest_module.validation_tree_evidence(
        raw,
        np.linspace(-1.0, 1.0, 10),
        np.ones(10),
        np.repeat(np.arange(5), 2),
        n_shadows=1,
        seeds=(2026,),
        task_name="market",
    )

    assert result["block_feature_evidence"].shape == (3, 1)


def test_assemble_manifest_keeps_task_contracts_and_history_pending() -> None:
    def selection(indices: list[int]) -> dict[str, object]:
        return {
            "selected_indices": indices,
            "representatives": indices[:1],
            "alternates": indices[1:],
            "evidence": [{"feature": index} for index in range(4)],
            "reasons": {str(index): ["linear"] for index in indices},
            "path_hyperedges": [
                {"features": [0, 1, 2], "support": 3, "blocks": [0, 1, 2]}
            ],
            "thresholds": {"linear_shadow_floor": 0.1},
        }

    manifest = assemble_manifest(
        ridge_selection=selection([0, 2]),
        xs_selection=selection([1]),
        market_selection=selection([2, 3]),
        feature_names=["feature_00", "feature_01", "feature_02", "feature_03"],
        protocol={"inner_splits": 3, "tree_rounds": 80},
    )

    assert manifest["schema_version"] == 1
    assert manifest["ridge"]["selected_indices"] == [0, 2]
    assert manifest["ridge"]["selected_names"] == ["feature_00", "feature_02"]
    assert manifest["ridge"]["selected_count"] == 2
    assert manifest["xs"]["selected_indices"] == [1]
    assert manifest["market"]["selected_indices"] == [2, 3]
    assert manifest["market"]["path_hyperedges"][0]["features"] == [0, 1, 2]
    assert manifest["protocol"] == {"inner_splits": 3, "tree_rounds": 80}
    assert manifest["history"] == {
        "status": "pending_task_2b",
        "selected_indices": None,
        "selected_names": None,
        "selected_count": None,
    }


def test_selection_task_views_use_unweighted_market_means() -> None:
    features = np.array(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [2.0, 20.0],
            [6.0, 60.0],
        ]
    )
    target = np.array([1.0, 9.0, 2.0, 10.0])
    weight = np.array([100.0, 1.0, 50.0, 1.0])
    time_ids = np.array([10, 10, 11, 11])
    views = selection_task_views(features, target, weight, time_ids)
    np.testing.assert_allclose(views["market"]["features"], [[2.0, 20.0], [4.0, 40.0]])
    np.testing.assert_allclose(views["market"]["target"], [5.0, 6.0])
    np.testing.assert_array_equal(views["market"]["time_ids"], [10, 11])
    np.testing.assert_array_equal(views["market"]["weight"], [1.0, 1.0])



def test_shadow_containing_tree_path_is_dropped() -> None:
    pure = {
        "tree_info": [{"tree_structure": {
            "split_feature": 0,
            "left_child": {"split_feature": 1,
                           "left_child": {"split_feature": 2,
                                          "left_child": {"leaf_value": 1.0},
                                          "right_child": {"leaf_value": 0.0}},
                           "right_child": {"leaf_value": 0.0}},
            "right_child": {"leaf_value": 0.0},
        }}]
    }
    mixed = {
        "tree_info": [{"tree_structure": {
            "split_feature": 0,
            "left_child": {"split_feature": 1,
                           "left_child": {"split_feature": 3,
                                          "left_child": {"leaf_value": 1.0},
                                          "right_child": {"leaf_value": 0.0}},
                           "right_child": {"leaf_value": 0.0}},
            "right_child": {"leaf_value": 0.0},
        }}]
    }
    assert manifest_module._original_paths(pure, 3, 5) == {(0, 1, 2)}
    assert manifest_module._original_paths(mixed, 3, 5) == set()


def test_selection_task_views_market_protocol_is_unweighted() -> None:
    features = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0], [6.0, 60.0]])
    target = np.array([1.0, 9.0, 2.0, 10.0])
    weight = np.array([100.0, 1.0, 50.0, 1.0])
    time_ids = np.array([10, 10, 11, 11])
    views = selection_task_views(features, target, weight, time_ids)
    np.testing.assert_allclose(views["market"]["features"], [[2.0, 20.0], [4.0, 40.0]])
    np.testing.assert_allclose(views["market"]["target"], [5.0, 6.0])
    np.testing.assert_array_equal(views["market"]["time_ids"], [10, 11])
    np.testing.assert_array_equal(views["market"]["weight"], [1.0, 1.0])


def test_manifest_cli_parses_smoke_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["manifest", "--smoke", "--smoke-time-ids", "8",
                                       "--smoke-tree-rounds", "3", "--smoke-row-cap", "20",
                                       "--force"])
    args = manifest_module.parse_args()
    assert args.smoke is True

    assert args.smoke_time_ids == 8
    assert args.smoke_tree_rounds == 3
    assert args.smoke_row_cap == 20
    assert args.force is True

def test_tree_budget_is_frozen_outside_smoke() -> None:
    normal = type("Args", (), {"smoke": False, "smoke_tree_rounds": None,
                               "smoke_row_cap": None})()
    assert manifest_module.resolve_tree_budget(normal) == (
        manifest_module.TREE_ROUNDS, manifest_module.TREE_ROW_CAP
    )
    invalid = type("Args", (), {"smoke": False, "smoke_tree_rounds": 3,
                                "smoke_row_cap": None})()
    with pytest.raises(ValueError, match="require --smoke"):
        manifest_module.resolve_tree_budget(invalid)


def test_tree_budget_uses_smoke_defaults_and_overrides() -> None:
    default = type("Args", (), {"smoke": True, "smoke_tree_rounds": None,
                                "smoke_row_cap": None})()
    assert manifest_module.resolve_tree_budget(default) == (5, 10_000)
    custom = type("Args", (), {"smoke": True, "smoke_tree_rounds": 3,
                               "smoke_row_cap": 20})()
    assert manifest_module.resolve_tree_budget(custom) == (3, 20)
    zero = type("Args", (), {"smoke": True, "smoke_tree_rounds": 0,
                             "smoke_row_cap": 20})()
    with pytest.raises(ValueError, match="rounds"):
        manifest_module.resolve_tree_budget(zero)
    too_large = type("Args", (), {"smoke": True,
                                  "smoke_tree_rounds": manifest_module.TREE_ROUNDS + 1,
                                  "smoke_row_cap": manifest_module.TREE_ROW_CAP + 1})()
    with pytest.raises(ValueError, match="rounds"):
        manifest_module.resolve_tree_budget(too_large)


def test_smoke_time_ids_requires_smoke_flag() -> None:
    args = type(
        "Args",
        (),
        {"train_window": 100, "smoke": False, "smoke_time_ids": 8,
         "smoke_tree_rounds": None, "smoke_row_cap": None},
    )()

    with pytest.raises(ValueError, match="smoke_time_ids requires --smoke"):
        manifest_module.validate_manifest_args(args)


def test_run_manifest_rejects_invalid_smoke_controls_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("data load must not start")

    monkeypatch.setattr(manifest_module, "load_rows", fail_load)
    args = type(
        "Args",
        (),
        {
            "data_root": "unused",
            "sample_modulo": 5,
            "sampling": "phase_balanced",
            "train_window": 100,
            "smoke": False,
            "smoke_time_ids": 8,
            "smoke_tree_rounds": None,
            "smoke_row_cap": None,
        },
    )()

    with pytest.raises(ValueError, match="smoke_time_ids requires --smoke"):
        manifest_module.run_manifest(args)


def test_manifest_cli_does_not_advertise_unimplemented_history_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["manifest"])
    assert not hasattr(manifest_module.parse_args(), "history_row_cap")


def test_manifest_bundle_serializes_protocol_and_refuses_overwrite(tmp_path) -> None:
    task = {"selected_count": 2, "path_hyperedges": []}
    manifest = {"protocol": {
                    "tree_rounds": 80,
                    "training_window": {"time_start": 0, "time_end": 1,
                                        "effective_time_ids": 2, "rows": 4},
                    "tree_evidence": {"inner_splits": 3, "num_boost_round": 80,
                                      "max_depth": 4, "num_leaves": 15,
                                      "seeds": [2026], "row_cap": 10, "n_shadows": 2}},
                "ridge": task, "xs": task, "market": task,
                "history": {"status": "pending_task_2b"}}
    paths = manifest_module.write_manifest_bundle(manifest, tmp_path, "unit")
    assert paths["json"].exists() and paths["markdown"].exists()
    assert '"tree_rounds": 80' in paths["json"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        manifest_module.write_manifest_bundle(manifest, tmp_path, "unit")
    manifest_module.write_manifest_bundle(manifest, tmp_path, "unit", force=True)
