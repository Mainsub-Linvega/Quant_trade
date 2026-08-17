from __future__ import annotations

import numpy as np

from experiments.v3_adaptive_selection import (
    aggregate_path_support,
    circular_shift_shadows,
    extract_tree_paths,
    linear_evidence,
    select_from_clusters,
    select_task_features,
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
