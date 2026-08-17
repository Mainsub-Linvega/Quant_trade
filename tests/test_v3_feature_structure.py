from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3_feature_structure import (
    build_task_views,
    contiguous_time_blocks,
    evenly_spaced_rows,
    feature_quality_by_blocks,
    stable_redundancy,
    weighted_component_gram,
)
from experiments.v3_feature_structure_audit import (
    extract_dense_matrices,
    render_markdown,
    write_report_bundle,
)


def test_build_task_views_separates_market_and_cross_section() -> None:
    time_id = np.repeat([10, 11], 3)
    features = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 3.0],
            [5.0, 6.0],
            [6.0, 9.0],
        ]
    )
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
    weight = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0])

    views = build_task_views(features, target, weight, time_id)

    np.testing.assert_allclose(views.market_features, [[2.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(views.market_target, [2.0, 6.75])
    np.testing.assert_allclose(
        np.add.reduceat(weight * views.cross_target, views.starts),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        views.cross_features.reshape(2, 3, 2).mean(axis=1),
        0.0,
        atol=1e-12,
    )


def test_build_task_views_rejects_unsorted_time_ids() -> None:
    with pytest.raises(ValueError, match="sorted"):
        build_task_views(np.ones((3, 2)), np.ones(3), np.ones(3), np.array([1, 0, 1]))


def test_contiguous_time_blocks_do_not_split_time_ids() -> None:
    time_ids = np.repeat(np.arange(8), 2)

    blocks = contiguous_time_blocks(time_ids, 4)

    grouped_ids = [np.unique(time_ids[block]).tolist() for block in blocks]
    assert grouped_ids == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_feature_quality_marks_stable_and_drifting_columns() -> None:
    target = np.arange(16, dtype=float)
    features = np.column_stack(
        [
            target,
            np.r_[np.arange(8, dtype=float), -np.arange(8, dtype=float)],
        ]
    )
    time_ids = np.repeat(np.arange(8), 2)

    report = feature_quality_by_blocks(
        features,
        target,
        np.ones(16),
        time_ids,
        n_blocks=4,
    )

    assert report["direction_consistency"][0] == 1.0
    assert report["direction_consistency"][1] < 1.0
    assert report["block_correlation"].shape == (4, 2)
    assert report["early_late_delta"][1] < 0.0




def test_stable_redundancy_clusters_duplicate_columns() -> None:
    base = np.arange(24, dtype=float)
    features = np.column_stack(
        [base, base * 2.0, -base, np.tile([0.0, 1.0, 0.0], 8)]
    )
    time_ids = np.repeat(np.arange(12), 2)

    result = stable_redundancy(features, time_ids, n_blocks=4, threshold=0.05)

    assert result.pearson.shape == (4, 4)
    assert result.spearman.shape == (4, 4)
    assert result.labels[0] == result.labels[1] == result.labels[2]
    assert result.labels[3] != result.labels[0]


def test_weighted_component_gram_exposes_market_cross_coupling() -> None:
    target = np.array([1.0, -1.0, 2.0, -2.0])
    baseline = np.array([0.2, -0.2, 0.3, -0.3])
    market_delta = np.array([1.0, 0.0, 1.0, 0.0])
    cross_delta = np.array([1.0, 0.0, 1.0, 0.0])

    result = weighted_component_gram(
        target, np.ones(4), baseline, market_delta, cross_delta
    )

    assert result["labels"] == ["b", "u", "v", "y"]
    assert result["gram"][1, 2] > 0.0
    np.testing.assert_allclose(result["gram"], result["gram"].T)


def test_render_markdown_contains_task_and_cluster_summaries() -> None:
    report = {
        "config": {"n_blocks": 4},
        "folds": [
            {
                "fold": 0,
                "tasks": {"ridge": {"n_features": 2, "cluster_count": 1}},
            }
        ],
        "gram": {"status": "ok", "uv_coupling": 0.25},
    }
    text = render_markdown(report)
    assert "# V3 Feature Structure Audit" in text
    assert "ridge" in text
    assert "0.25" in text


def test_build_task_views_preserves_float32_feature_storage() -> None:
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    views = build_task_views(
        features,
        np.arange(6, dtype=float),
        np.ones(6),
        np.repeat([0, 1], 3),
    )
    assert views.raw_features.dtype == np.float32
    assert views.cross_features.dtype == np.float32


def test_evenly_spaced_rows_is_deterministic_and_keeps_endpoints() -> None:
    np.testing.assert_array_equal(evenly_spaced_rows(10, 4), [0, 3, 6, 9])
    np.testing.assert_array_equal(evenly_spaced_rows(10, 4), evenly_spaced_rows(10, 4))


def test_stable_redundancy_row_cap_preserves_duplicate_cluster() -> None:
    base = np.arange(400, dtype=float)
    features = np.column_stack([base, base * 2.0, np.sin(base)])
    time_ids = np.repeat(np.arange(200), 2)

    result = stable_redundancy(
        features,
        time_ids,
        n_blocks=4,
        threshold=0.05,
        max_rows_per_block=25,
    )

    assert result.labels[0] == result.labels[1]

    assert result.sampled_rows_per_block == [25, 25, 25, 25]

def test_extract_dense_matrices_replaces_arrays_with_npz_references() -> None:
    report = {
        "folds": [
            {
                "fold": 0,
                "tasks": {
                    "ridge": {
                        "redundancy": {
                            "stability": np.eye(20),
                            "labels": np.array([1, 2]),
                        }
                    }
                },
            }
        ]
    }

    summary, matrices = extract_dense_matrices(report)

    key = "fold_0.ridge.redundancy.stability"
    assert key in matrices
    assert summary["folds"][0]["tasks"]["ridge"]["redundancy"]["stability"] == {
        "npz_key": key,
        "shape": [20, 20],
    }
    np.testing.assert_array_equal(
        summary["folds"][0]["tasks"]["ridge"]["redundancy"]["labels"], [1, 2]
    )


def test_write_report_bundle_round_trips_json_and_npz(tmp_path: Path) -> None:
    report = {
        "config": {"n_blocks": 4},
        "folds": [
            {
                "fold": 0,
                "tasks": {
                    "ridge": {
                        "status": "ok",
                        "n_features": 20,
                        "cluster_count": 1,
                        "redundancy": {"stability": np.eye(20)},
                    }
                },
            }
        ],
        "gram": {"status": "not_available"},
    }

    paths = write_report_bundle(report, tmp_path, "audit")

    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    with np.load(paths["npz"]) as matrices:
        assert matrices["fold_0.ridge.redundancy.stability"].shape == (20, 20)
    assert loaded["matrix_artifact"] == "audit_matrices.npz"
