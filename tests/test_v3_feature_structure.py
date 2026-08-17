from __future__ import annotations

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
    feature_quality_by_blocks,
    stable_redundancy,
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
