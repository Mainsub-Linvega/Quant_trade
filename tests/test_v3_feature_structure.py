from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3_feature_structure import build_task_views


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


