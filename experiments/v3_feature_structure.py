"""Pure algorithms for the V3 feature-structure audit.

The module intentionally performs no file I/O. Fold orchestration and report writing live in
``v3_feature_structure_audit.py`` so these calculations can be tested on small synthetic arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskViews:
    raw_features: np.ndarray
    full_target: np.ndarray
    market_features: np.ndarray
    market_target: np.ndarray
    cross_features: np.ndarray
    cross_target: np.ndarray
    unique_time_ids: np.ndarray
    starts: np.ndarray
    counts: np.ndarray


def _validate_panel_inputs(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features64 = np.asarray(features, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    weight64 = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    time_ids64 = np.asarray(time_ids, dtype=np.int64)
    if features64.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    n_rows = len(features64)
    if target64.shape != (n_rows,) or weight64.shape != (n_rows,) or time_ids64.shape != (n_rows,):
        raise ValueError("target, weight, and time_ids must match feature rows")
    if n_rows == 0:
        raise ValueError("panel must not be empty")
    if np.any(np.diff(time_ids64) < 0):
        raise ValueError("time_ids must be sorted")
    if not np.all(np.isfinite(features64)) or not np.all(np.isfinite(target64)):
        raise ValueError("features and target must be finite")
    return features64, target64, weight64, time_ids64


def build_task_views(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
) -> TaskViews:
    """Split a sorted panel into full, market and cross-sectional task views.

    Market feature rows are unweighted cross-sectional means, matching the deployable feature
    information available at inference. Market targets use competition weights. Cross-sectional
    targets therefore have zero weighted sum within each time group, while cross features have
    zero unweighted mean.
    """
    features64, target64, weight64, time_ids64 = _validate_panel_inputs(
        features, target, weight, time_ids
    )
    starts = np.r_[0, np.flatnonzero(time_ids64[1:] != time_ids64[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids64)])
    unique_time_ids = time_ids64[starts]

    market_features = np.add.reduceat(features64, starts, axis=0) / counts[:, None]
    total_weight = np.add.reduceat(weight64, starts)
    if np.any(total_weight <= 0.0):
        raise ValueError("each time group must have positive total weight")
    market_target = np.add.reduceat(weight64 * target64, starts) / total_weight

    repeated_market_features = np.repeat(market_features, counts, axis=0)
    repeated_market_target = np.repeat(market_target, counts)
    return TaskViews(
        raw_features=features64,
        full_target=target64,
        market_features=market_features,
        market_target=market_target,
        cross_features=features64 - repeated_market_features,
        cross_target=target64 - repeated_market_target,
        unique_time_ids=unique_time_ids,
        starts=starts,
        counts=counts,
    )
