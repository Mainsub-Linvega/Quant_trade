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


def contiguous_time_blocks(time_ids: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    """Return contiguous row-index blocks without splitting a time group."""
    ids = np.asarray(time_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("time_ids must be a non-empty one-dimensional array")
    if np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be sorted")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")

    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    if n_blocks > len(starts):
        raise ValueError("n_blocks cannot exceed the number of unique time_ids")
    group_blocks = np.array_split(np.arange(len(starts)), n_blocks)
    row_blocks: list[np.ndarray] = []
    for groups in group_blocks:
        row_start = int(starts[groups[0]])
        next_group = int(groups[-1]) + 1
        row_stop = int(starts[next_group]) if next_group < len(starts) else len(ids)
        row_blocks.append(np.arange(row_start, row_stop, dtype=np.int64))
    return row_blocks


def _weighted_corr_columns(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    correlations = np.full(features.shape[1], np.nan, dtype=np.float64)
    finite_target = np.isfinite(target) & np.isfinite(weight)
    for column in range(features.shape[1]):
        finite = finite_target & np.isfinite(features[:, column])
        if finite.sum() < 2:
            continue
        w = np.maximum(weight[finite], 0.0)
        total_weight = float(w.sum())
        if total_weight <= 0.0:
            continue
        x = features[finite, column]
        y = target[finite]
        x_centered = x - float(np.dot(w, x) / total_weight)
        y_centered = y - float(np.dot(w, y) / total_weight)
        covariance = float(np.dot(w, x_centered * y_centered))
        x_energy = float(np.dot(w, x_centered * x_centered))
        y_energy = float(np.dot(w, y_centered * y_centered))
        denominator = np.sqrt(max(x_energy * y_energy, 0.0))
        if denominator > 0.0:
            correlations[column] = covariance / denominator
    return correlations


def _nan_median(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmedian(values, axis=axis)


def feature_quality_by_blocks(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
    n_blocks: int = 4,
) -> dict[str, np.ndarray]:
    """Measure marginal quality, temporal stability and simple distribution drift."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if y.shape != (len(x),) or w.shape != (len(x),) or ids.shape != (len(x),):
        raise ValueError("target, weight, and time_ids must match feature rows")
    if not np.all(np.isfinite(y)):
        raise ValueError("target must be finite")

    blocks = contiguous_time_blocks(ids, n_blocks)
    finite = np.isfinite(x)
    clean = np.where(finite, x, np.nan)
    finite_rate = finite.mean(axis=0)
    with np.errstate(invalid="ignore"):
        standard_deviation = np.nanstd(clean, axis=0)
        quartiles = np.nanquantile(clean, [0.25, 0.75], axis=0)
    iqr = quartiles[1] - quartiles[0]
    pooled_correlation = _weighted_corr_columns(x, y, w)
    block_correlation = np.vstack(
        [_weighted_corr_columns(x[block], y[block], w[block]) for block in blocks]
    )
    median_correlation = _nan_median(block_correlation, axis=0)
    median_abs_correlation = _nan_median(np.abs(block_correlation), axis=0)
    correlation_mad = _nan_median(
        np.abs(block_correlation - median_correlation[None, :]), axis=0
    )

    direction_consistency = np.zeros(x.shape[1], dtype=np.float64)
    for column in range(x.shape[1]):
        values = block_correlation[:, column]
        values = values[np.isfinite(values) & (values != 0.0)]
        if len(values):
            direction_consistency[column] = max(
                float(np.mean(values > 0.0)), float(np.mean(values < 0.0))
            )

    early_late_delta = block_correlation[-1] - block_correlation[0]
    first_clean = clean[blocks[0]]
    last_clean = clean[blocks[-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        standardized_mean_shift = (
            np.nanmean(last_clean, axis=0) - np.nanmean(first_clean, axis=0)
        ) / np.where(standard_deviation > 0.0, standard_deviation, np.nan)

    return {
        "finite_rate": finite_rate,
        "standard_deviation": standard_deviation,
        "iqr": iqr,
        "pooled_correlation": pooled_correlation,
        "block_correlation": block_correlation,
        "median_abs_correlation": median_abs_correlation,
        "correlation_mad": correlation_mad,
        "direction_consistency": direction_consistency,
        "early_late_delta": early_late_delta,
        "standardized_mean_shift": standardized_mean_shift,
    }
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata


@dataclass(frozen=True)
class RedundancyResult:
    pearson: np.ndarray
    spearman: np.ndarray
    stability: np.ndarray
    distance: np.ndarray
    labels: np.ndarray


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be two-dimensional")
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    sums = np.where(finite, x, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(x.shape[1]), where=counts > 0)
    centered = np.where(finite, x - means[None, :], 0.0)
    energy = np.sqrt(np.sum(centered * centered, axis=0))
    denominator = np.outer(energy, energy)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((x.shape[1], x.shape[1]), dtype=np.float64),
        where=denominator > 0.0,
    )
    np.clip(correlation, -1.0, 1.0, out=correlation)
    np.fill_diagonal(correlation, np.where(energy > 0.0, 1.0, 0.0))
    return correlation


def _rank_columns(values: np.ndarray) -> np.ndarray:
    ranked = np.full(values.shape, np.nan, dtype=np.float64)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.any():
            ranked[finite, column] = rankdata(values[finite, column], method="average")
    return ranked


def stable_redundancy(
    features: np.ndarray,
    time_ids: np.ndarray,
    n_blocks: int = 4,
    threshold: float = 0.15,
) -> RedundancyResult:
    """Cluster columns whose Pearson and Spearman relationships persist across time blocks."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    blocks = contiguous_time_blocks(time_ids, n_blocks)
    pearson_blocks = np.stack([_correlation_matrix(x[block]) for block in blocks])
    spearman_blocks = np.stack(
        [_correlation_matrix(_rank_columns(x[block])) for block in blocks]
    )
    pearson = np.median(np.abs(pearson_blocks), axis=0)
    spearman = np.median(np.abs(spearman_blocks), axis=0)
    stability = np.min(
        np.minimum(np.abs(pearson_blocks), np.abs(spearman_blocks)), axis=0
    )
    stability = np.nan_to_num(stability, nan=0.0, posinf=0.0, neginf=0.0)
    stability = np.clip((stability + stability.T) * 0.5, 0.0, 1.0)
    np.fill_diagonal(stability, 1.0)
    distance = 1.0 - stability
    distance = np.clip((distance + distance.T) * 0.5, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    if x.shape[1] == 1:
        labels = np.ones(1, dtype=np.int64)
    else:
        tree = linkage(squareform(distance, checks=True), method="average")
        labels = fcluster(tree, t=threshold, criterion="distance").astype(np.int64)
    return RedundancyResult(
        pearson=pearson,
        spearman=spearman,
        stability=stability,
        distance=distance,
        labels=labels,
    )


def weighted_component_gram(
    target: np.ndarray,
    weight: np.ndarray,
    baseline: np.ndarray,
    market_delta: np.ndarray,
    cross_delta: np.ndarray,
) -> dict[str, object]:
    """Return the target-energy-normalized weighted Gram matrix for ``[b, u, v, y]``."""
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (target, weight, baseline, market_delta, cross_delta)
    ]
    target64, weight64, baseline64, market64, cross64 = arrays
    if target64.ndim != 1:
        raise ValueError("component arrays must be one-dimensional")
    if any(values.shape != target64.shape for values in arrays[1:]):
        raise ValueError("component arrays must have identical shapes")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("component arrays must be finite")

    nonnegative_weight = np.maximum(weight64, 0.0)
    denominator = float(np.dot(nonnegative_weight, target64 * target64))
    if denominator <= 0.0:
        raise ValueError("target must have positive weighted energy")
    components = np.column_stack(
        [baseline64, market64, cross64, target64]
    )
    gram = components.T @ (nonnegative_weight[:, None] * components) / denominator
    gram = (gram + gram.T) * 0.5
    return {
        "labels": ["b", "u", "v", "y"],
        "gram": gram,
        "uv_coupling": float(gram[1, 2]),
        "target_energy": denominator,
    }
