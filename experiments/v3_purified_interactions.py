"""Leakage-safe primitives for purified residual interaction diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np


_FROZEN_OUTER = {
    "n_folds": 5,
    "train_window": 78_960,
    "embargo": 6,
    "sample_modulo": 5,
    "sampling": "phase_balanced",
}
_FROZEN_FUSION = {
    "market_lambda": 0.7,
    "blend_weight": 1.17,
    "prediction_scale": 1.16,
}
_TASK_BINS = {"ridge": 8, "xs": 8, "market": 4}


def default_purified_protocol() -> dict[str, object]:
    """Return an independent copy of the pre-registered P0 protocol."""
    return copy.deepcopy({
        "schema_version": 1,
        "outer": _FROZEN_OUTER,
        "inner_blocks": 4,
        "tasks": {
            "ridge": {"bins": 8, "min_cell_weight": 64.0},
            "xs": {"bins": 8, "min_cell_weight": 64.0},
            "market": {"bins": 4, "min_cell_weight": 16.0},
        },
        "null": {
            "quantile": 0.95,
            "seeds": [2026, 2027, 2028, 2029],
            "minimum_time_shift": 7,
        },
        "stability": {
            "minimum_positive_blocks": 2,
            "minimum_coverage": 0.80,
            "maximum_single_cell_gain_share": 0.50,
        },
        "budgets": {
            "max_pairs": 52_003,
            "max_output_candidates": 256,
            "max_surface_cells": 1_000_000,
        },
        "history_enabled": False,
        "fusion": _FROZEN_FUSION,
    })


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def validate_purified_protocol(protocol: Mapping[str, object]) -> None:
    """Fail closed when the P0 protocol is incomplete or no longer frozen."""
    root = _mapping(protocol, "protocol")
    if root.get("schema_version") != 1:
        raise ValueError("unsupported purified interaction schema version")
    if dict(_mapping(root.get("outer"), "outer")) != _FROZEN_OUTER:
        raise ValueError("outer validation protocol is not frozen")
    if root.get("history_enabled") is not False:
        raise ValueError("history must remain disabled in P0")
    if dict(_mapping(root.get("fusion"), "fusion")) != _FROZEN_FUSION:
        raise ValueError("frozen fusion must remain 0.7/1.17/1.16")

    inner_blocks = _positive_number(root.get("inner_blocks"), "inner_blocks")
    if inner_blocks < 4 or int(inner_blocks) != inner_blocks:
        raise ValueError("inner_blocks must be a positive integer of at least four")

    tasks = _mapping(root.get("tasks"), "tasks")
    if set(tasks) != set(_TASK_BINS):
        raise ValueError("tasks must contain ridge, xs, and market")
    for task, expected_bins in _TASK_BINS.items():
        settings = _mapping(tasks[task], f"tasks.{task}")
        if set(settings) != {"bins", "min_cell_weight"}:
            raise ValueError("each task requires one primary bin count")
        bins = _positive_number(settings.get("bins"), f"{task} bins")
        if int(bins) != bins or int(bins) != expected_bins:
            raise ValueError("each task requires one primary bin count")
        _positive_number(
            settings.get("min_cell_weight"), f"{task} min_cell_weight"
        )

    null = _mapping(root.get("null"), "null")
    quantile = null.get("quantile")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(float(quantile))
        or not 0.5 < float(quantile) < 1.0
    ):
        raise ValueError("null quantile must be between 0.5 and 1")
    seeds = null.get("seeds")
    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or len(seeds) == 0
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("null seeds must be unique integers")
    _positive_number(null.get("minimum_time_shift"), "minimum_time_shift")

    stability = _mapping(root.get("stability"), "stability")
    _positive_number(
        stability.get("minimum_positive_blocks"), "minimum_positive_blocks"
    )
    coverage = _positive_number(
        stability.get("minimum_coverage"), "minimum_coverage"
    )
    concentration = _positive_number(
        stability.get("maximum_single_cell_gain_share"),
        "maximum_single_cell_gain_share",
    )
    if coverage > 1.0 or concentration > 1.0:
        raise ValueError("stability fractions must not exceed one")

    budgets = _mapping(root.get("budgets"), "budgets")
    for name in ("max_pairs", "max_output_candidates", "max_surface_cells"):
        value = _positive_number(budgets.get(name), name)
        if int(value) != value:
            raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PrebinnedFeatureSplit:
    """Train-fitted compact bins for one chronological split."""

    train_bins: np.ndarray
    valid_bins: np.ndarray
    edges: np.ndarray
    bins: int
    missing_sentinel: int = 255


@dataclass(frozen=True)
class PurifiedPairSurface:
    """Training-fitted pure interaction surface for one feature pair."""

    left_feature: int
    right_feature: int
    edges_left: np.ndarray
    edges_right: np.ndarray
    values: np.ndarray
    cell_weights: np.ndarray
    coverage: float


def fit_quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Fit a fixed-width empirical quantile grid on finite training values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("quantile values must be one-dimensional")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two")
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("quantile values contain no finite training data")
    return np.asarray(
        np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)),
        dtype=np.float64,
    )


def assign_quantile_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values to a fitted grid; missing values receive bin -1."""
    array = np.asarray(values, dtype=np.float64)
    grid = np.asarray(edges, dtype=np.float64)
    if array.ndim != 1 or grid.ndim != 1 or len(grid) < 3:
        raise ValueError("values must be 1D and edges must define at least two bins")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) < 0.0):
        raise ValueError("quantile edges must be finite and nondecreasing")
    result = np.full(len(array), -1, dtype=np.int64)
    finite = np.isfinite(array)
    result[finite] = np.searchsorted(
        grid[1:-1], array[finite], side="right"
    )
    return result


def purify_pair_surface(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Separate a weighted pair surface into pure, main, and intercept terms."""
    surface = np.asarray(scores, dtype=np.float64)
    cell_weight = np.asarray(weights, dtype=np.float64)
    if (
        surface.ndim != 2
        or surface.shape != cell_weight.shape
        or surface.size == 0
    ):
        raise ValueError("scores and weights must be equal nonempty 2D arrays")
    if (
        not np.all(np.isfinite(surface))
        or not np.all(np.isfinite(cell_weight))
        or np.any(cell_weight < 0.0)
        or float(np.sum(cell_weight)) <= 0.0
    ):
        raise ValueError("surface scores and nonnegative weights must be finite")
    if tolerance < 0.0 or max_iterations <= 0:
        raise ValueError("purification controls are invalid")

    total_weight = float(np.sum(cell_weight))
    intercept = float(np.sum(surface * cell_weight) / total_weight)
    pure = surface - intercept
    left_main = np.zeros(surface.shape[0], dtype=np.float64)
    right_main = np.zeros(surface.shape[1], dtype=np.float64)
    row_weight = np.sum(cell_weight, axis=1)
    column_weight = np.sum(cell_weight, axis=0)

    for _ in range(max_iterations):
        row_effect = np.divide(
            np.sum(pure * cell_weight, axis=1),
            row_weight,
            out=np.zeros_like(row_weight),
            where=row_weight > 0.0,
        )
        pure -= row_effect[:, None]
        left_main += row_effect

        column_effect = np.divide(
            np.sum(pure * cell_weight, axis=0),
            column_weight,
            out=np.zeros_like(column_weight),
            where=column_weight > 0.0,
        )
        pure -= column_effect[None, :]
        right_main += column_effect

        row_error = np.max(np.abs(np.sum(pure * cell_weight, axis=1)))
        column_error = np.max(np.abs(np.sum(pure * cell_weight, axis=0)))
        if max(float(row_error), float(column_error)) <= tolerance:
            break
    else:
        raise RuntimeError("pair-surface purification did not converge")

    return pure, (left_main, right_main), intercept


def _validated_pair_arrays(
    left: np.ndarray,
    right: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(residual, dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("pair inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1 or len(arrays[0]) == 0:
        raise ValueError("pair inputs must be nonempty and row-aligned")
    if (
        not np.all(np.isfinite(arrays[2]))
        or not np.all(np.isfinite(arrays[3]))
        or np.any(arrays[3] < 0.0)
    ):
        raise ValueError("residual and nonnegative weight must be finite")
    return arrays


def fit_weighted_residual_surface(
    left: np.ndarray,
    right: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    bins: int,
    min_cell_weight: float,
    max_surface_cells: int,
    left_feature: int,
    right_feature: int,
) -> PurifiedPairSurface:
    """Fit and purify one binned residual surface using training rows only."""
    if bins * bins > max_surface_cells:
        raise MemoryError(
            f"surface requires {bins * bins} cells; budget={max_surface_cells}"
        )
    if min_cell_weight <= 0.0:
        raise ValueError("min_cell_weight must be positive")
    x_left, x_right, y, w = _validated_pair_arrays(
        left, right, residual, weight
    )
    edges_left = fit_quantile_edges(x_left, bins)
    edges_right = fit_quantile_edges(x_right, bins)
    left_bin = assign_quantile_bins(x_left, edges_left)
    right_bin = assign_quantile_bins(x_right, edges_right)
    valid = (left_bin >= 0) & (right_bin >= 0) & (w > 0.0)
    flat = left_bin[valid] * bins + right_bin[valid]
    cell_weights = np.bincount(
        flat, weights=w[valid], minlength=bins * bins
    ).reshape(bins, bins)
    weighted_sum = np.bincount(
        flat, weights=w[valid] * y[valid], minlength=bins * bins
    ).reshape(bins, bins)
    supported = cell_weights >= float(min_cell_weight)
    supported_weights = np.where(supported, cell_weights, 0.0)
    means = np.divide(
        weighted_sum,
        cell_weights,
        out=np.zeros_like(weighted_sum),
        where=supported,
    )

    if float(np.sum(supported_weights)) > 0.0:
        pure, _, _ = purify_pair_surface(means, supported_weights)
        pure = np.where(supported, pure, 0.0)
    else:
        pure = np.zeros((bins, bins), dtype=np.float64)
    total_valid_weight = float(np.sum(w[valid]))
    coverage = (
        float(np.sum(supported_weights)) / total_valid_weight
        if total_valid_weight > 0.0
        else 0.0
    )
    return PurifiedPairSurface(
        left_feature=int(left_feature),
        right_feature=int(right_feature),
        edges_left=edges_left,
        edges_right=edges_right,
        values=np.asarray(pure, dtype=np.float64),
        cell_weights=np.asarray(supported_weights, dtype=np.float64),
        coverage=float(coverage),
    )


def transform_purified_surface(
    surface: PurifiedPairSurface,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Apply a training-fitted surface; unsupported or missing cells map to zero."""
    x_left = np.asarray(left, dtype=np.float64)
    x_right = np.asarray(right, dtype=np.float64)
    if x_left.ndim != 1 or x_left.shape != x_right.shape:
        raise ValueError("surface transform inputs must be aligned 1D arrays")
    left_bin = assign_quantile_bins(x_left, surface.edges_left)
    right_bin = assign_quantile_bins(x_right, surface.edges_right)
    result = np.zeros(len(x_left), dtype=np.float64)
    valid = (left_bin >= 0) & (right_bin >= 0)
    if np.any(valid):
        rows = left_bin[valid]
        columns = right_bin[valid]
        supported = surface.cell_weights[rows, columns] > 0.0
        positions = np.flatnonzero(valid)[supported]
        result[positions] = surface.values[
            rows[supported], columns[supported]
        ]
    if not np.all(np.isfinite(result)):
        raise AssertionError("purified interaction transform produced non-finite values")
    return result


def weighted_residual_gain(
    residual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return normalized weighted SSE reduction against a zero prediction."""
    y = np.asarray(residual, dtype=np.float64)
    fitted = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if (
        y.ndim != 1
        or y.shape != fitted.shape
        or y.shape != w.shape
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(fitted))
        or not np.all(np.isfinite(w))
        or np.any(w < 0.0)
    ):
        raise ValueError("gain inputs must be aligned finite arrays with nonnegative weight")
    denominator = float(np.dot(w, y * y))
    if denominator <= 0.0:
        return 0.0
    error = y - fitted
    return float((denominator - np.dot(w, error * error)) / denominator)


def _surface_checksum(surface: PurifiedPairSurface) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(
        [surface.left_feature, surface.right_feature], dtype=np.int64
    ).tobytes())
    for array in (
        surface.edges_left,
        surface.edges_right,
        surface.values,
        surface.cell_weights,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validation_surface_diagnostics(
    surface: PurifiedPairSurface,
    left: np.ndarray,
    right: np.ndarray,
    residual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    left_bin = assign_quantile_bins(left, surface.edges_left)
    right_bin = assign_quantile_bins(right, surface.edges_right)
    w = np.asarray(weight, dtype=np.float64)
    y = np.asarray(residual, dtype=np.float64)
    fitted = np.asarray(prediction, dtype=np.float64)
    valid = (left_bin >= 0) & (right_bin >= 0) & (w > 0.0)
    supported = np.zeros(len(w), dtype=bool)
    supported[valid] = (
        surface.cell_weights[left_bin[valid], right_bin[valid]] > 0.0
    )
    valid_weight = float(np.sum(w[valid]))
    coverage = (
        float(np.sum(w[supported])) / valid_weight
        if valid_weight > 0.0
        else 0.0
    )

    denominator = float(np.dot(w, y * y))
    if denominator <= 0.0 or not np.any(supported):
        return coverage, 0.0
    contributions = w * (2.0 * y * fitted - fitted * fitted) / denominator
    bins = surface.values.shape[0]
    flat = left_bin[supported] * bins + right_bin[supported]
    by_cell = np.bincount(
        flat,
        weights=contributions[supported],
        minlength=bins * bins,
    )
    absolute = np.abs(by_cell)
    total = float(np.sum(absolute))
    dominant = float(np.max(absolute) / total) if total > 0.0 else 0.0
    return coverage, dominant


def score_pair_split(
    train_features: np.ndarray,
    valid_features: np.ndarray,
    train_residual: np.ndarray,
    valid_residual: np.ndarray,
    train_weight: np.ndarray,
    valid_weight: np.ndarray,
    *,
    pair: tuple[int, int],
    bins: int,
    min_cell_weight: float,
    max_surface_cells: int,
) -> dict[str, object]:
    """Fit one pure pair on training rows and score it on later rows."""
    train = np.asarray(train_features)
    valid = np.asarray(valid_features)
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("train and validation features must have equal 2D width")
    left_feature, right_feature = pair
    if (
        isinstance(left_feature, bool)
        or isinstance(right_feature, bool)
        or not isinstance(left_feature, int)
        or not isinstance(right_feature, int)
        or left_feature == right_feature
        or min(left_feature, right_feature) < 0
        or max(left_feature, right_feature) >= train.shape[1]
    ):
        raise ValueError("pair must contain two distinct valid feature indices")
    surface = fit_weighted_residual_surface(
        train[:, left_feature],
        train[:, right_feature],
        train_residual,
        train_weight,
        bins=bins,
        min_cell_weight=min_cell_weight,
        max_surface_cells=max_surface_cells,
        left_feature=left_feature,
        right_feature=right_feature,
    )
    prediction = transform_purified_surface(
        surface, valid[:, left_feature], valid[:, right_feature]
    )
    gain = weighted_residual_gain(valid_residual, prediction, valid_weight)
    coverage, dominant = _validation_surface_diagnostics(
        surface,
        valid[:, left_feature],
        valid[:, right_feature],
        valid_residual,
        prediction,
        valid_weight,
    )
    return {
        "pair": [int(left_feature), int(right_feature)],
        "gain": float(gain),
        "coverage": float(coverage),
        "dominant_cell_gain_share": float(dominant),
        "finite": bool(np.all(np.isfinite(prediction))),
        "surface_checksum": _surface_checksum(surface),
        "surface": surface,
    }


def prebin_feature_split(
    train_features: np.ndarray,
    valid_features: np.ndarray,
    *,
    bins: int,
) -> PrebinnedFeatureSplit:
    """Fit each feature grid on training rows and return compact bin matrices."""
    train = np.asarray(train_features)
    valid = np.asarray(valid_features)
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("train and validation features must have equal 2D width")
    if len(train) == 0 or len(valid) == 0 or bins < 2 or bins >= 255:
        raise ValueError("pre-binning requires rows and between 2 and 254 bins")
    edges = np.empty((train.shape[1], bins + 1), dtype=np.float64)
    train_bins = np.full(train.shape, 255, dtype=np.uint8)
    valid_bins = np.full(valid.shape, 255, dtype=np.uint8)
    for column in range(train.shape[1]):
        if not np.any(np.isfinite(train[:, column])):
            edges[column] = 0.0
            continue

        grid = fit_quantile_edges(train[:, column], bins)
        edges[column] = grid
        assigned_train = assign_quantile_bins(train[:, column], grid)
        assigned_valid = assign_quantile_bins(valid[:, column], grid)
        train_finite = assigned_train >= 0
        valid_finite = assigned_valid >= 0
        train_bins[train_finite, column] = assigned_train[train_finite]
        valid_bins[valid_finite, column] = assigned_valid[valid_finite]
    return PrebinnedFeatureSplit(
        train_bins=train_bins,
        valid_bins=valid_bins,
        edges=edges,
        bins=int(bins),
    )


def score_prebinned_pair_split(
    prebinned: PrebinnedFeatureSplit,
    train_residual: np.ndarray,
    valid_residual: np.ndarray,
    train_weight: np.ndarray,
    valid_weight: np.ndarray,
    *,
    pair: tuple[int, int],
    min_cell_weight: float,
    max_surface_cells: int,
) -> dict[str, object]:
    """Score one pair from train-fitted bins using the standard purification."""
    train_bins = np.asarray(prebinned.train_bins)
    valid_bins = np.asarray(prebinned.valid_bins)
    edges = np.asarray(prebinned.edges, dtype=np.float64)
    bins = int(prebinned.bins)
    if (
        train_bins.ndim != 2
        or valid_bins.ndim != 2
        or train_bins.dtype != np.uint8
        or valid_bins.dtype != np.uint8
        or train_bins.shape[1] != valid_bins.shape[1]
        or edges.shape != (train_bins.shape[1], bins + 1)
    ):
        raise ValueError("pre-binned split arrays are invalid")
    if bins * bins > max_surface_cells:
        raise MemoryError(
            f"surface requires {bins * bins} cells; budget={max_surface_cells}"
        )
    if min_cell_weight <= 0.0:
        raise ValueError("min_cell_weight must be positive")
    left_feature, right_feature = pair
    if (
        isinstance(left_feature, bool)
        or isinstance(right_feature, bool)
        or not isinstance(left_feature, int)
        or not isinstance(right_feature, int)
        or left_feature >= right_feature
        or left_feature < 0
        or right_feature >= train_bins.shape[1]
    ):
        raise ValueError("pair must contain two ordered valid feature indices")
    train_y = np.asarray(train_residual, dtype=np.float64)
    valid_y = np.asarray(valid_residual, dtype=np.float64)
    train_w = np.asarray(train_weight, dtype=np.float64)
    valid_w = np.asarray(valid_weight, dtype=np.float64)
    if (
        train_y.shape != (len(train_bins),)
        or train_w.shape != train_y.shape
        or valid_y.shape != (len(valid_bins),)
        or valid_w.shape != valid_y.shape
        or not np.all(np.isfinite(train_y))
        or not np.all(np.isfinite(valid_y))
        or not np.all(np.isfinite(train_w))
        or not np.all(np.isfinite(valid_w))
        or np.any(train_w < 0.0)
        or np.any(valid_w < 0.0)
    ):
        raise ValueError("pre-binned score arrays must be aligned and finite")

    left_train = train_bins[:, left_feature]
    right_train = train_bins[:, right_feature]
    train_valid = (
        (left_train != prebinned.missing_sentinel)
        & (right_train != prebinned.missing_sentinel)
        & (train_w > 0.0)
    )
    flat_train = (
        left_train[train_valid].astype(np.int64) * bins
        + right_train[train_valid]
    )
    cell_weights = np.bincount(
        flat_train, weights=train_w[train_valid], minlength=bins * bins
    ).reshape(bins, bins)
    weighted_sum = np.bincount(
        flat_train,
        weights=train_w[train_valid] * train_y[train_valid],
        minlength=bins * bins,
    ).reshape(bins, bins)
    supported = cell_weights >= float(min_cell_weight)
    supported_weights = np.where(supported, cell_weights, 0.0)
    means = np.divide(
        weighted_sum, cell_weights, out=np.zeros((bins, bins)), where=supported
    )
    if float(np.sum(supported_weights)) > 0.0:
        pure, _, _ = purify_pair_surface(means, supported_weights)
        pure = np.where(supported, pure, 0.0)
    else:
        pure = np.zeros((bins, bins), dtype=np.float64)
    train_valid_weight = float(np.sum(train_w[train_valid]))
    surface = PurifiedPairSurface(
        left_feature=left_feature,
        right_feature=right_feature,
        edges_left=edges[left_feature],
        edges_right=edges[right_feature],
        values=pure,
        cell_weights=supported_weights,
        coverage=(
            float(np.sum(supported_weights)) / train_valid_weight
            if train_valid_weight > 0.0 else 0.0
        ),
    )

    left_valid = valid_bins[:, left_feature]
    right_valid = valid_bins[:, right_feature]
    finite_valid = (
        (left_valid != prebinned.missing_sentinel)
        & (right_valid != prebinned.missing_sentinel)
    )
    prediction = np.zeros(len(valid_bins), dtype=np.float64)
    supported_valid = np.zeros(len(valid_bins), dtype=bool)
    if np.any(finite_valid):
        rows = left_valid[finite_valid].astype(np.int64)
        columns = right_valid[finite_valid].astype(np.int64)
        cell_supported = surface.cell_weights[rows, columns] > 0.0
        positions = np.flatnonzero(finite_valid)[cell_supported]
        prediction[positions] = surface.values[rows[cell_supported], columns[cell_supported]]
        supported_valid[positions] = True
    positive_valid = finite_valid & (valid_w > 0.0)
    valid_weight_sum = float(np.sum(valid_w[positive_valid]))
    coverage = (
        float(np.sum(valid_w[supported_valid])) / valid_weight_sum
        if valid_weight_sum > 0.0 else 0.0
    )
    denominator = float(np.dot(valid_w, valid_y * valid_y))
    dominant = 0.0
    if denominator > 0.0 and np.any(supported_valid):
        contributions = valid_w * (2.0 * valid_y * prediction - prediction * prediction) / denominator
        flat_valid = (
            left_valid[supported_valid].astype(np.int64) * bins
            + right_valid[supported_valid]
        )
        by_cell = np.bincount(
            flat_valid, weights=contributions[supported_valid], minlength=bins * bins
        )
        absolute = np.abs(by_cell)
        total = float(np.sum(absolute))
        dominant = float(np.max(absolute) / total) if total > 0.0 else 0.0
    return {
        "pair": [left_feature, right_feature],
        "gain": weighted_residual_gain(valid_y, prediction, valid_w),
        "coverage": coverage,
        "dominant_cell_gain_share": dominant,
        "finite": bool(np.all(np.isfinite(prediction))),
        "surface_checksum": _surface_checksum(surface),
        "surface": surface,
    }


def _group_bounds(time_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(time_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be nonempty and nondecreasing")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    return starts, counts


def _within_group_nonidentity_shuffle(
    values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.asarray(values, dtype=np.float64).copy()
    for start, count in zip(starts, counts):
        if count <= 1:
            continue
        shift = int(rng.integers(1, int(count)))
        rows = slice(int(start), int(start + count))
        shuffled[rows] = np.roll(shuffled[rows], shift)
    return shuffled


def _shift_complete_time_groups(
    values: np.ndarray,
    counts: np.ndarray,
    shift: int,
) -> np.ndarray:
    base = np.asarray(values, dtype=np.float64)
    if len(set(int(value) for value in counts)) == 1:
        width = int(counts[0])
        matrix = base.reshape(len(counts), width)
        return np.roll(matrix, int(shift), axis=0).reshape(-1)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    means = np.add.reduceat(base, starts) / counts
    centered = base - np.repeat(means, counts)
    return centered + np.repeat(np.roll(means, int(shift)), counts)


def make_task_null(
    task: str,
    residual: np.ndarray,
    time_ids: np.ndarray,
    *,
    seed: int,
    embargo: int,
) -> np.ndarray:
    """Break task alignment while retaining the relevant panel structure."""
    values = np.asarray(residual, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if values.ndim != 1 or values.shape != ids.shape or not np.all(np.isfinite(values)):
        raise ValueError("null residual and time_ids must be aligned finite arrays")
    if embargo < 0:
        raise ValueError("embargo must be nonnegative")
    starts, counts = _group_bounds(ids)
    rng = np.random.default_rng(seed)
    if task == "xs":
        return _within_group_nonidentity_shuffle(values, starts, counts, rng)
    if task not in {"market", "ridge"}:
        raise ValueError(f"unknown null task: {task}")
    if len(counts) <= embargo + 1:
        raise ValueError("not enough time groups for an embargo-safe null shift")
    base = (
        _within_group_nonidentity_shuffle(values, starts, counts, rng)
        if task == "ridge"
        else values.copy()
    )
    return _shift_complete_time_groups(base, counts, embargo + 1)


def make_split_task_nulls(
    task: str,
    train_residual: np.ndarray,
    valid_residual: np.ndarray,
    train_time_ids: np.ndarray,
    valid_time_ids: np.ndarray,
    *,
    seeds: Sequence[int],
    embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct null residuals independently on each side of a split."""
    if (
        len(seeds) == 0
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("split null seeds must be unique integers")
    return [
        (
            make_task_null(
                task,
                train_residual,
                train_time_ids,
                seed=int(seed),
                embargo=embargo,
            ),
            make_task_null(
                task,
                valid_residual,
                valid_time_ids,
                seed=int(seed),
                embargo=embargo,
            ),
        )
        for seed in seeds
    ]


def empirical_null_threshold(
    null_gains: np.ndarray,
    quantile: float,
) -> float:
    """Return a finite empirical upper-tail gain threshold."""
    values = np.asarray(null_gains, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("null gains must be a nonempty finite vector")
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("null quantile must be between 0.5 and 1")
    return float(np.quantile(values, float(quantile)))


def interaction_stability_gate(
    block_scores: Sequence[Mapping[str, object]],
    *,
    null_threshold: float,
    minimum_positive_blocks: int,
    minimum_coverage: float,
    maximum_single_cell_gain_share: float,
) -> dict[str, object]:
    """Apply pre-registered chronological stability checks to one pair family."""
    if len(block_scores) == 0 or not math.isfinite(float(null_threshold)):
        raise ValueError("stability gate requires block scores and a finite null")
    gains = np.asarray(
        [score.get("gain") for score in block_scores], dtype=np.float64
    )
    coverages = np.asarray(
        [score.get("coverage") for score in block_scores], dtype=np.float64
    )
    concentrations = np.asarray(
        [score.get("dominant_cell_gain_share") for score in block_scores],
        dtype=np.float64,
    )
    if not (
        np.all(np.isfinite(gains))
        and np.all(np.isfinite(coverages))
        and np.all(np.isfinite(concentrations))
    ):
        raise ValueError("block stability metrics must be finite")
    if minimum_positive_blocks <= 0 or minimum_positive_blocks > len(gains):
        raise ValueError("minimum_positive_blocks is invalid")
    drop_best = (
        float(np.mean(np.delete(gains, int(np.argmax(gains)))))
        if len(gains) > 1
        else float(gains[0])
    )
    checks = {
        "positive_blocks": int(np.sum(gains > 0.0)) >= minimum_positive_blocks,
        "above_null": float(np.median(gains)) > float(null_threshold),
        "coverage": bool(np.all(coverages >= float(minimum_coverage))),
        "tail_concentration": bool(np.all(
            concentrations <= float(maximum_single_cell_gain_share)
        )),
        "positive_drop_best": drop_best > 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_blocks": int(np.sum(gains > 0.0)),
        "median_gain": float(np.median(gains)),
        "mean_gain": float(np.mean(gains)),
        "drop_best_mean_gain": drop_best,
        "minimum_coverage": float(np.min(coverages)),
        "maximum_single_cell_gain_share": float(np.max(concentrations)),
        "null_threshold": float(null_threshold),
    }
