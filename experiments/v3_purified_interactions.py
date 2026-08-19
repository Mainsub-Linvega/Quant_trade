"""Leakage-safe primitives for purified residual interaction diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
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
