"""Leakage-safe primitives for purified residual interaction diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from typing import Any


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
