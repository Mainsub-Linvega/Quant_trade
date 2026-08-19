"""Paired strict OOF evaluation for additive interaction columns.

The public helpers in this module keep acceptance and early stopping machine-readable.
The experiment runner is added around these primitives so a failed screen can never
fall through to final training or submission generation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def positive_fold_gate_impossible(
    deltas: list[float] | np.ndarray,
    *,
    total_folds: int = 5,
    required_positive: int = 4,
) -> bool:
    """Return whether remaining folds cannot reach the required positive count."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) > total_folds:
        raise ValueError("fold deltas must be a one-dimensional partial fold sequence")
    if total_folds <= 0 or not 0 < required_positive <= total_folds:
        raise ValueError("invalid positive-fold gate")
    if not np.all(np.isfinite(values)):
        raise ValueError("fold deltas must be finite")
    positives = int(np.sum(values > 0.0))
    remaining = total_folds - len(values)
    return positives + remaining < required_positive


def interaction_gate(
    deltas: np.ndarray,
    delta_a: float,
    delta_b: float,
    *,
    required_positive: int = 4,
) -> dict[str, object]:
    """Apply the frozen five-fold interaction acceptance criteria."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("interaction gate requires five finite paired fold deltas")
    if not np.isfinite(delta_a) or not np.isfinite(delta_b):
        raise ValueError("delta_a and delta_b must be finite")
    positive_folds = int(np.sum(values > 0.0))
    mean_delta = float(np.mean(values))
    drop_best_mean = float(np.mean(np.delete(values, int(np.argmax(values)))))
    checks = {
        "positive_mean": mean_delta > 0.0,
        "four_of_five_positive": positive_folds >= required_positive,
        "positive_drop_best": drop_best_mean > 0.0,
        "target_alignment": 2.0 * float(delta_a) > float(delta_b),
    }
    return {
        "passed": all(checks.values()),
        "mean_delta": mean_delta,
        "positive_folds": positive_folds,
        "drop_best_mean": drop_best_mean,
        "delta_a": float(delta_a),
        "delta_b": float(delta_b),
        "checks": checks,
    }


def append_interactions_before_asset(
    base_design: np.ndarray,
    interactions: np.ndarray,
) -> np.ndarray:
    """Append derived columns while preserving the final categorical asset column."""
    base = np.asarray(base_design, dtype=np.float32)
    added = np.asarray(interactions, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] < 1:
        raise ValueError("base design must be 2D with a final asset column")
    if added.ndim != 2 or len(added) != len(base):
        raise ValueError("interaction columns must be row-aligned 2D")
    if not np.all(np.isfinite(added)):
        raise ValueError("interaction columns must be finite")
    return np.ascontiguousarray(
        np.column_stack([base[:, :-1], added, base[:, -1]]),
        dtype=np.float32,
    )


def run_paired_fold_sequence(
    total_folds: int,
    run_fold: Callable[[int], dict[str, float]],
    *,
    required_positive: int = 4,
) -> dict[str, object]:
    """Run paired folds in order and stop once the positive-fold gate is impossible."""
    if total_folds <= 0 or not 0 < required_positive <= total_folds:
        raise ValueError("invalid paired fold configuration")
    folds: list[dict[str, float]] = []
    stopped_early = False
    stop_reason: str | None = None
    for fold_index in range(total_folds):
        payload = dict(run_fold(fold_index))
        required = {"peak_delta", "delta_a", "delta_b"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"fold payload is missing metrics: {missing}")
        if any(not np.isfinite(float(payload[name])) for name in required):
            raise ValueError("fold metrics must be finite")
        payload["fold"] = fold_index
        folds.append(payload)
        deltas = [float(item["peak_delta"]) for item in folds]
        if positive_fold_gate_impossible(
            deltas,
            total_folds=total_folds,
            required_positive=required_positive,
        ):
            stopped_early = True
            stop_reason = "four_of_five_positive_is_impossible"
            break
    return {
        "folds": folds,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "completed_folds": len(folds),
        "total_folds": total_folds,
        "required_positive": required_positive,
    }
