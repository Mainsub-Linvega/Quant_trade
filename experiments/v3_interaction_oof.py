"""Paired strict OOF evaluation for additive interaction columns.

The public helpers in this module keep acceptance and early stopping machine-readable.
The experiment runner is added around these primitives so a failed screen can never
fall through to final training or submission generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCREEN = {
    "n_folds": 5,
    "train_window": 78_960,
    "embargo": 6,
    "sample_modulo": 5,
    "sampling": "phase_balanced",
    "n_seeds": 1,
    "num_iteration": 160,
    "market_lambda": 0.7,
    "blend_weight": 1.17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument(
        "--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments")
    )
    parser.add_argument("--label", default="v3_interactions_screen_1s160_07_117")
    parser.add_argument("--n-folds", type=int, default=FROZEN_SCREEN["n_folds"])
    parser.add_argument("--train-window", type=int, default=FROZEN_SCREEN["train_window"])
    parser.add_argument("--embargo", type=int, default=FROZEN_SCREEN["embargo"])
    parser.add_argument(
        "--sample-modulo", type=int, default=FROZEN_SCREEN["sample_modulo"]
    )
    parser.add_argument(
        "--sampling",
        choices=["periodic", "phase_balanced"],
        default=FROZEN_SCREEN["sampling"],
    )
    parser.add_argument("--n-seeds", type=int, default=FROZEN_SCREEN["n_seeds"])
    parser.add_argument(
        "--num-iteration", type=int, default=FROZEN_SCREEN["num_iteration"]
    )
    parser.add_argument(
        "--market-lambda", type=float, default=FROZEN_SCREEN["market_lambda"]
    )
    parser.add_argument(
        "--blend-weight", type=float, default=FROZEN_SCREEN["blend_weight"]
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--miner-row-cap", type=int, default=150_000)
    parser.add_argument("--max-source-cells", type=int, default=130_000_000)
    parser.add_argument("--max-interaction-cells", type=int, default=100_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_frozen_screen(args: argparse.Namespace) -> None:
    for name, expected in FROZEN_SCREEN.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"{name}={actual!r} changes the frozen screen; expected {expected!r}"
            )
    if args.num_threads <= 0 or args.history_window != 5:
        raise ValueError("num_threads must be positive and history_window must remain 5")
    if args.miner_row_cap <= 0 or args.max_source_cells <= 0:
        raise ValueError("interaction memory budgets must be positive")


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


def _group_mean(values: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if prediction.shape != ids.shape or prediction.ndim != 1:
        raise ValueError("group mean inputs must be aligned one-dimensional arrays")
    if len(ids) == 0 or np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be nonempty and nondecreasing")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    means = np.add.reduceat(prediction, starts) / counts
    return np.repeat(means, counts)


def compose_hybrid_raw(
    ridge_prediction: np.ndarray,
    xs_prediction: np.ndarray,
    market_prediction: np.ndarray,
    time_ids: np.ndarray,
    *,
    market_lambda: float,
    blend_weight: float,
) -> np.ndarray:
    """Compose unscaled, unclipped V3 output under fixed market/cross weights."""
    ridge = np.asarray(ridge_prediction, dtype=np.float64)
    xs = np.asarray(xs_prediction, dtype=np.float64)
    market_rows = np.asarray(market_prediction, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if not (ridge.shape == xs.shape == market_rows.shape == ids.shape):
        raise ValueError("hybrid component predictions must have identical shapes")
    if not np.isfinite(market_lambda) or not np.isfinite(blend_weight):
        raise ValueError("hybrid weights must be finite")
    market_ridge = _group_mean(ridge, ids)
    e_ridge = ridge - market_ridge
    e_lgbm = xs - _group_mean(xs, ids)
    market_lgbm = _group_mean(market_rows, ids)
    market = (1.0 - market_lambda) * market_ridge + market_lambda * market_lgbm
    cross = (1.0 - blend_weight) * e_ridge + blend_weight * e_lgbm
    return market + cross


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
