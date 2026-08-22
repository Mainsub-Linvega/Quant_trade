"""Deterministic selectors and paired runner for ROADMAP P4.

The selector functions in this module are pure: they accept training-fold
arrays and return global feature indices without reading files or model state.
"""

from __future__ import annotations

import argparse
import os
import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import sys
import time
from typing import Any
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import numpy as np

from experiments.v3_feature_structure import contiguous_time_blocks
from strategies.v3_hybrid.history import AssetHistory

P4_ARMS = (
    "baseline_corr",
    "market_task_aligned",
    "xs_time_stable",
    "history_lag_aligned",
)
P4_COUNTS = {"ridge": 200, "xs": 200, "market": 200, "history": 40}
P4_COMMON_PROTOCOL = {
    "n_folds": 5,
    "train_window": 78_960,
    "embargo": 6,
    "sample_modulo": 5,
    "sampling": "phase_balanced",
    "market_lambda": 0.7,
    "blend_weight": 1.17,
    "prediction_scale": 1.16,
    "prediction_clip": 0.5,
    "history_window": 5,
}
def parse_p4_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(P4_MODE_PROTOCOL), default="screen")
    parser.add_argument("--arm-set", default=",".join(P4_ARMS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--label", default="v3_p4_task_aligned_screen_1s160_phasebal_prodwindow")
    parser.add_argument("--n-folds", type=int, default=P4_COMMON_PROTOCOL["n_folds"])
    parser.add_argument("--train-window", type=int, default=P4_COMMON_PROTOCOL["train_window"])
    parser.add_argument("--embargo", type=int, default=P4_COMMON_PROTOCOL["embargo"])
    parser.add_argument("--sample-modulo", type=int, default=P4_COMMON_PROTOCOL["sample_modulo"])
    parser.add_argument("--sampling", choices=["periodic", "phase_balanced"], default=P4_COMMON_PROTOCOL["sampling"])
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--num-iteration", type=int, default=None)
    parser.add_argument("--market-lambda", type=float, default=P4_COMMON_PROTOCOL["market_lambda"])
    parser.add_argument("--blend-weight", type=float, default=P4_COMMON_PROTOCOL["blend_weight"])
    parser.add_argument("--prediction-scale", type=float, default=P4_COMMON_PROTOCOL["prediction_scale"])
    parser.add_argument("--prediction-clip", type=float, default=P4_COMMON_PROTOCOL["prediction_clip"])
    parser.add_argument("--history-window", type=int, default=P4_COMMON_PROTOCOL["history_window"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    mode_defaults = P4_MODE_PROTOCOL[args.mode]
    if args.n_seeds is None:
        args.n_seeds = mode_defaults["n_seeds"]
    if args.num_iteration is None:
        args.num_iteration = mode_defaults["num_iteration"]
    arms = [item.strip() for item in str(args.arm_set).split(",") if item.strip()]
    if not arms or len(set(arms)) != len(arms) or any(item not in P4_ARMS for item in arms):
        parser.error(f"--arm-set must contain unique registered arms: {P4_ARMS}")
    args.arm_set = arms
    protocol = {
        name: getattr(args, name)
        for name in P4_COMMON_PROTOCOL
        if hasattr(args, name)
    }
    protocol.update({"n_seeds": args.n_seeds, "num_iteration": args.num_iteration})
    try:
        validate_frozen_protocol(args.mode, protocol)
    except ValueError as error:
        parser.error(str(error))
    if args.seed != 2026 or args.num_threads <= 0:
        parser.error("P4 seed must be 2026 and num_threads must be positive")
    return args
P4_MODE_PROTOCOL = {
    "screen": {"n_seeds": 1, "num_iteration": 160},
    "confirmation": {"n_seeds": 3, "num_iteration": 480},
}


def validate_frozen_protocol(mode: str, protocol: Mapping[str, object]) -> dict[str, object]:
    if mode not in P4_MODE_PROTOCOL:
        raise ValueError(f"unknown P4 mode: {mode}")
    expected = {**P4_COMMON_PROTOCOL, **P4_MODE_PROTOCOL[mode]}
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise ValueError(f"{name}={protocol.get(name)!r}; expected {value!r}")
    return expected
def _rank_baseline_features(
    features: np.ndarray,
    target: np.ndarray,
    count: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("baseline ranking inputs must be aligned")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(labels)):
        raise ValueError("baseline ranking inputs must be finite")
    if weights is None:
        mass = np.ones(len(labels), dtype=np.float64)
    else:
        mass = np.asarray(weights, dtype=np.float64)
        if mass.shape != labels.shape or not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
            raise ValueError("weights must be finite, non-negative, and aligned")
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("weights must contain positive mass")
    mean_x = np.sum(values * mass[:, None], axis=0) / total
    mean_y = float(np.sum(labels * mass) / total)
    centered_x = values - mean_x
    centered_y = labels - mean_y
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.abs(
            np.sum(centered_x * (centered_y * mass)[:, None], axis=0)
            / np.sqrt(
                np.sum(np.square(centered_x) * mass[:, None], axis=0)
                * np.sum(np.square(centered_y) * mass)
            )
        )
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    indices = np.arange(values.shape[1], dtype=np.int64)
    return indices[np.lexsort((indices, -correlations))[:count]]


def derive_p4_selections(
    transformed: np.ndarray,
    target: np.ndarray,
    cross_target: np.ndarray,
    time_ids: np.ndarray,
    asset_ids: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    counts: Mapping[str, int] = P4_COUNTS,
) -> dict[str, Any]:
    """Derive the baseline and all three registered candidate arms for one fold."""
    values, labels, ids = _validate_panel(
        transformed, target, time_ids, int(counts["ridge"]),
    )
    cross = np.asarray(cross_target, dtype=np.float64)
    assets = np.asarray(asset_ids, dtype=np.int64)
    if cross.shape != labels.shape or assets.shape != labels.shape:
        raise ValueError("target views and asset_ids must match training rows")
    if not np.all(np.isfinite(cross)):
        raise ValueError("cross_target must be finite")
    width = values.shape[1]
    for task in P4_COUNTS:
        if task not in counts or int(counts[task]) <= 0 or int(counts[task]) > width:
            raise ValueError(f"invalid P4 count for {task}")
    ridge = _rank_baseline_features(values, labels, int(counts["ridge"]), weights)
    xs = _rank_baseline_features(values, cross, int(counts["xs"]))
    history = select_history_lag_aligned(
        values[:, np.sort(xs)], cross, ids, assets, np.sort(xs),
        count=int(counts["history"]), window=P4_COMMON_PROTOCOL["history_window"],
    )["selected_indices"]
    baseline = {"ridge": ridge, "xs": xs, "market": xs.copy(), "history": history}
    candidates = {
        "market_task_aligned": {"market": select_market_task_aligned(
            values, labels, ids, count=int(counts["market"]),
        )},
        "xs_time_stable": {"xs": select_xs_time_stable(
            values, cross, ids, count=int(counts["xs"]),
        )},
        "history_lag_aligned": {"history": select_history_lag_aligned(
            values[:, np.sort(xs)], cross, ids, assets, np.sort(xs),
            count=int(counts["history"]), window=P4_COMMON_PROTOCOL["history_window"],
        )["selected_indices"]},
    }
    arms = {
        arm: resolve_p4_arm(arm, baseline, candidates, counts)
        for arm in P4_ARMS
    }
    return {"baseline": baseline, "candidates": candidates, "arms": arms}


def _validate_panel(
    features: np.ndarray,
    target: np.ndarray,
    time_ids: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    if labels.shape != (len(values),) or ids.shape != (len(values),):
        raise ValueError("target and time_ids must match feature rows")
    if np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be sorted")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(labels)):
        raise ValueError("features and target must be finite")
    if count <= 0 or count > values.shape[1]:
        raise ValueError("count must be within the feature width")
    return values, labels, ids


def _group_layout(time_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    return starts, counts


def _correlation_columns(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(
        np.sum(np.square(x_centered), axis=0) * float(y_centered @ y_centered)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=denominator > 0.0,
    )


def select_market_task_aligned(
    transformed: np.ndarray,
    target: np.ndarray,
    time_ids: np.ndarray,
    count: int = 200,
) -> np.ndarray:
    """Rank features by market-mean association within the training fold."""
    values, labels, ids = _validate_panel(transformed, target, time_ids, count)
    starts, counts = _group_layout(ids)
    market_features = np.add.reduceat(values, starts, axis=0) / counts[:, None]
    market_target = np.add.reduceat(labels, starts) / counts
    correlations = np.abs(_correlation_columns(market_features, market_target))
    feature_indices = np.arange(values.shape[1], dtype=np.int64)
    order = np.lexsort((feature_indices, -correlations))
    return feature_indices[order[:count]]


def select_xs_time_stable(
    transformed: np.ndarray,
    cross_target: np.ndarray,
    time_ids: np.ndarray,
    count: int = 200,
    n_blocks: int = 4,
) -> np.ndarray:
    """Rank cross-sectional features by stable blockwise target association."""
    values, labels, ids = _validate_panel(
        transformed, cross_target, time_ids, count
    )
    blocks = contiguous_time_blocks(ids, n_blocks)
    correlations = np.vstack(
        [_correlation_columns(values[block], labels[block]) for block in blocks]
    )
    median_abs = np.median(np.abs(correlations), axis=0)
    positive = np.mean(correlations > 0.0, axis=0)
    negative = np.mean(correlations < 0.0, axis=0)
    consistency = np.maximum(positive, negative)
    score = median_abs * consistency
    feature_indices = np.arange(values.shape[1], dtype=np.int64)
    order = np.lexsort((feature_indices, -median_abs, -score))
    return feature_indices[order[:count]]


def _cross_sectional_center(values: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64).copy()
    starts, counts = _group_layout(time_ids)
    means = np.add.reduceat(centered, starts, axis=0) / counts[:, None]
    centered -= np.repeat(means, counts, axis=0)
    return centered


def select_history_lag_aligned(
    transformed_xs: np.ndarray,
    cross_target: np.ndarray,
    time_ids: np.ndarray,
    asset_ids: np.ndarray | None,
    candidate_indices: np.ndarray,
    count: int = 40,
    window: int = 5,
    n_blocks: int = 4,
) -> dict[str, object]:
    """Rank baseline XS bases by stable causal history association."""
    values = np.asarray(transformed_xs, dtype=np.float64)
    labels = np.asarray(cross_target, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if asset_ids is None:
        raise ValueError("asset_ids are required for causal history selection")
    assets = np.asarray(asset_ids, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("transformed_xs must be a non-empty 2D array")
    if labels.shape != (len(values),) or ids.shape != (len(values),):
        raise ValueError("cross_target and time_ids must match history rows")
    if assets.shape != (len(values),):
        raise ValueError("asset_ids must match history rows")
    if np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be sorted")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(labels)):
        raise ValueError("history inputs must be finite")
    if candidates.ndim != 1 or len(candidates) != values.shape[1]:
        raise ValueError("candidate_indices must match transformed_xs columns")
    if len(set(candidates.tolist())) != len(candidates) or np.any(candidates < 0):
        raise ValueError("candidate_indices must be unique and non-negative")
    if count <= 0 or count > len(candidates) or window <= 0:
        raise ValueError("history count and window must be valid")

    history = AssetHistory(feature_count=values.shape[1], window_size=window)
    history_blocks = history.transform(values.astype(np.float32), assets)
    family_names = (
        "previous", "difference", "rolling_mean", "rolling_deviation",
    )
    row_blocks = contiguous_time_blocks(ids, n_blocks)
    correlations = np.zeros(
        (n_blocks, len(family_names), values.shape[1]), dtype=np.float64
    )
    for family_index, family_values in enumerate(history_blocks):
        centered = _cross_sectional_center(family_values, ids)
        for block_index, rows in enumerate(row_blocks):
            correlations[block_index, family_index] = _correlation_columns(
                centered[rows], labels[rows]
            )

    median_abs = np.median(np.abs(correlations), axis=0)
    positive = np.mean(correlations > 0.0, axis=0)
    negative = np.mean(correlations < 0.0, axis=0)
    consistency = np.maximum(positive, negative)
    family_scores = median_abs * consistency
    best_family = np.argmax(family_scores, axis=0)
    columns = np.arange(values.shape[1], dtype=np.int64)
    best_score = family_scores[best_family, columns]
    best_median_abs = median_abs[best_family, columns]
    order = np.lexsort((candidates, -best_median_abs, -best_score))
    chosen = order[:count]
    return {
        "selected_indices": candidates[chosen].copy(),
        "selected_count": int(count),
        "families": [family_names[int(best_family[column])] for column in chosen],
        "evidence": [
            {
                "feature": int(candidates[column]),
                "family": family_names[int(best_family[column])],
                "score": float(best_score[column]),
                "median_abs_correlation": float(best_median_abs[column]),
                "direction_consistency": float(
                    consistency[int(best_family[column]), column]
                ),
            }
            for column in order
        ],
    }


def _validated_feature_set(
    task: str,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != count:
        raise ValueError(f"{task} must contain exactly {count} indices")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"{task} indices must be unique")
    if np.any(indices < 0):
        raise ValueError(f"{task} indices must be non-negative")
    return indices.copy()


def resolve_p4_arm(
    arm: str,
    baseline: Mapping[str, np.ndarray],
    candidates: Mapping[str, Mapping[str, np.ndarray]],
    counts: Mapping[str, int] = P4_COUNTS,
) -> dict[str, np.ndarray]:
    """Apply exactly the selection stage registered for one P4 arm."""
    if arm not in P4_ARMS:
        raise ValueError(f"unknown P4 arm: {arm}")
    required = tuple(P4_COUNTS)
    if set(counts) != set(required) or set(baseline) != set(required):
        raise ValueError("P4 counts and baseline must define all four tasks")
    resolved = {
        task: _validated_feature_set(task, baseline[task], int(counts[task]))
        for task in required
    }
    baseline_xs = set(resolved["xs"])
    if not set(resolved["history"]).issubset(baseline_xs):
        raise ValueError("baseline history features must be a subset of baseline XS")
    changed_task = {
        "market_task_aligned": "market",
        "xs_time_stable": "xs",
        "history_lag_aligned": "history",
    }.get(arm)
    if changed_task is None:
        return resolved
    arm_candidates = candidates.get(arm)
    if arm_candidates is None or set(arm_candidates) != {changed_task}:
        raise ValueError(f"{arm} must provide only the {changed_task} selection")
    resolved[changed_task] = _validated_feature_set(
        changed_task,
        arm_candidates[changed_task],
        int(counts[changed_task]),
    )
    if changed_task == "history" and not set(resolved["history"]).issubset(baseline_xs):
        raise ValueError("history candidate must be a subset of baseline XS")
    return resolved


def paired_gate(fold_rows: list[Mapping[str, object]]) -> dict[str, object]:
    """Evaluate the frozen five-fold paired P4 acceptance gate."""
    if len(fold_rows) != 5:
        raise ValueError("paired gate requires exactly five fold rows")
    peak_delta: list[float] = []
    relative_delta_a: list[float] = []
    relative_delta_b: list[float] = []
    fold_deltas: list[dict[str, float | int]] = []
    tiny = np.finfo(np.float64).tiny
    all_values: list[float] = []
    for row in fold_rows:
        baseline = row.get("baseline")
        candidate = row.get("candidate")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError("each fold row requires baseline and candidate metrics")
        base_peak = float(baseline["peak"])
        base_a = float(baseline["A"])
        base_b = float(baseline["B"])
        cand_peak = float(candidate["peak"])
        cand_a = float(candidate["A"])
        cand_b = float(candidate["B"])
        values = [base_peak, base_a, base_b, cand_peak, cand_a, cand_b]
        all_values.extend(values)
        delta_peak = cand_peak - base_peak
        delta_a = cand_a - base_a
        delta_b = cand_b - base_b
        rel_a = delta_a / max(abs(base_a), tiny)
        rel_b = delta_b / max(abs(base_b), tiny)
        peak_delta.append(delta_peak)
        relative_delta_a.append(rel_a)
        relative_delta_b.append(rel_b)
        fold_deltas.append({
            "fold": int(row["fold"]),
            "peak_delta": float(delta_peak),
            "relative_delta_A": float(rel_a),
            "relative_delta_B": float(rel_b),
        })

    peak = np.asarray(peak_delta, dtype=np.float64)
    rel_a = np.asarray(relative_delta_a, dtype=np.float64)
    rel_b = np.asarray(relative_delta_b, dtype=np.float64)
    all_finite = bool(np.all(np.isfinite(all_values)))
    mean_peak_delta = float(np.mean(peak))
    positive_folds = int(np.sum(peak > 0.0))
    drop_best = np.delete(peak, int(np.argmax(peak)))
    drop_best_mean = float(np.mean(drop_best))
    mean_relative_delta_a = float(np.mean(rel_a))
    mean_relative_delta_b = float(np.mean(rel_b))
    alignment_energy_passed = bool(
        2.0 * mean_relative_delta_a > mean_relative_delta_b
    )
    passed = bool(
        all_finite
        and mean_peak_delta > 0.0
        and positive_folds >= 4
        and drop_best_mean > 0.0
        and alignment_energy_passed
    )
    return {
        "passed": passed,
        "all_finite": all_finite,
        "mean_peak_delta": mean_peak_delta,
        "positive_folds": positive_folds,
        "drop_best_mean_peak_delta": drop_best_mean,
        "mean_relative_delta_A": mean_relative_delta_a,
        "mean_relative_delta_B": mean_relative_delta_b,
        "alignment_energy_passed": alignment_energy_passed,
        "fold_deltas": fold_deltas,
    }


def candidate_gate_impossible(
    deltas: list[float] | np.ndarray,
    *,
    total_folds: int = 5,
    required_positive: int = 4,
) -> bool:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) > total_folds:
        raise ValueError("fold deltas must be a partial one-dimensional sequence")
    if total_folds <= 0 or not 0 < required_positive <= total_folds:
        raise ValueError("invalid positive-fold gate")
    if not np.all(np.isfinite(values)):
        raise ValueError("fold deltas must be finite")
    positives = int(np.sum(values > 0.0))
    remaining = total_folds - len(values)
    return positives + remaining < required_positive


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# P4 Task-Aligned Feature Reselection",
        "",
        f"Status: **{payload.get('status', 'unknown')}**",
        "",
        "| Fold | Arm | Peak | A | B |",
        "|---:|---|---:|---:|---:|",
    ]
    for fold in payload.get("folds", []):
        if not isinstance(fold, Mapping):
            continue
        arms = fold.get("arms", {})
        if not isinstance(arms, Mapping):
            continue
        for arm, metrics in arms.items():
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"| {fold['fold']} | {arm} | {float(metrics.get('peak', np.nan)):.8f} | "
                f"{float(metrics.get('A', np.nan)):.8f} | {float(metrics.get('B', np.nan)):.8f} |"
            )
    lines.extend([
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(payload.get("gates", {}), ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "No candidate arms were combined after observing results. No submission CSV was generated.",
    ])
    return "\n".join(lines) + "\n"


def write_p4_bundle(payload: Mapping[str, object], output_dir: str | Path, label: str) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fold_dir = directory / f"{label}_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold in payload.get("folds", []):
        if not isinstance(fold, Mapping) or "fold" not in fold:
            raise ValueError("every fold artifact requires a fold index")
        fold_path = fold_dir / f"fold_{int(fold['fold'])}.json"
        _atomic_write_text(
            fold_path,
            json.dumps(fold, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        )
    json_path = directory / f"{label}.json"
    markdown_path = directory / f"{label}.md"
    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )
    _atomic_write_text(markdown_path, _render_markdown(payload))
    return {"json": json_path, "markdown": markdown_path, "fold_dir": fold_dir}
def _selection_hash(selection: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for task in ("ridge", "xs", "market", "history"):
        digest.update(task.encode("ascii"))
        digest.update(np.asarray(selection[task], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _group_mean_1d(values: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if values.shape != ids.shape or values.ndim != 1 or len(ids) == 0:
        raise ValueError("group mean inputs must be aligned and non-empty")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    return np.repeat(np.add.reduceat(values, starts) / counts, counts)


def _p4_fold_metric(
    target: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    *,
    scale: float,
    clip: float,
) -> dict[str, float]:
    from src.metric import scale_invariant_score, weighted_zero_mean_r2

    raw = scale_invariant_score(target, prediction, weight)
    frozen_prediction = np.clip(prediction * scale, -clip, clip)
    frozen = scale_invariant_score(target, frozen_prediction, weight)
    return {
        "score": float(weighted_zero_mean_r2(target, frozen_prediction, weight)),
        "peak": float(raw["peak"]),
        "A": float(raw["A"]),
        "B": float(raw["B"]),
        "optimal_scale": float(raw["optimal_scale"]),
        "frozen_score": float(weighted_zero_mean_r2(target, frozen_prediction, weight)),
        "frozen_peak": float(frozen["peak"]),
        "frozen_A": float(frozen["A"]),
        "frozen_B": float(frozen["B"]),
    }


def runner_import_paths(repo_root: Path) -> tuple[str, str]:
    """Return the legacy module paths required by the reused V3 OOF helpers."""
    return (
        str(repo_root / "strategies" / "v1_ridge"),
        str(repo_root / "experiments"),
    )

def allocate_p4_arrays(
    n_rows: int, feature_count: int, feature_path: str | Path,
) -> dict[str, np.ndarray]:
    """Allocate P4 arrays without creating an in-memory feature-matrix copy."""
    if n_rows <= 0 or feature_count <= 0:
        raise ValueError("P4 array dimensions must be positive")
    path = Path(feature_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return {
        "features": np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(n_rows, feature_count),
        ),
        "target": np.empty(n_rows, dtype=np.float64),
        "weight": np.empty(n_rows, dtype=np.float64),
        "time_id": np.empty(n_rows, dtype=np.int64),
        "asset_id": np.empty(n_rows, dtype=np.int64),
    }


def reuse_p4_feature_memmap(
    feature_path: str | Path, n_rows: int, feature_count: int,
) -> np.memmap:
    """Open a previously completed P4 feature memmap after validating shape."""
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(path)
    mapped = np.load(path, mmap_mode="r")
    if not isinstance(mapped, np.memmap):
        raise ValueError("P4 feature cache is not a NumPy memmap")
    if mapped.shape != (n_rows, feature_count) or mapped.dtype != np.dtype(np.float32):
        raise ValueError(
            f"invalid P4 feature cache shape/dtype: {mapped.shape}/{mapped.dtype}"
        )
    return mapped

def load_p4_rows(
    data_root: Path,
    sample_modulo: int,
    sampling: str,
    feature_path: str | Path,
) -> dict[str, np.ndarray]:
    """Two-pass sampled loader that writes features directly to disk."""
    import pyarrow.parquet as pq

    from src.io import FEATURE_COLUMNS, time_sample_mask, train_files

    files = list(train_files(data_root))
    if not files:
        raise ValueError("no training partitions found")
    count = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=120_000, columns=["time_id"],
        ):
            time_id = batch.column(0).to_numpy(zero_copy_only=False)
            count += int(np.sum(time_sample_mask(time_id, sample_modulo, sampling=sampling)))
    arrays = allocate_p4_arrays(count, len(FEATURE_COLUMNS), feature_path)
    columns = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]
    offset = 0
    for path in files:
        kept = 0
        started = time.perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            mask = time_sample_mask(
                frame["time_id"].to_numpy(copy=False), sample_modulo, sampling=sampling,
            )
            rows = int(np.sum(mask))
            if rows == 0:
                continue
            end = offset + rows
            arrays["features"][offset:end] = frame.loc[mask, FEATURE_COLUMNS].to_numpy(
                dtype=np.float32, copy=False,
            )
            arrays["target"][offset:end] = frame.loc[mask, "target"].to_numpy(
                dtype=np.float64, copy=False,
            )
            arrays["weight"][offset:end] = frame.loc[mask, "weight"].to_numpy(
                dtype=np.float64, copy=False,
            )
            arrays["time_id"][offset:end] = frame.loc[mask, "time_id"].to_numpy(
                dtype=np.int64, copy=False,
            )
            arrays["asset_id"][offset:end] = frame.loc[mask, "asset_id"].to_numpy(
                dtype=np.int64, copy=False,
            )
            offset = end
            kept += rows
        print(f"  {path.name}: {kept:,} rows ({time.perf_counter()-started:.0f}s)", flush=True)
    if offset != count:
        raise RuntimeError(f"P4 loader wrote {offset} rows after counting {count}")
    arrays["features"].flush()
    return arrays

def load_p4_metadata(
    data_root: Path,
    sample_modulo: int,
    sampling: str,
    n_rows: int,
) -> dict[str, np.ndarray]:
    """Load sampled labels and row keys while reusing an existing feature map."""
    import pyarrow.parquet as pq

    from src.io import time_sample_mask, train_files

    arrays = {
        "target": np.empty(n_rows, dtype=np.float64),
        "weight": np.empty(n_rows, dtype=np.float64),
        "time_id": np.empty(n_rows, dtype=np.int64),
        "asset_id": np.empty(n_rows, dtype=np.int64),
    }
    offset = 0
    columns = ["time_id", "asset_id", "weight", "target"]
    for path in train_files(data_root):
        kept = 0
        started = time.perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            mask = time_sample_mask(
                frame["time_id"].to_numpy(copy=False), sample_modulo, sampling=sampling,
            )
            rows = int(np.sum(mask))
            if rows == 0:
                continue
            end = offset + rows
            if end > n_rows:
                raise RuntimeError("P4 metadata rows exceed the feature cache")
            for name, dtype in (("target", np.float64), ("weight", np.float64),
                                ("time_id", np.int64), ("asset_id", np.int64)):
                arrays[name][offset:end] = frame.loc[mask, name].to_numpy(
                    dtype=dtype, copy=False,
                )
            offset = end
            kept += rows
        print(f"  {path.name}: {kept:,} metadata rows ({time.perf_counter()-started:.0f}s)", flush=True)
    if offset != n_rows:
        raise RuntimeError(f"P4 metadata wrote {offset} rows for {n_rows} cached features")
    return arrays


def spill_p4_features(
    data: dict[str, np.ndarray], path: str | Path,
) -> np.memmap:
    """Release the loaded feature matrix after moving it to a read-only memmap."""
    from experiments.v3_production_oof import spill_feature_matrix

    if "features" not in data:
        raise ValueError("loaded P4 data must contain features")
    loaded_features = data.pop("features")
    mapped = spill_feature_matrix(loaded_features, path)
    del loaded_features
    return mapped

    """Run the registered paired P4 screen or confirmation experiment."""
def run_p4(args: argparse.Namespace) -> dict[str, object]:
    validate_frozen_protocol(args.mode, {
        name: getattr(args, name)
        for name in P4_COMMON_PROTOCOL
        if hasattr(args, name)
    } | {"n_seeds": args.n_seeds, "num_iteration": args.num_iteration})
    repo_root = Path(__file__).resolve().parents[1]
    for path in reversed(runner_import_paths(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from experiments.v3_production_oof import (
        build_task_lgbm_designs,
        fit_predict_lgbm,
        row_slice,
    )
    from experiments.history_peak import fit_ridge, ridge_designs
    from src.io import FEATURE_COLUMNS
    from src.validation import rolling_time_folds
    from strategies.v3_hybrid.train import stream_history_blocks
    from train import robust_transform_fit
    from experiments.v3_interaction_oof import compose_hybrid_raw
    from features import cross_sectional_deviation

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    npz_path = cache_dir / f"{args.label}.npz"
    if not args.force and any(path.exists() for path in (json_path, md_path, npz_path)):
        raise SystemExit(f"output exists; use --force: {json_path}")

    started = time.perf_counter()
    feature_spill_path = cache_dir / f".{args.label}_features.npy"
    if feature_spill_path.exists():
        cached = np.load(feature_spill_path, mmap_mode="r")
        features = reuse_p4_feature_memmap(
            feature_spill_path, cached.shape[0], len(FEATURE_COLUMNS),
        )
        data = load_p4_metadata(
            data_root, args.sample_modulo, args.sampling, cached.shape[0],
        )
        del cached
    else:
        print(f"streaming sampled data: modulo {args.sample_modulo}/{args.sampling}", flush=True)
        data = load_p4_rows(data_root, args.sample_modulo, args.sampling, feature_spill_path)
        features = data["features"]
    gc.collect()
    target = data["target"].astype(np.float64, copy=False)
    weight = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    time_ids = np.asarray(data["time_id"], dtype=np.int64)
    asset_ids = np.asarray(data["asset_id"], dtype=np.int64)
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)
    n = len(target)
    evaluated_arms = ["baseline_corr", *[arm for arm in args.arm_set if arm != "baseline_corr"]]
    oof = {arm: np.full(n, np.nan, dtype=np.float64) for arm in evaluated_arms}
    fold_id = np.full(n, -1, dtype=np.int16)
    fold_records: list[dict[str, object]] = []
    paired_rows = {arm: [] for arm in evaluated_arms if arm != "baseline_corr"}
    protocol = {
        **P4_COMMON_PROTOCOL,
        "n_seeds": args.n_seeds,
        "num_iteration": args.num_iteration,
        "mode": args.mode,
    }
    partial: dict[str, object] = {
        "experiment": "p4_task_aligned_feature_reselection",
        "status": "running",
        "protocol": protocol,
        "arms": evaluated_arms,
        "folds": fold_records,
        "gates": {},
        "submission_generated": False,
    }

    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        tr = row_slice(time_ids, train_ids)
        va = row_slice(time_ids, valid_ids)
        raw_train = np.asarray(features[tr])
        transformed_train, stats = robust_transform_fit(raw_train.copy())
        transformed_valid = np.asarray(features[va]).copy()
        from features import apply_robust_transform
        apply_robust_transform(
            transformed_valid, stats["lower"], stats["upper"],
            stats["center"], stats["scale"],
        )
        y_tr, y_va = target[tr], target[va]
        w_tr, w_va = weight[tr], weight[va]
        tid_tr, tid_va = time_ids[tr], time_ids[va]
        aid_tr, aid_va = asset_ids[tr], asset_ids[va]
        cross_target = y_tr - _group_mean_1d(y_tr, tid_tr)
        selections = derive_p4_selections(
            transformed_train, y_tr, cross_target, tid_tr, aid_tr,
            weights=w_tr, counts=P4_COUNTS,
        )
        baseline = selections["baseline"]
        history_cache: dict[tuple[int, ...], tuple[list[np.ndarray], list[np.ndarray]]] = {}

        ridge_train = ridge_designs(transformed_train, tid_tr, baseline["ridge"], None)
        ridge_valid = ridge_designs(transformed_valid, tid_va, baseline["ridge"], None)
        ridge_alpha = 2_000_000.0 * len(train_ids) / 78_960
        ridge = fit_ridge(ridge_train, y_tr, w_tr, ridge_alpha)
        ridge_prediction = ridge.predict(ridge_valid).astype(np.float64)
        del ridge_train, ridge_valid, ridge
        ridge_market = _group_mean_1d(ridge_prediction, tid_va)
        ridge_residual = ridge_prediction - ridge_market

        def get_history(selection: Mapping[str, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
            key = tuple(int(v) for v in selection["history"])
            if key not in history_cache:
                names = [FEATURE_COLUMNS[index] for index in selection["history"]]
                history_stats = tuple(stats[name][selection["history"]]
                                      for name in ("lower", "upper", "center", "scale"))
                all_blocks = stream_history_blocks(
                    data_root, args.sample_modulo, args.sampling, names,
                    history_stats, args.history_window,
                )
                history_cache[key] = (
                    [np.asarray(block[tr]) for block in all_blocks],
                    [np.asarray(block[va]) for block in all_blocks],
                )
                del all_blocks
            return history_cache[key]

        component_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def fit_components(
            selection: Mapping[str, np.ndarray], name: str,
            tasks: tuple[str, ...] = ("xs", "market"),
        ) -> tuple[np.ndarray, np.ndarray]:
            history_tr, history_va = get_history(selection)
            designs_tr = build_task_lgbm_designs(
                transformed_train, tid_tr, aid_tr,
                xs_indices=selection["xs"], market_indices=selection["market"],
                history_blocks=history_tr,
            )
            designs_va = build_task_lgbm_designs(
                transformed_valid, tid_va, aid_va,
                xs_indices=selection["xs"], market_indices=selection["market"],
                history_blocks=history_va,
            )
            if tasks == ("market",):
                return np.empty(0, dtype=np.float64), fit_predict_lgbm(
                    designs_tr["market"], y_tr, None, designs_va["market"],
                    args, f"fold {fold_index} {name} Market", {
                        "num_leaves": 15, "learning_rate": 0.02,
                        "feature_fraction": 0.4, "lambda_l2": 30.0,
                    }, min_data_scale=25.0 / 3.0, num_iteration=args.num_iteration,
                )
            xs_prediction = fit_predict_lgbm(
                designs_tr["xs"], cross_target, w_tr, designs_va["xs"],
                args, f"fold {fold_index} {name} XS", {
                    "num_leaves": 63, "learning_rate": 0.03,
                    "feature_fraction": 0.7, "lambda_l2": 1.0,
                }, num_iteration=args.num_iteration,
            )
            if tasks == ("xs",):
                return xs_prediction, np.empty(0, dtype=np.float64)
            market_prediction = fit_predict_lgbm(
                designs_tr["market"], y_tr, None, designs_va["market"],
                args, f"fold {fold_index} {name} Market", {
                    "num_leaves": 15, "learning_rate": 0.02,
                    "feature_fraction": 0.4, "lambda_l2": 30.0,
                }, min_data_scale=25.0 / 3.0, num_iteration=args.num_iteration,
            )
            return xs_prediction, market_prediction

        def get_components(arm: str, selection: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            if arm in component_cache:
                return component_cache[arm]
            if arm == "baseline_corr" or arm == "history_lag_aligned":
                result = fit_components(selection, arm)
            else:
                base_xs, base_market = component_cache["baseline_corr"]
                if arm == "market_task_aligned":
                    _, market = fit_components(selection, arm, ("market",))
                    result = base_xs, market
                elif arm == "xs_time_stable":
                    xs, _ = fit_components(selection, arm, ("xs",))
                    result = xs, base_market
                else:
                    raise ValueError(f"unknown P4 arm: {arm}")
            component_cache[arm] = result
            return result

        # Baseline is always materialized first so candidate arms can reuse it.
        get_components("baseline_corr", selections["arms"]["baseline_corr"])
        arm_metrics: dict[str, object] = {}
        out_indices = np.arange(va.start, va.stop)
        for arm in evaluated_arms:
            xs_prediction, market_prediction = get_components(arm, selections["arms"][arm])
            raw_prediction = compose_hybrid_raw(
                ridge_prediction, xs_prediction, market_prediction, tid_va,
                market_lambda=args.market_lambda, blend_weight=args.blend_weight,
            )
            frozen_prediction = np.clip(
                raw_prediction * args.prediction_scale,
                -args.prediction_clip, args.prediction_clip,
            )
            metric = _p4_fold_metric(
                y_va, raw_prediction, w_va,
                scale=args.prediction_scale, clip=args.prediction_clip,
            )
            oof[arm][out_indices] = frozen_prediction
            arm_metrics[arm] = {
                **metric,
                "selection_hash": _selection_hash(selections["arms"][arm]),
                "ridge": [int(v) for v in selections["arms"][arm]["ridge"]],
                "xs": [int(v) for v in selections["arms"][arm]["xs"]],
                "market": [int(v) for v in selections["arms"][arm]["market"]],
                "history": [int(v) for v in selections["arms"][arm]["history"]],
            }
            if arm != "baseline_corr":
                paired_rows[arm].append({
                    "fold": fold_index,
                    "baseline": arm_metrics["baseline_corr"],
                    "candidate": arm_metrics[arm],
                })
        fold_id[out_indices] = fold_index
        fold_records.append({
            "fold": fold_index,
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "train_rows": int(len(y_tr)),
            "valid_rows": int(len(y_va)),
            "arms": arm_metrics,
            "elapsed_seconds": float(time.perf_counter() - fold_started),
        })
        partial["folds"] = fold_records
        write_p4_bundle(partial, output_dir, args.label)
        del transformed_train, transformed_valid, stats, component_cache, history_cache
        gc.collect()

    gates: dict[str, object] = {}
    for arm, rows in paired_rows.items():
        gates[arm] = paired_gate(rows) if len(rows) == 5 else {
            "passed": False, "reason": "incomplete_five_fold_evidence", "completed_folds": len(rows),
        }
    partial["status"] = "completed"
    partial["gates"] = gates
    partial["elapsed_seconds"] = float(time.perf_counter() - started)
    del features
    feature_spill_path.unlink(missing_ok=True)
    partial["pooled"] = {
        arm: _p4_fold_metric(target[fold_id >= 0], oof[arm][fold_id >= 0], weight[fold_id >= 0],
                             scale=args.prediction_scale, clip=args.prediction_clip)
        for arm in evaluated_arms
    }
    partial["submission_generated"] = False
    np.savez_compressed(npz_path, target=target, weight=weight, time_id=time_ids,
                        asset_id=asset_ids, fold=fold_id, **oof)
    write_p4_bundle(partial, output_dir, args.label)
    return partial


def main() -> None:
    args = parse_p4_args()
    run_p4(args)




if __name__ == "__main__":
    main()
