"""Deterministic selectors and paired runner for ROADMAP P4.

The selector functions in this module are pure: they accept training-fold
arrays and return global feature indices without reading files or model state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

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
