"""Post-hoc, training-only diagnostics for ROADMAP P4 Market200 selection."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from experiments.v3_feature_structure import contiguous_time_blocks


def market_overlap_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    """Summarize the Market200 replacement set for one paired fold."""
    base = set(np.asarray(baseline, dtype=np.int64).tolist())
    cand = set(np.asarray(candidate, dtype=np.int64).tolist())
    if not base or not cand:
        raise ValueError("market sets must be non-empty")
    overlap = sorted(base & cand)
    union = base | cand
    return {
        "baseline_count": len(base),
        "candidate_count": len(cand),
        "overlap_count": len(overlap),
        "jaccard": float(len(overlap) / len(union)),
        "added": sorted(cand - base),
        "removed": sorted(base - cand),
    }


def attribute_peak_delta(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    scale: float,
) -> dict[str, object]:
    """Attribute a paired Peak change to alignment, energy, or both."""
    base_a, base_b, base_peak = (float(baseline[name]) for name in ("A", "B", "peak"))
    cand_a, cand_b, cand_peak = (float(candidate[name]) for name in ("A", "B", "peak"))
    delta_a = cand_a - base_a
    delta_b = cand_b - base_b
    delta_peak = cand_peak - base_peak
    score_delta_alignment = 2.0 * scale * delta_a
    score_delta_energy = -(scale ** 2) * delta_b
    if delta_peak > 0.0:
        cause = "improved"
    elif delta_a < 0.0 and delta_b <= 0.0:
        cause = "alignment_loss"
    elif delta_a >= 0.0 and delta_b > 0.0:
        cause = "energy_inflation"
    else:
        cause = "mixed"
    return {
        "delta_A": float(delta_a),
        "delta_B": float(delta_b),
        "delta_peak": float(delta_peak),
        "frozen_scale_alignment_term": float(score_delta_alignment),
        "frozen_scale_energy_term": float(score_delta_energy),
        "primary_cause": cause,
        "all_finite": bool(np.all(np.isfinite([base_a, base_b, base_peak, cand_a, cand_b, cand_peak]))),
    }


def _column_correlations(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("market stability inputs must be aligned")
    centered_x = values - values.mean(axis=0, keepdims=True)
    centered_y = labels - labels.mean()
    denominator = np.sqrt(
        np.sum(np.square(centered_x), axis=0) * float(centered_y @ centered_y)
    )
    return np.divide(
        centered_x.T @ centered_y,
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=denominator > 0.0,
    )


def _time_means(
    features: np.ndarray,
    target: np.ndarray,
    time_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    labels = np.asarray(target, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (len(values),) or ids.shape != (len(values),):
        raise ValueError("market mean inputs must be row-aligned")
    if len(ids) == 0 or np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be non-empty and sorted")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    return (
        np.add.reduceat(values, starts, axis=0) / counts[:, None],
        np.add.reduceat(labels, starts) / counts,
        ids[starts],
    )


def market_block_stability(
    features: np.ndarray,
    target: np.ndarray,
    time_ids: np.ndarray,
    *,
    n_blocks: int = 4,
    top_count: int = 200,
) -> dict[str, np.ndarray]:
    """Compute training-only four-block stability of market-mean correlations."""
    market_features, market_target, market_ids = _time_means(features, target, time_ids)
    if top_count <= 0 or top_count > market_features.shape[1]:
        raise ValueError("top_count must be within feature width")
    blocks = contiguous_time_blocks(market_ids, n_blocks)
    full_corr = _column_correlations(market_features, market_target)
    block_corr = np.vstack([
        _column_correlations(market_features[rows], market_target[rows])
        for rows in blocks
    ])
    feature_index = np.arange(market_features.shape[1], dtype=np.int64)
    full_order = np.lexsort((feature_index, -np.abs(full_corr)))
    full_rank = np.empty_like(feature_index)
    full_rank[full_order] = np.arange(1, len(feature_index) + 1)
    block_ranks = np.empty((n_blocks, len(feature_index)), dtype=np.int64)
    membership = np.zeros(len(feature_index), dtype=np.int64)
    for block_index, correlations in enumerate(block_corr):
        order = np.lexsort((feature_index, -np.abs(correlations)))
        block_ranks[block_index, order] = np.arange(1, len(feature_index) + 1)
        membership[order[:top_count]] += 1
    positive = np.mean(block_corr > 0.0, axis=0)
    negative = np.mean(block_corr < 0.0, axis=0)
    return {
        "full_correlation": full_corr,
        "full_rank": full_rank,
        "block_correlation": block_corr,
        "block_ranks": block_ranks,
        "rank_std": np.std(block_ranks, axis=0),
        "sign_consistency": np.maximum(positive, negative),
        "top_count_frequency": membership,
    }


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


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Market Task-Aligned Diagnostic",
        "",
        f"Status: **{payload.get('status', 'unknown')}**",
        "",
        "| Fold | Overlap | Jaccard | Peak delta | A delta | B delta | Cause |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fold in payload.get("folds", []):
        if not isinstance(fold, Mapping):
            continue
        overlap = fold.get("overlap", {})
        attribution = fold.get("attribution", {})
        lines.append(
            f"| {fold.get('fold')} | {overlap.get('overlap_count')} | "
            f"{float(overlap.get('jaccard', np.nan)):.3f} | "
            f"{float(attribution.get('delta_peak', np.nan)):.8f} | "
            f"{float(attribution.get('delta_A', np.nan)):.8f} | "
            f"{float(attribution.get('delta_B', np.nan)):.8f} | "
            f"{attribution.get('primary_cause')} |"
        )
    lines += [
        "",
        "No production model, confirmation run, or submission CSV was generated.",
    ]
    return "\n".join(lines) + "\n"




def scan_market_difference_scale(
    target: np.ndarray,
    weight: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fold: np.ndarray,
    *,
    alphas: np.ndarray,
) -> dict[str, object]:
    """Scan candidate-minus-baseline amplitudes without retraining."""
    from src.metric import scale_invariant_score, weighted_zero_mean_r2

    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    folds = np.asarray(fold, dtype=np.int64)
    values = np.asarray(alphas, dtype=np.float64)
    if not (y.shape == w.shape == base.shape == cand.shape == folds.shape):
        raise ValueError("energy scan inputs must be aligned")
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("alphas must be a non-empty finite vector")
    curve: list[dict[str, object]] = []
    for alpha in values:
        prediction = base + float(alpha) * (cand - base)
        metric = scale_invariant_score(y, prediction, w)
        curve.append({
            "alpha": float(alpha),
            "peak": float(metric["peak"]),
            "A": float(metric["A"]),
            "B": float(metric["B"]),
            "score": float(weighted_zero_mean_r2(y, prediction, w)),
        })
    best = max(curve, key=lambda row: float(row["peak"]))
    return {"best_alpha": float(best["alpha"]), "curve": curve}


def write_energy_scan_bundle(
    payload: Mapping[str, object], output_dir: str | Path, label: str,
) -> dict[str, Path]:
    """Atomically persist post-hoc energy-scan evidence."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{label}.json"
    markdown_path = directory / f"{label}.md"
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )
    lines = [
        "# Market Difference Energy Scan",
        "",
        f"Best alpha: **{payload.get('best_alpha')}**",
        "",
        "| Alpha | Peak | A | B | Score |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("curve", []):
        lines.append(
            f"| {row.get('alpha')} | {float(row.get('peak', np.nan)):.10f} | "
            f"{float(row.get('A', np.nan)):.10f} | {float(row.get('B', np.nan)):.10f} | "
            f"{float(row.get('score', np.nan)):.10f} |"
        )
    lines.append("\nNo model was retrained and no submission CSV was generated.\n")
    _atomic_write(markdown_path, "\n".join(lines))
    return {"json": json_path, "markdown": markdown_path}
def write_diagnostic_bundle(
    payload: Mapping[str, object], output_dir: str | Path, label: str,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fold_dir = directory / f"{label}_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold in payload.get("folds", []):
        if not isinstance(fold, Mapping) or "fold" not in fold:
            raise ValueError("each diagnostic fold requires a fold index")
        _atomic_write(
            fold_dir / f"fold_{int(fold['fold'])}.json",
            json.dumps(fold, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        )
    json_path = directory / f"{label}.json"
    markdown_path = directory / f"{label}.md"
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    _atomic_write(markdown_path, _render_markdown(payload))
    return {"json": json_path, "markdown": markdown_path, "fold_dir": fold_dir}
