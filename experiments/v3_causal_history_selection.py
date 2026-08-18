"""Causal, full-stream selection evidence for V3 history base features."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "strategies" / "v3_hybrid", ROOT / "strategies" / "v1_ridge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from features import apply_robust_transform
from history import AssetHistory
from src.io import FEATURE_COLUMNS, time_sample_mask, train_files


HISTORY_BLOCK_NAMES = (
    "previous",
    "difference",
    "rolling_mean",
    "rolling_deviation",
)
SHADOW_QUANTILE = 0.95
MIN_DIRECTION_CONSISTENCY = 0.75


def _history_stats_tuple(
    history_stats: Sequence[np.ndarray],
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(history_stats) != 4:
        raise ValueError("history_stats must contain lower, upper, center, and scale")
    values = tuple(np.asarray(value, dtype=np.float32) for value in history_stats)
    if any(value.shape != (feature_count,) for value in values):
        raise ValueError("history_stats must match the selected feature count")
    return values  # type: ignore[return-value]


def causal_history_rows_from_batches(
    batches: Iterable[Mapping[str, np.ndarray]],
    *,
    history_stats: Sequence[np.ndarray],
    selected_time_ids: np.ndarray,
    sample_modulo: int,
    sampling: str,
    window_size: int,
) -> Iterator[dict[str, Any]]:
    """Yield history blocks for selected sample rows while advancing on every row.

    Input batches must preserve the original chronological order. Rows which are not
    part of the sampled training window still update ``AssetHistory`` before being
    discarded, so a sampled row never uses the previous sampled observation as lag1.
    """
    selected_ids = np.asarray(selected_time_ids, dtype=np.int64)
    if selected_ids.ndim != 1 or len(selected_ids) == 0:
        raise ValueError("selected_time_ids must be a non-empty 1D array")
    if np.any(np.diff(selected_ids) <= 0):
        raise ValueError("selected_time_ids must be strictly increasing")
    feature_count: int | None = None
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    history: AssetHistory | None = None
    last_time: int | None = None

    for batch in batches:
        time_ids = np.asarray(batch["time_id"], dtype=np.int64)
        asset_ids = np.asarray(batch["asset_id"], dtype=np.int64)
        target = np.asarray(batch["target"], dtype=np.float64)
        features = np.asarray(batch["features"], dtype=np.float32)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("history batches must contain non-empty 2D features")
        if len(time_ids) != len(asset_ids) or len(target) != len(features):
            raise ValueError("history batch arrays must have matching row counts")
        if np.any(np.diff(time_ids) < 0) or (last_time is not None and time_ids[0] < last_time):
            raise ValueError("history batches must be sorted by time_id")
        if time_ids[0] > selected_ids[-1]:
            break
        last_time = int(time_ids[-1])
        if feature_count is None:
            feature_count = features.shape[1]
            stats = _history_stats_tuple(history_stats, feature_count)
            history = AssetHistory(feature_count=feature_count, window_size=window_size)
        elif features.shape[1] != feature_count:
            raise ValueError("history feature count changed between batches")

        transformed = features.copy()
        apply_robust_transform(transformed, *stats)
        blocks = history.transform(transformed, asset_ids)
        sampled = time_sample_mask(time_ids, sample_modulo, sampling=sampling)
        positions = np.searchsorted(selected_ids, time_ids)
        bounded = positions < len(selected_ids)
        exact = np.zeros(len(time_ids), dtype=bool)
        exact[bounded] = selected_ids[positions[bounded]] == time_ids[bounded]
        keep = sampled & exact
        if keep.any():
            yield {
                "time_id": time_ids[keep].copy(),
                "target": target[keep].copy(),
                "blocks": tuple(block[keep].copy() for block in blocks),
            }


def _complete_time_groups(
    chunks: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[int, np.ndarray, tuple[np.ndarray, ...]]]:
    pending: dict[str, Any] | None = None
    for chunk in chunks:
        time_ids = np.asarray(chunk["time_id"], dtype=np.int64)
        target = np.asarray(chunk["target"], dtype=np.float64)
        blocks = tuple(np.asarray(block, dtype=np.float32) for block in chunk["blocks"])
        if pending is not None:
            time_ids = np.concatenate([pending["time_id"], time_ids])
            target = np.concatenate([pending["target"], target])
            blocks = tuple(
                np.concatenate([old, new])
                for old, new in zip(pending["blocks"], blocks, strict=True)
            )
        starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
        stops = np.r_[starts[1:], len(time_ids)]
        for start, stop in zip(starts[:-1], stops[:-1], strict=True):
            yield int(time_ids[start]), target[start:stop], tuple(
                block[start:stop] for block in blocks
            )
        start = int(starts[-1])
        pending = {
            "time_id": time_ids[start:],
            "target": target[start:],
            "blocks": tuple(block[start:] for block in blocks),
        }
    if pending is not None:
        yield int(pending["time_id"][0]), pending["target"], pending["blocks"]


def _empty_moments(n_blocks: int, n_columns: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(n_blocks, dtype=np.float64),
        "sum_y": np.zeros(n_blocks, dtype=np.float64),
        "sum_y2": np.zeros(n_blocks, dtype=np.float64),
        "sum_x": np.zeros((n_blocks, len(HISTORY_BLOCK_NAMES), n_columns), dtype=np.float64),
        "sum_x2": np.zeros((n_blocks, len(HISTORY_BLOCK_NAMES), n_columns), dtype=np.float64),
        "sum_xy": np.zeros((n_blocks, len(HISTORY_BLOCK_NAMES), n_columns), dtype=np.float64),
    }


def _accumulate_design(
    moments: dict[str, np.ndarray],
    block_index: int,
    kind_index: int,
    values: np.ndarray,
    target: np.ndarray,
) -> None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("history evidence design and target must align")
    moments["sum_x"][block_index, kind_index] += x.sum(axis=0)
    moments["sum_x2"][block_index, kind_index] += np.square(x).sum(axis=0)
    moments["sum_xy"][block_index, kind_index] += x.T @ y


def _finalize_correlations(moments: Mapping[str, np.ndarray]) -> np.ndarray:
    count = moments["count"][:, None, None]
    sum_y = moments["sum_y"][:, None, None]
    sum_y2 = moments["sum_y2"][:, None, None]
    covariance = moments["sum_xy"] - moments["sum_x"] * sum_y / np.maximum(count, 1.0)
    x_variance = moments["sum_x2"] - np.square(moments["sum_x"]) / np.maximum(count, 1.0)
    y_variance = sum_y2 - np.square(sum_y) / np.maximum(count, 1.0)
    denominator = np.sqrt(np.maximum(x_variance * y_variance, 0.0))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-30)


def history_correlation_evidence(
    rows: Iterable[Mapping[str, Any]],
    *,
    selected_time_ids: np.ndarray,
    n_features: int,
    n_blocks: int,
    n_shadows: int,
) -> dict[str, Any]:
    """Accumulate causal four-block correlations without retaining all history rows."""
    if n_features <= 0 or n_blocks < 2 or n_shadows <= 0:
        raise ValueError("invalid history evidence dimensions")
    selected_ids = np.asarray(selected_time_ids, dtype=np.int64)
    if len(selected_ids) < n_blocks:
        raise ValueError("selected_time_ids must cover every history evidence block")
    block_ends = np.array_split(selected_ids, n_blocks)
    block_end_ids = np.asarray([block[-1] for block in block_ends], dtype=np.int64)
    actual = _empty_moments(n_blocks, n_features)
    shadow = _empty_moments(n_blocks, n_shadows)
    time_counts = np.zeros(n_blocks, dtype=np.int64)

    for time_id, target, blocks in _complete_time_groups(rows):
        block_index = int(np.searchsorted(block_end_ids, time_id, side="left"))
        selected_position = int(np.searchsorted(selected_ids, time_id))
        if (
            block_index >= n_blocks
            or selected_position >= len(selected_ids)
            or selected_ids[selected_position] != time_id
        ):
            raise ValueError("history row is outside the selected training window")
        y = np.asarray(target, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError("history target must be finite")
        y = y - y.mean()
        actual["count"][block_index] += len(y)
        actual["sum_y"][block_index] += y.sum()
        actual["sum_y2"][block_index] += np.square(y).sum()
        shadow["count"][block_index] += len(y)
        shadow["sum_y"][block_index] += y.sum()
        shadow["sum_y2"][block_index] += np.square(y).sum()
        time_counts[block_index] += 1
        for kind_index, values in enumerate(blocks):
            centered = np.asarray(values, dtype=np.float64)
            centered = centered - centered.mean(axis=0, keepdims=True)
            _accumulate_design(actual, block_index, kind_index, centered, y)
            shadow_values = np.empty((len(centered), n_shadows), dtype=np.float64)
            for shadow_index in range(n_shadows):
                source = shadow_index % n_features
                if len(centered) < 2:
                    shadow_values[:, shadow_index] = 0.0
                else:
                    shift = 1 + (shadow_index + kind_index) % (len(centered) - 1)
                    shadow_values[:, shadow_index] = np.roll(centered[:, source], shift)
            _accumulate_design(shadow, block_index, kind_index, shadow_values, y)

    if np.any(time_counts == 0):
        raise ValueError("at least one history evidence block has no selected time_ids")
    if int(time_counts.sum()) != len(selected_ids):
        raise ValueError("history evidence did not cover every selected time_id exactly once")
    return {
        "block_correlations": _finalize_correlations(actual),
        "shadow_block_correlations": _finalize_correlations(shadow),
        "rows_per_block": actual["count"].astype(int),
        "time_ids_per_block": time_counts,
    }


def select_history_bases(
    *,
    feature_indices: np.ndarray,
    block_correlations: np.ndarray,
    shadow_block_correlations: np.ndarray,
    shadow_quantile: float = SHADOW_QUANTILE,
    min_direction_consistency: float = MIN_DIRECTION_CONSISTENCY,
) -> dict[str, Any]:
    """Select whole history bases; each accepted base contributes all four blocks."""
    indices = np.asarray(feature_indices, dtype=np.int64)
    correlations = np.asarray(block_correlations, dtype=np.float64)
    shadows = np.asarray(shadow_block_correlations, dtype=np.float64)
    expected_kinds = len(HISTORY_BLOCK_NAMES)
    if indices.ndim != 1 or len(indices) == 0 or len(indices) != len(set(indices.tolist())):
        raise ValueError("feature_indices must be unique and non-empty")
    if correlations.shape != (correlations.shape[0], expected_kinds, len(indices)):
        raise ValueError("block_correlations has an invalid shape")
    if shadows.ndim != 3 or shadows.shape[:2] != correlations.shape[:2] or shadows.shape[2] == 0:
        raise ValueError("shadow_block_correlations has an invalid shape")
    if not np.all(np.isfinite(correlations)) or not np.all(np.isfinite(shadows)):
        raise ValueError("history correlations must be finite")
    if not 0.0 < shadow_quantile <= 1.0 or not 0.0 < min_direction_consistency <= 1.0:
        raise ValueError("history gate settings must be within (0, 1]")

    floors = np.quantile(np.abs(shadows), shadow_quantile, axis=(0, 2))
    evidence: list[dict[str, Any]] = []
    selected: list[int] = []
    for column, feature_index in enumerate(indices):
        derived: list[dict[str, Any]] = []
        for kind_index, name in enumerate(HISTORY_BLOCK_NAMES):
            values = correlations[:, kind_index, column]
            signed_median = float(np.median(values))
            direction = float(np.sign(signed_median))
            consistency = 0.0 if direction == 0.0 else float(np.mean(values * direction > 0.0))
            median_abs = float(np.median(np.abs(values)))
            derived.append({
                "name": name,
                "median_abs_correlation": median_abs,
                "signed_median_correlation": signed_median,
                "direction_consistency": consistency,
                "shadow_floor": float(floors[kind_index]),
                "passed": bool(median_abs > floors[kind_index] and consistency >= min_direction_consistency),
            })
        passed = any(item["passed"] for item in derived)
        best = max(derived, key=lambda item: item["median_abs_correlation"])
        evidence.append({
            "feature": int(feature_index),
            "passed": passed,
            "strongest_derived": best["name"],
            "derived": derived,
        })
        if passed:
            selected.append(int(feature_index))
    return {
        "selected_indices": selected,
        "selected_count": len(selected),
        "derived_columns": list(HISTORY_BLOCK_NAMES),
        "model_column_count": len(selected) * len(HISTORY_BLOCK_NAMES),
        "evidence": evidence,
        "thresholds": {
            "shadow_quantile": shadow_quantile,
            "shadow_floors": {name: float(floor) for name, floor in zip(HISTORY_BLOCK_NAMES, floors, strict=True)},
            "min_direction_consistency": min_direction_consistency,
        },
    }


def _parquet_history_batches(
    data_root: Path,
    feature_names: Sequence[str],
    batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    import pyarrow.parquet as pq

    columns = ["time_id", "asset_id", "target", *feature_names]
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            yield {
                "time_id": frame["time_id"].to_numpy(dtype=np.int64, copy=False),
                "asset_id": frame["asset_id"].to_numpy(dtype=np.int64, copy=False),
                "target": frame["target"].to_numpy(dtype=np.float64, copy=False),
                "features": frame.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True),
            }


def run_causal_history_selection(
    *,
    data_root: Path,
    selected_time_ids: np.ndarray,
    feature_indices: np.ndarray,
    robust_stats: Mapping[str, np.ndarray],
    sample_modulo: int,
    sampling: str,
    n_blocks: int,
    n_shadows: int,
    window_size: int = 5,
    batch_size: int = 120_000,
) -> dict[str, Any]:
    """Run history selection on the full chronological stream up to the final window."""
    indices = np.asarray(feature_indices, dtype=np.int64)
    if batch_size <= 0 or window_size <= 0:
        raise ValueError("history batch_size and window_size must be positive")
    if indices.ndim != 1:
        raise ValueError("history feature indices must be one-dimensional")
    if np.any(indices < 0) or np.any(indices >= len(FEATURE_COLUMNS)):
        raise ValueError("history feature index is out of range")
    if len(indices) == 0:
        return {
            "status": "selected_causal_history",
            "selected_indices": [],
            "selected_names": [],
            "selected_count": 0,
            "derived_columns": list(HISTORY_BLOCK_NAMES),
            "model_column_count": 0,
            "evidence": [],
            "thresholds": {},
            "source": "adaptive_xs_selection",
            "source_candidate_count": 0,
            "protocol": {
                "state": "not scanned because the XS candidate set was empty",
                "window_size": window_size,
            },
            "coverage": {"rows_per_block": [], "time_ids_per_block": []},
        }
    feature_names = [FEATURE_COLUMNS[index] for index in indices]
    stats = tuple(np.asarray(robust_stats[name], dtype=np.float32)[indices]
                  for name in ("lower", "upper", "center", "scale"))
    lag_cell_budget = 24_000_000
    effective_batch_size = min(
        batch_size,
        max(1_000, lag_cell_budget // (len(indices) * window_size)),
    )
    rows = causal_history_rows_from_batches(
        _parquet_history_batches(data_root, feature_names, effective_batch_size),
        history_stats=stats,
        selected_time_ids=selected_time_ids,
        sample_modulo=sample_modulo,
        sampling=sampling,
        window_size=window_size,
    )
    evidence = history_correlation_evidence(
        rows,
        selected_time_ids=selected_time_ids,
        n_features=len(indices),
        n_blocks=n_blocks,
        n_shadows=n_shadows,
    )
    selection = select_history_bases(
        feature_indices=indices,
        block_correlations=evidence["block_correlations"],
        shadow_block_correlations=evidence["shadow_block_correlations"],
    )
    selection.update({
        "status": "selected_causal_history",
        "selected_names": [FEATURE_COLUMNS[index] for index in selection["selected_indices"]],
        "source": "adaptive_xs_selection",
        "source_candidate_count": len(indices),
        "protocol": {
            "state": "AssetHistory advanced on every full-stream row",
            "derived_columns": list(HISTORY_BLOCK_NAMES),
            "window_size": window_size,
            "target": "unweighted cross-sectional target deviation",
            "evidence": "four chronological block correlations versus within-time centered history columns",
            "n_shadows": n_shadows,
            "batch_size": effective_batch_size,
            "lag_cell_budget": lag_cell_budget,
        },
        "coverage": {
            "rows_per_block": evidence["rows_per_block"].tolist(),
            "time_ids_per_block": evidence["time_ids_per_block"].tolist(),
        },
    })
    return selection
