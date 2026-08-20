"""Leakage-safe low-sample proposal scan for purified feature pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

import numpy as np

from experiments.v3_purified_interactions import default_purified_protocol


from experiments.v3_purified_interactions import (
    prebin_feature_split,
    score_prebinned_pair_split,
)
_PURIFIED = default_purified_protocol()
_FROZEN_PROTOCOL: dict[str, object] = {
    "schema_version": 1,
    "source_feature_count": 323,
    "source_pair_count": 52_003,
    "proposal_folds": [0, 1, 2],
    "gate_folds": [3, 4],
    "proposal_blocks": 4,
    "proposal_bins": 4,
    "sampling": {
        "ridge_xs": {
            "row_cap_per_block": 40_000,
            "fallback_row_cap_per_block": 20_000,
            "min_cell_weight": 32.0,
        },
        "market": {
            "time_cap_per_block": 20_000,
            "min_cell_weight": 8.0,
        },
    },
    "benchmark": {
        "lexical_pair_count": 1_024,
        "runtime_ceiling_seconds": 1_800.0,
        "peak_rss_ceiling_bytes": 4 * 1024**3,
    },
    "eligibility": {
        "minimum_positive_blocks": 2,
        "positive_drop_best_mean_gain": True,
        "minimum_coverage": 0.80,
        "maximum_dominant_cell_gain_share": 0.50,
    },
    "candidate_budget": {
        "core": 192,
        "diversity": 64,
        "maximum": 256,
        "diversity_parent_cap": 4,
    },
    "outer": copy.deepcopy(_PURIFIED["outer"]),
    "fusion": copy.deepcopy(_PURIFIED["fusion"]),
}


def default_proposal_protocol() -> dict[str, object]:
    """Return an independent copy of the frozen proposal protocol."""
    return copy.deepcopy(_FROZEN_PROTOCOL)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_proposal_protocol(protocol: Mapping[str, object]) -> None:
    """Fail closed if any pre-registered proposal decision has changed."""
    root = _mapping(protocol, "proposal protocol")
    if root.get("schema_version") != 1:
        raise ValueError("unsupported proposal protocol schema")

    proposal_folds = root.get("proposal_folds")
    gate_folds = root.get("gate_folds")
    if proposal_folds != [0, 1, 2] or gate_folds != [3, 4]:
        raise ValueError("proposal and gate fold assignments are frozen")
    if set(proposal_folds).intersection(gate_folds):
        raise ValueError("proposal and gate folds must not overlap")

    if dict(_mapping(root.get("fusion"), "fusion")) != _FROZEN_PROTOCOL["fusion"]:
        raise ValueError("frozen fusion must remain 0.7/1.17/1.16")
    if dict(_mapping(root.get("outer"), "outer")) != _FROZEN_PROTOCOL["outer"]:
        raise ValueError("frozen outer validation settings changed")
    if dict(_mapping(root.get("eligibility"), "eligibility")) != _FROZEN_PROTOCOL["eligibility"]:
        raise ValueError("proposal eligibility thresholds are frozen")

    budget = _mapping(root.get("candidate_budget"), "candidate budget")
    if dict(budget) != _FROZEN_PROTOCOL["candidate_budget"]:
        raise ValueError("candidate budget is frozen at 192 + 64 <= 256")
    if int(budget["core"]) + int(budget["diversity"]) > int(budget["maximum"]):
        raise ValueError("candidate budget exceeds maximum")

    fixed_names = (
        "source_feature_count",
        "source_pair_count",
        "proposal_blocks",
        "proposal_bins",
        "sampling",
        "benchmark",
    )
    for name in fixed_names:
        if root.get(name) != _FROZEN_PROTOCOL[name]:
            raise ValueError(f"frozen proposal setting changed: {name}")
    if set(root) != set(_FROZEN_PROTOCOL):
        raise ValueError("proposal protocol keys do not match the frozen schema")


def _group_bounds(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    return starts, counts


def validate_proposal_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate task input without inferring any missing fold boundary."""
    required = {
        "features",
        "residual",
        "weight",
        "time_id",
        "fold",
        "feature_indices",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"proposal input is missing arrays: {missing}")
    features = np.asarray(arrays["features"])
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("proposal features must be a nonempty 2D matrix")
    output = {
        "features": features,
        "residual": np.asarray(arrays["residual"], dtype=np.float64),
        "weight": np.asarray(arrays["weight"], dtype=np.float64),
        "time_id": np.asarray(arrays["time_id"], dtype=np.int64),
        "fold": np.asarray(arrays["fold"], dtype=np.int8),
        "feature_indices": np.asarray(arrays["feature_indices"], dtype=np.int64),
    }
    row_arrays = ("residual", "weight", "time_id", "fold")
    if any(output[name].shape != (len(features),) for name in row_arrays):
        raise ValueError("proposal input arrays must be row-aligned")
    if output["feature_indices"].shape != (features.shape[1],):
        raise ValueError("feature_indices must align with feature columns")
    if (
        not np.all(np.isfinite(output["residual"]))
        or not np.all(np.isfinite(output["weight"]))
        or np.any(output["weight"] <= 0.0)
    ):
        raise ValueError("proposal residual and positive weights must be finite")
    if np.any(np.diff(output["time_id"]) < 0):
        raise ValueError("proposal time_id must be nondecreasing")
    starts, counts = _group_bounds(output["time_id"])
    fold_by_time = output["fold"][starts]
    if not np.array_equal(output["fold"], np.repeat(fold_by_time, counts)):
        raise ValueError("each proposal time_id must belong to exactly one fold")
    if not set(np.unique(output["fold"])).issubset({0, 1, 2, 3, 4}):
        raise ValueError("proposal fold values must be within 0..4")
    return output


def split_proposal_gate_rows(
    fold: np.ndarray,
    protocol: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint row indices for proposal and untouched gate folds."""
    validate_proposal_protocol(protocol)
    values = np.asarray(fold)
    if values.ndim != 1:
        raise ValueError("fold must be one-dimensional")
    proposal = np.flatnonzero(np.isin(values, protocol["proposal_folds"]))
    gate = np.flatnonzero(np.isin(values, protocol["gate_folds"]))
    if len(proposal) == 0 or len(gate) == 0:
        raise ValueError("proposal input must contain both proposal and gate folds")
    if np.intersect1d(proposal, gate).size:
        raise AssertionError("proposal and gate rows overlap")
    return proposal, gate


def chronological_time_blocks(
    time_id: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    n_blocks: int,
) -> list[np.ndarray]:
    """Split ordered candidate rows into complete chronological time groups."""
    ids = np.asarray(time_id, dtype=np.int64)
    rows = np.asarray(candidate_rows, dtype=np.int64)
    if ids.ndim != 1 or rows.ndim != 1 or len(rows) == 0:
        raise ValueError("time blocks require nonempty one-dimensional arrays")
    if np.any(np.diff(rows) <= 0) or min(rows) < 0 or max(rows) >= len(ids):
        raise ValueError("candidate rows must be strictly increasing valid indices")
    if np.any(np.diff(ids) < 0):
        raise ValueError("time_id must be nondecreasing")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks < 2:
        raise ValueError("n_blocks must be an integer of at least two")
    selected_times = np.unique(ids[rows])
    if len(selected_times) < n_blocks:
        raise ValueError("not enough complete time groups for proposal blocks")
    candidate_mask = np.zeros(len(ids), dtype=bool)
    candidate_mask[rows] = True
    blocks = []
    for time_block in np.array_split(selected_times, n_blocks):
        block_rows = np.flatnonzero(candidate_mask & np.isin(ids, time_block))
        for value in time_block:
            full_group = np.flatnonzero(ids == value)
            if not np.array_equal(block_rows[ids[block_rows] == value], full_group):
                raise ValueError("candidate rows split a complete time_id group")
        blocks.append(block_rows)
    return blocks


def sample_complete_time_groups(
    time_id: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    row_cap: int,
) -> np.ndarray:
    """Choose evenly spaced complete time groups without exceeding row_cap."""
    ids = np.asarray(time_id, dtype=np.int64)
    rows = np.asarray(candidate_rows, dtype=np.int64)
    if isinstance(row_cap, bool) or not isinstance(row_cap, int) or row_cap <= 0:
        raise ValueError("row_cap must be a positive integer")
    if ids.ndim != 1 or rows.ndim != 1 or len(rows) == 0:
        raise ValueError("sampling requires nonempty one-dimensional arrays")
    selected_ids = ids[rows]
    if np.any(np.diff(rows) <= 0) or np.any(np.diff(selected_ids) < 0):
        raise ValueError("candidate rows must be ordered")
    starts, counts = _group_bounds(selected_ids)
    for start, count in zip(starts, counts):
        value = selected_ids[start]
        if not np.array_equal(rows[start:start + count], np.flatnonzero(ids == value)):
            raise ValueError("candidate rows must contain complete time_id groups")
    affordable = int(np.searchsorted(np.cumsum(np.sort(counts)), row_cap, side="right"))
    if affordable == 0:
        raise ValueError("row_cap is smaller than every complete time_id group")
    targets = np.rint(np.linspace(0, len(starts) - 1, affordable)).astype(int)
    distance = np.min(
        np.abs(np.arange(len(starts))[:, None] - targets[None, :]), axis=1
    )
    priority = sorted(
        range(len(starts)), key=lambda index: (int(distance[index]), index)
    )
    chosen: list[int] = []
    used = 0
    for index in priority:
        count = int(counts[index])
        if used + count <= row_cap:
            chosen.append(index)
            used += count
    pieces = [
        rows[starts[index]:starts[index] + counts[index]]
        for index in sorted(chosen)
    ]
    return np.concatenate(pieces).astype(np.int64, copy=False)


def enumerate_lexical_pairs(n_features: int) -> list[tuple[int, int]]:
    """Return every ordered feature pair exactly once in lexical order."""
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 2:
        raise ValueError("n_features must be an integer of at least two")
    return [
        (left, right)
        for left in range(n_features - 1)
        for right in range(left + 1, n_features)
    ]


def proposal_split_rows(
    sampled_blocks: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct expanding-train/next-block row indices."""
    if len(sampled_blocks) != 4:
        raise ValueError("proposal scan requires exactly four sampled blocks")
    blocks = [np.asarray(block, dtype=np.int64) for block in sampled_blocks]
    if any(block.ndim != 1 or len(block) == 0 for block in blocks):
        raise ValueError("proposal blocks must be nonempty row vectors")
    joined = np.concatenate(blocks)
    if len(np.unique(joined)) != len(joined) or np.any(np.diff(joined) <= 0):
        raise ValueError("proposal blocks must be ordered and disjoint")
    return [
        (np.concatenate(blocks[:index]), blocks[index])
        for index in range(1, len(blocks))
    ]


def aggregate_pair_scores(
    split_scores: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate the frozen three-split proposal statistics for one pair."""
    if len(split_scores) != 3:
        raise ValueError("proposal pair requires exactly three split scores")
    gains = np.asarray([score.get("gain") for score in split_scores], dtype=np.float64)
    coverage = np.asarray(
        [score.get("coverage") for score in split_scores], dtype=np.float64
    )
    dominant = np.asarray(
        [score.get("dominant_cell_gain_share") for score in split_scores],
        dtype=np.float64,
    )
    finite = np.asarray(
        [bool(score.get("finite")) for score in split_scores], dtype=bool
    )
    all_finite = bool(
        np.all(finite)
        and np.all(np.isfinite(gains))
        and np.all(np.isfinite(coverage))
        and np.all(np.isfinite(dominant))
    )
    if not all_finite:
        return {
            "median_gain": float("nan"),
            "mean_gain": float("nan"),
            "drop_best_mean_gain": float("nan"),
            "positive_blocks": int(np.sum(gains > 0.0)),
            "minimum_coverage": float("nan"),
            "maximum_dominant_cell_gain_share": float("nan"),
            "all_finite": False,
        }
    return {
        "median_gain": float(np.median(gains)),
        "mean_gain": float(np.mean(gains)),
        "drop_best_mean_gain": float(
            np.mean(np.delete(gains, int(np.argmax(gains))))
        ),
        "positive_blocks": int(np.sum(gains > 0.0)),
        "minimum_coverage": float(np.min(coverage)),
        "maximum_dominant_cell_gain_share": float(np.max(dominant)),
        "all_finite": True,
    }


def proposal_eligible(
    summary: Mapping[str, object],
    protocol: Mapping[str, object],
) -> bool:
    """Apply the frozen cheap-screen eligibility criteria."""
    validate_proposal_protocol(protocol)
    settings = _mapping(protocol["eligibility"], "eligibility")
    try:
        return bool(
            summary["all_finite"]
            and int(summary["positive_blocks"])
            >= int(settings["minimum_positive_blocks"])
            and float(summary["drop_best_mean_gain"]) > 0.0
            and float(summary["minimum_coverage"])
            >= float(settings["minimum_coverage"])
            and float(summary["maximum_dominant_cell_gain_share"])
            <= float(settings["maximum_dominant_cell_gain_share"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def scan_prebinned_pairs(
    features: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    sampled_blocks: Sequence[np.ndarray],
    *,
    bins: int,
    min_cell_weight: float,
    max_surface_cells: int,
    pairs: Sequence[tuple[int, int]] | None = None,
    protocol: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Score requested pairs on three chronological pre-binned splits."""
    x = np.asarray(features)
    y = np.asarray(residual, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or w.shape != y.shape
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(w))
        or np.any(w < 0.0)
    ):
        raise ValueError("scan arrays must be row-aligned and finite")
    requested = list(pairs) if pairs is not None else enumerate_lexical_pairs(x.shape[1])
    if (
        len(requested) == 0
        or len(set(requested)) != len(requested)
        or any(left < 0 or left >= right or right >= x.shape[1] for left, right in requested)
    ):
        raise ValueError("scan pairs must be unique ordered valid indices")
    split_rows = proposal_split_rows(sampled_blocks)
    pair_indices = np.asarray(requested, dtype=np.int16)
    shape = (len(requested), len(split_rows))
    split_gain = np.empty(shape, dtype=np.float64)
    split_coverage = np.empty(shape, dtype=np.float64)
    split_dominant = np.empty(shape, dtype=np.float64)
    split_finite = np.empty(shape, dtype=bool)
    checksums = np.empty(shape, dtype="U64")
    for split_index, (train_rows, valid_rows) in enumerate(split_rows):
        prebinned = prebin_feature_split(
            x[train_rows], x[valid_rows], bins=bins
        )
        for pair_index, pair in enumerate(requested):
            score = score_prebinned_pair_split(
                prebinned,
                y[train_rows],
                y[valid_rows],
                w[train_rows],
                w[valid_rows],
                pair=pair,
                min_cell_weight=min_cell_weight,
                max_surface_cells=max_surface_cells,
            )
            split_gain[pair_index, split_index] = score["gain"]
            split_coverage[pair_index, split_index] = score["coverage"]
            split_dominant[pair_index, split_index] = score["dominant_cell_gain_share"]
            split_finite[pair_index, split_index] = score["finite"]
            checksums[pair_index, split_index] = score["surface_checksum"]
    all_finite = (
        np.all(split_finite, axis=1)
        & np.all(np.isfinite(split_gain), axis=1)
        & np.all(np.isfinite(split_coverage), axis=1)
        & np.all(np.isfinite(split_dominant), axis=1)
    )
    positive_blocks = np.sum(split_gain > 0.0, axis=1).astype(np.int8)
    median_gain = np.median(split_gain, axis=1)
    mean_gain = np.mean(split_gain, axis=1)
    drop_best = (np.sum(split_gain, axis=1) - np.max(split_gain, axis=1)) / 2.0
    minimum_coverage = np.min(split_coverage, axis=1)
    maximum_dominant = np.max(split_dominant, axis=1)
    frozen = protocol if protocol is not None else default_proposal_protocol()
    eligibility = _mapping(frozen["eligibility"], "eligibility")
    eligible = (
        all_finite
        & (positive_blocks >= int(eligibility["minimum_positive_blocks"]))
        & (drop_best > 0.0)
        & (minimum_coverage >= float(eligibility["minimum_coverage"]))
        & (maximum_dominant <= float(eligibility["maximum_dominant_cell_gain_share"]))
    )
    return {
        "pair_indices": pair_indices,
        "split_gain": split_gain,
        "split_coverage": split_coverage,
        "split_dominant_cell_gain_share": split_dominant,
        "split_finite": split_finite,
        "surface_checksum": checksums,
        "median_gain": median_gain,
        "mean_gain": mean_gain,
        "drop_best_mean_gain": drop_best,
        "positive_blocks": positive_blocks,
        "minimum_coverage": minimum_coverage,
        "maximum_dominant_cell_gain_share": maximum_dominant,
        "all_finite": all_finite,
        "eligible": eligible,
        "scored_pair_split_count": len(requested) * len(split_rows),
    }
