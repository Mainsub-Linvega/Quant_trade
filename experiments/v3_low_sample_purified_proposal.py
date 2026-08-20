"""Leakage-safe low-sample proposal scan for purified feature pairs."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

import numpy as np

from experiments.v3_purified_interactions import default_purified_protocol


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
