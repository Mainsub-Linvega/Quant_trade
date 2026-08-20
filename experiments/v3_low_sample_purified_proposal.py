"""Leakage-safe low-sample proposal scan for purified feature pairs."""

from __future__ import annotations
import argparse

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import re
import time
from typing import Any

import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline_indices(
    task: str,
    candidate_dir: str | Path,
) -> dict[str, object]:
    """Load and hash the frozen task-specific direct Top200 reference."""
    directory = Path(candidate_dir).resolve()
    if task == "ridge":
        source = directory / "baseline_model.json"
        key = "selected_indices"
    elif task in {"xs", "market"}:
        source = directory / "hybrid_meta.json"
        key = "lgbm_features"
    else:
        raise ValueError(f"unknown proposal task: {task}")
    data = json.loads(source.read_text(encoding="utf-8"))
    values = data.get(key)
    if not isinstance(values, list):
        raise ValueError(f"baseline reference is missing {key}")
    if task == "ridge":
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("Ridge baseline indices must be integers")
        indices = [int(value) for value in values]
    else:
        pattern = re.compile(r"^feature_(\d{3})$")
        matches = [pattern.fullmatch(str(value)) for value in values]
        if any(match is None for match in matches):
            raise ValueError("LGBM baseline names must use feature_###")
        indices = [int(match.group(1)) for match in matches if match is not None]
    if (
        len(indices) != 200
        or len(set(indices)) != 200
        or min(indices) < 0
        or max(indices) >= 323
    ):
        raise ValueError("baseline reference must contain 200 unique source features")
    return {
        "indices": indices,
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
        "source_key": key,
    }


def _ranked_eligible_rows(scores: Mapping[str, object]) -> list[int]:
    required = {
        "pair_indices",
        "drop_best_mean_gain",
        "median_gain",
        "mean_gain",
        "eligible",
    }
    missing = sorted(required.difference(scores))
    if missing:
        raise ValueError(f"proposal scores are missing arrays: {missing}")
    pairs = np.asarray(scores["pair_indices"])
    drop_best = np.asarray(scores["drop_best_mean_gain"], dtype=np.float64)
    median = np.asarray(scores["median_gain"], dtype=np.float64)
    mean = np.asarray(scores["mean_gain"], dtype=np.float64)
    eligible = np.asarray(scores["eligible"], dtype=bool)
    count = len(pairs)
    if (
        pairs.shape != (count, 2)
        or drop_best.shape != (count,)
        or median.shape != (count,)
        or mean.shape != (count,)
        or eligible.shape != (count,)
    ):
        raise ValueError("proposal score arrays must be pair-aligned")
    eligible_rows = np.flatnonzero(eligible)
    if not (
        np.all(np.isfinite(drop_best[eligible_rows]))
        and np.all(np.isfinite(median[eligible_rows]))
        and np.all(np.isfinite(mean[eligible_rows]))
    ):
        raise ValueError("eligible proposal ranking metrics must be finite")
    return sorted(
        (int(row) for row in eligible_rows),
        key=lambda row: (
            -float(drop_best[row]),
            -float(median[row]),
            -float(mean[row]),
            int(pairs[row, 0]),
            int(pairs[row, 1]),
        ),
    )


def select_proposal_candidates(
    scores: Mapping[str, object],
    *,
    baseline_indices: set[int],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Select unrestricted core plus parent-capped diversity candidates."""
    validate_proposal_protocol(protocol)
    if any(index < 0 or index >= 323 for index in baseline_indices):
        raise ValueError("baseline indices must be source feature indices")
    budget = _mapping(protocol["candidate_budget"], "candidate budget")
    pairs = np.asarray(scores["pair_indices"])
    ranked = _ranked_eligible_rows(scores)
    core_rows = ranked[:int(budget["core"])]
    core_set = set(core_rows)
    rank_position = {row: position for position, row in enumerate(ranked)}
    remaining = [row for row in ranked if row not in core_set]
    remaining.sort(
        key=lambda row: (
            not (
                int(pairs[row, 0]) not in baseline_indices
                or int(pairs[row, 1]) not in baseline_indices
            ),
            rank_position[row],
        )
    )
    parent_counts: dict[int, int] = {}
    diversity_rows: list[int] = []
    parent_cap = int(budget["diversity_parent_cap"])
    for row in remaining:
        left, right = (int(value) for value in pairs[row])
        if parent_counts.get(left, 0) >= parent_cap or parent_counts.get(right, 0) >= parent_cap:
            continue
        diversity_rows.append(row)
        parent_counts[left] = parent_counts.get(left, 0) + 1
        parent_counts[right] = parent_counts.get(right, 0) + 1
        if len(diversity_rows) == int(budget["diversity"]):
            break
    core = [tuple(int(value) for value in pairs[row]) for row in core_rows]
    diversity = [tuple(int(value) for value in pairs[row]) for row in diversity_rows]
    manifest_pairs = sorted(core + diversity)
    manifest_json = json.dumps(
        {"pairs": [list(pair) for pair in manifest_pairs]},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    return {
        "core_ranked": core,
        "diversity_ranked": diversity,
        "pairs": manifest_pairs,
        "manifest_json": manifest_json,
        "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
        "eligible_count": len(ranked),
    }


def benchmark_runtime_decision(
    *,
    elapsed_seconds: float,
    benchmark_pairs: int,
    total_pairs: int,
    peak_rss_bytes: int,
    row_cap: int,
    fallback_row_cap: int,
    runtime_ceiling_seconds: float,
    peak_rss_ceiling_bytes: int,
) -> dict[str, object]:
    """Choose run/fallback/stop from resource measurements alone."""
    numeric = (
        elapsed_seconds,
        benchmark_pairs,
        total_pairs,
        peak_rss_bytes,
        row_cap,
        fallback_row_cap,
        runtime_ceiling_seconds,
        peak_rss_ceiling_bytes,
    )
    if any(float(value) <= 0.0 for value in numeric):
        raise ValueError("benchmark measurements and ceilings must be positive")
    estimated = float(elapsed_seconds) * int(total_pairs) / int(benchmark_pairs)
    if int(peak_rss_bytes) > int(peak_rss_ceiling_bytes):
        action = "stop"
        reason = "peak_rss_exceeded"
        selected_cap = int(row_cap)
    elif estimated <= float(runtime_ceiling_seconds):
        action = "run"
        reason = "within_resource_ceiling"
        selected_cap = int(row_cap)
    elif int(row_cap) != int(fallback_row_cap):
        action = "fallback"
        reason = "primary_runtime_exceeded"
        selected_cap = int(fallback_row_cap)
    else:
        action = "stop"
        reason = "fallback_runtime_exceeded"
        selected_cap = int(row_cap)
    return {
        "action": action,
        "reason": reason,
        "row_cap": selected_cap,
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_full_seconds": estimated,
        "peak_rss_bytes": int(peak_rss_bytes),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_proposal_report(
    *,
    label: str,
    task: str,
    protocol: Mapping[str, object],
    input_path: str | Path,
    input_sha256: str,
    baseline_reference: Mapping[str, object],
    sampling: Mapping[str, object],
    benchmark: Mapping[str, object],
    scores: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Build a JSON-safe manifest for one completed proposal scan."""
    validate_proposal_protocol(protocol)
    if task not in {"ridge", "xs", "market"}:
        raise ValueError("unknown proposal task")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("label must be a safe file stem")
    pairs = [list(pair) for pair in selection["pairs"]]
    pair_count = len(np.asarray(scores["pair_indices"]))
    return {
        "schema_version": 1,
        "label": label,
        "task": task,
        "protocol": _jsonable(protocol),
        "input": {
            "path": str(Path(input_path).resolve()),
            "sha256": input_sha256,
        },
        "baseline_reference": _jsonable(baseline_reference),
        "sampling": _jsonable(sampling),
        "benchmark": _jsonable(benchmark),
        "scan": {
            "pair_count": pair_count,
            "split_count": 3,
            "pair_split_count": int(scores["scored_pair_split_count"]),
            "eligible_count": int(selection["eligible_count"]),
        },
        "selection": {
            "core_ranked": _jsonable(selection["core_ranked"]),
            "diversity_ranked": _jsonable(selection["diversity_ranked"]),
            "pair_manifest_sha256": selection["manifest_sha256"],
        },
        "pairs": pairs,
        "candidate_generated": False,
        "submission_generated": False,
    }


def _proposal_markdown(report: Mapping[str, object]) -> str:
    scan = _mapping(report["scan"], "scan")
    benchmark = _mapping(report["benchmark"], "benchmark")
    return "\n".join([
        f"# Low-Sample Purified Pair Proposal: {report['label']}",
        "",
        f"- Task: `{report['task']}`",
        f"- Scanned pairs: `{scan['pair_count']}`",
        f"- Pair-split scores: `{scan['pair_split_count']}`",
        f"- Eligible pairs: `{scan['eligible_count']}`",
        f"- Frozen candidates: `{len(report['pairs'])}`",
        f"- Benchmark action: `{benchmark.get('action', 'recorded')}`",
        f"- Estimated full seconds: `{benchmark.get('estimated_full_seconds', 'n/a')}`",
        "- Candidate model generated: `false`",
        "- Submission generated: `false`",
        "",
    ])


def write_proposal_artifacts(
    output_dir: str | Path,
    label: str,
    scores: Mapping[str, object],
    report: Mapping[str, object],
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Atomically write NPZ scores plus JSON and Markdown experiment reports."""
    directory = Path(output_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("label must be a safe file stem")
    paths = {
        "scores": directory / f"{label}_pair_scores.npz",
        "manifest": directory / f"{label}_manifest.json",
        "markdown": directory / f"{label}.md",
    }
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError("proposal artifact exists; use force to overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = {
        "scores": directory / f".{label}_pair_scores.tmp.npz",
        "manifest": directory / f".{label}_manifest.json.tmp",
        "markdown": directory / f".{label}.md.tmp",
    }
    arrays = {
        name: value
        for name, value in scores.items()
        if isinstance(value, (np.ndarray, np.generic, int, float, bool))
    }
    try:
        np.savez_compressed(temporary["scores"], **arrays)
        temporary["manifest"].write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary["markdown"].write_text(
            _proposal_markdown(report), encoding="utf-8"
        )
        for name in ("scores", "manifest", "markdown"):
            temporary[name].replace(paths[name])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths


def _peak_rss_bytes() -> int:
    # Linux reports KiB; this CLI runs in the project's frozen WSL environment.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _sample_proposal_blocks(
    arrays: Mapping[str, np.ndarray],
    proposal_rows: np.ndarray,
    *,
    row_cap: int,
    n_blocks: int,
) -> list[np.ndarray]:
    blocks = chronological_time_blocks(
        arrays["time_id"], proposal_rows, n_blocks=n_blocks
    )
    return [
        sample_complete_time_groups(arrays["time_id"], block, row_cap=row_cap)
        for block in blocks
    ]


def _benchmark_scan(
    arrays: Mapping[str, np.ndarray],
    sampled_blocks: Sequence[np.ndarray],
    *,
    bins: int,
    min_cell_weight: float,
    benchmark_pairs: int,
) -> tuple[dict[str, object], float, int]:
    pairs = enumerate_lexical_pairs(arrays["features"].shape[1])[:benchmark_pairs]
    started = time.perf_counter()
    scores = scan_prebinned_pairs(
        arrays["features"],
        arrays["residual"],
        arrays["weight"],
        sampled_blocks,
        bins=bins,
        min_cell_weight=min_cell_weight,
        max_surface_cells=bins * bins,
        pairs=pairs,
    )
    return scores, time.perf_counter() - started, _peak_rss_bytes()


def run_proposal(
    *,
    task: str,
    input_path: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    label: str,
    force: bool = False,
) -> dict[str, Path]:
    """Benchmark, resource-gate, scan, select, and report one task."""
    protocol = default_proposal_protocol()
    validate_proposal_protocol(protocol)
    source = Path(input_path).resolve()
    with np.load(source, allow_pickle=False) as loaded:
        arrays = validate_proposal_arrays({name: loaded[name] for name in loaded.files})
    if arrays["features"].shape[1] != int(protocol["source_feature_count"]):
        raise ValueError("proposal input must contain all 323 source features")
    proposal_rows, _ = split_proposal_gate_rows(arrays["fold"], protocol)
    sampling_protocol = _mapping(protocol["sampling"], "sampling")
    if task in {"ridge", "xs"}:
        task_sampling = _mapping(sampling_protocol["ridge_xs"], "ridge_xs")
        row_cap = int(task_sampling["row_cap_per_block"])
        fallback_cap = int(task_sampling["fallback_row_cap_per_block"])
    elif task == "market":
        task_sampling = _mapping(sampling_protocol["market"], "market")
        row_cap = int(task_sampling["time_cap_per_block"])
        fallback_cap = row_cap
    else:
        raise ValueError("unknown proposal task")
    benchmark_protocol = _mapping(protocol["benchmark"], "benchmark")
    benchmark_pairs = int(benchmark_protocol["lexical_pair_count"])
    sampled = _sample_proposal_blocks(
        arrays, proposal_rows, row_cap=row_cap,
        n_blocks=int(protocol["proposal_blocks"]),
    )
    _, elapsed, rss = _benchmark_scan(
        arrays, sampled, bins=int(protocol["proposal_bins"]),
        min_cell_weight=float(task_sampling["min_cell_weight"]),
        benchmark_pairs=benchmark_pairs,
    )
    decision = benchmark_runtime_decision(
        elapsed_seconds=elapsed, benchmark_pairs=benchmark_pairs,
        total_pairs=int(protocol["source_pair_count"]), peak_rss_bytes=rss,
        row_cap=row_cap, fallback_row_cap=fallback_cap,
        runtime_ceiling_seconds=float(benchmark_protocol["runtime_ceiling_seconds"]),
        peak_rss_ceiling_bytes=int(benchmark_protocol["peak_rss_ceiling_bytes"]),
    )
    if decision["action"] == "fallback":
        sampled = _sample_proposal_blocks(
            arrays, proposal_rows, row_cap=fallback_cap,
            n_blocks=int(protocol["proposal_blocks"]),
        )
        _, elapsed, rss = _benchmark_scan(
            arrays, sampled, bins=int(protocol["proposal_bins"]),
            min_cell_weight=float(task_sampling["min_cell_weight"]),
            benchmark_pairs=benchmark_pairs,
        )
        decision = benchmark_runtime_decision(
            elapsed_seconds=elapsed, benchmark_pairs=benchmark_pairs,
            total_pairs=int(protocol["source_pair_count"]), peak_rss_bytes=rss,
            row_cap=fallback_cap, fallback_row_cap=fallback_cap,
            runtime_ceiling_seconds=float(benchmark_protocol["runtime_ceiling_seconds"]),
            peak_rss_ceiling_bytes=int(benchmark_protocol["peak_rss_ceiling_bytes"]),
        )
    if decision["action"] != "run":
        raise RuntimeError(f"proposal resource gate stopped: {decision['reason']}")
    scores = scan_prebinned_pairs(
        arrays["features"], arrays["residual"], arrays["weight"], sampled,
        bins=int(protocol["proposal_bins"]),
        min_cell_weight=float(task_sampling["min_cell_weight"]),
        max_surface_cells=int(protocol["proposal_bins"]) ** 2,
        protocol=protocol,
    )
    baseline = load_baseline_indices(task, candidate_dir)
    selection = select_proposal_candidates(
        scores, baseline_indices=set(baseline["indices"]), protocol=protocol
    )
    sampled_rows = np.concatenate(sampled).astype(np.int64, copy=False)
    sampling = {
        "row_cap": int(decision["row_cap"]),
        "block_row_counts": [len(block) for block in sampled],
        "selected_time_ids": [
            [int(value) for value in np.unique(arrays["time_id"][block])]
            for block in sampled
        ],
        "row_indices_sha256": hashlib.sha256(sampled_rows.tobytes()).hexdigest(),
    }
    report = build_proposal_report(
        label=label, task=task, protocol=protocol, input_path=source,
        input_sha256=_sha256_file(source), baseline_reference=baseline,
        sampling=sampling, benchmark=decision, scores=scores, selection=selection,
    )
    return write_proposal_artifacts(
        output_dir, label, scores, report, force=force
    )


def run_synthetic_smoke() -> dict[str, object]:
    """Exercise the production scanner on balanced XOR and additive controls."""
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    grid = np.stack(
        np.meshgrid(levels, levels, levels, levels, indexing="ij"), axis=-1
    ).reshape(-1, 4)
    rng = np.random.default_rng(2026)
    blocks: list[np.ndarray] = []
    feature_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    offset = 0
    for block_index in range(4):
        block = np.repeat(grid, 4, axis=0)
        order = rng.permutation(len(block))
        block = block[order]
        noise = rng.normal(size=(len(block), 2)).astype(np.float32)
        features = np.column_stack([block, noise]).astype(np.float32)
        xor = np.where(
            (features[:, 0] > 0.0) ^ (features[:, 1] > 0.0), 1.0, -1.0
        )
        additive = 0.08 * features[:, 2] + 0.04 * features[:, 3]
        feature_blocks.append(features)
        residual_blocks.append(xor + additive)
        blocks.append(np.arange(offset, offset + len(features), dtype=np.int64))
        offset += len(features)
    matrix = np.vstack(feature_blocks)
    residual = np.concatenate(residual_blocks)
    scores = scan_prebinned_pairs(
        matrix,
        residual,
        np.ones(len(residual), dtype=np.float64),
        blocks,
        bins=4,
        min_cell_weight=32.0,
        max_surface_cells=16,
    )
    selection = select_proposal_candidates(
        scores,
        baseline_indices=set(range(matrix.shape[1])),
        protocol=default_proposal_protocol(),
    )
    pair_rows = {
        tuple(int(value) for value in pair): index
        for index, pair in enumerate(scores["pair_indices"])
    }
    xor_pair = (0, 1)
    additive_pair = (2, 3)
    return {
        "xor_pair": list(xor_pair),
        "xor_eligible": bool(scores["eligible"][pair_rows[xor_pair]]),
        "additive_control_pair": list(additive_pair),
        "additive_control_eligible": bool(
            scores["eligible"][pair_rows[additive_pair]]
        ),
        "pairs": [list(pair) for pair in selection["pairs"]],
        "scanned_pairs": len(scores["pair_indices"]),
        "pair_split_count": int(scores["scored_pair_split_count"]),
        "manifest_sha256": selection["manifest_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ridge", "xs", "market"], default="ridge")
    parser.add_argument("--input-npz")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--label")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        print(json.dumps(run_synthetic_smoke(), indent=2, sort_keys=True))
        return
    required = (args.input_npz, args.candidate_dir, args.output_dir, args.label)
    if any(value is None for value in required):
        raise SystemExit(
            "--input-npz, --candidate-dir, --output-dir, and --label are required"
        )
    paths = run_proposal(
        task=args.task,
        input_path=args.input_npz,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        label=args.label,
        force=args.force,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
