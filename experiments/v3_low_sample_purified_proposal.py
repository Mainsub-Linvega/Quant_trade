"""Leakage-safe low-sample proposal scan for purified feature pairs."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

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
