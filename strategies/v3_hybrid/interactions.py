"""Versioned additive interaction definitions shared by V3 training and inference.

This module intentionally depends only on NumPy so the official submission package can
use exactly the same validation and column construction as offline training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


INTERACTION_SCHEMA_VERSION = 1
SOURCE_FAMILIES = frozenset({
    "current",
    "xs_deviation",
    "market_raw",
    "market_deviation",
    "history_previous",
    "history_difference",
    "history_rolling_mean",
    "history_rolling_deviation",
})
HISTORY_SOURCE_FAMILIES = frozenset(
    family for family in SOURCE_FAMILIES if family.startswith("history_")
)
CURRENT_SOURCE_FAMILIES = SOURCE_FAMILIES - HISTORY_SOURCE_FAMILIES
OPERATIONS = frozenset({"region", "gated_value", "current_history_gated"})
TASKS = ("ridge", "xs", "market")
_TASK_SOURCE_FAMILIES = {
    "ridge": frozenset({"current"}),
    "xs": frozenset({"xs_deviation", *HISTORY_SOURCE_FAMILIES}),
    "market": frozenset({"market_raw", "market_deviation", *HISTORY_SOURCE_FAMILIES}),
}
_STAT_NAMES = ("lower", "upper", "center", "scale")


def source_key(family: str, feature_index: int) -> str:
    if family not in SOURCE_FAMILIES:
        raise ValueError(f"unknown source family: {family}")
    if isinstance(feature_index, bool) or int(feature_index) != feature_index or feature_index < 0:
        raise ValueError("feature_index must be a non-negative integer")
    return f"{family}:{int(feature_index)}"


def _parse_source(value: object) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError("interaction source must be a string")
    family, separator, raw_index = value.partition(":")
    if not separator or family not in SOURCE_FAMILIES:
        raise ValueError(f"unknown source family in {value!r}")
    try:
        feature_index = int(raw_index)
    except ValueError as error:
        raise ValueError(f"invalid feature index in {value!r}") from error
    if feature_index < 0 or raw_index != str(feature_index):
        raise ValueError(f"invalid feature index in {value!r}")
    return family, feature_index


def _validated_condition(condition: object) -> str:
    if not isinstance(condition, Mapping):
        raise ValueError("interaction condition must be a mapping")
    source = condition.get("source")
    _parse_source(source)
    if condition.get("direction") not in {"gt", "le"}:
        raise ValueError("interaction direction must be 'gt' or 'le'")
    threshold = condition.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise ValueError("interaction condition requires a finite threshold")
    if not np.isfinite(float(threshold)):
        raise ValueError("interaction condition requires a finite threshold")
    if not isinstance(condition.get("missing_matches"), bool):
        raise ValueError("interaction condition requires boolean missing_matches")
    return str(source)


def validate_interaction_definitions(definitions: list[dict[str, object]]) -> None:
    """Reject ambiguous, non-finite, duplicated, or unsupported definitions."""
    if not isinstance(definitions, list):
        raise ValueError("interaction definitions must be a list")
    names: set[str] = set()
    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise ValueError("interaction definition must be a mapping")
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("interaction output name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate interaction output name: {name}")
        names.add(name)

        operation = definition.get("operation")
        if operation not in OPERATIONS:
            raise ValueError(f"unsupported interaction operation: {operation!r}")
        conditions = definition.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("interaction conditions must be a non-empty list")
        condition_sources = [_validated_condition(condition) for condition in conditions]
        if len(condition_sources) != len(set(condition_sources)):
            raise ValueError("interaction path contains a duplicate semantic source")

        all_sources = list(condition_sources)
        value_source = definition.get("value_source")
        if operation == "region":
            if value_source is not None:
                raise ValueError("region interaction cannot define value_source")
        else:
            _parse_source(value_source)
            value_source = str(value_source)
            if value_source in condition_sources:
                raise ValueError("interaction path contains a duplicate semantic source")
            all_sources.append(value_source)

        if not 2 <= len(all_sources) <= 4:
            raise ValueError("interaction must contain between 2 and 4 distinct sources")

        if operation == "current_history_gated":
            value_family, _ = _parse_source(value_source)
            if value_family not in HISTORY_SOURCE_FAMILIES:
                raise ValueError("current_history_gated value_source must be history")
            if any(_parse_source(source)[0] in HISTORY_SOURCE_FAMILIES
                   for source in condition_sources):
                raise ValueError("current_history_gated conditions must be current sources")


def interaction_source_keys(definitions: list[dict[str, object]]) -> list[str]:
    """Return unique source keys in deterministic first-use order."""
    validate_interaction_definitions(definitions)
    ordered: list[str] = []
    seen: set[str] = set()
    for definition in definitions:
        candidates = [str(condition["source"]) for condition in definition["conditions"]]
        if definition.get("value_source") is not None:
            candidates.append(str(definition["value_source"]))
        for key in candidates:
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def _condition_mask(values: np.ndarray, condition: Mapping[str, object]) -> np.ndarray:
    finite = np.isfinite(values)
    threshold = float(condition["threshold"])
    if condition["direction"] == "gt":
        result = values > threshold
    else:
        result = values <= threshold
    result = np.asarray(result, dtype=bool)
    result[~finite] = bool(condition["missing_matches"])
    return result


def build_interaction_columns(
    sources: Mapping[str, np.ndarray],
    definitions: list[dict[str, object]],
    *,
    max_cells: int,
) -> np.ndarray:
    """Build float32 columns in manifest order and fail before exceeding max_cells."""
    validate_interaction_definitions(definitions)
    if isinstance(max_cells, bool) or not isinstance(max_cells, (int, np.integer)):
        raise ValueError("max_cells must be a non-negative integer")
    if max_cells < 0:
        raise ValueError("max_cells must be a non-negative integer")

    required = interaction_source_keys(definitions)
    missing = [key for key in required if key not in sources]
    if missing:
        raise ValueError(f"interaction sources are missing: {missing}")

    row_count: int | None = None
    arrays: dict[str, np.ndarray] = {}
    for key in required:
        values = np.asarray(sources[key])
        if values.ndim != 1:
            raise ValueError(f"interaction source {key} must be one-dimensional")
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError("interaction sources must have equal row counts")
        arrays[key] = values

    if row_count is None:
        row_count = len(np.asarray(next(iter(sources.values())))) if sources else 0
    cells = row_count * len(definitions)
    if cells > max_cells:
        raise MemoryError(f"interaction matrix {cells} cells exceeds max_cells={max_cells}")

    output = np.empty((row_count, len(definitions)), dtype=np.float32)
    for column, definition in enumerate(definitions):
        gate = np.ones(row_count, dtype=bool)
        for condition in definition["conditions"]:
            gate &= _condition_mask(arrays[str(condition["source"])], condition)
        if definition["operation"] == "region":
            output[:, column] = gate
        else:
            values = arrays[str(definition["value_source"])]
            output[:, column] = np.where(gate, values, 0.0)
    return output


def _copy_direct_features(
    direct_features: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    missing = [task for task in TASKS if task not in direct_features]
    if missing:
        raise ValueError(f"direct feature contract is missing tasks: {missing}")
    copied = {task: [str(name) for name in direct_features[task]] for task in TASKS}
    for task, names in copied.items():
        if len(names) != len(set(names)):
            raise ValueError(f"{task} direct features contain duplicates")
    return copied


def _normalise_source_stats(task: str, payload: object) -> dict[str, list[Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{task} interaction_source_stats must be a mapping")
    result = {name: list(payload.get(name, [])) for name in ("features", *_STAT_NAMES)}
    count = len(result["features"])
    if any(len(result[name]) != count for name in _STAT_NAMES):
        raise ValueError(f"{task} source-only preprocessing arrays have inconsistent lengths")
    if len(result["features"]) != len(set(result["features"])):
        raise ValueError(f"{task} source-only preprocessing features contain duplicates")
    arrays = {name: np.asarray(result[name], dtype=np.float64) for name in _STAT_NAMES}
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError(f"{task} source-only preprocessing values must be finite")
    if np.any(arrays["scale"] <= 0):
        raise ValueError(f"{task} source-only preprocessing scale must be positive")
    if np.any(arrays["lower"] > arrays["upper"]):
        raise ValueError(f"{task} source-only preprocessing lower exceeds upper")
    result["features"] = [str(name) for name in result["features"]]
    return result


def resolve_interaction_contract(
    meta: Mapping[str, object],
    *,
    direct_features: Mapping[str, Sequence[str]],
    n_features: int = 323,
) -> dict[str, object]:
    """Validate interaction metadata while keeping source-only columns indirect."""
    direct = _copy_direct_features(direct_features)
    version = meta.get("interaction_schema_version", 0)
    if version == 0:
        return {
            "schema_version": 0,
            "definitions": {task: [] for task in TASKS},
            "source_columns": [],
            "source_stats": {
                task: {"features": [], **{name: [] for name in _STAT_NAMES}}
                for task in TASKS
            },
            "direct_features": direct,
        }
    if version != INTERACTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported interaction schema version: {version!r}")

    raw_definitions = meta.get("interactions")
    raw_stats = meta.get("interaction_source_stats")
    if not isinstance(raw_definitions, Mapping) or not isinstance(raw_stats, Mapping):
        raise ValueError("schema version 1 requires interactions and interaction_source_stats")

    definitions: dict[str, list[dict[str, object]]] = {}
    source_stats: dict[str, dict[str, list[Any]]] = {}
    source_columns: list[str] = []
    seen_columns: set[str] = set()

    for task in TASKS:
        task_definitions = raw_definitions.get(task)
        if not isinstance(task_definitions, list):
            raise ValueError(f"interactions.{task} must be a list")
        validate_interaction_definitions(task_definitions)
        definitions[task] = list(task_definitions)

        all_keys = interaction_source_keys(task_definitions)
        for key in all_keys:
            family, index = _parse_source(key)
            if family not in _TASK_SOURCE_FAMILIES[task]:
                raise ValueError(f"{task} interaction cannot use source family {family}")
            if index >= n_features:
                raise ValueError(f"{task} interaction source index is out of range: {index}")

        stats = _normalise_source_stats(task, raw_stats.get(task))
        source_stats[task] = stats
        direct_set = set(direct[task])
        required_external = {
            f"feature_{index:03d}"
            for key in all_keys
            for family, index in [_parse_source(key)]
            if family in CURRENT_SOURCE_FAMILIES
            and f"feature_{index:03d}" not in direct_set
        }
        available = set(stats["features"])
        missing_external = sorted(required_external - available)
        if missing_external:
            raise ValueError(
                f"{task} source-only preprocessing is missing {', '.join(missing_external)}"
            )
        extras = sorted(available - required_external)
        if extras:
            raise ValueError(
                f"{task} source-only preprocessing has unused features: {extras}"
            )
        for name in stats["features"]:
            if name not in seen_columns:
                source_columns.append(name)
                seen_columns.add(name)

    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "definitions": definitions,
        "source_columns": source_columns,
        "source_stats": source_stats,
        "direct_features": direct,
    }

