"""Training-only discovery of stable conditional interaction columns.

Shallow LightGBM models are used only to propose paths from strict out-of-sample
baseline residuals. The boosters are never part of the prediction ensemble.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.v3_adaptive_selection_manifest import chronological_inner_splits
from strategies.v3_hybrid.features import cross_sectional_deviation
from strategies.v3_hybrid.interactions import (
    CURRENT_SOURCE_FAMILIES,
    HISTORY_SOURCE_FAMILIES,
    SOURCE_FAMILIES,
    validate_interaction_definitions,
)


@dataclass(frozen=True, order=True)
class Source:
    family: str
    feature_index: int

    def __post_init__(self) -> None:
        if self.family not in SOURCE_FAMILIES:
            raise ValueError(f"unknown source family: {self.family}")
        if (
            isinstance(self.feature_index, bool)
            or not isinstance(self.feature_index, (int, np.integer))
            or self.feature_index < 0
        ):
            raise ValueError("feature index must be a non-negative integer")

    @property
    def key(self) -> str:
        return f"{self.family}:{self.feature_index}"


@dataclass(frozen=True)
class PathCondition:
    source: Source
    direction: str
    threshold: float
    missing_matches: bool


@dataclass(frozen=True)
class PathCandidate:
    conditions: tuple[PathCondition, ...]
    block_index: int
    tree_index: int
    leaf_index: int


@dataclass(frozen=True)
class CanonicalPath:
    support_key: str
    ordered_conditions: tuple[dict[str, object], ...]
    blocks: tuple[int, ...]
    block_index: int
    tree_index: int
    leaf_index: int


@dataclass(frozen=True)
class TaskSourceView:
    values: np.ndarray
    catalog: tuple[Source, ...]


_HISTORY_FAMILIES = (
    "history_previous",
    "history_difference",
    "history_rolling_mean",
    "history_rolling_deviation",
)


def _validate_source_view_inputs(
    transformed: np.ndarray,
    time_ids: np.ndarray,
    history_indices: np.ndarray,
    history_blocks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    values = np.asarray(transformed, dtype=np.float32)
    ids = np.asarray(time_ids, dtype=np.int64)
    indices = np.asarray(history_indices, dtype=np.int64)
    blocks = tuple(np.asarray(block, dtype=np.float32) for block in history_blocks)
    if values.ndim != 2 or values.shape[1] != 323:
        raise ValueError("interaction discovery requires exactly 323 current features")
    if ids.shape != (len(values),) or np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be row-aligned and nondecreasing")
    if indices.shape != (40,) or len(np.unique(indices)) != 40:
        raise ValueError("interaction history must contain exactly 40 unique bases")
    if np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise ValueError("history feature index is outside the current feature matrix")
    if len(blocks) != 4 or any(block.shape != (len(values), 40) for block in blocks):
        raise ValueError("history must contain four row-aligned 40-column blocks")
    return values, ids, indices, blocks


def _source_width(task: str) -> int:
    widths = {"ridge": 323, "xs": 323 + 4 * 40, "market": 2 * 323 + 4 * 40}
    if task not in widths:
        raise ValueError(f"unknown interaction task: {task}")
    return widths[task]


def build_interaction_source_view(
    task: str,
    transformed: np.ndarray,
    time_ids: np.ndarray,
    history_indices: np.ndarray,
    history_blocks: Sequence[np.ndarray],
    *,
    max_cells: int,
) -> TaskSourceView:
    """Materialize one task source universe after an explicit cell-budget check."""
    values, ids, indices, blocks = _validate_source_view_inputs(
        transformed, time_ids, history_indices, history_blocks
    )
    width = _source_width(task)
    cells = len(values) * width
    if cells > max_cells:
        raise MemoryError(
            f"{task} source matrix {cells} cells exceeds max_cells={max_cells}"
        )
    current_indices = range(values.shape[1])
    history_catalog = tuple(
        Source(family, int(feature_index))
        for family in _HISTORY_FAMILIES
        for feature_index in indices
    )
    if task == "ridge":
        matrix = np.ascontiguousarray(values, dtype=np.float32)
        catalog = tuple(Source("current", index) for index in current_indices)
    else:
        deviation = cross_sectional_deviation(values.copy(), ids)
        if task == "xs":
            matrix = np.ascontiguousarray(
                np.column_stack([deviation, *blocks]), dtype=np.float32
            )
            catalog = (
                tuple(Source("xs_deviation", index) for index in current_indices)
                + history_catalog
            )
        else:
            matrix = np.ascontiguousarray(
                np.column_stack([values, deviation, *blocks]), dtype=np.float32
            )
            catalog = (
                tuple(Source("market_raw", index) for index in current_indices)
                + tuple(Source("market_deviation", index) for index in current_indices)
                + history_catalog
            )
    if matrix.shape != (len(values), width) or len(catalog) != width:
        raise AssertionError("interaction source matrix and catalog width disagree")
    return TaskSourceView(matrix, catalog)


def build_interaction_source_views(
    transformed: np.ndarray,
    time_ids: np.ndarray,
    history_indices: np.ndarray,
    history_blocks: Sequence[np.ndarray],
    *,
    max_cells: int,
) -> dict[str, TaskSourceView]:
    """Small-data convenience wrapper; production should build one task at a time."""
    rows = len(np.asarray(transformed))
    for task in ("ridge", "xs", "market"):
        cells = rows * _source_width(task)
        if cells > max_cells:
            raise MemoryError(
                f"{task} source matrix {cells} cells exceeds max_cells={max_cells}"
            )
    return {
        task: build_interaction_source_view(
            task,
            transformed,
            time_ids,
            history_indices,
            history_blocks,
            max_cells=max_cells,
        )
        for task in ("ridge", "xs", "market")
    }


def _tree_roots(model_dump: Any) -> list[tuple[int, Mapping[str, Any]]]:
    if not isinstance(model_dump, Mapping):
        raise ValueError("model_dump must be a mapping")
    tree_info = model_dump.get("tree_info")
    if isinstance(tree_info, Sequence) and not isinstance(tree_info, (str, bytes)):
        roots: list[tuple[int, Mapping[str, Any]]] = []
        for tree_index, tree in enumerate(tree_info):
            if isinstance(tree, Mapping) and isinstance(tree.get("tree_structure"), Mapping):
                roots.append((tree_index, tree["tree_structure"]))
        return roots
    if isinstance(model_dump.get("tree_structure"), Mapping):
        return [(0, model_dump["tree_structure"])]
    if "split_feature" in model_dump or "leaf_value" in model_dump:
        return [(0, model_dump)]
    raise ValueError("model_dump does not contain a tree structure")


def extract_candidate_paths(
    model_dump: Any,
    catalog: Sequence[Source],
    *,
    block_index: int,
    min_sources: int = 2,
    max_sources: int = 4,
) -> list[PathCandidate]:
    """Extract complete LightGBM conditions from valid 2-4-source paths."""
    if min_sources <= 0 or max_sources < min_sources:
        raise ValueError("invalid interaction path bounds")
    sources = tuple(catalog)
    if len(sources) != len(set(sources)):
        raise ValueError("source catalog contains duplicates")
    paths: list[PathCandidate] = []
    leaf_index = 0

    def visit(
        node: Mapping[str, Any],
        conditions: tuple[PathCondition, ...],
        tree_index: int,
    ) -> None:
        nonlocal leaf_index
        if "split_feature" not in node:
            current_leaf = leaf_index
            leaf_index += 1
            distinct = [condition.source for condition in conditions]
            if (
                min_sources <= len(distinct) <= max_sources
                and len(distinct) == len(set(distinct))
            ):
                paths.append(
                    PathCandidate(
                        conditions=conditions,
                        block_index=int(block_index),
                        tree_index=tree_index,
                        leaf_index=current_leaf,
                    )
                )
            return

        feature = int(node["split_feature"])
        if feature < 0 or feature >= len(sources):
            raise ValueError(f"tree split_feature is outside source catalog: {feature}")
        if node.get("decision_type", "<=") != "<=":
            raise ValueError("interaction miner only supports numeric <= splits")
        threshold = float(node.get("threshold"))
        if not np.isfinite(threshold):
            raise ValueError("tree split threshold must be finite")
        default_left = bool(node.get("default_left", True))
        source = sources[feature]
        children = (
            (
                node.get("left_child"),
                PathCondition(source, "le", threshold, default_left),
            ),
            (
                node.get("right_child"),
                PathCondition(source, "gt", threshold, not default_left),
            ),
        )
        if not any(isinstance(child, Mapping) for child, _ in children):
            raise ValueError("split node has no child mappings")
        for child, condition in children:
            if isinstance(child, Mapping):
                visit(child, (*conditions, condition), tree_index)

    for tree_index, root in _tree_roots(model_dump):
        visit(root, (), tree_index)
    return sorted(
        paths,
        key=lambda path: (
            path.tree_index,
            path.leaf_index,
            tuple((item.source, item.direction) for item in path.conditions),
        ),
    )


def training_quantile_grids(
    source_values: np.ndarray,
    catalog: Sequence[Source],
    *,
    bins: int = 32,
) -> dict[Source, np.ndarray]:
    """Fit source-specific finite empirical quantile grids."""
    values = np.asarray(source_values)
    sources = tuple(catalog)
    if values.ndim != 2 or values.shape[1] != len(sources):
        raise ValueError("source_values columns must match the source catalog")
    if bins < 2:
        raise ValueError("quantile bins must be at least two")
    grids: dict[Source, np.ndarray] = {}
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    for column, source in enumerate(sources):
        finite = np.asarray(values[:, column], dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            raise ValueError(f"interaction source {source.key} has no finite training values")
        grids[source] = np.asarray(np.quantile(finite, quantiles), dtype=np.float64)
    return grids


def _canonical_condition(
    condition: PathCondition,
    grid: np.ndarray,
) -> dict[str, object]:
    bin_count = len(grid) - 1
    if bin_count < 1:
        raise ValueError("quantile grid must contain at least two values")
    quantile_bin = int(np.searchsorted(grid, condition.threshold, side="right") - 1)
    quantile_bin = int(np.clip(quantile_bin, 0, bin_count - 1))
    resolved_threshold = float(grid[quantile_bin + 1])
    return {
        "source": condition.source.key,
        "direction": condition.direction,
        "quantile_bin": quantile_bin,
        "threshold": resolved_threshold,
        "missing_matches": condition.missing_matches,
    }


def _condition_token(condition: Mapping[str, object]) -> str:
    return (
        f"{condition['source']}:{condition['direction']}:"
        f"q{int(condition['quantile_bin'])}:"
        f"m{int(bool(condition['missing_matches']))}"
    )


def canonicalize_path(
    path: PathCandidate,
    quantile_grids: Mapping[Source, np.ndarray],
) -> CanonicalPath:
    """Map thresholds into training quantile bins without losing path order."""
    ordered: list[dict[str, object]] = []
    for condition in path.conditions:
        if condition.source not in quantile_grids:
            raise ValueError(f"missing quantile grid for {condition.source.key}")
        ordered.append(
            _canonical_condition(condition, np.asarray(quantile_grids[condition.source]))
        )
    support_key = "|".join(sorted(_condition_token(item) for item in ordered))
    return CanonicalPath(
        support_key=support_key,
        ordered_conditions=tuple(ordered),
        blocks=(path.block_index,),
        block_index=path.block_index,
        tree_index=path.tree_index,
        leaf_index=path.leaf_index,
    )


def aggregate_repeated_paths(
    paths: Sequence[CanonicalPath],
    *,
    min_blocks: int = 2,
) -> list[CanonicalPath]:
    """Keep one deterministic representative for paths repeated across blocks."""
    if min_blocks <= 0:
        raise ValueError("min_blocks must be positive")
    grouped: dict[str, list[CanonicalPath]] = defaultdict(list)
    for path in paths:
        grouped[path.support_key].append(path)

    accepted: list[CanonicalPath] = []
    for support_key, matches in grouped.items():
        blocks = tuple(sorted({block for path in matches for block in path.blocks}))
        if len(blocks) < min_blocks:
            continue
        representative = min(
            matches,
            key=lambda path: (
                path.block_index,
                path.tree_index,
                path.leaf_index,
                tuple(_condition_token(item) for item in path.ordered_conditions),
            ),
        )
        accepted.append(
            CanonicalPath(
                support_key=support_key,
                ordered_conditions=representative.ordered_conditions,
                blocks=blocks,
                block_index=representative.block_index,
                tree_index=representative.tree_index,
                leaf_index=representative.leaf_index,
            )
        )
    return sorted(
        accepted,
        key=lambda path: (
            -len(path.blocks),
            path.support_key,
            path.block_index,
            path.tree_index,
            path.leaf_index,
        ),
    )


def _definition_conditions(
    conditions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "source": str(condition["source"]),
            "direction": str(condition["direction"]),
            "threshold": float(condition["threshold"]),
            "quantile_bin": int(condition["quantile_bin"]),
            "missing_matches": bool(condition["missing_matches"]),
        }
        for condition in conditions
    ]


def definitions_from_paths(
    task: str,
    paths: Sequence[CanonicalPath],
) -> list[dict[str, object]]:
    """Convert accepted paths into deterministic derived-column definitions."""
    if task not in {"ridge", "xs", "market"}:
        raise ValueError(f"unknown interaction task: {task}")
    definitions: list[dict[str, object]] = []
    for path_index, path in enumerate(paths):
        conditions = list(path.ordered_conditions)
        definitions.append({
            "name": f"{task}_region_{path_index:04d}",
            "operation": "region",
            "conditions": _definition_conditions(conditions),
        })
        definitions.append({
            "name": f"{task}_gated_{path_index:04d}",
            "operation": "gated_value",
            "value_source": str(conditions[-1]["source"]),
            "conditions": _definition_conditions(conditions[:-1]),
        })

        history = [
            condition
            for condition in conditions
            if str(condition["source"]).split(":", 1)[0] in HISTORY_SOURCE_FAMILIES
        ]
        current = [
            condition
            for condition in conditions
            if str(condition["source"]).split(":", 1)[0] in CURRENT_SOURCE_FAMILIES
        ]
        if history and current:
            definitions.append({
                "name": f"{task}_current_history_{path_index:04d}",
                "operation": "current_history_gated",
                "value_source": str(history[-1]["source"]),
                "conditions": _definition_conditions(current),
            })
    validate_interaction_definitions(definitions)
    return definitions


def _time_preserving_cap(
    rows: np.ndarray,
    time_ids: np.ndarray,
    row_cap: int,
) -> np.ndarray:
    if len(rows) <= row_cap:
        return rows
    row_ids = time_ids[rows]
    unique, starts, counts = np.unique(row_ids, return_index=True, return_counts=True)
    if counts.max() > row_cap:
        raise ValueError("one time_id exceeds the interaction miner row cap")
    desired_groups = max(1, int(row_cap // counts.max()))
    positions = np.unique(
        np.linspace(0, len(unique) - 1, min(desired_groups, len(unique)), dtype=np.int64)
    )
    selected: list[np.ndarray] = []
    used = 0
    for position in positions:
        count = int(counts[position])
        if used + count > row_cap:
            continue
        start = int(starts[position])
        selected.append(rows[start:start + count])
        used += count
    if not selected:
        raise ValueError("row cap cannot retain a complete time_id")
    return np.concatenate(selected)


def _train_residual_booster(
    features: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    params: dict[str, object],
    num_boost_round: int,
) -> Any:
    import lightgbm as lgb

    dataset = lgb.Dataset(
        features,
        label=residual,
        weight=weight,
        free_raw_data=False,
    )
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def mine_task_interactions(
    *,
    task: str,
    source_values: np.ndarray,
    catalog: Sequence[Source],
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
    baseline_predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_blocks: int = 4,
    min_blocks: int = 2,
    quantile_bins: int = 32,
    num_boost_round: int = 80,
    row_cap: int = 150_000,
    seed: int = 2026,
    num_threads: int = 1,
) -> dict[str, object]:
    """Mine repeated paths from strict expanding OOS baseline residuals."""
    if task not in {"ridge", "xs", "market"}:
        raise ValueError(f"unknown interaction task: {task}")
    values = np.asarray(source_values)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    sources = tuple(catalog)
    if values.ndim != 2 or values.shape[1] != len(sources):
        raise ValueError("source_values columns must match catalog")
    if not (len(values) == len(y) == len(w) == len(ids)):
        raise ValueError("interaction miner inputs must have equal rows")
    if len(sources) != len(set(sources)):
        raise ValueError("source catalog contains duplicates")
    if row_cap <= 0 or num_boost_round <= 0 or num_threads <= 0:
        raise ValueError("miner budgets must be positive")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("target and positive weights must be finite")

    params: dict[str, object] = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "num_threads": int(num_threads),
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
    }
    canonical_paths: list[CanonicalPath] = []
    split_payloads: list[dict[str, int]] = []
    splits = chronological_inner_splits(ids, n_blocks=n_blocks)
    for block_index, (train_rows, valid_rows) in enumerate(splits, start=1):
        prediction = np.asarray(
            baseline_predictor(train_rows.copy(), valid_rows.copy()),
            dtype=np.float64,
        )
        if prediction.shape != (len(valid_rows),) or not np.all(np.isfinite(prediction)):
            raise ValueError("baseline_predictor returned invalid OOS predictions")
        residual = y[valid_rows] - prediction
        kept_rows = _time_preserving_cap(valid_rows, ids, row_cap)
        valid_positions = np.searchsorted(valid_rows, kept_rows)
        booster = _train_residual_booster(
            values[kept_rows],
            residual[valid_positions],
            w[kept_rows],
            params=params,
            num_boost_round=num_boost_round,
        )
        grids = training_quantile_grids(
            values[train_rows],
            sources,
            bins=quantile_bins,
        )
        candidates = extract_candidate_paths(
            booster.dump_model(),
            sources,
            block_index=block_index,
        )
        canonical_paths.extend(canonicalize_path(path, grids) for path in candidates)
        split_payloads.append({
            "block": block_index,
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(valid_rows)),
            "miner_rows": int(len(kept_rows)),
            "candidate_paths": int(len(candidates)),
        })

    accepted = aggregate_repeated_paths(canonical_paths, min_blocks=min_blocks)
    definitions = definitions_from_paths(task, accepted)
    return {
        "task": task,
        "definitions": definitions,
        "accepted_paths": [
            {
                "support_key": path.support_key,
                "blocks": list(path.blocks),
                "support": len(path.blocks),
                "conditions": list(path.ordered_conditions),
                "representative": {
                    "block": path.block_index,
                    "tree": path.tree_index,
                    "leaf": path.leaf_index,
                },
            }
            for path in accepted
        ],
        "protocol": {
            "strict_oos_residuals": True,
            "inner_blocks": int(n_blocks),
            "inner_validation_splits": len(splits),
            "minimum_support_blocks": int(min_blocks),
            "quantile_bins": int(quantile_bins),
            "num_boost_round": int(num_boost_round),
            "max_depth": 4,
            "num_leaves": 15,
            "seed": int(seed),
            "row_cap": int(row_cap),
            "num_threads": int(num_threads),
            "splits": split_payloads,
        },
    }
