"""Pure adaptive feature-selection algorithms for V3 experiments.

The selector consumes block-level correlations, tree gains, redundancy labels, and
precomputed tree paths. Model fitting and file orchestration intentionally live elsewhere.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def circular_shift_shadows(
    target: np.ndarray,
    n_shadows: int,
) -> dict[str, np.ndarray]:
    """Create deterministic non-identity circular shifts of a training target."""
    values = np.asarray(target)
    if values.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if len(values) < 2:
        raise ValueError("target must contain at least two rows")
    if n_shadows <= 0:
        raise ValueError("n_shadows must be positive")
    if n_shadows >= len(values):
        raise ValueError("n_shadows must be smaller than the target length")

    shifts = (
        np.arange(1, n_shadows + 1, dtype=np.int64) * len(values)
        // (n_shadows + 1)
    )
    targets = np.stack([np.roll(values, int(shift)) for shift in shifts])
    return {"shifts": shifts, "targets": targets}


def _block_matrix(
    values: np.ndarray,
    name: str,
    *,
    require_finite: bool = False,
) -> np.ndarray:
    """Validate a block matrix, conservatively zeroing non-finite feature values."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    finite = np.isfinite(matrix)
    if require_finite and not np.all(finite):
        raise ValueError(f"{name} must contain only finite values")
    return np.where(finite, matrix, 0.0)


def _quantile(values: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("shadow_quantile must be between zero and one")
    return float(np.quantile(values, quantile, method="higher"))


def linear_evidence(
    block_correlations: np.ndarray,
    shadow_block_correlations: np.ndarray,
    *,
    min_direction_consistency: float = 0.75,
    shadow_quantile: float = 0.95,
) -> dict[str, Any]:
    """Score stable marginal correlation against a training-only shadow floor."""
    correlations = _block_matrix(block_correlations, "block_correlations")
    shadows = _block_matrix(
        shadow_block_correlations,
        "shadow_block_correlations",
        require_finite=True,
    )
    if shadows.shape[0] != correlations.shape[0]:
        raise ValueError("feature and shadow correlations must use the same blocks")
    if not 0.0 <= min_direction_consistency <= 1.0:
        raise ValueError("min_direction_consistency must be between zero and one")

    score = np.median(np.abs(correlations), axis=0)
    positive_fraction = np.mean(correlations > 0.0, axis=0)
    negative_fraction = np.mean(correlations < 0.0, axis=0)
    direction_consistency = np.maximum(positive_fraction, negative_fraction)
    shadow_scores = np.median(np.abs(shadows), axis=0)
    shadow_floor = _quantile(shadow_scores, shadow_quantile)
    passed = (score > shadow_floor) & (
        direction_consistency >= min_direction_consistency
    )
    return {
        "score": score,
        "direction_consistency": direction_consistency,
        "shadow_scores": shadow_scores,
        "shadow_floor": shadow_floor,
        "passed": passed,
    }


def _tree_roots(model_dump: Any) -> list[Mapping[str, Any]]:
    if isinstance(model_dump, Mapping):
        if "tree_info" in model_dump:
            return [
                tree["tree_structure"]
                for tree in model_dump["tree_info"]
                if isinstance(tree, Mapping)
                and isinstance(tree.get("tree_structure"), Mapping)
            ]
        if isinstance(model_dump.get("tree_structure"), Mapping):
            return [model_dump["tree_structure"]]
        return [model_dump]
    if isinstance(model_dump, Sequence) and not isinstance(
        model_dump, (str, bytes)
    ):
        roots: list[Mapping[str, Any]] = []
        for item in model_dump:
            roots.extend(_tree_roots(item))
        return roots
    raise ValueError("model_dump must contain tree mappings")


def extract_tree_paths(
    model_dump: Any,
    *,
    min_features: int = 3,
    max_features: int = 6,
) -> list[tuple[int, ...]]:
    """Extract distinct-feature root-to-leaf paths from LightGBM-style dumps."""
    if min_features <= 0 or max_features < min_features:
        raise ValueError("invalid path feature bounds")

    paths: set[tuple[int, ...]] = set()

    def visit(node: Mapping[str, Any], path: tuple[int, ...]) -> None:
        if "split_feature" not in node:
            distinct = tuple(dict.fromkeys(path))
            if min_features <= len(distinct) <= max_features:
                paths.add(distinct)
            return

        feature = int(node["split_feature"])
        next_path = (*path, feature)
        children = [
            child
            for child in (node.get("left_child"), node.get("right_child"))
            if isinstance(child, Mapping)
        ]
        if not children:
            distinct = tuple(dict.fromkeys(next_path))
            if min_features <= len(distinct) <= max_features:
                paths.add(distinct)
            return
        for child in children:
            visit(child, next_path)

    for root in _tree_roots(model_dump):
        visit(root, ())
    return sorted(paths)


def aggregate_path_support(
    paths_by_block: Sequence[Sequence[Sequence[int]]],
    *,
    min_blocks: int = 2,
    min_features: int = 3,
    max_features: int = 6,
) -> list[dict[str, Any]]:
    """Aggregate unordered path hyperedges, counting each at most once per block."""
    if min_blocks <= 0:
        raise ValueError("min_blocks must be positive")
    if min_features <= 0 or max_features < min_features:
        raise ValueError("invalid path feature bounds")

    blocks_by_edge: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for block_index, block_paths in enumerate(paths_by_block):
        block_edges: set[tuple[int, ...]] = set()
        for path in block_paths:
            features = tuple(sorted({int(feature) for feature in path}))
            if min_features <= len(features) <= max_features:
                block_edges.add(features)
        for edge in block_edges:
            blocks_by_edge[edge].append(block_index)

    accepted = [
        {
            "features": list(edge),
            "support": len(blocks),
            "blocks": blocks,
        }
        for edge, blocks in blocks_by_edge.items()
        if len(blocks) >= min_blocks
    ]
    return sorted(
        accepted,
        key=lambda item: (-int(item["support"]), item["features"]),
    )


def select_from_clusters(
    passed: np.ndarray,
    cluster_labels: np.ndarray,
    scores: np.ndarray,
    path_hyperedges: Sequence[Mapping[str, Any]],
    residual_alternate_passed: np.ndarray | None = None,
    residual_alternate_scores: np.ndarray | None = None,
) -> dict[str, list[int]]:
    """Select cluster representatives and path-distinct supported alternates."""
    passed_array = np.asarray(passed, dtype=bool)
    labels = np.asarray(cluster_labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if passed_array.ndim != 1:
        raise ValueError("passed must be one-dimensional")
    if labels.shape != passed_array.shape or score_array.shape != passed_array.shape:
        raise ValueError("cluster labels and scores must match passed")
    score_array = np.where(np.isfinite(score_array), score_array, -np.inf)
    if residual_alternate_passed is None and residual_alternate_scores is None:
        residual_passed = np.zeros_like(passed_array)
        residual_scores = np.zeros_like(score_array)
    elif residual_alternate_passed is None or residual_alternate_scores is None:
        raise ValueError("residual alternate pass flags and scores must be supplied together")
    else:
        residual_passed = np.asarray(residual_alternate_passed, dtype=bool)
        residual_scores = np.asarray(residual_alternate_scores, dtype=np.float64)
        if residual_passed.shape != passed_array.shape or residual_scores.shape != passed_array.shape:
            raise ValueError("residual alternate evidence must match passed")
        residual_scores = np.where(np.isfinite(residual_scores), residual_scores, -np.inf)

    representatives: list[int] = []
    representative_by_cluster: dict[Any, int] = {}
    passing_indices = np.flatnonzero(passed_array)
    for label in dict.fromkeys(labels[passing_indices].tolist()):
        members = [
            int(index)
            for index in passing_indices
            if labels[index] == label
        ]
        representative = min(
            members,
            key=lambda index: (-score_array[index], index),
        )
        representatives.append(representative)
        representative_by_cluster[label] = representative

    path_alternates: set[int] = set()
    for edge in path_hyperedges:
        features = {int(feature) for feature in edge.get("features", [])}
        if not features or any(
            feature < 0
            or feature >= len(passed_array)
            or not passed_array[feature]
            for feature in features
        ):
            continue
        for feature in features:
            representative = representative_by_cluster.get(labels[feature])
            if (
                representative is not None
                and feature != representative
                and representative not in features
            ):
                path_alternates.add(feature)

    residual_alternates: set[int] = set()
    for label, representative in representative_by_cluster.items():
        candidates = [
            int(index)
            for index in range(len(passed_array))
            if labels[index] == label
            and index != representative
            and residual_passed[index]
        ]
        if candidates:
            residual_alternates.add(
                min(candidates, key=lambda index: (-residual_scores[index], index))
            )

    representatives.sort()
    path_alternate_list = sorted(path_alternates)
    residual_alternate_list = sorted(residual_alternates)
    alternate_list = sorted(path_alternates | residual_alternates)
    return {
        "representatives": representatives,
        "path_alternates": path_alternate_list,
        "residual_alternates": residual_alternate_list,
        "alternates": alternate_list,
        "selected_indices": sorted({*representatives, *alternate_list}),
    }


def _tree_gain_evidence(
    block_tree_gains: np.ndarray,
    shadow_block_tree_gains: np.ndarray,
    *,
    min_block_fraction: float,
    shadow_quantile: float,
) -> dict[str, Any]:
    gains = np.maximum(_block_matrix(block_tree_gains, "block_tree_gains"), 0.0)
    shadows = np.maximum(
        _block_matrix(
            shadow_block_tree_gains,
            "shadow_block_tree_gains",
            require_finite=True,
        ),
        0.0,
    )
    if gains.shape[0] != shadows.shape[0]:
        raise ValueError("feature and shadow tree gains must use the same blocks")
    if not 0.0 <= min_block_fraction <= 1.0:
        raise ValueError("min_tree_block_fraction must be between zero and one")

    block_floors = np.array(
        [_quantile(block, shadow_quantile) for block in shadows],
        dtype=np.float64,
    )
    score = np.median(gains, axis=0)
    block_fraction = np.mean(gains > block_floors[:, None], axis=0)
    shadow_scores = np.median(shadows, axis=0)
    shadow_floor = _quantile(shadow_scores, shadow_quantile)
    passed = (score > shadow_floor) & (
        block_fraction >= min_block_fraction
    )
    return {
        "score": score,
        "block_fraction": block_fraction,
        "shadow_scores": shadow_scores,
        "shadow_floor": shadow_floor,
        "block_floors": block_floors,
        "passed": passed,
    }


def _cluster_ranked_accepted_strength(
    scores: np.ndarray,
    passed: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """Return gate-masked dense percentile ranks within each redundancy cluster."""
    score_array = np.asarray(scores, dtype=np.float64)
    passed_array = np.asarray(passed, dtype=bool)
    labels = np.asarray(cluster_labels)
    if (
        score_array.shape != passed_array.shape
        or labels.shape != passed_array.shape
    ):
        raise ValueError("scores, passed, and cluster_labels must have the same shape")

    strength = np.zeros_like(score_array)
    accepted_indices = np.flatnonzero(passed_array)
    if len(accepted_indices) == 0:
        return strength

    for label in dict.fromkeys(labels[accepted_indices].tolist()):
        cluster_indices = accepted_indices[labels[accepted_indices] == label]
        cluster_scores = np.maximum(score_array[cluster_indices], 0.0)
        unique_scores = np.unique(cluster_scores)
        dense_ranks = np.searchsorted(unique_scores, cluster_scores) + 1
        strength[cluster_indices] = dense_ranks / len(unique_scores)
    return strength


def _combine_cluster_strengths(
    linear_strength: np.ndarray,
    tree_strength: np.ndarray,
    path_strength: np.ndarray,
    stable_passed: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """Prioritize stable marginal/tree evidence over path-only evidence."""
    linear_array = np.asarray(linear_strength, dtype=np.float64)
    tree_array = np.asarray(tree_strength, dtype=np.float64)
    path_array = np.asarray(path_strength, dtype=np.float64)
    stable_array = np.asarray(stable_passed, dtype=bool)
    labels = np.asarray(cluster_labels)
    expected_shape = stable_array.shape
    if any(
        values.shape != expected_shape
        for values in (linear_array, tree_array, path_array, labels)
    ):
        raise ValueError("strengths, stable_passed, and cluster_labels must align")

    stable_strength = np.maximum(linear_array, tree_array)
    scores = np.zeros_like(stable_strength)
    for label in dict.fromkeys(labels.tolist()):
        cluster_mask = labels == label
        stable_mask = cluster_mask & stable_array
        if np.any(stable_mask):
            scores[cluster_mask] = path_array[cluster_mask] * 1e-6
            scores[stable_mask] = (
                2.0
                + stable_strength[stable_mask]
                + path_array[stable_mask] * 1e-6
            )
        else:
            scores[cluster_mask] = path_array[cluster_mask]
    return scores


def select_task_features(
    *,
    block_correlations: np.ndarray,
    shadow_block_correlations: np.ndarray,
    cluster_labels: np.ndarray,
    block_tree_gains: np.ndarray | None = None,
    shadow_block_tree_gains: np.ndarray | None = None,
    paths_by_block: Sequence[Sequence[Sequence[int]]] = (),
    residual_alternate_passed: np.ndarray | None = None,
    residual_alternate_scores: np.ndarray | None = None,
    min_direction_consistency: float = 0.75,
    min_tree_block_fraction: float = 0.75,
    shadow_quantile: float = 0.95,
    min_path_blocks: int = 2,
    min_path_features: int = 3,
    max_path_features: int = 6,
) -> dict[str, Any]:
    """Select task features without imposing a fixed final count."""
    linear = linear_evidence(
        block_correlations,
        shadow_block_correlations,
        min_direction_consistency=min_direction_consistency,
        shadow_quantile=shadow_quantile,
    )
    n_features = len(linear["score"])
    labels = np.asarray(cluster_labels)
    if labels.shape != (n_features,):
        raise ValueError("cluster_labels must match the feature count")

    if block_tree_gains is None and shadow_block_tree_gains is None:
        tree = {
            "score": np.zeros(n_features),
            "block_fraction": np.zeros(n_features),
            "shadow_floor": 0.0,
            "passed": np.zeros(n_features, dtype=bool),
        }
    elif block_tree_gains is None or shadow_block_tree_gains is None:
        raise ValueError("feature and shadow tree gains must be supplied together")
    else:
        tree = _tree_gain_evidence(
            block_tree_gains,
            shadow_block_tree_gains,
            min_block_fraction=min_tree_block_fraction,
            shadow_quantile=shadow_quantile,
        )
        if len(tree["score"]) != n_features:
            raise ValueError("tree gains must match the feature count")

    hyperedges = aggregate_path_support(
        paths_by_block,
        min_blocks=min_path_blocks,
        min_features=min_path_features,
        max_features=max_path_features,
    )
    path_support = np.zeros(n_features, dtype=np.int64)
    for edge in hyperedges:
        for feature in edge["features"]:
            if feature < 0 or feature >= n_features:
                raise ValueError("path feature index is outside the feature count")
            path_support[feature] = max(
                path_support[feature], int(edge["support"])
            )
    path_passed = path_support > 0

    passed = linear["passed"] | tree["passed"] | path_passed
    linear_strength = _cluster_ranked_accepted_strength(
        linear["score"], linear["passed"], labels
    )
    tree_strength = _cluster_ranked_accepted_strength(
        tree["score"], tree["passed"], labels
    )
    path_strength = _cluster_ranked_accepted_strength(
        path_support.astype(np.float64),
        path_passed,
        labels,
    )
    stable_passed = linear["passed"] | tree["passed"]
    scores = _combine_cluster_strengths(
        linear_strength,
        tree_strength,
        path_strength,
        stable_passed,
        labels,
    )
    cluster_selection = select_from_clusters(
        passed,
        labels,
        scores,
        hyperedges,
        residual_alternate_passed=residual_alternate_passed,
        residual_alternate_scores=residual_alternate_scores,
    )

    if residual_alternate_passed is None:
        residual_passed = np.zeros(n_features, dtype=bool)
        residual_scores = np.zeros(n_features, dtype=np.float64)
    else:
        residual_passed = np.asarray(residual_alternate_passed, dtype=bool)
        residual_scores = np.asarray(residual_alternate_scores, dtype=np.float64)

    evidence: list[dict[str, Any]] = []
    reason_by_feature: dict[int, list[str]] = {}
    for feature in range(n_features):
        reasons: list[str] = []
        if linear["passed"][feature]:
            reasons.append("linear")
        if tree["passed"][feature]:
            reasons.append("tree_gain")
        if path_passed[feature]:
            reasons.append("path")
        if feature in cluster_selection["residual_alternates"]:
            reasons.append("residual_increment")
        reason_by_feature[feature] = reasons
        evidence.append(
            {
                "feature": feature,
                "linear_score": float(linear["score"][feature]),
                "linear_direction_consistency": float(
                    linear["direction_consistency"][feature]
                ),
                "linear_passed": bool(linear["passed"][feature]),
                "tree_gain": float(tree["score"][feature]),
                "tree_block_fraction": float(
                    tree["block_fraction"][feature]
                ),
                "tree_passed": bool(tree["passed"][feature]),
                "path_support": int(path_support[feature]),
                "path_passed": bool(path_passed[feature]),
                "residual_alternate_score": float(residual_scores[feature]),
                "residual_alternate_passed": bool(residual_passed[feature]),
                "passed": bool(passed[feature]),
                "selection_score": float(scores[feature]),
            }
        )

    selected = cluster_selection["selected_indices"]
    return {
        "selected_indices": selected,
        "representatives": cluster_selection["representatives"],
        "path_alternates": cluster_selection["path_alternates"],
        "residual_alternates": cluster_selection["residual_alternates"],
        "alternates": cluster_selection["alternates"],
        "evidence": evidence,
        "reasons": {
            str(feature): reason_by_feature[feature] for feature in selected
        },
        "path_hyperedges": hyperedges,
        "thresholds": {
            "linear_shadow_floor": float(linear["shadow_floor"]),
            "tree_shadow_floor": float(tree["shadow_floor"]),
        },
    }
