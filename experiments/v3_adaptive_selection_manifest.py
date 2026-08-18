"""Build task-specific adaptive-selection manifests from the latest training window.

History selection is deliberately absent. Task 2B will replace the explicit pending
entry after causal, lag-aligned history evidence is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "strategies" / "v1_ridge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.lgbm_xs import load_rows
from experiments.v3_adaptive_selection import (
    extract_tree_paths,
    select_task_features,
)
from experiments.v3_feature_structure import (
    build_task_views,
    contiguous_time_blocks,
    evenly_spaced_rows,
    feature_quality_by_blocks,
    stable_redundancy,
)
from src.io import FEATURE_COLUMNS
from train import robust_transform_fit

TREE_ROUNDS = 80
TREE_MAX_DEPTH = 4
TREE_NUM_LEAVES = 15
TREE_SEEDS = (2026, 2027, 2028)
TREE_ROW_CAP = 150_000
N_SHADOWS = 32


def chronological_inner_splits(
    time_ids: np.ndarray,
    n_blocks: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-train/next-block row splits without splitting a time_id."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least two")
    blocks = contiguous_time_blocks(time_ids, n_blocks)
    return [
        (np.concatenate(blocks[:valid_block]), blocks[valid_block].copy())
        for valid_block in range(1, len(blocks))
    ]


def make_shadow_columns(
    features: np.ndarray,
    n_shadows: int = N_SHADOWS,
) -> np.ndarray:
    """Create deterministic non-identity row shifts of cycling feature columns."""
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if len(values) < 2:
        raise ValueError("features must contain at least two rows")
    if n_shadows <= 0:
        raise ValueError("n_shadows must be positive")

    shadows = np.empty(
        (len(values), n_shadows), dtype=np.result_type(values.dtype, np.float64)
    )
    for shadow_index in range(n_shadows):
        source_index = shadow_index % values.shape[1]
        if len(values) == 1:
            shadows[:, shadow_index] = (
                values[:, source_index]
                + (shadow_index + 1) * np.finfo(np.float64).eps
            )
            continue
        cycle = shadow_index // values.shape[1]
        shift = 1 + (shadow_index + cycle) % (len(values) - 1)
        shadows[:, shadow_index] = np.roll(
            values[:, source_index], shift=shift
        )
    return shadows


def _validate_task_arrays(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    expected = (len(x),)
    if y.shape != expected or w.shape != expected or ids.shape != expected:
        raise ValueError("target, weight, and time_ids must match feature rows")
    if len(x) == 0:
        raise ValueError("task arrays must not be empty")
    if np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be sorted")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and target must be finite")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weight must be finite and nonnegative")
    if float(w.sum()) <= 0.0:
        raise ValueError("weight must have positive mass")
    return x, y, w, ids


def _train_lightgbm_booster(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    params: dict[str, object],
    num_boost_round: int,
) -> Any:
    import lightgbm as lgb

    dataset = lgb.Dataset(
        features,
        label=target,
        weight=weight,
        free_raw_data=False,
    )
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def _weighted_mean_abs(
    values: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    total_weight = float(weight.sum())
    return np.sum(np.abs(values) * weight[:, None], axis=0) / total_weight


def _original_paths(
    model_dump: Any,
    n_features: int,
    n_total_features: int,
) -> set[tuple[int, ...]]:
    paths = extract_tree_paths(
        model_dump,
        min_features=3,
        max_features=max(6, n_total_features),
    )
    original_paths: set[tuple[int, ...]] = set()
    for path in paths:
        if any(feature < 0 or feature >= n_features for feature in path):
            continue
        original = tuple(dict.fromkeys(path))
        if 3 <= len(original) <= 6:
            original_paths.add(original)
    return original_paths


def validation_tree_evidence(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
    *,
    n_blocks: int = 4,
    n_shadows: int = N_SHADOWS,
    num_boost_round: int = TREE_ROUNDS,
    max_depth: int = TREE_MAX_DEPTH,
    num_leaves: int = TREE_NUM_LEAVES,
    seeds: Sequence[int] = TREE_SEEDS,
    row_cap: int = TREE_ROW_CAP,
    num_threads: int = 1,
) -> dict[str, Any]:
    """Measure feature contributions strictly on next-block validation rows."""
    x, y, w, ids = _validate_task_arrays(features, target, weight, time_ids)
    if num_boost_round <= 0:
        raise ValueError("num_boost_round must be positive")
    if max_depth <= 0 or num_leaves <= 1:
        raise ValueError("invalid shallow-tree shape")
    if row_cap < 2:
        raise ValueError("row_cap must be at least two")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must not be empty")
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")

    n_features = x.shape[1]
    feature_evidence: list[np.ndarray] = []
    shadow_evidence: list[np.ndarray] = []
    paths_by_block: list[list[tuple[int, ...]]] = []
    splits = chronological_inner_splits(ids, n_blocks=n_blocks)

    for train_rows, valid_rows in splits:
        if len(train_rows) > row_cap:
            train_rows = train_rows[
                evenly_spaced_rows(len(train_rows), row_cap)
            ]
        train_features = x[train_rows]
        valid_features = x[valid_rows]
        train_shadows = make_shadow_columns(train_features, n_shadows)
        if len(valid_features) >= 2:
            valid_shadows = make_shadow_columns(valid_features, n_shadows)
        else:
            # A one-row validation block cannot be permuted without leakage.
            valid_shadows = np.zeros((len(valid_features), n_shadows), dtype=np.float64)
        train_design = np.column_stack(
            [train_features, train_shadows]
        )
        valid_design = np.column_stack(
            [valid_features, valid_shadows]
        )
        n_total_features = train_design.shape[1]
        seed_evidence: list[np.ndarray] = []
        block_paths: set[tuple[int, ...]] = set()

        for seed in seed_values:
            params: dict[str, object] = {
                "objective": "regression",
                "metric": "None",
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "deterministic": True,
                "force_col_wise": True,
                "verbosity": -1,
                "num_threads": num_threads,
                "seed": seed,
                "feature_fraction_seed": seed,
                "bagging_seed": seed,
                "data_random_seed": seed,
            }
            booster = _train_lightgbm_booster(
                train_design,
                y[train_rows],
                w[train_rows],
                params=params,
                num_boost_round=num_boost_round,
            )
            contributions = np.asarray(
                booster.predict(valid_design, pred_contrib=True),
                dtype=np.float64,
            )
            expected_shape = (len(valid_rows), n_total_features + 1)
            if contributions.shape != expected_shape:
                raise ValueError(
                    "LightGBM pred_contrib returned an unexpected shape"
                )
            seed_evidence.append(
                _weighted_mean_abs(contributions[:, :-1], w[valid_rows])
            )
            block_paths.update(
                _original_paths(
                    booster.dump_model(), n_features, n_total_features
                )
            )

        block_evidence = np.mean(seed_evidence, axis=0)
        feature_evidence.append(block_evidence[:n_features])
        shadow_evidence.append(block_evidence[n_features:])
        paths_by_block.append(sorted(block_paths))

    return {
        "block_feature_evidence": np.vstack(feature_evidence),
        "block_shadow_evidence": np.vstack(shadow_evidence),
        "paths_by_block": paths_by_block,
        "protocol": {
            "method": "mean_absolute_oos_pred_contrib",
            "inner_splits": len(splits),
            "n_blocks": n_blocks,
            "num_boost_round": num_boost_round,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "seeds": list(seed_values),
            "row_cap": row_cap,
            "n_shadows": n_shadows,
        },
    }


def selection_task_views(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the exact weighted Ridge, XS, and unweighted market views."""
    x, y, w, ids = _validate_task_arrays(features, target, weight, time_ids)
    views = build_task_views(x, y, w, ids)
    market_target = np.add.reduceat(y, views.starts) / views.counts
    unit_rows = np.ones(len(y), dtype=np.float64)
    unit_times = np.ones(len(views.unique_time_ids), dtype=np.float64)
    return {
        "ridge": {
            "features": views.raw_features,
            "target": views.full_target,
            "weight": w,
            "time_ids": ids,
        },
        "xs": {
            "features": views.cross_features,
            "target": views.cross_target,
            "weight": unit_rows,
            "time_ids": ids,
        },
        "market": {
            "features": views.market_features,
            "target": market_target,
            "weight": unit_times,
            "time_ids": views.unique_time_ids,
        },
    }


def _task_manifest(
    selection: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    indices = [int(index) for index in selection.get("selected_indices", [])]
    if len(indices) != len(set(indices)):
        raise ValueError("selected feature indices must be unique")
    if any(index < 0 or index >= len(feature_names) for index in indices):
        raise ValueError("selected feature index is out of range")
    return {
        "selected_indices": indices,
        "selected_names": [str(feature_names[index]) for index in indices],
        "selected_count": len(indices),
        "representatives": list(selection.get("representatives", [])),
        "alternates": list(selection.get("alternates", [])),
        "evidence": list(selection.get("evidence", [])),
        "reasons": dict(selection.get("reasons", {})),
        "path_hyperedges": list(selection.get("path_hyperedges", [])),
        "thresholds": dict(selection.get("thresholds", {})),
    }


def assemble_manifest(
    *,
    ridge_selection: Mapping[str, Any],
    xs_selection: Mapping[str, Any],
    market_selection: Mapping[str, Any],
    feature_names: Sequence[str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble separate task selections while leaving history visibly pending."""
    names = [str(name) for name in feature_names]
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    return {
        "schema_version": 1,
        "stage": "prehistory",
        "feature_names": names,
        "ridge": _task_manifest(ridge_selection, names),
        "xs": _task_manifest(xs_selection, names),
        "market": _task_manifest(market_selection, names),
        "history": {
            "status": "pending_task_2b",
            "selected_indices": None,
            "selected_names": None,
            "selected_count": None,
        },
        "protocol": dict(protocol),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _select_task(
    task: Mapping[str, np.ndarray],
    *,
    n_blocks: int,
    n_shadows: int,
    cluster_threshold: float,
    redundancy_rows_per_block: int | None,
    tree_rounds: int,
    tree_row_cap: int,
    num_threads: int,
) -> dict[str, Any]:
    features = task["features"]
    target = task["target"]
    weight = task["weight"]
    time_ids = task["time_ids"]
    shadows = make_shadow_columns(features, n_shadows=n_shadows)
    quality = feature_quality_by_blocks(
        np.column_stack([features, shadows]),
        target,
        weight,
        time_ids,
        n_blocks=n_blocks,
    )
    n_features = features.shape[1]
    correlations = np.nan_to_num(
        quality["block_correlation"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    redundancy = stable_redundancy(
        features,
        time_ids,
        n_blocks=n_blocks,
        threshold=cluster_threshold,
        max_rows_per_block=redundancy_rows_per_block,
    )
    tree = validation_tree_evidence(
        features,
        target,
        weight,
        time_ids,
        n_blocks=n_blocks,
        n_shadows=n_shadows,
        num_boost_round=tree_rounds,
        row_cap=tree_row_cap,
        num_threads=num_threads,
    )
    return select_task_features(
        block_correlations=correlations[:, :n_features],
        shadow_block_correlations=correlations[:, n_features:],
        cluster_labels=redundancy.labels,
        block_tree_gains=tree["block_feature_evidence"],
        shadow_block_tree_gains=tree["block_shadow_evidence"],
        paths_by_block=tree["paths_by_block"],
    )


def _latest_window_mask(
    time_ids: np.ndarray,
    train_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_window <= 0:
        raise ValueError("train_window must be positive")
    unique_ids = time_ids[np.r_[True, time_ids[1:] != time_ids[:-1]]]
    if train_window > len(unique_ids):
        raise ValueError("train_window exceeds sampled time coverage")
    selected_ids = unique_ids[-train_window:]
    return time_ids >= selected_ids[0], selected_ids


def run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    time_ids = np.asarray(data["time_id"], dtype=np.int64)
    if np.any(np.diff(time_ids) < 0):
        raise ValueError("loaded time_ids must be sorted")

    train_window = args.train_window
    if args.smoke_time_ids is not None:
        if args.smoke_time_ids <= 0:
            raise ValueError("smoke_time_ids must be positive")
        train_window = min(train_window, args.smoke_time_ids)
    mask, selected_ids = _latest_window_mask(time_ids, train_window)
    features, _ = robust_transform_fit(data["features"][mask].copy())
    target = data["target"][mask]
    weight = data["weight"][mask]
    selected_time_ids = time_ids[mask]
    views = selection_task_views(features, target, weight, selected_time_ids)

    tree_rounds = args.smoke_tree_rounds or args.tree_rounds
    tree_row_cap = args.smoke_row_cap or args.tree_row_cap
    selections = {
        task_name: _select_task(
            task,
            n_blocks=args.n_blocks,
            n_shadows=N_SHADOWS,
            cluster_threshold=args.cluster_threshold,
            redundancy_rows_per_block=args.redundancy_rows_per_block,
            tree_rounds=tree_rounds,
            tree_row_cap=tree_row_cap,
            num_threads=args.num_threads,
        )
        for task_name, task in views.items()
    }
    protocol = {
        "training_window": {
            "requested_time_ids": args.train_window,
            "effective_time_ids": len(selected_ids),
            "time_start": int(selected_ids[0]),
            "time_end": int(selected_ids[-1]),
            "rows": int(mask.sum()),
            "latest_window": True,
        },
        "robust_transform": "fit_on_selected_training_window",
        "task_views": {
            "ridge": "full_target_with_competition_weights",
            "xs": "weighted_mean_cross_target_with_unit_evidence_weights",
            "market": "unweighted_time_feature_means_vs_unweighted_time_target_mean",
        },
        "quality_blocks": args.n_blocks,
        "cluster_threshold": args.cluster_threshold,
        "redundancy_rows_per_block": args.redundancy_rows_per_block,
        "tree_evidence": {
            "method": "mean_absolute_oos_pred_contrib",
            "inner_splits": args.n_blocks - 1,
            "num_boost_round": tree_rounds,
            "default_num_boost_round": TREE_ROUNDS,
            "max_depth": TREE_MAX_DEPTH,
            "num_leaves": TREE_NUM_LEAVES,
            "seeds": list(TREE_SEEDS),
            "row_cap": tree_row_cap,
            "default_row_cap": TREE_ROW_CAP,
            "n_shadows": N_SHADOWS,
        },
        "smoke": {
            "time_ids": args.smoke_time_ids,
            "tree_rounds": args.smoke_tree_rounds,
            "row_cap": args.smoke_row_cap,
        },
    }
    return assemble_manifest(
        ridge_selection=selections["ridge"],
        xs_selection=selections["xs"],
        market_selection=selections["market"],
        feature_names=FEATURE_COLUMNS,
        protocol=protocol,
    )


def render_markdown(manifest: Mapping[str, Any]) -> str:
    protocol = manifest["protocol"]
    window = protocol["training_window"]
    tree = protocol["tree_evidence"]
    lines = [
        "# V3 Adaptive Selection Prehistory Manifest",
        "",
        "## Selection",
        "",
    ]
    for task_name in ("ridge", "xs", "market"):
        task = manifest[task_name]
        lines.append(
            f"- `{task_name}`: {task['selected_count']} selected; "
            f"{len(task['path_hyperedges'])} supported path hyperedges"
        )
    lines.extend(
        [
            "- `history`: pending Task 2B; no history selection is present",
            "",
            "## Protocol",
            "",
            f"- latest training window: `{window['time_start']}` to `{window['time_end']}` "
            f"({window['effective_time_ids']} time_ids, {window['rows']} rows)",
            f"- chronological inner validations: `{tree['inner_splits']}`",
            f"- shallow trees: `{tree['num_boost_round']}` rounds, depth "
            f"`{tree['max_depth']}`, leaves `{tree['num_leaves']}`",
            f"- seeds: `{tree['seeds']}`; row cap: `{tree['row_cap']}`; "
            f"shadows: `{tree['n_shadows']}`",
            "- tree evidence: mean absolute out-of-sample validation contributions",
            "- market view: unweighted feature means versus unweighted target mean",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest_bundle(
    manifest: Mapping[str, Any],
    output_dir: Path,
    label: str,
    *,
    force: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_prehistory"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    if not force and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError(
            f"manifest exists: {json_path} / {markdown_path}; pass --force"
        )
    safe_manifest = _json_safe(manifest)
    json_path.write_text(
        json.dumps(
            safe_manifest,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(safe_manifest), encoding="utf-8"
    )
    return {"json": json_path, "markdown": markdown_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V3 adaptive-selection prehistory manifest."
    )
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "experiments")
    )
    parser.add_argument("--label", default="v3_adaptive_selection")
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument(
        "--sampling",
        choices=["periodic", "phase_balanced"],
        default="phase_balanced",
    )
    parser.add_argument("--train-window", type=int, default=78_960)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--cluster-threshold", type=float, default=0.15)
    parser.add_argument(
        "--redundancy-rows-per-block", type=int, default=100_000
    )
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--smoke-time-ids", type=int, default=None)
    parser.add_argument("--tree-rounds", type=int, default=TREE_ROUNDS)
    parser.add_argument("--tree-row-cap", type=int, default=TREE_ROW_CAP)
    parser.add_argument("--history-row-cap", type=int, default=TREE_ROW_CAP)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tree-rounds", type=int, default=None)
    parser.add_argument("--smoke-row-cap", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_manifest(args)
    paths = write_manifest_bundle(
        manifest, Path(args.output_dir), args.label, force=args.force
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
