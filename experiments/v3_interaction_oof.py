"""Paired strict OOF evaluation for additive interaction columns.

The public helpers in this module keep acceptance and early stopping machine-readable.
The experiment runner is added around these primitives so a failed screen can never
fall through to final training or submission generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
FROZEN_SCREEN = {
    "n_folds": 5,
    "train_window": 78_960,
    "embargo": 6,
    "sample_modulo": 5,
    "sampling": "phase_balanced",
    "n_seeds": 1,
    "num_iteration": 160,
    "market_lambda": 0.7,
    "blend_weight": 1.17,
}
FEATURE_COUNT = 200
HISTORY_COUNT = 40
RIDGE_ALPHA = 2_000_000.0
REFERENCE_TRAIN_WINDOW = 78_960
XS_SPEC = {
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "lambda_l2": 1.0,
}
MARKET_SPEC = {
    "num_leaves": 15,
    "learning_rate": 0.02,
    "feature_fraction": 0.4,
    "lambda_l2": 30.0,
}
MIN_DATA_FRAC = 12000 / 3_500_000
MARKET_MIN_DATA_SCALE = 25.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument(
        "--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments")
    )
    parser.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "outputs" / "cache")
    )
    parser.add_argument("--label", default="v3_interactions_screen_1s160_07_117")
    parser.add_argument("--n-folds", type=int, default=FROZEN_SCREEN["n_folds"])
    parser.add_argument("--train-window", type=int, default=FROZEN_SCREEN["train_window"])
    parser.add_argument("--embargo", type=int, default=FROZEN_SCREEN["embargo"])
    parser.add_argument(
        "--sample-modulo", type=int, default=FROZEN_SCREEN["sample_modulo"]
    )
    parser.add_argument(
        "--sampling",
        choices=["periodic", "phase_balanced"],
        default=FROZEN_SCREEN["sampling"],
    )
    parser.add_argument("--n-seeds", type=int, default=FROZEN_SCREEN["n_seeds"])
    parser.add_argument(
        "--num-iteration", type=int, default=FROZEN_SCREEN["num_iteration"]
    )
    parser.add_argument(
        "--market-lambda", type=float, default=FROZEN_SCREEN["market_lambda"]
    )
    parser.add_argument(
        "--blend-weight", type=float, default=FROZEN_SCREEN["blend_weight"]
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--miner-row-cap", type=int, default=150_000)
    parser.add_argument("--max-source-cells", type=int, default=130_000_000)
    parser.add_argument("--max-interaction-cells", type=int, default=100_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def spill_interaction_features(
    data: dict[str, np.ndarray],
    path: str | Path,
    *,
    chunk_rows: int = 100_000,
) -> np.memmap:
    """Move loaded features to a read-only disk mapping for the OOF screen."""
    from experiments.v3_production_oof import spill_feature_matrix

    loaded_features = data.pop("features")
    mapped = spill_feature_matrix(loaded_features, path, chunk_rows=chunk_rows)
    del loaded_features
    gc.collect()
    return mapped


def validate_frozen_screen(args: argparse.Namespace) -> None:
    for name, expected in FROZEN_SCREEN.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"{name}={actual!r} changes the frozen screen; expected {expected!r}"
            )
    if args.num_threads <= 0 or args.history_window != 5:
        raise ValueError("num_threads must be positive and history_window must remain 5")
    if args.miner_row_cap <= 0 or args.max_source_cells <= 0:
        raise ValueError("interaction memory budgets must be positive")


def positive_fold_gate_impossible(
    deltas: list[float] | np.ndarray,
    *,
    total_folds: int = 5,
    required_positive: int = 4,
) -> bool:
    """Return whether remaining folds cannot reach the required positive count."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) > total_folds:
        raise ValueError("fold deltas must be a one-dimensional partial fold sequence")
    if total_folds <= 0 or not 0 < required_positive <= total_folds:
        raise ValueError("invalid positive-fold gate")
    if not np.all(np.isfinite(values)):
        raise ValueError("fold deltas must be finite")
    positives = int(np.sum(values > 0.0))
    remaining = total_folds - len(values)
    return positives + remaining < required_positive


def interaction_gate(
    deltas: np.ndarray,
    delta_a: float,
    delta_b: float,
    *,
    required_positive: int = 4,
) -> dict[str, object]:
    """Apply the frozen five-fold interaction acceptance criteria."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("interaction gate requires five finite paired fold deltas")
    if not np.isfinite(delta_a) or not np.isfinite(delta_b):
        raise ValueError("delta_a and delta_b must be finite")
    positive_folds = int(np.sum(values > 0.0))
    mean_delta = float(np.mean(values))
    drop_best_mean = float(np.mean(np.delete(values, int(np.argmax(values)))))
    checks = {
        "positive_mean": mean_delta > 0.0,
        "four_of_five_positive": positive_folds >= required_positive,
        "positive_drop_best": drop_best_mean > 0.0,
        "target_alignment": 2.0 * float(delta_a) > float(delta_b),
    }
    return {
        "passed": all(checks.values()),
        "mean_delta": mean_delta,
        "positive_folds": positive_folds,
        "drop_best_mean": drop_best_mean,
        "delta_a": float(delta_a),
        "delta_b": float(delta_b),
        "checks": checks,
    }


def append_interactions_before_asset(
    base_design: np.ndarray,
    interactions: np.ndarray,
) -> np.ndarray:
    """Append derived columns while preserving the final categorical asset column."""
    base = np.asarray(base_design, dtype=np.float32)
    added = np.asarray(interactions, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] < 1:
        raise ValueError("base design must be 2D with a final asset column")
    if added.ndim != 2 or len(added) != len(base):
        raise ValueError("interaction columns must be row-aligned 2D")
    if not np.all(np.isfinite(added)):
        raise ValueError("interaction columns must be finite")
    return np.ascontiguousarray(
        np.column_stack([base[:, :-1], added, base[:, -1]]),
        dtype=np.float32,
    )


def _group_mean(values: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if prediction.shape != ids.shape or prediction.ndim != 1:
        raise ValueError("group mean inputs must be aligned one-dimensional arrays")
    if len(ids) == 0 or np.any(np.diff(ids) < 0):
        raise ValueError("time_ids must be nonempty and nondecreasing")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    means = np.add.reduceat(prediction, starts) / counts
    return np.repeat(means, counts)


def compose_hybrid_raw(
    ridge_prediction: np.ndarray,
    xs_prediction: np.ndarray,
    market_prediction: np.ndarray,
    time_ids: np.ndarray,
    *,
    market_lambda: float,
    blend_weight: float,
) -> np.ndarray:
    """Compose unscaled, unclipped V3 output under fixed market/cross weights."""
    ridge = np.asarray(ridge_prediction, dtype=np.float64)
    xs = np.asarray(xs_prediction, dtype=np.float64)
    market_rows = np.asarray(market_prediction, dtype=np.float64)
    ids = np.asarray(time_ids, dtype=np.int64)
    if not (ridge.shape == xs.shape == market_rows.shape == ids.shape):
        raise ValueError("hybrid component predictions must have identical shapes")
    if not np.isfinite(market_lambda) or not np.isfinite(blend_weight):
        raise ValueError("hybrid weights must be finite")
    market_ridge = _group_mean(ridge, ids)
    e_ridge = ridge - market_ridge
    e_lgbm = xs - _group_mean(xs, ids)
    market_lgbm = _group_mean(market_rows, ids)
    market = (1.0 - market_lambda) * market_ridge + market_lambda * market_lgbm
    cross = (1.0 - blend_weight) * e_ridge + blend_weight * e_lgbm
    return market + cross


def paired_component_predictions(
    base_train: np.ndarray,
    base_valid: np.ndarray,
    interaction_train: np.ndarray,
    interaction_valid: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    asset_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit paired component predictions, reusing A when no columns are added."""
    train_added = np.asarray(interaction_train, dtype=np.float32)
    valid_added = np.asarray(interaction_valid, dtype=np.float32)
    if train_added.ndim != 2 or valid_added.ndim != 2:
        raise ValueError("paired interaction matrices must be two-dimensional")
    if len(train_added) != len(base_train) or len(valid_added) != len(base_valid):
        raise ValueError("paired interaction matrices must be row-aligned")
    if train_added.shape[1] != valid_added.shape[1]:
        raise ValueError("paired interaction widths must match")
    if train_added.shape[1] == 0:
        baseline = np.asarray(fit_predict(base_train, base_valid), dtype=np.float64)
        return baseline, baseline.copy()
    if asset_last:
        augmented_train = append_interactions_before_asset(base_train, train_added)
        augmented_valid = append_interactions_before_asset(base_valid, valid_added)
    else:
        augmented_train = np.ascontiguousarray(
            np.column_stack([base_train, train_added]), dtype=np.float32
        )
        augmented_valid = np.ascontiguousarray(
            np.column_stack([base_valid, valid_added]), dtype=np.float32
        )
    baseline = np.asarray(fit_predict(base_train, base_valid), dtype=np.float64)
    interaction = np.asarray(
        fit_predict(augmented_train, augmented_valid), dtype=np.float64
    )
    return baseline, interaction


def run_paired_fold_sequence(
    total_folds: int,
    run_fold: Callable[[int], dict[str, float]],
    *,
    required_positive: int = 4,
) -> dict[str, object]:
    """Run paired folds in order and stop once the positive-fold gate is impossible."""
    if total_folds <= 0 or not 0 < required_positive <= total_folds:
        raise ValueError("invalid paired fold configuration")
    folds: list[dict[str, float]] = []
    stopped_early = False
    stop_reason: str | None = None
    for fold_index in range(total_folds):
        payload = dict(run_fold(fold_index))
        required = {"peak_delta", "delta_a", "delta_b"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"fold payload is missing metrics: {missing}")
        if any(not np.isfinite(float(payload[name])) for name in required):
            raise ValueError("fold metrics must be finite")
        payload["fold"] = fold_index
        folds.append(payload)
        deltas = [float(item["peak_delta"]) for item in folds]
        if positive_fold_gate_impossible(
            deltas,
            total_folds=total_folds,
            required_positive=required_positive,
        ):
            stopped_early = True
            stop_reason = "four_of_five_positive_is_impossible"
            break
    return {
        "folds": folds,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "completed_folds": len(folds),
        "total_folds": total_folds,
        "required_positive": required_positive,
    }


def _metric_payload(
    target: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    from src.metric import scale_invariant_score

    metric = scale_invariant_score(target, prediction, weight)
    return {name: float(metric[name]) for name in ("peak", "A", "B", "optimal_scale")}


def _fit_predict_lgbm(
    design_train: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray | None,
    design_valid: np.ndarray,
    *,
    spec: Mapping[str, float],
    min_data_scale: float,
    n_seeds: int,
    seed: int,
    num_iteration: int,
    num_threads: int,
) -> np.ndarray:
    import lightgbm as lgb

    if design_train.ndim != 2 or design_valid.ndim != 2:
        raise ValueError("LightGBM designs must be two-dimensional")
    if design_train.shape[1] != design_valid.shape[1] or design_train.shape[1] < 2:
        raise ValueError("LightGBM train/valid widths must match and include asset_id")
    min_data = max(
        20,
        int(round(MIN_DATA_FRAC * len(design_train) * min_data_scale)),
    )
    prediction = np.zeros(len(design_valid), dtype=np.float64)
    categorical = design_train.shape[1] - 1
    for seed_offset in range(n_seeds):
        model_seed = seed + seed_offset
        params: dict[str, object] = {
            **spec,
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "num_threads": num_threads,
            "min_data_in_leaf": min_data,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "deterministic": True,
            "force_row_wise": True,
            "feature_pre_filter": False,
            "seed": model_seed,
            "bagging_seed": model_seed + 1000,
            "feature_fraction_seed": model_seed + 2000,
        }
        dataset = lgb.Dataset(
            design_train,
            label=label,
            weight=weight,
            params=params,
            categorical_feature=[categorical],
            free_raw_data=False,
        )
        booster = lgb.train(params, dataset, num_boost_round=num_iteration)
        prediction += booster.predict(design_valid, num_iteration=num_iteration)
        del booster, dataset
    return prediction / n_seeds


def _task_design(
    task: str,
    transformed: np.ndarray,
    time_ids: np.ndarray,
    asset_ids: np.ndarray,
    selected: np.ndarray,
    history_blocks: tuple[np.ndarray, ...],
) -> np.ndarray:
    from strategies.v3_hybrid.features import cross_sectional_deviation

    raw = transformed[:, selected].copy()
    deviation = cross_sectional_deviation(raw.copy(), time_ids)
    asset = asset_ids.astype(np.float32)
    if task == "xs":
        return np.ascontiguousarray(
            np.column_stack([deviation, *history_blocks, asset]),
            dtype=np.float32,
        )
    if task == "market":
        return np.ascontiguousarray(
            np.column_stack([raw, deviation, *history_blocks, asset]),
            dtype=np.float32,
        )
    raise ValueError(f"unknown tree task: {task}")


def _write_partial_report(
    json_path: Path,
    markdown_path: Path,
    payload: dict[str, object],
) -> None:
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Additive interaction strict OOF screen",
        "",
        f"Status: **{payload['status']}**",
        "",
        "| Fold | Baseline Peak | Interaction Peak | Delta | Ridge | XS | Market |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in payload.get("folds", []):
        counts = fold["interaction_counts"]
        lines.append(
            f"| {fold['fold']} | {fold['baseline']['peak']:.8f} | "
            f"{fold['interaction']['peak']:.8f} | {fold['peak_delta']:+.8f} | "
            f"{counts['ridge']} | {counts['xs']} | {counts['market']} |"
        )
    if payload.get("gate") is not None:
        lines.extend(["", "## Gate", "", "```json", json.dumps(
            payload["gate"], ensure_ascii=False, indent=2), "```"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest_definitions(
    manifest: Mapping[str, object], task: str
) -> list[dict[str, object]]:
    task_payload = manifest["tasks"][task]
    return list(task_payload["definitions"])


def main() -> None:
    from experiments.history_peak import build_lag_cache, fit_ridge, history_blocks, ridge_designs
    from experiments.lgbm_xs import load_rows
    from experiments.v3_interaction_features import (
        build_interaction_source_view,
        interaction_source_arrays,
        mine_task_interactions,
        resolve_quantile_thresholds,
    )
    from src.io import FEATURE_COLUMNS, train_files
    from src.validation import rolling_time_folds
    from strategies.v1_ridge.train import robust_transform_fit, select_features
    from strategies.v3_hybrid.features import (
        apply_robust_transform,
        cross_sectional_deviation,
    )
    from strategies.v3_hybrid.interactions import build_interaction_columns

    args = parse_args()
    validate_frozen_screen(args)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / f"{args.label}_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    markdown_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or markdown_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    print(
        f"loading sampled rows: modulo {args.sample_modulo}/{args.sampling}",
        flush=True,
    )
    data = load_rows(data_root, args.sample_modulo, args.sampling)
    feature_spill_path = cache_dir / f".{args.label}_features.npy"
    features = spill_interaction_features(data, feature_spill_path)
    target = data["target"].astype(np.float64, copy=False)
    weight = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    time_ids = data["time_id"].astype(np.int64, copy=False)
    asset_ids = data["asset_id"].astype(np.int64, copy=False)
    del data
    if np.any(np.diff(time_ids) < 0):
        raise AssertionError("sampled rows must be sorted by time_id")
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(
        unique_time_ids, args.n_folds, args.train_window, args.embargo
    )

    completed: list[dict[str, Any]] = []
    pooled_target: list[np.ndarray] = []
    pooled_weight: list[np.ndarray] = []
    pooled_baseline: list[np.ndarray] = []
    pooled_interaction: list[np.ndarray] = []
    payload: dict[str, object] = {
        "experiment": "v3_additive_interaction_strict_oof",
        "status": "running",
        "config": {
            **FROZEN_SCREEN,
            "num_threads": args.num_threads,
            "miner_row_cap": args.miner_row_cap,
            "max_source_cells": args.max_source_cells,
            "max_interaction_cells": args.max_interaction_cells,
            "feature_spill_path": str(feature_spill_path),
        },
        "folds": completed,
        "gate": None,
        "candidate_generated": False,
        "submission_generated": False,
    }

    def row_slice(ids: np.ndarray) -> slice:
        left = int(np.searchsorted(time_ids, ids[0], side="left"))
        right = int(np.searchsorted(time_ids, ids[-1], side="right"))
        return slice(left, right)

    def run_fold(fold_index: int) -> dict[str, float]:
        fold_started = time.perf_counter()
        train_ids, valid_ids = folds[fold_index]
        tr = row_slice(train_ids)
        va = row_slice(valid_ids)
        raw_train = features[tr]
        raw_valid = features[va]
        y_train, y_valid = target[tr], target[va]
        w_train, w_valid = weight[tr], weight[va]
        tid_train, tid_valid = time_ids[tr], time_ids[va]
        aid_train, aid_valid = asset_ids[tr], asset_ids[va]

        transformed_train, outer_stats = robust_transform_fit(raw_train.copy())
        transformed_valid = raw_valid.copy()
        apply_robust_transform(
            transformed_valid,
            outer_stats["lower"], outer_stats["upper"],
            outer_stats["center"], outer_stats["scale"],
        )
        train_market = _group_mean(y_train, tid_train)
        e_train = y_train - train_market
        ridge_selected = select_features(
            transformed_train, y_train, w_train, FEATURE_COUNT
        )
        xs_selected = select_features(
            transformed_train, e_train, np.ones_like(e_train), FEATURE_COUNT
        )
        xs_deviation = cross_sectional_deviation(
            transformed_train[:, xs_selected].copy(), tid_train
        )
        history_positions = select_features(
            xs_deviation, e_train, np.ones_like(e_train), HISTORY_COUNT
        )
        history_indices = xs_selected[np.sort(history_positions.astype(np.int64))]
        del xs_deviation

        range_start = tr.start
        range_stop = va.stop
        lag_cache = build_lag_cache(
            train_files(data_root),
            history_indices,
            args.sample_modulo,
            args.history_window,
            sampling=args.sampling,
            verbose=False,
            minimum_time_id=int(tid_train[0]),
            maximum_time_id=int(tid_valid[-1]),
        )
        expected_ids = time_ids[range_start:range_stop]
        expected_assets = asset_ids[range_start:range_stop]
        if not (
            np.array_equal(lag_cache["time_id"], expected_ids)
            and np.array_equal(lag_cache["asset_id"], expected_assets)
        ):
            raise AssertionError("fold lag cache is not aligned with sampled rows")

        def fold_history(
            global_rows: np.ndarray,
            transformed_rows: np.ndarray,
            stats: Mapping[str, np.ndarray],
        ) -> tuple[np.ndarray, ...]:
            local = global_rows - range_start
            selected_stats = [
                np.asarray(stats[name])[history_indices]
                for name in ("lower", "upper", "center", "scale")
            ]
            return history_blocks(
                lag_cache["lags"][local],
                lag_cache["count"][local],
                transformed_rows[:, history_indices],
                *selected_stats,
            )

        outer_history_train = fold_history(
            np.arange(tr.start, tr.stop), transformed_train, outer_stats
        )
        outer_history_valid = fold_history(
            np.arange(va.start, va.stop), transformed_valid, outer_stats
        )

        split_cache: list[dict[str, object]] = []

        def prepare_split(
            local_train_rows: np.ndarray,
            local_valid_rows: np.ndarray,
        ) -> dict[str, object]:
            for cached in split_cache:
                if (
                    np.array_equal(cached["train_rows"], local_train_rows)
                    and np.array_equal(cached["valid_rows"], local_valid_rows)
                ):
                    return cached
            inner_train, inner_stats = robust_transform_fit(
                raw_train[local_train_rows].copy()
            )
            inner_valid = raw_train[local_valid_rows].copy()
            apply_robust_transform(
                inner_valid,
                inner_stats["lower"], inner_stats["upper"],
                inner_stats["center"], inner_stats["scale"],
            )
            history_train = fold_history(
                tr.start + local_train_rows, inner_train, inner_stats
            )
            history_valid = fold_history(
                tr.start + local_valid_rows, inner_valid, inner_stats
            )
            cached = {
                "train_rows": local_train_rows.copy(),
                "valid_rows": local_valid_rows.copy(),
                "train": inner_train,
                "valid": inner_valid,
                "history_train": history_train,
                "history_valid": history_valid,
            }
            split_cache.clear()
            split_cache.append(cached)
            return cached

        task_labels = {"ridge": y_train, "xs": e_train, "market": y_train}
        task_selections = {
            "ridge": ridge_selected,
            "xs": xs_selected,
            "market": xs_selected,
        }
        task_results: dict[str, object] = {}
        for task in ("ridge", "xs", "market"):
            def baseline_predictor(
                inner_train_rows: np.ndarray,
                inner_valid_rows: np.ndarray,
                *,
                task_name: str = task,
            ) -> np.ndarray:
                prepared = prepare_split(inner_train_rows, inner_valid_rows)
                if task_name == "ridge":
                    d_train = ridge_designs(
                        prepared["train"], tid_train[inner_train_rows],
                        ridge_selected, None,
                    )
                    d_valid = ridge_designs(
                        prepared["valid"], tid_train[inner_valid_rows],
                        ridge_selected, None,
                    )
                    alpha = RIDGE_ALPHA * len(np.unique(tid_train[inner_train_rows])) / REFERENCE_TRAIN_WINDOW
                    model = fit_ridge(
                        d_train, y_train[inner_train_rows], w_train[inner_train_rows], alpha
                    )
                    prediction = model.predict(d_valid).astype(np.float64)
                    del d_train, d_valid, model
                    return prediction
                d_train = _task_design(
                    task_name,
                    prepared["train"], tid_train[inner_train_rows], aid_train[inner_train_rows],
                    xs_selected, prepared["history_train"],
                )
                d_valid = _task_design(
                    task_name,
                    prepared["valid"], tid_train[inner_valid_rows], aid_train[inner_valid_rows],
                    xs_selected, prepared["history_valid"],
                )
                prediction = _fit_predict_lgbm(
                    d_train,
                    task_labels[task_name][inner_train_rows],
                    (w_train[inner_train_rows] if task_name == "xs" else None),
                    d_valid,
                    spec=(XS_SPEC if task_name == "xs" else MARKET_SPEC),
                    min_data_scale=(1.0 if task_name == "xs" else MARKET_MIN_DATA_SCALE),
                    n_seeds=args.n_seeds,
                    seed=2026,
                    num_iteration=args.num_iteration,
                    num_threads=args.num_threads,
                )
                del d_train, d_valid
                return prediction

            def source_builder(
                inner_train_rows: np.ndarray,
                inner_valid_rows: np.ndarray,
                *,
                task_name: str = task,
            ) -> tuple[np.ndarray, np.ndarray, tuple[object, ...]]:
                prepared = prepare_split(inner_train_rows, inner_valid_rows)
                train_view = build_interaction_source_view(
                    task_name,
                    prepared["train"], tid_train[inner_train_rows],
                    history_indices, prepared["history_train"],
                    max_cells=args.max_source_cells,
                )
                valid_view = build_interaction_source_view(
                    task_name,
                    prepared["valid"], tid_train[inner_valid_rows],
                    history_indices, prepared["history_valid"],
                    max_cells=args.max_source_cells,
                )
                if train_view.catalog != valid_view.catalog:
                    raise AssertionError("train/valid interaction catalogs differ")
                return train_view.values, valid_view.values, train_view.catalog

            task_results[task] = mine_task_interactions(
                task=task,
                source_values=None,
                catalog=None,
                source_builder=source_builder,
                target=task_labels[task],
                weight=(w_train if task != "market" else np.ones_like(w_train)),
                time_ids=tid_train,
                baseline_predictor=baseline_predictor,
                n_blocks=4,
                min_blocks=2,
                row_cap=args.miner_row_cap,
                num_threads=args.num_threads,
            )
            mined_definitions = list(task_results[task]["definitions"])
            if mined_definitions:
                outer_sources = interaction_source_arrays(
                    task,
                    mined_definitions,
                    transformed_train,
                    tid_train,
                    history_indices,
                    outer_history_train,
                    max_cells=args.max_interaction_cells,
                )
                task_results[task]["definitions"] = resolve_quantile_thresholds(
                    mined_definitions, outer_sources, bins=32
                )
                del outer_sources
            gc.collect()

        manifest: dict[str, object] = {
            "schema_version": 1,
            "outer_fold_training_only": True,
            "fold": fold_index,
            "training_window": {
                "time_start": int(tid_train[0]),
                "time_end": int(tid_train[-1]),
                "time_ids": int(len(np.unique(tid_train))),
                "rows": int(len(tid_train)),
            },
            "feature_sets": {
                "ridge": [int(index) for index in ridge_selected],
                "xs": [int(index) for index in xs_selected],
                "market": [int(index) for index in xs_selected],
                "history": [int(index) for index in history_indices],
            },
            "tasks": task_results,
        }
        manifest_path = manifest_dir / f"fold_{fold_index}.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        split_cache.clear()
        del lag_cache
        gc.collect()

        interactions_train: dict[str, np.ndarray] = {}
        interactions_valid: dict[str, np.ndarray] = {}
        for task in ("ridge", "xs", "market"):
            definitions = _manifest_definitions(manifest, task)
            if not definitions:
                interactions_train[task] = np.empty((len(y_train), 0), dtype=np.float32)
                interactions_valid[task] = np.empty((len(y_valid), 0), dtype=np.float32)
                continue
            sources_train = interaction_source_arrays(
                task, definitions, transformed_train, tid_train,
                history_indices, outer_history_train,
                max_cells=args.max_interaction_cells,
            )
            sources_valid = interaction_source_arrays(
                task, definitions, transformed_valid, tid_valid,
                history_indices, outer_history_valid,
                max_cells=args.max_interaction_cells,
            )
            interactions_train[task] = build_interaction_columns(
                sources_train, definitions, max_cells=args.max_interaction_cells
            )
            interactions_valid[task] = build_interaction_columns(
                sources_valid, definitions, max_cells=args.max_interaction_cells
            )

        fold_alpha = RIDGE_ALPHA * len(train_ids) / REFERENCE_TRAIN_WINDOW
        ridge_train_base = ridge_designs(
            transformed_train, tid_train, ridge_selected, None
        )
        ridge_valid_base = ridge_designs(
            transformed_valid, tid_valid, ridge_selected, None
        )

        def fit_predict_ridge(
            train_design: np.ndarray, valid_design: np.ndarray
        ) -> np.ndarray:
            model = fit_ridge(train_design, y_train, w_train, fold_alpha)
            prediction = model.predict(valid_design).astype(np.float64)
            del model
            return prediction

        ridge_base_prediction, ridge_added_prediction = paired_component_predictions(
            ridge_train_base,
            ridge_valid_base,
            interactions_train["ridge"],
            interactions_valid["ridge"],
            fit_predict_ridge,
            asset_last=False,
        )
        del ridge_train_base, ridge_valid_base
        gc.collect()

        tree_predictions: dict[str, dict[str, np.ndarray]] = {
            "baseline": {},
            "interaction": {},
        }
        for task in ("xs", "market"):
            label = e_train if task == "xs" else y_train
            sample_weight = w_train if task == "xs" else None
            spec = XS_SPEC if task == "xs" else MARKET_SPEC
            min_scale = 1.0 if task == "xs" else MARKET_MIN_DATA_SCALE
            base_train = _task_design(
                task,
                transformed_train,
                tid_train,
                aid_train,
                xs_selected,
                outer_history_train,
            )
            base_valid = _task_design(
                task,
                transformed_valid,
                tid_valid,
                aid_valid,
                xs_selected,
                outer_history_valid,
            )

            def fit_predict_tree(
                train_design: np.ndarray,
                valid_design: np.ndarray,
            ) -> np.ndarray:
                return _fit_predict_lgbm(
                    train_design,
                    label,
                    sample_weight,
                    valid_design,
                    spec=spec,
                    min_data_scale=min_scale,
                    n_seeds=args.n_seeds,
                    seed=2026,
                    num_iteration=args.num_iteration,
                    num_threads=args.num_threads,
                )

            baseline_task, interaction_task = paired_component_predictions(
                base_train,
                base_valid,
                interactions_train[task],
                interactions_valid[task],
                fit_predict_tree,
                asset_last=True,
            )
            tree_predictions["baseline"][task] = baseline_task
            tree_predictions["interaction"][task] = interaction_task
            del base_train, base_valid
            gc.collect()

        baseline_prediction = compose_hybrid_raw(
            ridge_base_prediction,
            tree_predictions["baseline"]["xs"],
            tree_predictions["baseline"]["market"],
            tid_valid,
            market_lambda=args.market_lambda,
            blend_weight=args.blend_weight,
        )
        interaction_prediction = compose_hybrid_raw(
            ridge_added_prediction,
            tree_predictions["interaction"]["xs"],
            tree_predictions["interaction"]["market"],
            tid_valid,
            market_lambda=args.market_lambda,
            blend_weight=args.blend_weight,
        )
        baseline_metric = _metric_payload(y_valid, baseline_prediction, w_valid)
        interaction_metric = _metric_payload(y_valid, interaction_prediction, w_valid)
        peak_delta = interaction_metric["peak"] - baseline_metric["peak"]
        delta_a = interaction_metric["A"] / baseline_metric["A"] - 1.0
        delta_b = interaction_metric["B"] / baseline_metric["B"] - 1.0
        counts = {
            task: len(_manifest_definitions(manifest, task))
            for task in ("ridge", "xs", "market")
        }
        fold_payload: dict[str, Any] = {
            "fold": fold_index,
            "train_time_range": [int(tid_train[0]), int(tid_train[-1])],
            "valid_time_range": [int(tid_valid[0]), int(tid_valid[-1])],
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "baseline": baseline_metric,
            "interaction": interaction_metric,
            "peak_delta": float(peak_delta),
            "delta_a": float(delta_a),
            "delta_b": float(delta_b),
            "interaction_counts": counts,
            "manifest": str(manifest_path),
            "elapsed_seconds": float(time.perf_counter() - fold_started),
        }
        completed.append(fold_payload)
        pooled_target.append(y_valid.copy())
        pooled_weight.append(w_valid.copy())
        pooled_baseline.append(np.asarray(baseline_prediction, dtype=np.float64))
        pooled_interaction.append(np.asarray(interaction_prediction, dtype=np.float64))
        payload["folds"] = completed
        _write_partial_report(json_path, markdown_path, payload)
        print(
            f"fold {fold_index}: baseline={baseline_metric['peak']:.8f}, "
            f"interaction={interaction_metric['peak']:.8f}, delta={peak_delta:+.8f}, "
            f"counts={counts}, elapsed={fold_payload['elapsed_seconds']:.0f}s",
            flush=True,
        )
        del (
            transformed_train, transformed_valid,
            outer_history_train, outer_history_valid, split_cache,
            interactions_train, interactions_valid, tree_predictions,
        )
        gc.collect()
        return {
            "peak_delta": float(peak_delta),
            "delta_a": float(delta_a),
            "delta_b": float(delta_b),
        }

    sequence = run_paired_fold_sequence(
        len(folds), run_fold, required_positive=4
    )
    if sequence["stopped_early"]:
        gate: dict[str, object] = {
            "passed": False,
            "checks": {"four_of_five_positive": False},
            "reason": sequence["stop_reason"],
        }
        status = "failed_early"
    else:
        pooled_y = np.concatenate(pooled_target)
        pooled_w = np.concatenate(pooled_weight)
        pooled_base = np.concatenate(pooled_baseline)
        pooled_added = np.concatenate(pooled_interaction)
        baseline_pooled_metric = _metric_payload(pooled_y, pooled_base, pooled_w)
        interaction_pooled_metric = _metric_payload(pooled_y, pooled_added, pooled_w)
        pooled_delta_a = (
            interaction_pooled_metric["A"] / baseline_pooled_metric["A"] - 1.0
        )
        pooled_delta_b = (
            interaction_pooled_metric["B"] / baseline_pooled_metric["B"] - 1.0
        )
        gate = interaction_gate(
            np.asarray([fold["peak_delta"] for fold in completed]),
            pooled_delta_a,
            pooled_delta_b,
        )
        gate["pooled_baseline"] = baseline_pooled_metric
        gate["pooled_interaction"] = interaction_pooled_metric
        status = "passed" if gate["passed"] else "failed"
    payload["status"] = status
    payload["gate"] = gate
    payload["sequence"] = sequence
    payload["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_partial_report(json_path, markdown_path, payload)
    del features
    gc.collect()
    feature_spill_path.unlink(missing_ok=True)
    print(f"wrote {json_path}\nwrote {markdown_path}\nstatus={status}", flush=True)


if __name__ == "__main__":
    main()
