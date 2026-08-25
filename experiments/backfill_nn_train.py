from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge", ROOT / "strategies" / "v4_mlp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.metric import scale_invariant_score, weighted_zero_mean_r2
from features import apply_robust_transform, cross_sectional_deviation
from mt_predictability import group_starts
from train import robust_transform_fit, select_features
from mlp_numpy import NumpyMLP

RESPONDER_COLUMNS = [f"responder_{i:02d}" for i in range(47)]
RESPONDER_SETS = {
    "ladder": ("responder_00", "responder_02", "responder_03", "responder_04", "responder_05"),
    "shortlist": ("responder_04", "responder_28", "responder_05", "responder_29", "responder_06"),
}


def experiment_distinction() -> dict[str, object]:
    return {
        "uses_public_backfill_labels": True,
        "new_data": "20260823 public backfill labels at /mnt/e/量化/public_release_20260823/data/train",
        "different_from": [
            "target_mlp_screen used only the original train rolling OOF setup and cached v3 OOF blend comparison; it did not train on the 20260823 public backfill labels.",
            "This experiment trains on original train plus the non-held-out part of the public backfill, then scores only on the held-out public backfill tail.",
            "The holdout is a contiguous public backfill time tail, not the old rolling_time_folds validation window.",
        ],
        "holdout": "public backfill tail reserved by real time_id; default is the last 60000 public-backfill time_id values.",
        "training_reopen_reason": (
            "previous NN did not show stable signal on the old original-train rolling setup; "
            "20260823 public backfill creates a new training distribution and a public-period holdout, "
            "so this is a fresh training experiment rather than a parameter tweak of the old NN."
        ),
        "scope_boundary": (
            "not a continuation of the old target_mlp_screen verdict; it re-tests NN training only under "
            "the expanded label set, and current responder auxiliary results remain unstable across sample density"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NN holdout on original train + public backfill labels.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--backfill-root", default="/mnt/e/量化/public_release_20260823/data")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--artifact-dir", default=None,
                        help="Optional directory to save target-only per-seed NumPy MLP artifacts.")
    parser.add_argument("--label", default="backfill_nn_shortlist_s20")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-modulo", type=int, default=20)
    parser.add_argument("--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced")
    parser.add_argument("--reserve-time-ids", type=int, default=60000)
    parser.add_argument("--train-backfill-end-time-id", type=int, default=None)
    parser.add_argument("--validation-start-time-id", type=int, default=None)
    parser.add_argument("--validation-end-time-id", type=int, default=None)
    parser.add_argument("--current-feature-count", type=int, default=100)
    parser.add_argument("--arms", choices=["target_only", "both"], default="both",
                        help="target_only skips responder loading/training; both also trains the auxiliary arm.")
    parser.add_argument("--responder-set", choices=["ladder", "shortlist", "custom"], default="shortlist")
    parser.add_argument("--responders", default=None)
    parser.add_argument("--aux-lambda", type=float, default=0.3)
    parser.add_argument("--market-hidden", type=int, nargs="+", default=[32])
    parser.add_argument("--cross-hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Train multiple target-only seeds after one data load; averaged prediction is scored as ensemble.")
    return parser.parse_args()


def arm_mode(name: str) -> dict[str, bool]:
    if name == "target_only":
        return {"target_only": True, "aux": False}
    if name == "both":
        return {"target_only": True, "aux": True}
    raise ValueError(f"unknown arm mode: {name}")


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    seeds = list(args.seeds) if getattr(args, "seeds", None) else [int(args.seed)]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not allowed")
    return [int(seed) for seed in seeds]


def ensemble_prediction(predictions: list[np.ndarray]) -> np.ndarray:
    if not predictions:
        raise ValueError("at least one prediction is required")
    first = predictions[0].shape
    if any(pred.shape != first for pred in predictions):
        raise ValueError("all predictions must have the same shape")
    return np.mean(np.vstack([pred.astype(np.float64, copy=False) for pred in predictions]), axis=0)


def selected_responders(kind: str, custom: str | None) -> list[str]:
    if kind in RESPONDER_SETS:
        return list(RESPONDER_SETS[kind])
    if kind != "custom":
        raise ValueError(f"unknown responder set: {kind}")
    if not custom:
        raise ValueError("--responders is required when --responder-set custom")
    names = [item.strip() for item in custom.split(",") if item.strip()]
    unknown = [name for name in names if name not in RESPONDER_COLUMNS]
    if unknown:
        raise ValueError(f"unknown responders: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate responders in custom set")
    return names


def split_backfill_holdout(time_id: np.ndarray, is_backfill: np.ndarray,
                           reserve_time_ids: int) -> tuple[np.ndarray, np.ndarray, int]:
    if reserve_time_ids <= 0:
        raise ValueError("reserve_time_ids must be positive")
    backfill_times = np.asarray(time_id[is_backfill], dtype=np.int64)
    if backfill_times.size == 0:
        raise ValueError("no backfill rows are available")
    cutoff = int(backfill_times.max() - reserve_time_ids + 1)
    if cutoff <= int(backfill_times.min()):
        raise ValueError("reserve_time_ids covers the whole backfill span")
    holdout = is_backfill & (time_id >= cutoff)
    train = ~holdout
    if not holdout.any():
        raise ValueError("holdout split is empty after sampling")
    return train, holdout, cutoff


def split_backfill_window(
    time_id: np.ndarray,
    is_backfill: np.ndarray,
    *,
    train_backfill_end_exclusive: int,
    valid_start_inclusive: int,
    valid_end_exclusive: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_backfill_end_exclusive > valid_start_inclusive:
        raise ValueError("training backfill end must not exceed validation start")
    if valid_start_inclusive >= valid_end_exclusive:
        raise ValueError("validation window must have positive width")
    ids = np.asarray(time_id, dtype=np.int64)
    backfill = np.asarray(is_backfill, dtype=bool)
    train = ~backfill | (backfill & (ids < train_backfill_end_exclusive))
    valid = backfill & (ids >= valid_start_inclusive) & (ids < valid_end_exclusive)
    if not train.any() or not valid.any():
        raise ValueError("training or validation window is empty")
    if np.any(train & valid):
        raise ValueError("training and validation windows overlap")
    return train, valid


def resolve_backfill_split(
    args: argparse.Namespace, time_id: np.ndarray, is_backfill: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, str]:
    values = (
        args.train_backfill_end_time_id,
        args.validation_start_time_id,
        args.validation_end_time_id,
    )
    if all(value is None for value in values):
        train, valid, cutoff = split_backfill_holdout(
            time_id, is_backfill, args.reserve_time_ids
        )
        return train, valid, cutoff, "backfill_tail"
    if any(value is None for value in values):
        raise ValueError("all explicit backfill window arguments must be provided together")
    train, valid = split_backfill_window(
        time_id, is_backfill,
        train_backfill_end_exclusive=int(args.train_backfill_end_time_id),
        valid_start_inclusive=int(args.validation_start_time_id),
        valid_end_exclusive=int(args.validation_end_time_id),
    )
    return train, valid, int(args.validation_start_time_id), "explicit_backfill_window"


def standardize(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float, float]:
    w = np.maximum(weight.astype(np.float64), 0.0)
    v = values.astype(np.float64)
    total = float(w.sum())
    mean = float(np.dot(w, v) / total)
    var = float(np.dot(w, (v - mean) ** 2) / total)
    std = max(float(np.sqrt(max(var, 0.0))), 1e-8)
    return ((v - mean) / std).astype(np.float64), mean, std


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts, axis=0) / counts[:, None]



def compose_market_cross_prediction(
    market_prediction: np.ndarray, cross_prediction: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    market = np.asarray(market_prediction, dtype=np.float64)
    cross = np.asarray(cross_prediction, dtype=np.float64).copy()
    group_counts = np.asarray(counts, dtype=np.int64)
    if len(market) != len(group_counts) or len(cross) != int(group_counts.sum()):
        raise ValueError("market, cross, and counts are not aligned")
    starts = np.r_[0, np.cumsum(group_counts)[:-1]]
    cross -= np.repeat(np.add.reduceat(cross, starts) / group_counts, group_counts)
    return np.repeat(market, group_counts) + cross

def impute_missing(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    arr = np.array(values, dtype=np.float64, copy=True)
    w = np.maximum(weight.astype(np.float64), 0.0)
    counts: dict[int, int] = {}
    for j in range(arr.shape[1]):
        col = arr[:, j]
        missing = ~np.isfinite(col)
        counts[j] = int(missing.sum())
        if missing.all():
            raise ValueError(f"auxiliary responder column {j} is all missing")
        if missing.any():
            observed = ~missing
            col[missing] = float(np.dot(w[observed], col[observed]) / max(w[observed].sum(), 1e-12))
    return arr, counts


def build_multitask_targets(residual: np.ndarray, responder_dev: np.ndarray,
                            weight: np.ndarray, aux_lambda: float) -> tuple[np.ndarray, float, float]:
    y0, mean, std = standardize(residual, weight)
    columns = [y0]
    root = float(np.sqrt(aux_lambda))
    for j in range(responder_dev.shape[1]):
        aux, _, _ = standardize(responder_dev[:, j], weight)
        columns.append(root * aux)
    out = np.column_stack(columns)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite multitask target matrix")
    return out, mean, std


def replay_target_only_from_models(
    market_model: NumpyMLP,
    cross_model: NumpyMLP,
    market_design: np.ndarray,
    cross_design: np.ndarray,
    holdout_counts: np.ndarray,
    *,
    market_mean: float,
    market_std: float,
    cross_mean: float,
    cross_std: float,
) -> np.ndarray:
    market_prediction = market_model.predict(market_design) * market_std + market_mean
    residual = cross_model.predict(cross_design) * cross_std + cross_mean
    starts = np.r_[0, np.cumsum(holdout_counts)[:-1]]
    residual -= np.repeat(np.add.reduceat(residual, starts) / holdout_counts, holdout_counts)
    return np.repeat(market_prediction, holdout_counts) + residual


def _selected_indices_from_metadata(metadata: dict[str, object]) -> np.ndarray:
    selected = metadata.get("selected_indices", metadata.get("selected"))
    if selected is None:
        raise ValueError("artifact metadata is missing selected feature indices")
    return np.asarray(selected, dtype=np.int64)


def _transform_stats_from_metadata(
    metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in ("lower", "upper", "center", "scale") if name not in metadata]
    if missing:
        raise ValueError(f"artifact metadata is missing transform stats: {missing}")
    return tuple(np.asarray(metadata[name], dtype=np.float32) for name in ("lower", "upper", "center", "scale"))


def transform_selected_features_from_metadata(features: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    selected = _selected_indices_from_metadata(metadata)
    lower, upper, center, scale = _transform_stats_from_metadata(metadata)
    selected_features = np.asarray(features[:, selected], dtype=np.float32, copy=True)
    return apply_robust_transform(selected_features, lower, upper, center, scale)


def predict_target_only_from_artifacts(
    artifacts: list[dict[str, object]],
    features: np.ndarray,
    time_id: np.ndarray,
    asset_id: np.ndarray,
) -> np.ndarray:
    if not artifacts:
        raise ValueError("at least one artifact row is required")
    tid = np.asarray(time_id, dtype=np.int64)
    aid = np.asarray(asset_id, dtype=np.int64)
    if len(tid) != len(aid) or len(features) != len(tid):
        raise ValueError("features, time_id, and asset_id must have identical row counts")
    if len(tid) == 0:
        raise ValueError("no rows are available for prediction")
    starts = group_starts(tid)
    counts = np.diff(np.r_[starts, len(tid)])
    predictions: list[np.ndarray] = []
    for row in artifacts:
        market_model, market_meta = NumpyMLP.load(row["market"])
        cross_model, cross_meta = NumpyMLP.load(row["cross"])
        market_features = transform_selected_features_from_metadata(features, market_meta)
        cross_features = transform_selected_features_from_metadata(features, cross_meta)
        market_design = group_mean(market_features, starts, counts)
        cross_dev = cross_sectional_deviation(cross_features, tid)
        asset_one_hot = int(cross_meta.get("asset_one_hot", 15))
        if asset_one_hot <= 0:
            raise ValueError("asset_one_hot must be positive")
        if aid.min() < 0 or aid.max() >= asset_one_hot:
            raise ValueError("asset_id is outside the saved asset_one_hot width")
        cross_design = np.ascontiguousarray(np.column_stack([
            cross_dev,
            np.eye(asset_one_hot, dtype=np.float32)[aid],
        ]))
        predictions.append(replay_target_only_from_models(
            market_model,
            cross_model,
            market_design,
            cross_design,
            counts,
            market_mean=float(market_meta["mean"]),
            market_std=float(market_meta["std"]),
            cross_mean=float(cross_meta["mean"]),
            cross_std=float(cross_meta["std"]),
        ))
    return ensemble_prediction(predictions)


def fit_mlp(design: np.ndarray, target: np.ndarray, weight: np.ndarray,
            hidden: tuple[int, ...], args: argparse.Namespace, seed: int) -> MLPRegressor:
    estimator = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=args.alpha,
        batch_size=args.batch_size, learning_rate_init=args.learning_rate,
        max_iter=args.max_iter, shuffle=True, random_state=seed, tol=0.0,
        early_stopping=False, n_iter_no_change=args.max_iter + 1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(design, target, sample_weight=np.maximum(weight, 0.0))
    return estimator


def load_combined_rows(data_root: Path, backfill_root: Path, responders: list[str],
                       sample_modulo: int, sampling: str) -> dict[str, np.ndarray]:
    columns = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target", *responders]
    keys = ("features", "responders", "target", "weight", "time_id", "asset_id", "is_backfill")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    jobs = [(False, [Path(p) for p in train_files(data_root)]),
            (True, [Path(p) for p in train_files(backfill_root)])]
    for is_backfill, paths in jobs:
        if not paths:
            raise SystemExit(f"no train files under {backfill_root if is_backfill else data_root}")
        for path in paths:
            kept, started = 0, time.perf_counter()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=120000, columns=columns):
                frame = batch.to_pandas()
                mask = time_sample_mask(frame["time_id"].to_numpy(copy=False), sample_modulo, sampling=sampling)
                if not mask.any():
                    continue
                n = int(mask.sum())
                parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
                if responders:
                    parts["responders"].append(frame.loc[mask, responders].to_numpy(dtype=np.float32, copy=True))
                parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float64, copy=True))
                parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float64, copy=True))
                parts["time_id"].append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
                parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int64, copy=True))
                parts["is_backfill"].append(np.full(n, is_backfill, dtype=bool))
                kept += n
            label = "backfill" if is_backfill else "original"
            print(f"  {label} {path.name}: {kept:,} rows ({time.perf_counter()-started:.0f}s)", flush=True)
    return {key: np.concatenate(value) for key, value in parts.items() if value}


def _json_safe_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def arm_score(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> dict[str, float | None]:
    peak = scale_invariant_score(y, pred, weight)
    return {**{key: _json_safe_float(value) for key, value in peak.items()},
            "score_at_unit_scale_unclipped": _json_safe_float(weighted_zero_mean_r2(y, pred, weight))}


def format_score_cell(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(float(value), spec)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.label}.json"
    md_path = out / f"{args.label}.md"
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else out / f"{args.label}_artifacts"
    if not args.force and (json_path.exists() or md_path.exists() or artifact_dir.exists()):
        raise SystemExit(f"{json_path}, {md_path}, or {artifact_dir} exists; pass --force")
    if args.force and artifact_dir.exists():
        import shutil
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    modes = arm_mode(args.arms)
    responders = selected_responders(args.responder_set, args.responders) if modes["aux"] else []
    print(f"loading original + backfill rows; arms={args.arms}; responders={responders}", flush=True)
    data = load_combined_rows(Path(args.data_root), Path(args.backfill_root), responders,
                              args.sample_modulo, args.sampling)
    tid = data["time_id"]
    aid = data["asset_id"]
    train_mask, holdout_mask, cutoff, split_mode = resolve_backfill_split(args, tid, data["is_backfill"])
    print(f"validation cutoff time_id >= {cutoff} ({split_mode})", flush=True)
    print(f"sample rows: total={len(tid):,} train={int(train_mask.sum()):,} holdout={int(holdout_mask.sum()):,}", flush=True)

    from history_peak import transform_with
    t_train, stats = robust_transform_fit(data["features"][train_mask].copy())
    t_holdout = transform_with(data["features"][holdout_mask], stats)
    y_train = data["target"][train_mask].astype(np.float64)
    y_holdout = data["target"][holdout_mask].astype(np.float64)
    w_train = np.maximum(data["weight"][train_mask].astype(np.float64), 0.0)
    w_holdout = np.maximum(data["weight"][holdout_mask].astype(np.float64), 0.0)
    tid_train, tid_holdout = tid[train_mask], tid[holdout_mask]
    aid_train, aid_holdout = aid[train_mask], aid[holdout_mask]

    starts_train = group_starts(tid_train)
    counts_train = np.diff(np.r_[starts_train, len(tid_train)])
    starts_holdout = group_starts(tid_holdout)
    counts_holdout = np.diff(np.r_[starts_holdout, len(tid_holdout)])
    market_y_train = np.add.reduceat(y_train, starts_train) / counts_train
    market_weight_train = np.add.reduceat(w_train, starts_train)
    residual_train = y_train - np.repeat(market_y_train, counts_train)

    market_selected = select_features(t_train, y_train, w_train, args.current_feature_count)
    cross_selected = select_features(t_train, residual_train, np.ones_like(residual_train), args.current_feature_count)
    market_stats = {name: stats[name][market_selected].tolist() for name in ("lower", "upper", "center", "scale")}
    cross_stats = {name: stats[name][cross_selected].tolist() for name in ("lower", "upper", "center", "scale")}
    market_train = group_mean(t_train[:, market_selected], starts_train, counts_train)
    market_holdout = group_mean(t_holdout[:, market_selected], starts_holdout, counts_holdout)
    cross_train = cross_sectional_deviation(t_train[:, cross_selected].copy(), tid_train)
    cross_holdout = cross_sectional_deviation(t_holdout[:, cross_selected].copy(), tid_holdout)
    asset_train = np.eye(15, dtype=np.float32)[aid_train]
    asset_holdout = np.eye(15, dtype=np.float32)[aid_holdout]
    design_train = np.ascontiguousarray(np.column_stack([cross_train, asset_train]))
    design_holdout = np.ascontiguousarray(np.column_stack([cross_holdout, asset_holdout]))
    del cross_train, cross_holdout, asset_train, asset_holdout
    gc.collect()

    seeds = resolve_seeds(args)
    market_target, market_mean, market_std = standardize(market_y_train, market_weight_train)
    unit = np.ones_like(residual_train)
    target_y, cross_mean, cross_std = standardize(residual_train, unit)
    target_predictions: list[np.ndarray] = []
    market_predictions: list[np.ndarray] = []
    replay_predictions: list[np.ndarray] = []
    seed_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    last_market_est = None
    last_target_est = None
    for seed in seeds:
        market_est = fit_mlp(market_train, market_target, market_weight_train, tuple(args.market_hidden), args, seed)
        market_pred = market_est.predict(market_holdout) * market_std + market_mean
        target_est = fit_mlp(design_train, target_y, unit, tuple(args.cross_hidden), args, seed + 100)
        target_resid = target_est.predict(design_holdout) * cross_std + cross_mean
        target_pred = compose_market_cross_prediction(market_pred, target_resid, counts_holdout)
        target_predictions.append(target_pred.astype(np.float64, copy=False))
        market_predictions.append(market_pred.astype(np.float64, copy=False))
        market_path = artifact_dir / f"seed_{seed}_market.npz"
        cross_path = artifact_dir / f"seed_{seed}_cross.npz"
        market_metadata = {"seed": int(seed), "head": "market", "mean": market_mean, "std": market_std,
                           "selected": [int(i) for i in market_selected], **market_stats}
        cross_metadata = {"seed": int(seed), "head": "cross", "mean": cross_mean, "std": cross_std,
                          "selected": [int(i) for i in cross_selected], "asset_one_hot": 15, **cross_stats}
        NumpyMLP.from_sklearn(market_est).save(market_path, market_metadata)
        NumpyMLP.from_sklearn(target_est).save(cross_path, cross_metadata)
        loaded_market, loaded_market_meta = NumpyMLP.load(market_path)
        loaded_cross, loaded_cross_meta = NumpyMLP.load(cross_path)
        replay_pred = replay_target_only_from_models(
            loaded_market, loaded_cross, market_holdout, design_holdout, counts_holdout,
            market_mean=loaded_market_meta["mean"], market_std=loaded_market_meta["std"],
            cross_mean=loaded_cross_meta["mean"], cross_std=loaded_cross_meta["std"])
        replay_predictions.append(replay_pred.astype(np.float64, copy=False))
        replay_max_abs = float(np.max(np.abs(replay_pred - target_pred)))
        seed_rows.append({
            "seed": int(seed),
            "iterations": {"market": int(market_est.n_iter_), "target_only_cross": int(target_est.n_iter_)},
            "target_only": arm_score(y_holdout, target_pred, w_holdout),
            "artifact_replay_max_abs": replay_max_abs,
        })
        artifact_rows.append({"seed": int(seed), "market": str(market_path), "cross": str(cross_path),
                              "replay_max_abs": replay_max_abs})
        last_market_est, last_target_est = market_est, target_est
    target_pred = ensemble_prediction(target_predictions)
    replay_target_pred = ensemble_prediction(replay_predictions)
    artifact_replay_ensemble_max_abs = float(np.max(np.abs(replay_target_pred - target_pred)))
    cold_start_target_pred = predict_target_only_from_artifacts(
        artifact_rows, data["features"][holdout_mask], tid_holdout, aid_holdout
    )
    cold_start_replay_ensemble_max_abs = float(np.max(np.abs(cold_start_target_pred - target_pred)))

    aux_name = f"aux_{args.responder_set}"
    missing_by_index: dict[int, int] = {}
    aux_est = None
    arms = {
        "zero": arm_score(y_holdout, np.zeros_like(y_holdout), w_holdout),
        "target_only": arm_score(y_holdout, target_pred, w_holdout),
    }
    if modes["aux"]:
        raw_resp, missing_by_index = impute_missing(data["responders"][train_mask], unit)
        responder_dev = cross_sectional_deviation(raw_resp.astype(np.float32), tid_train)
        multitask_y, aux_mean, aux_std = build_multitask_targets(residual_train, responder_dev, unit, args.aux_lambda)
        aux_predictions: list[np.ndarray] = []
        for index, seed in enumerate(seeds):
            aux_est = fit_mlp(design_train, multitask_y, unit, tuple(args.cross_hidden), args, seed + 200)
            aux_raw = aux_est.predict(design_holdout)
            aux_resid = (aux_raw[:, 0] if aux_raw.ndim == 2 else aux_raw) * aux_std + aux_mean
            aux_pred_seed = compose_market_cross_prediction(market_predictions[index], aux_resid, counts_holdout)
            aux_predictions.append(aux_pred_seed.astype(np.float64, copy=False))
        aux_pred = ensemble_prediction(aux_predictions)
        arms[aux_name] = arm_score(y_holdout, aux_pred, w_holdout)
        arms["aux_delta_vs_target_only"] = {
            "peak_delta": float(arms[aux_name]["peak"] - arms["target_only"]["peak"]),
            "relative_peak_gain": float(arms[aux_name]["peak"] / max(arms["target_only"]["peak"], 1e-300) - 1.0),
        }

    payload: dict[str, Any] = {
        "experiment": "backfill_nn_train",
        "question": "Train NN on original train plus public backfill labels, reserving the public tail for scoring.",
        "configuration": vars(args) | {"responders": responders},
        "distinction_from_prior_nn": experiment_distinction(),
        "split": {
            "holdout_cutoff_time_id": cutoff,
            "mode": split_mode,
            "train_backfill_end_exclusive": args.train_backfill_end_time_id,
            "validation_start_inclusive": args.validation_start_time_id,
            "validation_end_exclusive": args.validation_end_time_id,
            "reserve_time_ids": args.reserve_time_ids,
            "rows_total_sampled": int(len(tid)),
            "rows_train_sampled": int(train_mask.sum()),
            "rows_holdout_sampled": int(holdout_mask.sum()),
            "train_time_range": [int(tid_train.min()), int(tid_train.max())],
            "holdout_time_range": [int(tid_holdout.min()), int(tid_holdout.max())],
            "backfill_rows_sampled": int(data["is_backfill"].sum()),
        },
        "architecture": {
            "market": "sampled time-level mean features -> MLP",
            "cross": "current cross-sectional deviation + asset one-hot -> MLP",
            "history_used": False,
            "target_only_control": True,
            "auxiliary_loss": {"lambda": args.aux_lambda, "responders": responders},
        },
        "selected_features": {"market": [int(i) for i in market_selected], "cross": [int(i) for i in cross_selected]},
        "aux_missing_imputed": {responders[int(k)]: int(v) for k, v in missing_by_index.items()},
        "iterations": {"market": None if last_market_est is None else int(last_market_est.n_iter_),
                       "target_only_cross": None if last_target_est is None else int(last_target_est.n_iter_),
                       "aux_cross": None if aux_est is None else int(aux_est.n_iter_)},
        "seed_results": seed_rows,
        "ensemble": {"seeds": seeds, "target_only_prediction": "rowwise mean of seed target_only predictions"},
        "artifacts": {"dir": str(artifact_dir), "seeds": artifact_rows,
                      "ensemble_replay_max_abs": artifact_replay_ensemble_max_abs,
                      "cold_start_replay_max_abs": cold_start_replay_ensemble_max_abs},
        "arms": arms,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    distinction = payload["distinction_from_prior_nn"]
    lines = [
        f"# Backfill NN holdout ({args.label})", "",
        "## Distinction from prior NN records", "",
        f"- New data: {distinction['new_data']}",
        f"- Prior NN difference: {distinction['different_from'][0]}",
        f"- Training reopen reason: {distinction['training_reopen_reason']}",
        f"- Scope boundary: {distinction['scope_boundary']}",
        f"- Current holdout: {distinction['holdout']}", "",
        f"Sample: {payload['split']['rows_total_sampled']:,} rows; train {payload['split']['rows_train_sampled']:,}; holdout {payload['split']['rows_holdout_sampled']:,}.",
        f"Holdout: public backfill time_id >= {cutoff} ({payload['split']['holdout_time_range'][0]}..{payload['split']['holdout_time_range'][1]}).",
        f"Arms: {args.arms}; seeds: {', '.join(map(str, seeds))}; responders: {', '.join(responders) if responders else 'none'}; aux lambda={args.aux_lambda}.",
        f"Artifacts: `{artifact_dir}`; replay max abs={artifact_replay_ensemble_max_abs:.3e}; cold-start max abs={cold_start_replay_ensemble_max_abs:.3e}.", "",
    ]
    if len(seed_rows) > 1:
        lines += ["| individual seed | peak | optimal scale | score@unit |", "|---|---:|---:|---:|"]
        for row in seed_rows:
            block = row["target_only"]
            lines.append(
                f"| `{row['seed']}` | {format_score_cell(block['peak'], '.8f')} | "
                f"{format_score_cell(block['optimal_scale'], '.4f')} | "
                f"{format_score_cell(block['score_at_unit_scale'], '.8f')} |"
            )
        lines += ["", "Ensemble target-only prediction is the rowwise mean across seeds.", ""]
    lines += ["| arm | peak | optimal scale | score@unit | A | B |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in (["zero", "target_only"] + ([aux_name] if modes["aux"] else [])):
        block = arms[name]
        lines.append(
            f"| `{name}` | {format_score_cell(block['peak'], '.8f')} | "
            f"{format_score_cell(block['optimal_scale'], '.4f')} | "
            f"{format_score_cell(block['score_at_unit_scale'], '.8f')} | "
            f"{format_score_cell(block['A'], '+.4e')} | "
            f"{format_score_cell(block['B'], '.4e')} |"
        )
    if modes["aux"]:
        delta = arms["aux_delta_vs_target_only"]
        lines += ["", f"Aux delta vs target-only: {delta['peak_delta']:+.3e} ({delta['relative_peak_gain']:+.2%}).", ""]
    else:
        lines += ["", "Auxiliary arm skipped for target-only training experiment.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    summary_out = {"target_only_peak": arms["target_only"]["peak"], "json": str(json_path), "md": str(md_path)}
    if modes["aux"]:
        summary_out.update({"aux_peak": arms[aux_name]["peak"],
                            "aux_relative_gain": arms["aux_delta_vs_target_only"]["relative_peak_gain"]})
    print(json.dumps(summary_out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
