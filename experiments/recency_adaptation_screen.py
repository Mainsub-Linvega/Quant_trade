"""Screen fitting-loss recency policies for the current v3 decomposition."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "strategies" / "v1_ridge")]
from backfill_v3_retrain import selection_manifest_from_model
from backfill_v3_run import load_history_blocks
from features import apply_robust_transform, cross_sectional_deviation
from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.metric import scale_invariant_score, weighted_zero_mean_r2
from train import fit_model, predict_array, robust_transform_fit

POLICIES = ("none", "backfill_x2", "half_life_39480", "recent_window_78960")
BACKFILL_START = 888480
DEFAULT_SOURCE_MODEL = ROOT / "outputs" / "candidates" / "v3_frozen_retrained"


def effective_fit_weight(metric_weight: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    weight = np.maximum(np.asarray(metric_weight, dtype=np.float64), 0.0)
    factor = np.asarray(multiplier, dtype=np.float64)
    if weight.shape != factor.shape:
        raise ValueError("metric_weight and multiplier must have matching shapes")
    if not np.all(np.isfinite(factor)) or np.any(factor < 0.0):
        raise ValueError("multiplier must be finite and nonnegative")
    return weight * factor


def market_fit_weight(metric_weight: np.ndarray | None, policy: str) -> None:
    if policy != "frozen_unweighted":
        raise ValueError("market fitting must remain frozen_unweighted")
    return None


def policy_multiplier(time_id: np.ndarray, is_backfill: np.ndarray, policy: str, *, fit_end: int) -> np.ndarray:
    ids = np.asarray(time_id, dtype=np.int64)
    backfill = np.asarray(is_backfill, dtype=bool)
    if ids.ndim != 1 or backfill.shape != ids.shape:
        raise ValueError("time_id and is_backfill must be matching 1D arrays")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    result = np.ones(len(ids), dtype=np.float64)
    if policy == "backfill_x2":
        result[backfill] = 2.0
    elif policy == "half_life_39480":
        age = np.maximum(ids.astype(np.float64) - BACKFILL_START, 0.0)
        result[backfill] = np.power(2.0, -age[backfill] / 39480.0)
    elif policy == "recent_window_78960":
        result[backfill & (ids < int(fit_end) - 78960)] = 0.0
    return result


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform != "darwin" else value


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _load_window(data_root: Path, backfill_root: Path, end_time: int, sample_modulo: int, sampling: str) -> dict[str, np.ndarray]:
    columns = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("features", "target", "weight", "time_id", "asset_id", "is_backfill")}
    for source_root, is_backfill in ((data_root, False), (backfill_root, True)):
        for path in train_files(source_root):
            kept = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
                frame = batch.to_pandas()
                ids = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
                mask = time_sample_mask(ids, sample_modulo, sampling=sampling)
                if is_backfill:
                    mask &= ids < int(end_time)
                if not mask.any():
                    continue
                parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
                parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float64, copy=True))
                parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float64, copy=True))
                parts["time_id"].append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
                parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int64, copy=True))
                parts["is_backfill"].append(np.full(int(mask.sum()), is_backfill, dtype=bool))
                kept += int(mask.sum())
            print(f"load {source_root.name}/{path.name}: {kept:,} rows", flush=True)
    if not parts["features"]:
        raise ValueError("empty sampled window")
    result = {key: np.concatenate(value) for key, value in parts.items()}
    if np.any(np.diff(result["time_id"]) < 0):
        raise ValueError("combined sampled rows are not time ordered")
    return result


def _row_masks(data: dict[str, np.ndarray], *, train_end: int, valid_start: int, valid_end: int, embargo: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, backfill = data["time_id"], data["is_backfill"]
    train = ~backfill | (backfill & (ids < train_end))
    valid = backfill & (ids >= valid_start) & (ids < valid_end)
    embargo_mask = backfill & (ids >= train_end) & (ids < valid_start)
    expected = np.arange(train_end, valid_start, dtype=np.int64)
    observed = np.unique(ids[embargo_mask])
    if len(expected) != embargo or np.any(~np.isin(observed, expected)):
        raise ValueError("embargo time_ids are incomplete or have the wrong width")
    if not train.any() or not valid.any() or np.any(train & valid) or np.any(train & embargo_mask) or np.any(valid & embargo_mask):
        raise ValueError("invalid or overlapping train/validation masks")
    if valid_end <= valid_start:
        raise ValueError("validation interval must be positive")
    return train, valid, embargo_mask


def _group_mean(values: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    return np.repeat(np.add.reduceat(values, starts) / counts, counts)


def _metric(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray, scale: float) -> dict[str, float | int]:
    invariant = scale_invariant_score(target, prediction, weight)
    published = np.clip(prediction * scale, -0.5, 0.5)
    return {"A": float(invariant["A"]), "B": float(invariant["B"]), "peak": float(invariant["peak"]), "optimal_scale": float(invariant["optimal_scale"]), "score": float(weighted_zero_mean_r2(target, published, weight)), "scale": float(scale), "clip_count": int(np.count_nonzero(np.abs(published) >= 0.5 - 1e-12))}


def _train_predict(design_train: np.ndarray, label: np.ndarray, weight: np.ndarray | None, design_valid: np.ndarray, *, spec: dict[str, object], min_data: int, rounds: int, seed: int, threads: int) -> np.ndarray:
    import lightgbm as lgb
    params = dict(spec)
    for key in ("min_data_in_leaf", "bagging_fraction", "bagging_freq"):
        params.pop(key, None)
    params.update({"objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": threads, "min_data_in_leaf": min_data, "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True, "force_row_wise": True, "feature_pre_filter": False, "seed": seed, "bagging_seed": seed + 1000, "feature_fraction_seed": seed + 2000})
    dataset = lgb.Dataset(design_train, label=label, weight=weight, params=params, categorical_feature=[design_train.shape[1] - 1], free_raw_data=False)
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    prediction = np.asarray(booster.predict(design_valid, num_iteration=rounds), dtype=np.float64)
    del booster, dataset
    gc.collect()
    return prediction


def _blocks(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray, time_id: np.ndarray, scale: float, count: int = 5) -> list[dict[str, float | int]]:
    unique = np.unique(time_id)
    edges = np.linspace(0, len(unique), count + 1, dtype=int)
    result = []
    for index in range(count):
        mask = np.isin(time_id, unique[edges[index]:edges[index + 1]])
        result.append({"block": index, **_metric(target[mask], prediction[mask], weight[mask], scale)})
    return result


def paired_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    deltas = [float(c["peak"]) - float(b["peak"]) for b, c in zip(baseline["blocks"], candidate["blocks"])]
    delta_a = [float(c["A"]) - float(b["A"]) for b, c in zip(baseline["blocks"], candidate["blocks"])]
    delta_b = [float(c["B"]) - float(b["B"]) for b, c in zip(baseline["blocks"], candidate["blocks"])]
    result = {"block_peak_delta": deltas, "mean_peak_delta": float(np.mean(deltas)), "positive_blocks": int(sum(x > 0 for x in deltas)), "n_blocks": len(deltas), "drop_best_mean_peak_delta": float(np.mean(sorted(deltas)[:-1])), "mean_delta_A": float(np.mean(delta_a)), "mean_delta_B": float(np.mean(delta_b))}
    result["two_delta_A_gt_delta_B"] = bool(2 * result["mean_delta_A"] > result["mean_delta_B"])
    result["passed_gate"] = bool(result["mean_peak_delta"] > 0 and result["positive_blocks"] >= 4 and result["drop_best_mean_peak_delta"] > 0 and result["two_delta_A_gt_delta_B"])
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "frozen"), default="calibration")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--backfill-root", default="/mnt/e/量化/public_release_20260823/data")
    parser.add_argument("--source-model-dir", default=str(DEFAULT_SOURCE_MODEL))
    parser.add_argument("--selection-json")
    parser.add_argument("--output")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--sampling", choices=("periodic", "phase_balanced"), default="phase_balanced")
    parser.add_argument("--train-backfill-end", type=int, default=948480)
    parser.add_argument("--valid-start", type=int, default=948486)
    parser.add_argument("--valid-end", type=int, default=1008480)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = Path(args.output or ROOT / "outputs/experiments" / f"recency_adaptation_{args.stage}.json")
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() and not args.force else {}
    if existing and existing.get("stage") != args.stage:
        raise SystemExit("existing output belongs to another stage; use --force")
    if args.stage == "frozen":
        if not args.selection_json:
            raise SystemExit("--selection-json is required for frozen stage")
        selected = json.loads(Path(args.selection_json).read_text(encoding="utf-8")).get("selected_policy")
        if selected not in POLICIES:
            raise SystemExit("calibration report has no valid selected_policy")
        policies = ["none"] if selected == "none" else ["none", selected]
    else:
        policies = list(dict.fromkeys(args.policies))
    source_dir = Path(args.source_model_dir)
    source_meta = json.loads((source_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    manifest = selection_manifest_from_model(source_dir)
    scale = float(source_meta.get("prediction_scale", 1.16))
    data = _load_window(Path(args.data_root), Path(args.backfill_root), args.valid_end, args.sample_modulo, args.sampling)
    train_mask, valid_mask, embargo_mask = _row_masks(data, train_end=args.train_backfill_end, valid_start=args.valid_start, valid_end=args.valid_end, embargo=args.embargo)
    features, target, time_id = data["features"], data["target"], data["time_id"]
    xs_indices = np.asarray(manifest["xs"]["selected_indices"], dtype=np.int64)
    market_indices = np.asarray(manifest["market"]["selected_indices"], dtype=np.int64)
    _, xs_stats = robust_transform_fit(features[train_mask][:, xs_indices].copy())
    xs_all = features[:, xs_indices].copy()
    apply_robust_transform(xs_all, xs_stats["lower"], xs_stats["upper"], xs_stats["center"], xs_stats["scale"])
    xs_dev = cross_sectional_deviation(xs_all, time_id)
    del xs_all
    gc.collect()
    history_positions = [int(x) for x in source_meta.get("history_positions", [])]
    history_names = [FEATURE_COLUMNS[int(xs_indices[pos])] for pos in history_positions]
    history_stats = tuple(xs_stats[name][history_positions] for name in ("lower", "upper", "center", "scale"))
    history_blocks = load_history_blocks(Path(args.data_root), Path(args.backfill_root), args.valid_end, args.sample_modulo, args.sampling, history_names, history_stats, int(source_meta.get("history_window", 5)))
    if any(len(block) != len(target) for block in history_blocks):
        raise ValueError("history blocks do not align with sampled data")
    _, market_stats = robust_transform_fit(features[train_mask][:, market_indices].copy())
    market_all = features[:, market_indices].copy()
    apply_robust_transform(market_all, market_stats["lower"], market_stats["upper"], market_stats["center"], market_stats["scale"])
    market_dev = cross_sectional_deviation(market_all.copy(), time_id)
    market_train = np.column_stack([market_all[train_mask], market_dev[train_mask], *(b[train_mask] for b in history_blocks), data["asset_id"][train_mask].astype(np.float32)]).astype(np.float32, copy=False)
    market_valid = np.column_stack([market_all[valid_mask], market_dev[valid_mask], *(b[valid_mask] for b in history_blocks), data["asset_id"][valid_mask].astype(np.float32)]).astype(np.float32, copy=False)
    market_prediction = _train_predict(market_train, target[train_mask], None, market_valid, spec=dict(source_meta.get("market_lgbm_params") or source_meta.get("lgbm_params") or {}), min_data=max(20, round(12000 / 3500000 * len(market_train))), rounds=args.rounds, seed=args.seed, threads=args.num_threads)
    market_prediction = _group_mean(market_prediction, time_id[valid_mask])
    del market_all, market_dev, market_train, market_valid, market_stats
    gc.collect()
    report: dict[str, Any] = {"experiment": "recency_adaptation_screen", "stage": args.stage, "policies": policies, "boundaries": vars(args) | {"data_root": str(args.data_root), "backfill_root": str(args.backfill_root)}, "sampling": {"fit_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum()), "embargo_rows": int(embargo_mask.sum())}, "fixed_identity": {"manifest_hash": _hash_json(manifest), "rounds": args.rounds, "prediction_scale": scale, "market_fit": "unweighted"}, "results": existing.get("results", {}), "peak_rss_bytes": _peak_rss_bytes()}
    _atomic_json(output, report)
    for policy in policies:
        if policy in report["results"] and report["results"][policy].get("metric"):
            continue
        started = time.perf_counter()
        multiplier = policy_multiplier(time_id[train_mask], data["is_backfill"][train_mask], policy, fit_end=args.train_backfill_end)
        fit_weight = effective_fit_weight(data["weight"][train_mask], multiplier)
        artifact, _ = fit_model(features[train_mask].copy(), target[train_mask], fit_weight, time_id[train_mask], len(manifest["ridge"]["selected_indices"]), 2000000.0, selected_indices=np.asarray(manifest["ridge"]["selected_indices"], dtype=np.int64), ridge_tol=1e-8, ridge_max_iter=2000)
        ridge = predict_array(artifact, features[valid_mask], time_id[valid_mask], np.asarray(manifest["ridge"]["selected_indices"], dtype=np.int64), 1.0, 1e9).astype(np.float64)
        residual = target[train_mask] - _group_mean(target[train_mask], time_id[train_mask])
        xs_train = np.column_stack([xs_dev[train_mask], *(b[train_mask] for b in history_blocks), data["asset_id"][train_mask].astype(np.float32)]).astype(np.float32, copy=False)
        xs_valid = np.column_stack([xs_dev[valid_mask], *(b[valid_mask] for b in history_blocks), data["asset_id"][valid_mask].astype(np.float32)]).astype(np.float32, copy=False)
        xs_prediction = _train_predict(xs_train, residual, fit_weight, xs_valid, spec=dict(source_meta.get("lgbm_params") or {}), min_data=max(20, round(12000 / 3500000 * len(xs_train))), rounds=args.rounds, seed=args.seed, threads=args.num_threads)
        valid_time = time_id[valid_mask]
        xs_prediction -= _group_mean(xs_prediction, valid_time)
        raw = 0.5 * _group_mean(ridge, valid_time) + 0.5 * market_prediction + xs_prediction
        result = {"policy": policy, "fit_weight": {"sum": float(fit_weight.sum()), "mean": float(fit_weight.mean()), "nonzero_rows": int(np.count_nonzero(fit_weight)), "rows": int(len(fit_weight)), "multiplier_min": float(multiplier.min()), "multiplier_max": float(multiplier.max())}, "metric": _metric(target[valid_mask], raw, data["weight"][valid_mask], scale), "blocks": _blocks(target[valid_mask], raw, data["weight"][valid_mask], time_id[valid_mask], scale), "elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": _peak_rss_bytes()}
        report["results"][policy] = result
        report["peak_rss_bytes"] = max(report["peak_rss_bytes"], result["peak_rss_bytes"])
        _atomic_json(output, report)
        del artifact, ridge, residual, xs_train, xs_valid, xs_prediction, raw, fit_weight, multiplier
        gc.collect()
    baseline = report["results"].get("none")
    if baseline:
        for policy, result in report["results"].items():
            if policy != "none":
                result["paired_vs_none"] = paired_gate(baseline, result)
    if args.stage == "calibration":
        passing = [p for p in policies if p != "none" and report["results"][p].get("paired_vs_none", {}).get("passed_gate", False)]
        report["selected_policy"] = passing[0] if passing else "none"
        report["calibration_gate"] = {"passing_policies": passing, "selected_policy": report["selected_policy"]}
    else:
        report["selected_policy"] = policies[-1]
        report["frozen_selection_source"] = str(args.selection_json)
    report["peak_rss_bytes"] = max(report["peak_rss_bytes"], _peak_rss_bytes())
    _atomic_json(output, report)
    lines = [f"# Recency adaptation {args.stage}", "", f"Selected policy: {report['selected_policy']}", f"Peak RSS: {report['peak_rss_bytes'] / 1024**3:.2f} GiB", "", "| Policy | Score | Peak | A | B | Delta peak | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
    for policy in policies:
        m = report["results"][policy]["metric"]
        gate = report["results"][policy].get("paired_vs_none", {}).get("passed_gate", False)
        delta = report["results"][policy].get("paired_vs_none", {}).get("mean_peak_delta", 0.0)
        lines.append(f"| {policy} | {m['score']:.8f} | {m['peak']:.8f} | {m['A']:.6g} | {m['B']:.6g} | {delta:+.3e} | {'PASS' if gate else ('REF' if policy == 'none' else 'FAIL')} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(output.with_suffix('.md')), "selected_policy": report["selected_policy"], "peak_rss_bytes": report["peak_rss_bytes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
