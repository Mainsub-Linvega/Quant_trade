"""Screen independent shrunk-market LightGBM checkpoints with the XS OOF block fixed.

The base cache supplies strict OOF ``e_lgbm`` and Ridge-market predictions. This script rebuilds
only the fold-local market design, trains one shrunk market booster to the largest checkpoint,
and evaluates 160/240/320/400/480 rounds without retraining the XS forest.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import apply_robust_transform, cross_sectional_deviation
from lgbm_xs import load_rows
from mt_predictability import group_starts
from src.metric import scale_invariant_score, weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import robust_transform_fit, select_features
from strategies.v3_hybrid.train import stream_history_blocks
from v3_production_oof import (FEATURE_COUNT, HISTORY_COUNT, HISTORY_WINDOW, MARKET_LAMBDA,
                               MARKET_MIN_DATA_SCALE, MARKET_SPEC, MIN_DATA_FRAC,
                               REFERENCE_TRAIN_WINDOW, group_mean, row_slice)

CHECKPOINTS = (160, 240, 320, 400, 480)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--base-oof", default=str(_REPO_ROOT / "outputs" / "cache" /
                                              "v3_production_oof_phasebal_prodwindow_exact.npz"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_market_round_scan_phasebal_prodwindow")
    p.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=REFERENCE_TRAIN_WINDOW)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced")
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--prediction-scale", type=float, default=1.16)
    p.add_argument("--prediction-clip", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray, scale: float, clip: float) -> dict[str, float]:
    peak = scale_invariant_score(y, pred, w)
    emitted = np.clip(pred * scale, -clip, clip)
    return {"peak": float(peak["peak"]), "A": float(peak["A"]), "B": float(peak["B"]),
            "optimal_scale": float(peak["optimal_scale"]),
            "unit_score": weighted_zero_mean_r2(y, pred, w),
            "public_scale_score": weighted_zero_mean_r2(y, emitted, w)}


def paired(values: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    delta = values - baseline
    drop = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    base = float(baseline.mean())
    return {"mean": float(values.mean()), "mean_delta": float(delta.mean()),
            "relative_gain": float(delta.mean() / base),
            "positive_folds": int((delta > 0).sum()), "n_folds": int(len(delta)),
            "drop_best_delta": float(drop.mean()),
            "drop_best_relative_gain": float(drop.mean() / base),
            "per_fold_delta": [float(v) for v in delta]}


def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] != 160 or any(v <= 0 for v in checkpoints):
        raise SystemExit("checkpoints must be positive and include 160 as baseline")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force")

    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")
    if v3_path not in sys.path:
        sys.path.append(v3_path)
    started = time.perf_counter()
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features = data["features"]
    target = data["target"].astype(np.float64, copy=False)
    weight = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    time_ids = data["time_id"]
    asset_ids = data["asset_id"]
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)

    with np.load(args.base_oof, allow_pickle=False) as base:
        for name, reference in (("target", target), ("weight", weight), ("time_id", time_ids),
                                ("asset_id", asset_ids)):
            if not np.array_equal(base[name], reference):
                raise AssertionError(f"base OOF {name} is not aligned with sampled data")
        base_fold = base["fold"].astype(np.int16)
        base_e = base["e_lgbm"].astype(np.float64)
        base_market_ridge = base["market_ridge"].astype(np.float64)

    fold_rows: list[dict[str, Any]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        tr, va = row_slice(time_ids, train_ids), row_slice(time_ids, valid_ids)
        if not np.all(base_fold[va] == index):
            raise AssertionError(f"base OOF fold {index} is not aligned")
        transformed_train, stats = robust_transform_fit(features[tr].copy())
        transformed_valid = features[va].copy()
        apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                               stats["center"], stats["scale"])
        y_tr, y_va = target[tr], target[va]
        w_va = weight[va]
        tid_tr, tid_va = time_ids[tr], time_ids[va]
        aid_tr, aid_va = asset_ids[tr], asset_ids[va]
        tr_starts, va_starts = group_starts(tid_tr), group_starts(tid_va)
        tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)
        va_counts = np.diff(np.r_[va_starts, len(tid_va)]).astype(np.float64)
        e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
        xs_selected = select_features(transformed_train, e_tr, np.ones_like(e_tr), FEATURE_COUNT)
        xs_tr = cross_sectional_deviation(transformed_train[:, xs_selected].copy(), tid_tr)
        xs_va = cross_sectional_deviation(transformed_valid[:, xs_selected].copy(), tid_va)
        history_positions = np.sort(select_features(
            xs_tr, e_tr, np.ones_like(e_tr), HISTORY_COUNT).astype(np.int64))
        history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
        history_stats = tuple(stats[key][xs_selected[history_positions]]
                              for key in ("lower", "upper", "center", "scale"))
        print(f"fold {index}: rebuilding causal history", flush=True)
        all_history = stream_history_blocks(Path(args.data_root), args.sample_modulo, args.sampling,
                                            history_names, history_stats, args.history_window)
        history_tr = [block[tr] for block in all_history]
        history_va = [block[va] for block in all_history]
        del all_history
        d_tr = np.ascontiguousarray(np.column_stack(
            [transformed_train[:, xs_selected], xs_tr, *history_tr, aid_tr.astype(np.float32)]))
        d_va = np.ascontiguousarray(np.column_stack(
            [transformed_valid[:, xs_selected], xs_va, *history_va, aid_va.astype(np.float32)]))
        min_data = max(20, int(round(MIN_DATA_FRAC * len(d_tr) * MARKET_MIN_DATA_SCALE)))
        params = {**MARKET_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                  "num_threads": args.num_threads, "min_data_in_leaf": min_data,
                  "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                  "force_row_wise": True, "feature_pre_filter": False,
                  "seed": args.seed, "bagging_seed": args.seed + 1000,
                  "feature_fraction_seed": args.seed + 2000}
        dataset = lgb.Dataset(d_tr, label=y_tr, params=params, categorical_feature=[d_tr.shape[1]-1],
                              free_raw_data=False)
        booster = lgb.train(params, dataset, num_boost_round=max(checkpoints))
        arms: dict[str, Any] = {}
        for checkpoint in checkpoints:
            row_pred = booster.predict(d_va, num_iteration=checkpoint).astype(np.float64)
            market_lgbm = group_mean(row_pred, va_starts, va_counts)
            market = (1.0 - MARKET_LAMBDA) * base_market_ridge[va] + MARKET_LAMBDA * market_lgbm
            full = market + base_e[va]
            arms[str(checkpoint)] = {"full": metrics(y_va, full, w_va, args.prediction_scale,
                                                      args.prediction_clip),
                                     "market": metrics(group_mean(y_va, va_starts, va_counts),
                                                       market, w_va, 1.0, 1e9)}
        fold_rows.append({"fold": index, "train_rows": len(y_tr), "valid_rows": len(y_va),
                          "arms": arms, "elapsed_seconds": time.perf_counter() - fold_started})
        print(f"  fold {index}: " + "  ".join(
            f"{cp}={arms[str(cp)]['full']['peak']:.8f}" for cp in checkpoints), flush=True)
        del transformed_train, transformed_valid, stats, e_tr, xs_tr, xs_va, history_tr, history_va
        del d_tr, d_va, dataset, booster
        gc.collect()

    baseline = np.array([row["arms"]["160"]["full"]["peak"] for row in fold_rows])
    base_a = np.mean([row["arms"]["160"]["full"]["A"] for row in fold_rows])
    base_b = np.mean([row["arms"]["160"]["full"]["B"] for row in fold_rows])
    summary: dict[str, Any] = {}
    for cp in checkpoints:
        peaks = np.array([row["arms"][str(cp)]["full"]["peak"] for row in fold_rows])
        stats = paired(peaks, baseline)
        a = np.mean([row["arms"][str(cp)]["full"]["A"] for row in fold_rows])
        b = np.mean([row["arms"][str(cp)]["full"]["B"] for row in fold_rows])
        stats.update({"delta_A": float(a / base_a - 1), "delta_B": float(b / base_b - 1),
                      "mechanism_2dA_gt_dB": bool(2 * (a / base_a - 1) > (b / base_b - 1))})
        summary[str(cp)] = stats

    for i, cp in enumerate(checkpoints):
        entry = summary[str(cp)]
        if cp == 160:
            entry.update({"neighbor_support": True, "pass": True})
            continue
        neighbors = [checkpoints[j] for j in (i - 1, i + 1) if 0 <= j < len(checkpoints)]
        support = any(summary[str(n)]["drop_best_delta"] > 0 for n in neighbors if n != 160)
        entry["neighbor_support"] = support
        entry["pass"] = bool(entry["relative_gain"] >= 0.01 and entry["positive_folds"] >= 4
                             and entry["drop_best_delta"] > 0 and support)
    passing = [cp for cp in checkpoints if cp != 160 and summary[str(cp)]["pass"]]
    selected = None
    if passing:
        best_mean = max(summary[str(cp)]["mean"] for cp in passing)
        selected = min(cp for cp in passing if summary[str(cp)]["mean"] >= best_mean * 0.9975)

    payload = {"experiment": "v3_market_round_scan", "base_oof": str(args.base_oof),
               "checkpoints": checkpoints, "selected_checkpoint": selected,
               "selection_rule": "pass gate, then lowest checkpoint within 0.25% of best mean peak",
               "folds": fold_rows, "summary": summary,
               "elapsed_seconds": time.perf_counter() - started}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v3 shrunk market checkpoint scan", "", f"Selected: `{selected}`", "",
             "| Rounds | Mean peak | Gain | Positive | Drop-best | ΔA | ΔB | Neighbor | Gate |",
             "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|"]
    for cp in checkpoints:
        row = summary[str(cp)]
        lines.append(f"| {cp} | {row['mean']:.8f} | {row['relative_gain']*100:+.2f}% | "
                     f"{row['positive_folds']}/{row['n_folds']} | "
                     f"{row['drop_best_relative_gain']*100:+.2f}% | {row['delta_A']*100:+.2f}% | "
                     f"{row['delta_B']*100:+.2f}% | {'Y' if row['neighbor_support'] else 'N'} | "
                     f"{'PASS' if row['pass'] else 'FAIL'} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}\nselected={selected}", flush=True)


if __name__ == "__main__":
    main()
