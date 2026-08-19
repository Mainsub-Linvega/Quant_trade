"""Memory-safe local full-resolution v3 resource smoke.

This measures whether fixed-production 200-feature XS and market forests can be
trained sequentially on one real-time fold. It deliberately skips Ridge, OOF arrays,
checkpoints, and promotion scoring. Long runs must be launched outside the chat via
systemd; this script itself is short enough to run as a background service.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# lgbm_xs imports the short module name ``train`` and must resolve v1_ridge.train;
# only add v3_hybrid after that import so the short-name import cannot collide.
for p in (ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.lgbm_xs import load_rows_disk_backed
from features import apply_robust_transform, cross_sectional_deviation
v3_path = str(ROOT / "strategies" / "v3_hybrid")
if v3_path not in sys.path:
    sys.path.insert(0, v3_path)
from strategies.v3_hybrid.train import stream_history_range_blocks


def group_mean(values: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    return np.repeat(np.add.reduceat(values, starts) / counts, counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--meta", default=str(ROOT / "strategies/v3_hybrid/model/hybrid_meta.json"))
    p.add_argument("--fold-grid", default=str(ROOT / "outputs/experiments/v3_real_time_fold0_grid.json"))
    p.add_argument("--cache", default=str(ROOT / "outputs/cache/fullres_rows_mod1"))
    p.add_argument("--design-cache", default="/tmp/v3_fullres_smoke_design")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--train-timeids", type=int, default=78_960,
                   help="resource smoke only: keep the most recent N real train time_ids")
    p.add_argument("--valid-timeids", type=int, default=20_000,
                   help="resource smoke only: keep the first N real validation time_ids")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--output", default=str(ROOT / "outputs/experiments/v3_fullres_resource_smoke.json"))
    return p.parse_args()


def fit(design: np.ndarray, y: np.ndarray, weight: np.ndarray | None,
        params: dict, rounds: int, threads: int, min_leaf: int):
    import lightgbm as lgb
    config = {**params, "objective": "regression", "metric": "l2", "verbosity": -1,
              "num_threads": threads, "min_data_in_leaf": min_leaf,
              "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
              "force_col_wise": True, "force_row_wise": False, "feature_pre_filter": False,
              "seed": 2026, "bagging_seed": 3026, "feature_fraction_seed": 4026}
    cat = design.shape[1] - 1
    # Do not construct before lgb.train: LightGBM sets categorical metadata during
    # train(); with free_raw_data=True, pre-constructing makes that second set fail.
    ds = lgb.Dataset(design, label=y, weight=weight, params=config,
                     categorical_feature=[cat], free_raw_data=True)
    booster = lgb.train(config, ds, num_boost_round=rounds)
    return booster


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    grid = json.loads(Path(args.fold_grid).read_text(encoding="utf-8"))
    fold = grid["folds"][0]
    train_range = tuple(map(int, fold["train_time_range"]))
    valid_range = tuple(map(int, fold["valid_time_range"]))
    fixed_names = list(meta["lgbm_features"])
    fixed_indices = np.array([int(name.split("_")[1]) for name in fixed_names], dtype=np.int64)
    if len(fixed_indices) != 200 or len(np.unique(fixed_indices)) != 200:
        raise RuntimeError("production meta does not contain 200 unique features")
    cache = load_rows_disk_backed(Path(args.data_root), 1, "periodic", Path(args.cache),
                                  feature_indices=fixed_indices)
    x = cache["features"]
    tid = cache["time_id"]; aid = cache["asset_id"]
    all_unique = np.unique(tid)
    train_ids = all_unique[(all_unique >= train_range[0]) & (all_unique <= train_range[1])]
    valid_ids = all_unique[(all_unique >= valid_range[0]) & (all_unique <= valid_range[1])]
    if args.train_timeids <= 0 or args.valid_timeids <= 0:
        raise ValueError("train/valid time-id counts must be positive")
    train_ids = train_ids[-args.train_timeids:]
    valid_ids = valid_ids[:args.valid_timeids]
    train_range = (int(train_ids[0]), int(train_ids[-1]))
    valid_range = (int(valid_ids[0]), int(valid_ids[-1]))
    tr0 = int(np.searchsorted(tid, train_range[0], side="left"))
    tr1 = int(np.searchsorted(tid, train_range[1], side="right"))
    va0 = int(np.searchsorted(tid, valid_range[0], side="left"))
    va1 = int(np.searchsorted(tid, valid_range[1], side="right"))
    tr = slice(tr0, tr1); va = slice(va0, va1)
    y_tr = np.asarray(cache["target"][tr]); w_tr = np.maximum(np.asarray(cache["weight"][tr]), 0.0)
    tid_tr = np.asarray(tid[tr]); tid_va = np.asarray(tid[va])
    aid_tr = np.asarray(aid[tr], dtype=np.int64); aid_va = np.asarray(aid[va], dtype=np.int64)
    train_features = np.asarray(x[tr], dtype=np.float32)
    valid_features = np.asarray(x[va], dtype=np.float32)
    stats = {k: np.asarray(meta[k], dtype=np.float32) for k in ("lower", "upper", "center", "scale")}
    apply_robust_transform(train_features, **stats)
    apply_robust_transform(valid_features, **stats)
    positions = np.asarray(meta["history_positions"], dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= len(fixed_names)):
        raise RuntimeError("production history_positions out of range")
    history_names = [fixed_names[int(pos)] for pos in positions]
    history_stats = tuple(stats[k][positions] for k in ("lower", "upper", "center", "scale"))
    history_tr, history_va = stream_history_range_blocks(
        Path(args.data_root), history_names, history_stats, int(meta["history_window"]),
        train_range, valid_range)
    if len(history_tr[0]) != len(y_tr):
        raise RuntimeError("history/train alignment mismatch")
    e_tr = y_tr - group_mean(y_tr, tid_tr)
    t_xs = time.perf_counter()
    xs_tr = cross_sectional_deviation(train_features, tid_tr)
    xs_va = cross_sectional_deviation(valid_features, tid_va)
    dtr = np.ascontiguousarray(np.column_stack([xs_tr, *history_tr, aid_tr.astype(np.float32)]), dtype=np.float32)
    dva = np.ascontiguousarray(np.column_stack([xs_va, *history_va, aid_va.astype(np.float32)]), dtype=np.float32)
    xs_booster = fit(dtr, e_tr, w_tr, meta["lgbm_params"], args.rounds, args.num_threads,
                     int(meta["lgbm_params"]["min_data_in_leaf"]))
    xs_pred = xs_booster.predict(dva, num_iteration=args.rounds)
    if not np.all(np.isfinite(xs_pred)):
        raise RuntimeError("XS smoke produced non-finite predictions")
    del xs_booster, dtr, dva, xs_pred, xs_tr, xs_va
    gc.collect()
    xs_seconds = time.perf_counter() - t_xs
    t_market = time.perf_counter()
    market_dev_tr = cross_sectional_deviation(train_features, tid_tr)
    market_dev_va = cross_sectional_deviation(valid_features, tid_va)
    dtr = np.ascontiguousarray(np.column_stack([train_features, market_dev_tr, *history_tr,
                                                aid_tr.astype(np.float32)]), dtype=np.float32)
    dva = np.ascontiguousarray(np.column_stack([valid_features, market_dev_va, *history_va,
                                                aid_va.astype(np.float32)]), dtype=np.float32)
    market_booster = fit(dtr, y_tr, None, meta["market_lgbm_params"], args.rounds, args.num_threads,
                         int(meta["market_lgbm_params"]["min_data_in_leaf"]))
    market_pred = market_booster.predict(dva, num_iteration=args.rounds)
    if not np.all(np.isfinite(market_pred)):
        raise RuntimeError("market smoke produced non-finite predictions")
    del market_booster, dtr, dva, market_pred, market_dev_tr, market_dev_va
    gc.collect()
    payload = {"status": "ok", "oof_valid": False, "rows": {"train": tr1 - tr0, "valid": va1 - va0},
               "rounds": args.rounds, "train_range": train_range, "valid_range": valid_range,
               "train_timeids": len(train_ids), "valid_timeids": len(valid_ids),
               "xs_seconds": xs_seconds, "market_seconds": time.perf_counter() - t_market,
               "elapsed_seconds": time.perf_counter() - started,
               "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "note": "resource smoke only; fixed production stats/features; Ridge and OOF score skipped"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
