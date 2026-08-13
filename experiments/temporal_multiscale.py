"""V4-T: pre-registered multi-scale per-asset temporal states for the LGBM cross-section block.

Arms share folds, preprocessing, selected columns, Ridge market prediction, LGBM target and parameters.
The only difference is the temporal blocks appended after ``xs_dev``:

- baseline: validated v3 history (lag1/difference/mean5/deviation5)
- t1_lags: baseline + lag2/lag5
- t2_state: baseline + EWM3/EWM10/std5/std20/slope5/slope20
- t3_full: T1 + T2 + observation gap

The first pass is deliberately cheap (one seed / 160 rounds). It is a directional gate, not a production
candidate. A candidate must beat the same strong baseline in absolute peak, not merely a weakened arm.
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
_IMPORT_PATHS = [ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge",
                 ROOT / "strategies" / "v3_hybrid"]
sys.path[:0] = [str(path) for path in _IMPORT_PATHS if str(path) not in sys.path]

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.metric import scale_invariant_score
from src.validation import rolling_time_folds
from features import apply_robust_transform, cross_sectional_deviation
from history_peak import (LGBM_MIN_DATA_FRAC, LGBM_SPEC, fit_ridge, ridge_designs,
                          transform_with)
from lgbm_xs import load_rows
from mt_predictability import group_starts
from temporal import (ARMS, MarketRegimeHistory, MultiScaleAssetHistory, temporal_arm_blocks,
                      temporal_atoms_from_lags)
from train import robust_transform_fit, select_features
from walk_forward_rolling import PROD_SAMPLED_WINDOW

TRAIN_WINDOW = 39_480
MAX_LAG = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 multi-scale temporal LGBM screen.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="temporal_multiscale_screen")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=TRAIN_WINDOW)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--sampling", default="phase_balanced", choices=["periodic", "phase_balanced"])
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--history-count", type=int, default=40)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--lgbm-rounds", type=int, default=160)
    parser.add_argument("--lgbm-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=30_000)
    parser.add_argument("--transform-chunk", type=int, default=20_000)
    parser.add_argument("--prediction-cache", default=None,
                        help="Optional npz path for concatenated outer-fold predictions/targets/weights.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_raw_lag_cache(files: list[Path], history_columns: np.ndarray, sample_modulo: int,
                        sampling: str, expected_rows: int, batch_size: int) -> dict[str, np.ndarray]:
    """Stream every row, retain 20 raw lags only for sampled rows, and avoid concatenate spikes."""
    names = [FEATURE_COLUMNS[int(i)] for i in history_columns]
    columns = ["time_id", "asset_id", *names]
    width = len(names)
    lags_out = np.empty((expected_rows, MAX_LAG, width), dtype=np.float32)
    counts_out = np.empty(expected_rows, dtype=np.int16)
    gap_out = np.empty(expected_rows, dtype=np.float32)
    tid_out = np.empty(expected_rows, dtype=np.int64)
    aid_out = np.empty(expected_rows, dtype=np.int64)
    buffers: dict[int, np.ndarray] = {}
    last_time: dict[int, int] = {}
    cursor = 0

    for path in files:
        kept, started = 0, time.perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            current = frame[names].to_numpy(dtype=np.float32, copy=True)
            n = len(frame)
            lags = np.zeros((n, MAX_LAG, width), dtype=np.float32)
            counts = np.zeros(n, dtype=np.int16)
            gaps = np.zeros(n, dtype=np.float32)
            for asset in np.unique(aid):
                index = np.flatnonzero(aid == asset)
                key = int(asset)
                buffer = buffers.get(key, np.empty((0, width), dtype=np.float32))
                combined = np.vstack([buffer, current[index]])
                positions = len(buffer) + np.arange(len(index))
                for lag in range(MAX_LAG):
                    source = positions - lag - 1
                    usable = source >= 0
                    if usable.any():
                        lags[index[usable], lag] = combined[source[usable]]
                counts[index] = np.minimum(positions, MAX_LAG)
                previous_times = np.r_[last_time.get(key, tid[index[0]]), tid[index[:-1]]]
                gaps[index] = tid[index] - previous_times
                if key not in last_time:
                    gaps[index[0]] = 0.0
                buffers[key] = combined[-MAX_LAG:].astype(np.float32, copy=True)
                last_time[key] = int(tid[index[-1]])
            mask = time_sample_mask(tid, sample_modulo, sampling=sampling)
            count = int(mask.sum())
            stop = cursor + count
            if stop > expected_rows:
                raise RuntimeError("sampled lag cache exceeded expected row count")
            lags_out[cursor:stop] = lags[mask]
            counts_out[cursor:stop] = counts[mask]
            gap_out[cursor:stop] = gaps[mask]
            tid_out[cursor:stop] = tid[mask]
            aid_out[cursor:stop] = aid[mask]
            cursor = stop; kept += count
            del frame, current, lags, counts, gaps
        print(f"  lag20 {path.name}: {kept:,} sampled rows ({time.perf_counter()-started:.1f}s)",
              flush=True)
    if cursor != expected_rows:
        raise RuntimeError(f"lag cache rows {cursor:,} != sampled rows {expected_rows:,}")
    return {"lags": lags_out, "count": counts_out, "gap": gap_out,
            "time_id": tid_out, "asset_id": aid_out}


def transformed_atoms(cache: dict[str, np.ndarray], indices: np.ndarray,
                      transformed_current: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      center: np.ndarray, scale: np.ndarray, chunk: int) -> dict[str, np.ndarray]:
    """Transform raw cached lags with fold-only statistics and materialise all temporal atoms."""
    probe = MultiScaleAssetHistory(feature_count=transformed_current.shape[1]).transform(
        np.empty((0, transformed_current.shape[1]), np.float32),
        np.empty(0, np.int64), np.empty(0, np.int64))
    output = {key: np.empty((len(indices), value.shape[1]), dtype=np.float32)
              for key, value in probe.items()}
    for start in range(0, len(indices), chunk):
        stop = min(start + chunk, len(indices))
        source = indices[start:stop]
        raw_lags = cache["lags"][source].copy()
        counts = cache["count"][source]
        for lag in range(MAX_LAG):
            apply_robust_transform(raw_lags[:, lag], lower, upper, center, scale)
            raw_lags[counts <= lag, lag] = 0.0
        atoms = temporal_atoms_from_lags(transformed_current[start:stop], raw_lags, counts,
                                         cache["gap"][source])
        for key in output:
            output[key][start:stop] = atoms[key]
        del raw_lags, atoms
    return output


def paired_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    base = np.array([row["arms"]["baseline"]["peak"] for row in rows], dtype=np.float64)
    candidate = np.array([row["arms"][arm]["peak"] for row in rows], dtype=np.float64)
    delta = candidate - base
    without_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    a_base = np.array([row["arms"]["baseline"]["A"] for row in rows])
    a_cand = np.array([row["arms"][arm]["A"] for row in rows])
    b_base = np.array([row["arms"]["baseline"]["B"] for row in rows])
    b_cand = np.array([row["arms"][arm]["B"] for row in rows])
    da = float(np.mean(a_cand / a_base - 1.0))
    db = float(np.mean(b_cand / b_base - 1.0))
    relative = float(delta.mean() / base.mean())
    checks = {
        "mean_delta_positive": float(delta.mean()) > 0,
        "positive_at_least_4of5_or_80pct": int((delta > 0).sum()) >= int(np.ceil(0.8 * len(delta))),
        "survives_drop_best": float(without_best.mean()) > 0,
        "relative_gain_at_least_3pct": relative >= 0.03,
        "mechanism_2dA_gt_dB": 2.0 * da > db,
    }
    return {"baseline_peak_mean": float(base.mean()), "candidate_peak_mean": float(candidate.mean()),
            "mean_delta": float(delta.mean()), "relative_gain": relative,
            "positive_folds": int((delta > 0).sum()), "n_folds": len(delta),
            "mean_delta_drop_best": float(without_best.mean()),
            "delta_A_relative": da, "delta_B_relative": db,
            "per_fold_delta": delta.tolist(), "checks": checks, "pass": all(checks.values())}


def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    if "baseline" not in args.arms:
        raise SystemExit("arms must include baseline")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.label}.json"; md_path = out / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} or {md_path} exists; pass --force")
    started = time.perf_counter()
    print(f"loading sampled rows: modulo={args.sample_modulo}, sampling={args.sampling}", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    tid = data["time_id"]; aid = data["asset_id"]
    folds = rolling_time_folds(np.unique(tid), args.n_folds, args.train_window, args.embargo)

    # Freeze the history column set on fold-0 train only; all arms/folds reuse it.
    tr0 = np.isin(tid, folds[0][0])
    transformed0, _ = robust_transform_fit(data["features"][tr0].copy())
    y0 = data["target"][tr0]
    starts0 = group_starts(tid[tr0]); counts0 = np.diff(np.r_[starts0, int(tr0.sum())])
    e0 = y0 - np.repeat(np.add.reduceat(y0, starts0) / counts0, counts0)
    pool = select_features(transformed0, e0, np.ones_like(e0), args.feature_count)
    inner = select_features(transformed0[:, pool], e0, np.ones_like(e0), args.history_count)
    history_columns = np.sort(pool[inner])
    del transformed0
    print(f"frozen temporal columns: {history_columns.tolist()}", flush=True)

    cache = build_raw_lag_cache(train_files(Path(args.data_root)), history_columns,
                                args.sample_modulo, args.sampling, len(tid), args.cache_batch_size)
    if not (np.array_equal(cache["time_id"], tid) and np.array_equal(cache["asset_id"], aid)):
        raise SystemExit("raw temporal cache is not row-aligned with sampled matrix")

    fold_rows: list[dict[str, Any]] = []
    prediction_parts: dict[str, list[np.ndarray]] = {arm: [] for arm in args.arms}
    cache_rows: list[np.ndarray] = []
    cache_targets: list[np.ndarray] = []
    cache_weights: list[np.ndarray] = []
    cache_time_ids: list[np.ndarray] = []
    cache_asset_ids: list[np.ndarray] = []
    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_start = time.perf_counter()
        tr = np.isin(tid, train_ids); va = np.isin(tid, valid_ids)
        tr_index = np.flatnonzero(tr); va_index = np.flatnonzero(va)
        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        t_valid = transform_with(data["features"][va], stats)
        y_train = data["target"][tr].astype(np.float64)
        y_valid = data["target"][va].astype(np.float64)
        w_train = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
        w_valid = np.maximum(data["weight"][va].astype(np.float64), 0.0)
        tid_train, tid_valid = tid[tr], tid[va]
        aid_train, aid_valid = aid[tr], aid[va]

        fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW
        ridge_selected = select_features(t_train, y_train, w_train, args.feature_count)
        ridge = fit_ridge(ridge_designs(t_train, tid_train, ridge_selected, None),
                          y_train, w_train, fold_alpha)
        ridge_valid = (ridge.intercept_ + ridge_designs(
            t_valid, tid_valid, ridge_selected, None) @ ridge.coef_).astype(np.float64)
        valid_starts = group_starts(tid_valid)
        valid_counts = np.diff(np.r_[valid_starts, len(tid_valid)])
        market = np.repeat(np.add.reduceat(ridge_valid, valid_starts) / valid_counts, valid_counts)
        del ridge, ridge_valid

        train_starts = group_starts(tid_train)
        train_counts = np.diff(np.r_[train_starts, len(tid_train)])
        e_train = y_train - np.repeat(np.add.reduceat(y_train, train_starts) / train_counts,
                                      train_counts)
        lgbm_selected = select_features(t_train, e_train, np.ones_like(e_train), args.feature_count)
        xs_train = cross_sectional_deviation(t_train[:, lgbm_selected].copy(), tid_train)
        xs_valid = cross_sectional_deviation(t_valid[:, lgbm_selected].copy(), tid_valid)
        lo, hi, ce, sc = (stats[key][history_columns] for key in
                          ("lower", "upper", "center", "scale"))
        atoms_train = transformed_atoms(cache, tr_index, t_train[:, history_columns],
                                        lo, hi, ce, sc, args.transform_chunk)
        atoms_valid = transformed_atoms(cache, va_index, t_valid[:, history_columns],
                                        lo, hi, ce, sc, args.transform_chunk)
        if "t4_regime" in args.arms:
            # regime 必须从训练窗之前一路推进到验证窗，不能在验证开头冷启动。
            all_regime_features = data["features"][:, history_columns].copy()
            apply_robust_transform(all_regime_features, lo, hi, ce, sc)
            regime_all = MarketRegimeHistory(len(history_columns)).transform(all_regime_features, tid)
            for key, values in regime_all.items():
                atoms_train[key] = values[tr]
                atoms_valid[key] = values[va]
            del all_regime_features, regime_all
        row: dict[str, Any] = {"fold": fold_index, "train_rows": int(tr.sum()),
                               "valid_rows": int(va.sum()), "arms": {}}
        min_data = max(20, int(round(LGBM_MIN_DATA_FRAC * int(tr.sum()))))
        for arm in args.arms:
            design_train = np.ascontiguousarray(np.column_stack(
                [xs_train, *temporal_arm_blocks(atoms_train, arm), aid_train.astype(np.float32)]))
            design_valid = np.ascontiguousarray(np.column_stack(
                [xs_valid, *temporal_arm_blocks(atoms_valid, arm), aid_valid.astype(np.float32)]))
            cat = design_train.shape[1] - 1
            e_hat = np.zeros(len(design_valid), dtype=np.float64)
            for seed_offset in range(args.lgbm_seeds):
                seed = args.seed + seed_offset
                params = {**LGBM_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                          "num_threads": args.num_threads, "min_data_in_leaf": min_data,
                          "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                          "force_row_wise": True, "feature_pre_filter": False,
                          "seed": seed, "bagging_seed": seed + 1000,
                          "feature_fraction_seed": seed + 2000}
                dataset = lgb.Dataset(design_train, label=e_train, params=params,
                                      categorical_feature=[cat], free_raw_data=False)
                booster = lgb.train(params, dataset, num_boost_round=args.lgbm_rounds)
                e_hat += booster.predict(design_valid, num_iteration=args.lgbm_rounds)
                del dataset, booster
            e_hat /= args.lgbm_seeds
            e_hat -= np.repeat(np.add.reduceat(e_hat, valid_starts) / valid_counts, valid_counts)
            full = market + e_hat
            metric = scale_invariant_score(y_valid, full, w_valid)
            metric["design_columns"] = int(design_train.shape[1])
            row["arms"][arm] = metric
            prediction_parts[arm].append(full.astype(np.float32))
            print(f"  fold {fold_index} {arm}: peak={metric['peak']:.8f} "
                  f"A={metric['A']:.6g} B={metric['B']:.6g}", flush=True)
            del design_train, design_valid, e_hat, full
            gc.collect()
        cache_rows.append(va_index.astype(np.int64))
        cache_targets.append(y_valid.astype(np.float32))
        cache_weights.append(w_valid.astype(np.float32))
        cache_time_ids.append(tid_valid.astype(np.int64))
        cache_asset_ids.append(aid_valid.astype(np.int16))
        row["elapsed_seconds"] = time.perf_counter() - fold_start
        fold_rows.append(row)
        del t_train, t_valid, atoms_train, atoms_valid, xs_train, xs_valid
        gc.collect()
        print(f"fold {fold_index} done in {row['elapsed_seconds']:.1f}s", flush=True)

    if args.prediction_cache:
        cache_path = Path(args.prediction_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload_cache = {
            "row_index": np.concatenate(cache_rows), "target": np.concatenate(cache_targets),
            "weight": np.concatenate(cache_weights), "time_id": np.concatenate(cache_time_ids),
            "asset_id": np.concatenate(cache_asset_ids),
        }
        payload_cache.update({f"prediction_{arm}": np.concatenate(parts)
                              for arm, parts in prediction_parts.items()})
        np.savez_compressed(cache_path, **payload_cache)
        print(f"prediction cache: {cache_path}", flush=True)

    comparisons = {arm: paired_summary(fold_rows, arm) for arm in args.arms if arm != "baseline"}
    passed = [arm for arm, value in comparisons.items() if value["pass"]]
    payload = {
        "question": "Do pre-registered multi-scale temporal states beat the validated v3 history block?",
        "configuration": vars(args),
        "history_columns": history_columns.tolist(),
        "arm_definitions": {
            "baseline": "lag1+difference+mean5+deviation5",
            "t1_lags": "baseline+lag2+lag5",
            "t2_state": "baseline+EWM3/EWM10/std5/std20/slope5/slope20",
            "t3_full": "T1+T2+observation_gap",
            "t4_regime": "baseline+20D current compressed regime+lag1+difference",
        },
        "folds": fold_rows, "comparisons": comparisons,
        "verdict": {"passed_arms": passed, "enter_confirmation": bool(passed),
                    "next": "3-seed/480-round confirmation" if passed else "stop temporal expansion"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    lines = [f"# 多尺度时间状态筛选（`{args.label}`）", "",
             f"配置：{args.lgbm_seeds} seed × {args.lgbm_rounds} rounds，{args.n_folds} folds，"
             f"modulo {args.sample_modulo}/{args.sampling}。", "",
             "| arm | baseline peak | candidate peak | relative | +folds | drop best | 2ΔA>ΔB | pass |",
             "|---|---:|---:|---:|---:|---:|:---:|:---:|"]
    for arm, result in comparisons.items():
        lines.append(f"| {arm} | {result['baseline_peak_mean']:.8f} | "
                     f"{result['candidate_peak_mean']:.8f} | {result['relative_gain']*100:+.2f}% | "
                     f"{result['positive_folds']}/{result['n_folds']} | "
                     f"{result['mean_delta_drop_best']:.3g} | "
                     f"{'✅' if result['checks']['mechanism_2dA_gt_dB'] else '❌'} | "
                     f"{'✅' if result['pass'] else '❌'} |")
    lines += ["", f"**{payload['verdict']['next']}**", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"reports: {json_path}, {md_path}")


if __name__ == "__main__":
    main()
