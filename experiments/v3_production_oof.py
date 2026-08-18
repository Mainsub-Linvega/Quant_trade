"""严格 OOF：复现当前 v3_hybrid 生产架构并保存逐行组件预测。

这不是公榜提交脚本。它的目的，是把当前生产组合拆成可诊断的 OOF 组件：

    Ridge market + weighted XS LGBM + unweighted market LGBM

默认配置先采用 1 seed × 160 rounds 的 screening 档；确认残差结构稳定后，再用
3 seeds × 480 rounds 做确认。每一折的预处理、选列、history 状态和模型都只用
训练窗拟合，validation 只做 transform/predict，避免把最终全量模型的统计量泄漏进
OOF。history 特征允许使用验证日前的历史 feature 值，这是因果的，不使用 target。

输出：outputs/cache/<label>.npz 和 outputs/experiments/<label>.{json,md}
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
from history_peak import fit_ridge, ridge_designs
from mt_predictability import group_starts
from src.metric import scale_invariant_score, weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import robust_transform_fit, select_features
from strategies.v3_hybrid.train import stream_history_blocks

FEATURE_COUNT = 200
HISTORY_COUNT = 40
HISTORY_WINDOW = 5
RIDGE_ALPHA = 2_000_000.0
REFERENCE_TRAIN_WINDOW = 78_960
XS_SPEC = {"num_leaves": 63, "learning_rate": 0.03,
           "feature_fraction": 0.7, "lambda_l2": 1.0}
MARKET_SPEC = {"num_leaves": 15, "learning_rate": 0.02,
               "feature_fraction": 0.4, "lambda_l2": 30.0}
MARKET_MIN_DATA_SCALE = 25.0 / 3.0
MIN_DATA_FRAC = 12000 / 3_500_000
PREDICTION_SCALE = 1.16
PREDICTION_CLIP = 0.5
MARKET_LAMBDA = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--cache-dir", default=str(_REPO_ROOT / "outputs" / "cache"))
    p.add_argument("--label", default="v3_production_oof_phasebal_prodwindow_exact")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=REFERENCE_TRAIN_WINDOW)
    p.add_argument("--train-truncate", type=int, default=None,
                   help="P4 recency：**保持 fold 版图不变**，只把每折训练段截到最后 N 个采样 "
                        "time_id。⚠️ 不要改 --train-window 来做这件事 —— rolling_time_folds 的 "
                        "first_valid_idx = train_window + embargo，改它会把**验证段**也挪走，"
                        "各臂就落在不同数据上，配对比较失效。")
    p.add_argument("--expanding-train", action="store_true",
                   help="P4 recency 的反方向：**保持 fold 版图不变**，把每折训练段扩到 embargo "
                        "之前的全部历史（滑动窗在后面几折丢掉了本可用的数据）。"
                        "fold 0 不变、验证段不变 ⟹ 与滑动窗基准天然配对。与 --train-truncate 互斥。")
    p.add_argument("--expanding-cap", type=int, default=None,
                   help="给 --expanding-train 封顶（内存不够时用），不影响配对性。")
    p.add_argument("--freeze-min-data", action="store_true",
                   help="P4 recency：把 min_data_in_leaf 冻结在 --train-window 那档的行数上，"
                        "不随截断后的行数缩放。用来把「数据量」与「有效容量」两个混淆项拆开。")
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced")
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--num-iteration", type=int, default=160,
                   help="backward-compatible default for both forests")
    p.add_argument("--cross-num-iteration", type=int, default=None)
    p.add_argument("--market-num-iteration", type=int, default=None)
    p.add_argument("--market-checkpoints", type=int, nargs="+", default=[160, 480])
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--prediction-scale", type=float, default=PREDICTION_SCALE)
    p.add_argument("--prediction-clip", type=float, default=PREDICTION_CLIP)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))


def row_slice(time_ids: np.ndarray, ids: np.ndarray) -> slice:
    if len(ids) == 0:
        raise ValueError("empty time-id slice")
    left = int(np.searchsorted(time_ids, ids[0], side="left"))
    right = int(np.searchsorted(time_ids, ids[-1], side="right"))
    got = time_ids[left:right]
    if len(got) == 0 or got[0] != ids[0] or got[-1] != ids[-1]:
        raise AssertionError("time-id fold is not a contiguous slice")
    return slice(left, right)


def fit_predict_lgbm(design_train: np.ndarray, label: np.ndarray, weight: np.ndarray | None,
                     design_valid: np.ndarray, args: argparse.Namespace,
                     prefix: str, spec: dict[str, float], min_data_scale: float = 1.0,
                     num_iteration: int | None = None,
                     min_data_rows: int | None = None) -> np.ndarray:
    import lightgbm as lgb

    rounds = args.num_iteration if num_iteration is None else num_iteration
    cat = design_train.shape[1] - 1
    # min_data_rows 只在 P4 的「冻结容量」臂里被指定；默认仍按实际训练行数缩放。
    rows_for_min_data = len(design_train) if min_data_rows is None else min_data_rows
    min_data = max(20, int(round(MIN_DATA_FRAC * rows_for_min_data * min_data_scale)))
    result = np.zeros(len(design_valid), dtype=np.float64)
    for seed_offset in range(args.n_seeds):
        seed = args.seed + seed_offset
        params = {
            **spec, "objective": "regression", "metric": "l2", "verbosity": -1,
            "num_threads": args.num_threads, "min_data_in_leaf": min_data,
            "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
            "force_row_wise": True, "feature_pre_filter": False,
            "seed": seed, "bagging_seed": seed + 1000,
            "feature_fraction_seed": seed + 2000,
        }
        dataset = lgb.Dataset(design_train, label=label, weight=weight, params=params,
                              categorical_feature=[cat], free_raw_data=False)
        booster = lgb.train(params, dataset, num_boost_round=rounds)
        result += booster.predict(design_valid, num_iteration=rounds)
        del booster, dataset
    print(f"    {prefix}: {len(design_train):,} rows × {design_train.shape[1]} cols, "
          f"{args.n_seeds} seed(s) × {rounds} rounds, min_leaf={min_data:,}",
          flush=True)
    return result / args.n_seeds



def fit_predict_lgbm_checkpoints(design_train: np.ndarray, label: np.ndarray,
                                  weight: np.ndarray | None, design_valid: np.ndarray,
                                  args: argparse.Namespace, prefix: str, spec: dict[str, float],
                                  min_data_scale: float, checkpoints: list[int],
                                  min_data_rows: int | None = None) -> dict[int, np.ndarray]:
    import lightgbm as lgb

    checkpoints = sorted(set(int(v) for v in checkpoints))
    max_rounds = max(checkpoints)
    cat = design_train.shape[1] - 1
    rows_for_min_data = len(design_train) if min_data_rows is None else min_data_rows
    min_data = max(20, int(round(MIN_DATA_FRAC * rows_for_min_data * min_data_scale)))
    result = {v: np.zeros(len(design_valid), dtype=np.float64) for v in checkpoints}
    for seed_offset in range(args.n_seeds):
        seed = args.seed + seed_offset
        params = {
            **spec, "objective": "regression", "metric": "l2", "verbosity": -1,
            "num_threads": args.num_threads, "min_data_in_leaf": min_data,
            "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
            "force_row_wise": True, "feature_pre_filter": False,
            "seed": seed, "bagging_seed": seed + 1000,
            "feature_fraction_seed": seed + 2000,
        }
        dataset = lgb.Dataset(design_train, label=label, weight=weight, params=params,
                              categorical_feature=[cat], free_raw_data=False)
        booster = lgb.train(params, dataset, num_boost_round=max_rounds)
        for checkpoint in checkpoints:
            result[checkpoint] += booster.predict(design_valid, num_iteration=checkpoint)
        del booster, dataset
    print(f"    {prefix}: {len(design_train):,} rows × {design_train.shape[1]} cols, "
          f"{args.n_seeds} seed(s) × max {max_rounds} rounds, checkpoints={checkpoints}, "
          f"min_leaf={min_data:,}", flush=True)
    return {v: pred / args.n_seeds for v, pred in result.items()}

def metric_payload(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    peak = scale_invariant_score(target, prediction, weight)
    return {
        "score": weighted_zero_mean_r2(target, prediction, weight),
        "peak": float(peak["peak"]),
        "optimal_scale": float(peak["optimal_scale"]),
        "A": float(peak["A"]),
        "B": float(peak["B"]),
    }


def main() -> None:
    args = parse_args()
    cross_rounds = args.num_iteration if args.cross_num_iteration is None else args.cross_num_iteration
    market_rounds = args.num_iteration if args.market_num_iteration is None else args.market_num_iteration
    checkpoints = sorted(set([*args.market_checkpoints, market_rounds]))
    if cross_rounds <= 0 or market_rounds <= 0 or any(v <= 0 for v in checkpoints):
        raise SystemExit("all iteration counts must be positive")
    if args.expanding_train and args.train_truncate is not None:
        raise SystemExit("--expanding-train 与 --train-truncate 互斥")
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{args.label}.npz"
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (npz_path.exists() or json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {npz_path}; use --force to overwrite")

    # stream_history_blocks imports the v3 history module by its short name.
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")
    if v3_path not in sys.path:
        sys.path.append(v3_path)
    started = time.perf_counter()
    print(f"loading sampled data: modulo {args.sample_modulo}/{args.sampling}", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features = data["features"]
    target = data["target"].astype(np.float64, copy=False)
    weight = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    time_ids = data["time_id"]
    asset_ids = data["asset_id"]
    if not np.all(np.diff(time_ids) >= 0):
        raise AssertionError("sampled rows must be sorted by time_id")
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)
    print(f"{len(features):,} rows / {len(unique_time_ids):,} sampled time_ids / "
          f"{len(folds)} folds", flush=True)

    n = len(target)
    oof = {name: np.full(n, np.nan, dtype=np.float64) for name in (
        "prediction", "prediction_raw", "market_ridge", "market_lgbm", "market", "e_lgbm",
        "e_target", "xs_lgbm")}
    for checkpoint in checkpoints:
        oof[f"prediction_raw_checkpoint_{checkpoint}"] = np.full(n, np.nan, dtype=np.float64)
        oof[f"market_checkpoint_{checkpoint}"] = np.full(n, np.nan, dtype=np.float64)
        oof[f"market_lgbm_checkpoint_{checkpoint}"] = np.full(n, np.nan, dtype=np.float64)
    fold_id = np.full(n, -1, dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []

    for index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        full_train_ids = train_ids
        if args.expanding_train:
            # 训练段扩到 embargo 之前的全部历史；valid_ids 一个不动 ⟹ 与滑动窗基准配对
            v_start = int(np.searchsorted(unique_time_ids, valid_ids[0]))
            expanded = unique_time_ids[: max(0, v_start - args.embargo)]
            if args.expanding_cap is not None:
                expanded = expanded[-args.expanding_cap:]
            if len(expanded) < len(train_ids):
                raise AssertionError("expanding window is shorter than the sliding window")
            train_ids = expanded
            print(f"fold {index}: 训练段 {len(full_train_ids):,} → {len(train_ids):,} 个 time_id"
                  f"（+{len(train_ids)-len(full_train_ids):,}；验证段不变：{len(valid_ids):,}）",
                  flush=True)
        if args.train_truncate is not None:
            if args.train_truncate > len(train_ids):
                raise SystemExit(f"--train-truncate {args.train_truncate} 超过本折训练段 "
                                 f"{len(train_ids)} 个 time_id")
            # 只砍训练段的**前端**，valid_ids 不动 ⟹ 各臂落在完全相同的验证行上
            train_ids = train_ids[-args.train_truncate:]
            print(f"fold {index}: 训练段 {len(full_train_ids):,} → {len(train_ids):,} 个 time_id"
                  f"（验证段不变：{len(valid_ids):,}）", flush=True)
        tr = row_slice(time_ids, train_ids)
        va = row_slice(time_ids, valid_ids)
        # 冻结容量臂：min_data_in_leaf 按**未截断**训练段的行数算，不随截断缩小
        full_tr = row_slice(time_ids, full_train_ids)
        frozen_rows = (full_tr.stop - full_tr.start) if args.freeze_min_data else None
        train_features = features[tr].copy()
        valid_features = features[va].copy()
        y_tr, y_va = target[tr], target[va]
        w_tr, w_va = weight[tr], weight[va]
        tid_tr, tid_va = time_ids[tr], time_ids[va]
        aid_tr, aid_va = asset_ids[tr], asset_ids[va]
        tr_starts, va_starts = group_starts(tid_tr), group_starts(tid_va)
        tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)
        va_counts = np.diff(np.r_[va_starts, len(tid_va)]).astype(np.float64)

        transformed_train, stats = robust_transform_fit(train_features)
        transformed_valid = valid_features
        apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                               stats["center"], stats["scale"])

        # Ridge market: same 200-feature raw_dev basis and fixed production alpha.
        ridge_selected = select_features(transformed_train, y_tr, w_tr, FEATURE_COUNT)
        ridge_train_design = ridge_designs(transformed_train, tid_tr, ridge_selected, None)
        ridge_valid_design = ridge_designs(transformed_valid, tid_va, ridge_selected, None)
        fold_alpha = RIDGE_ALPHA * len(train_ids) / REFERENCE_TRAIN_WINDOW
        ridge = fit_ridge(ridge_train_design, y_tr, w_tr, fold_alpha)
        ridge_raw = ridge.predict(ridge_valid_design).astype(np.float64)
        market_ridge = group_mean(ridge_raw, va_starts, va_counts)
        del ridge_train_design, ridge_valid_design, ridge, ridge_raw

        # Cross-sectional target and selected LGBM raw features.
        e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
        xs_selected = select_features(transformed_train, e_tr, np.ones_like(e_tr), FEATURE_COUNT)
        xs_tr = cross_sectional_deviation(transformed_train[:, xs_selected].copy(), tid_tr)
        xs_va = cross_sectional_deviation(transformed_valid[:, xs_selected].copy(), tid_va)
        history_positions = select_features(xs_tr, e_tr, np.ones_like(e_tr), HISTORY_COUNT)
        history_positions = np.sort(history_positions.astype(np.int64))
        history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
        history_stats = tuple(stats[key][xs_selected[history_positions]]
                              for key in ("lower", "upper", "center", "scale"))
        print(f"fold {index}: train {len(y_tr):,}, valid {len(y_va):,}, "
              f"history={len(history_names)}; streaming causal history", flush=True)
        all_history = stream_history_blocks(
            Path(args.data_root), args.sample_modulo, args.sampling, history_names,
            history_stats, args.history_window)
        history_tr = [block[tr] for block in all_history]
        history_va = [block[va] for block in all_history]
        del all_history

        d_tr_xs = np.ascontiguousarray(np.column_stack(
            [xs_tr, *history_tr, aid_tr.astype(np.float32)]))
        d_va_xs = np.ascontiguousarray(np.column_stack(
            [xs_va, *history_va, aid_va.astype(np.float32)]))
        e_pred = fit_predict_lgbm(d_tr_xs, e_tr, w_tr, d_va_xs, args, "cross", XS_SPEC,
                                   num_iteration=cross_rounds, min_data_rows=frozen_rows)
        e_lgbm = e_pred - group_mean(e_pred, va_starts, va_counts)

        d_tr_market = np.ascontiguousarray(np.column_stack(
            [transformed_train[:, xs_selected], xs_tr, *history_tr, aid_tr.astype(np.float32)]))
        d_va_market = np.ascontiguousarray(np.column_stack(
            [transformed_valid[:, xs_selected], xs_va, *history_va, aid_va.astype(np.float32)]))
        # ⚠️ 内存：market 设计矩阵是全流程最大的一次分配（fold 越大越夸张）。
        # 到这里 transformed_*/xs_*/history_* 都已经并进设计矩阵、不再被引用，
        # 但原本要到折末才 del —— 于是它们在**峰值时刻**白占约 6 GB。
        # 训练前先放掉，峰值直接降一大截（swap=0，没有缓冲，必须省）。
        del (train_features, valid_features, transformed_train, transformed_valid,
             xs_tr, xs_va, history_tr, history_va)
        gc.collect()
        market_preds = fit_predict_lgbm_checkpoints(
            d_tr_market, y_tr, None, d_va_market, args, "market", MARKET_SPEC,
            MARKET_MIN_DATA_SCALE, checkpoints, min_data_rows=frozen_rows)
        checkpoint_components: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for checkpoint, market_pred_at_checkpoint in market_preds.items():
            market_lgbm_at_checkpoint = group_mean(market_pred_at_checkpoint, va_starts, va_counts)
            market_at_checkpoint = ((1.0 - MARKET_LAMBDA) * market_ridge
                                    + MARKET_LAMBDA * market_lgbm_at_checkpoint)
            raw_at_checkpoint = market_at_checkpoint + e_lgbm
            checkpoint_components[checkpoint] = (market_lgbm_at_checkpoint,
                                                 market_at_checkpoint, raw_at_checkpoint)
        market_lgbm, market, prediction_raw = checkpoint_components[market_rounds]
        prediction = np.clip(prediction_raw * args.prediction_scale,
                              -args.prediction_clip, args.prediction_clip)

        out_slice = np.arange(va.start, va.stop)
        for name, value in (("prediction", prediction), ("prediction_raw", prediction_raw),
                            ("market_ridge", market_ridge), ("market_lgbm", market_lgbm),
                            ("market", market), ("e_lgbm", e_lgbm),
                            ("e_target", e_tr[:0]), ("xs_lgbm", e_lgbm)):
            if name == "e_target":
                continue
            oof[name][out_slice] = value
        for checkpoint, (market_lgbm_at_checkpoint, market_at_checkpoint, raw_at_checkpoint) in checkpoint_components.items():
            oof[f"market_lgbm_checkpoint_{checkpoint}"][out_slice] = market_lgbm_at_checkpoint
            oof[f"market_checkpoint_{checkpoint}"][out_slice] = market_at_checkpoint
            oof[f"prediction_raw_checkpoint_{checkpoint}"][out_slice] = raw_at_checkpoint
        fold_id[out_slice] = index
        fold_metric = metric_payload(y_va, prediction, w_va)
        raw_metric = metric_payload(y_va, prediction_raw, w_va)
        fold_rows.append({
            "fold": index,
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "train_time_ids": int(len(train_ids)),
            "valid_time_ids": int(len(valid_ids)),
            "train_rows": int(len(y_tr)),
            "valid_rows": int(len(y_va)),
            "ridge_alpha": float(fold_alpha),
            "cross_num_iteration": int(cross_rounds),
            "market_num_iteration": int(market_rounds),
            "checkpoint_metrics": {str(cp): metric_payload(y_va, checkpoint_components[cp][2], w_va)
                                   for cp in checkpoints},
            "metric": fold_metric,
            "raw_metric": raw_metric,
            "elapsed_seconds": float(time.perf_counter() - fold_started),
            "xs_selected": [int(v) for v in xs_selected],
            "history_positions": [int(v) for v in history_positions],
            "history_features": history_names,
        })
        print(f"  fold {index}: score={fold_metric['score']:.8f}, peak={fold_metric['peak']:.8f}, "
              f"raw_peak={raw_metric['peak']:.8f}, elapsed={fold_rows[-1]['elapsed_seconds']:.0f}s",
              flush=True)
        # 上面那批已在 market 训练前提前释放，这里只收剩下的
        del (stats, e_tr, d_tr_xs, d_va_xs, d_tr_market, d_va_market,
             e_pred, market_preds, prediction)
        gc.collect()

    valid_mask = fold_id >= 0
    if not np.any(valid_mask):
        raise AssertionError("OOF coverage is empty")
    print(f"OOF coverage: {int(valid_mask.sum()):,}/{n:,} rows; "
          f"training-prefix rows without OOF: {int((~valid_mask).sum()):,}", flush=True)
    arrays = {
        "target": target, "weight": weight, "time_id": time_ids, "asset_id": asset_ids,
        "fold": fold_id, **oof,
    }
    np.savez_compressed(npz_path, **arrays)
    pooled = {name: metric_payload(target[valid_mask], oof[name][valid_mask], weight[valid_mask])
              for name in ("prediction", "prediction_raw", "market", "market_ridge",
                           "market_lgbm", "e_lgbm")}
    payload = {
        "experiment": "v3_production_oof",
        "config": {k: getattr(args, k) for k in (
            "n_folds", "train_window", "embargo", "sample_modulo", "sampling",
            "history_window", "num_iteration", "n_seeds", "seed", "num_threads", "prediction_scale",
            "prediction_clip", "train_truncate", "freeze_min_data",
            "expanding_train", "expanding_cap")},
        "rounds": {"cross": int(cross_rounds), "market": int(market_rounds),
                   "market_checkpoints": checkpoints},
        "architecture": {
            "ridge_market": True, "xs_lgbm_weighted": True, "market_lgbm_weighted": False,
            "feature_count": FEATURE_COUNT, "history_count": HISTORY_COUNT,
            "history_window": HISTORY_WINDOW, "blend_weight": 1.0,
            "market_lambda": MARKET_LAMBDA, "xs_spec": XS_SPEC,
            "market_spec": MARKET_SPEC, "market_min_data_scale": MARKET_MIN_DATA_SCALE,
        },
        "cache": str(npz_path), "folds": fold_rows, "pooled": pooled,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v3 production architecture strict OOF", "", f"Cache: `{npz_path}`", "",
             "## Configuration", "", "```json", json.dumps(payload["config"], indent=2), "```", "",
             "## Pooled metrics", "", "| Component | Score | Peak | Optimal scale |", "|---|---:|---:|---:|"]
    for name, metric in pooled.items():
        lines.append(f"| `{name}` | {metric['score']:.8f} | {metric['peak']:.8f} | "
                     f"{metric['optimal_scale']:.6f} |")
    lines += ["", "## Fold scores", "", "| Fold | Score | Peak | Raw peak |", "|---:|---:|---:|---:|"]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | {row['metric']['score']:.8f} | {row['metric']['peak']:.8f} | "
                     f"{row['raw_metric']['peak']:.8f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {npz_path}\nwrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
