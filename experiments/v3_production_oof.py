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
from lgbm_xs import load_rows, load_rows_disk_backed
from history_peak import fit_ridge, ridge_designs
from mt_predictability import group_starts
from src.metric import scale_invariant_score, weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import robust_transform_fit, select_features
from strategies.v3_hybrid.train import stream_history_blocks, stream_history_range_blocks

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
    p.add_argument("--train-time-id-max", type=int, default=None,
                   help="把每折训练段的**后端**封顶在这个 real time_id（含）。"
                        "验证段一个不动 ⟹ 与不封顶的臂逐位配对。"
                        "\n⚠️ 2026-08-24 新增，用来问「多给一段更近的训练数据值多少」："
                        "扩展数据 888,480–1,045,889 落在原 OOF 折的**验证段之后**，"
                        "塞进那些折的训练段就是拿未来训练 ⟹ 预注册的 D2b 在因果上不成立。"
                        "合法问法是把验证段挪到新数据**之后**，两臂只差训练段后端封顶。"
                        "\n⚠️ 与 --train-truncate 正交：那个砍前端（recency），这个封后端（新鲜度）。")
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
    p.add_argument("--phase-feature", action="store_true",
                   help="将 time_id %% 10 作为第二个 categorical 输入（只影响实验，不改生产模型）")
    p.add_argument("--fold-grid", default=None,
                   help="JSON fold grid with train_time_range/valid_time_range;用于不同 sampling/modulo 间保持真实时间边界一致")
    p.add_argument("--disk-cache", default=None,
                   help="将输入行写入 disk-backed memmap，适合 full-resolution 长任务，避免内存中 list+concat 峰值")
    p.add_argument("--design-cache-dir", default=None,
                   help="将超大 XS/market 设计矩阵写入临时 memmap；每片森林训练完成后立即删除")
    p.add_argument("--fixed-production-features", default=None,
                   help="使用 hybrid_meta.json 的 200 个生产特征和固定统计量；仅用于 full-resolution 资源 smoke/筛选，不能替代最终 fold-local 选列")
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    # ⚠️ 选列宽度：三块各一个开关。此前是同一个模块常量 FEATURE_COUNT 驱动全部，
    # 而 xs_selected **同时喂截面块和市场块** ⟹ 动一个数等于同时改两块 = 组合臂，
    # 违反 CLAUDE.md §5.2「一次只回答一个问题」。默认值全部保持 200，行为逐位不变。
    p.add_argument("--ridge-feature-count", type=int, default=FEATURE_COUNT,
                   help="Ridge 市场块的选列宽度（生产 Ridge 已冻结，一般不动）")
    p.add_argument("--xs-feature-count", type=int, default=FEATURE_COUNT,
                   help="截面 LGBM 块的选列宽度；323 = 不筛选")
    p.add_argument("--market-feature-count", type=int, default=FEATURE_COUNT,
                   help="市场 LGBM 块（raw ‖ xs_dev 两段）的选列宽度；323 = 不筛选")
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
                     min_data_rows: int | None = None,
                     categorical_columns: list[int] | None = None) -> np.ndarray:
    import lightgbm as lgb

    rounds = args.num_iteration if num_iteration is None else num_iteration
    cat = [design_train.shape[1] - 1] if categorical_columns is None else list(categorical_columns)
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
                              categorical_feature=cat, free_raw_data=False)
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
                                  min_data_rows: int | None = None,
                                  categorical_columns: list[int] | None = None) -> dict[int, np.ndarray]:
    import lightgbm as lgb

    checkpoints = sorted(set(int(v) for v in checkpoints))
    max_rounds = max(checkpoints)
    cat = [design_train.shape[1] - 1] if categorical_columns is None else list(categorical_columns)
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
                              categorical_feature=cat, free_raw_data=False)
        booster = lgb.train(params, dataset, num_boost_round=max_rounds)
        for checkpoint in checkpoints:
            result[checkpoint] += booster.predict(design_valid, num_iteration=checkpoint)
        del booster, dataset
    print(f"    {prefix}: {len(design_train):,} rows × {design_train.shape[1]} cols, "
          f"{args.n_seeds} seed(s) × max {max_rounds} rounds, checkpoints={checkpoints}, "
          f"min_leaf={min_data:,}", flush=True)
    return {v: pred / args.n_seeds for v, pred in result.items()}

def stack_design(blocks: list[np.ndarray], path: Path | None = None) -> np.ndarray:
    """Column-stack blocks in RAM or into a fixed-size disk-backed memmap."""
    normalised = [np.asarray(block).reshape(len(block), -1) for block in blocks]
    rows = len(normalised[0])
    if any(len(block) != rows for block in normalised):
        raise ValueError("design blocks have inconsistent row counts")
    columns = sum(block.shape[1] for block in normalised)
    if path is None:
        return np.ascontiguousarray(np.column_stack(normalised), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    out = np.memmap(path, dtype=np.float32, mode="w+", shape=(rows, columns))
    cursor = 0
    for block in normalised:
        stop = cursor + block.shape[1]
        out[:, cursor:stop] = block
        cursor = stop
    out.flush()
    return out


def close_design(array: np.ndarray, path: Path | None) -> None:
    if path is None:
        return
    if isinstance(array, np.memmap):
        array.flush()
        array._mmap.close()
    path.unlink(missing_ok=True)


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
    fixed_meta = None
    fixed_indices = None
    if args.fixed_production_features:
        fixed_meta = json.loads(Path(args.fixed_production_features).read_text(encoding="utf-8"))
        fixed_names = list(fixed_meta["lgbm_features"])
        fixed_indices = np.array([int(name.split("_")[1]) for name in fixed_names], dtype=np.int64)
        if len(fixed_indices) != 200 or len(np.unique(fixed_indices)) != 200:
            raise SystemExit("fixed production feature meta must contain 200 unique lgbm_features")
    if args.disk_cache:
        data = load_rows_disk_backed(Path(args.data_root), args.sample_modulo, args.sampling,
                                     Path(args.disk_cache), feature_indices=fixed_indices)
    else:
        if fixed_indices is not None:
            raise SystemExit("--fixed-production-features requires --disk-cache to avoid full-schema materialization")
        data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features = data["features"]
    target = data["target"].astype(np.float64, copy=False)
    weight = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    time_ids = data["time_id"]
    asset_ids = data["asset_id"]
    phase_ids = (time_ids % 10).astype(np.float32)
    if not np.all(np.diff(time_ids) >= 0):
        raise AssertionError("sampled rows must be sorted by time_id")
    unique_time_ids = np.unique(time_ids)
    if args.fold_grid:
        grid_payload = json.loads(Path(args.fold_grid).read_text(encoding="utf-8"))
        folds = []
        for item in grid_payload["folds"]:
            train_lo, train_hi = map(int, item["train_time_range"])
            valid_lo, valid_hi = map(int, item["valid_time_range"])
            train_ids = unique_time_ids[(unique_time_ids >= train_lo) & (unique_time_ids <= train_hi)]
            valid_ids = unique_time_ids[(unique_time_ids >= valid_lo) & (unique_time_ids <= valid_hi)]
            if len(train_ids) == 0 or len(valid_ids) == 0:
                raise SystemExit(f"fold grid range not covered by loaded data: {item}")
            folds.append((train_ids, valid_ids))
        print(f"{len(features):,} rows / {len(unique_time_ids):,} time_ids / "
              f"{len(folds)} folds from real-time fold grid {args.fold_grid}", flush=True)
    else:
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

    history_names_per_fold: list[list[str]] = []
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
        if args.train_time_id_max is not None:
            kept = train_ids[train_ids <= args.train_time_id_max]
            if len(kept) == 0:
                raise SystemExit(f"fold {index}: --train-time-id-max "
                                 f"{args.train_time_id_max} 把训练段整个砍空了")
            # 只封**后端**，valid_ids 一个不动 ⟹ 各臂落在完全相同的验证行上
            if int(valid_ids[0]) <= args.train_time_id_max:
                raise SystemExit(
                    f"fold {index}: 封顶 {args.train_time_id_max} 落在验证段内部 "
                    f"（验证段起点 {int(valid_ids[0])}）⟹ 该折没有意义，拒绝继续")
            print(f"fold {index}: 训练段后端封顶 {int(train_ids[-1]):,} → {int(kept[-1]):,}"
                  f"（{len(train_ids):,} → {len(kept):,} 个 time_id；验证段不变："
                  f"{len(valid_ids):,}）", flush=True)
            train_ids = kept
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

        if fixed_meta is not None:
            stats = {key: np.asarray(fixed_meta[key], dtype=np.float32)
                     for key in ("lower", "upper", "center", "scale")}
            transformed_train = train_features
            apply_robust_transform(transformed_train, stats["lower"], stats["upper"],
                                   stats["center"], stats["scale"])
            transformed_valid = valid_features
            apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                                   stats["center"], stats["scale"])
            ridge_selected = np.arange(transformed_train.shape[1], dtype=np.int64)
        else:
            transformed_train, stats = robust_transform_fit(train_features)
            transformed_valid = valid_features
            apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                                   stats["center"], stats["scale"])
            ridge_selected = select_features(transformed_train, y_tr, w_tr, args.ridge_feature_count)
        ridge_train_design = ridge_designs(transformed_train, tid_tr, ridge_selected, None)
        ridge_valid_design = ridge_designs(transformed_valid, tid_va, ridge_selected, None)
        fold_alpha = RIDGE_ALPHA * len(train_ids) / REFERENCE_TRAIN_WINDOW
        ridge = fit_ridge(ridge_train_design, y_tr, w_tr, fold_alpha)
        ridge_raw = ridge.predict(ridge_valid_design).astype(np.float64)
        market_ridge = group_mean(ridge_raw, va_starts, va_counts)
        del ridge_train_design, ridge_valid_design, ridge, ridge_raw

        # Cross-sectional target and selected LGBM raw features.
        e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
        # 截面块与市场块各有独立的选列宽度。两者是同一判据下的 top-N ⟹ **嵌套**，
        # 所以只算一次「较宽那份」的截面偏差再切片：截面去均值是逐列独立的，
        # 切片与先切后算逐位等价，但省掉一整个设计矩阵的分配（OOM 事故的教训）。
        wide_count = max(args.xs_feature_count, args.market_feature_count)
        unit = np.ones_like(e_tr)
        wide_selected = (np.arange(transformed_train.shape[1], dtype=np.int64)
                         if fixed_meta is not None else select_features(transformed_train, e_tr, unit, wide_count))
        xs_selected = (wide_selected if fixed_meta is not None or args.xs_feature_count == wide_count
                       else select_features(transformed_train, e_tr, unit, args.xs_feature_count))
        market_selected = (wide_selected if fixed_meta is not None or args.market_feature_count == wide_count
                           else select_features(transformed_train, e_tr, unit,
                                                args.market_feature_count))
        if not (np.isin(xs_selected, wide_selected).all()
                and np.isin(market_selected, wide_selected).all()):
            raise AssertionError("选列不嵌套 ⟹ 切片路径失效")
        wide_dev_tr = cross_sectional_deviation(transformed_train[:, wide_selected].copy(), tid_tr)
        wide_dev_va = cross_sectional_deviation(transformed_valid[:, wide_selected].copy(), tid_va)
        xs_in_wide = np.searchsorted(wide_selected, xs_selected)
        market_in_wide = np.searchsorted(wide_selected, market_selected)
        xs_tr = wide_dev_tr if args.xs_feature_count == wide_count else wide_dev_tr[:, xs_in_wide]
        xs_va = wide_dev_va if args.xs_feature_count == wide_count else wide_dev_va[:, xs_in_wide]
        market_dev_tr = (wide_dev_tr if args.market_feature_count == wide_count
                         else wide_dev_tr[:, market_in_wide])
        market_dev_va = (wide_dev_va if args.market_feature_count == wide_count
                         else wide_dev_va[:, market_in_wide])
        history_positions = (np.asarray(fixed_meta["history_positions"], dtype=np.int64)
                             if fixed_meta is not None else
                             select_features(xs_tr, e_tr, np.ones_like(e_tr), HISTORY_COUNT))
        history_positions = np.sort(history_positions.astype(np.int64))
        if fixed_meta is not None:
            fixed_names = list(fixed_meta["lgbm_features"])
            if len(fixed_names) != len(stats["center"]):
                raise SystemExit("fixed meta feature/stat length mismatch")
            if np.any(history_positions < 0) or np.any(history_positions >= len(fixed_names)):
                raise SystemExit("fixed meta history_positions out of range")
            history_names = [fixed_names[int(pos)] for pos in history_positions]
        else:
            history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
        history_names_per_fold.append(list(history_names))
        history_stats = tuple(stats[key][xs_selected[history_positions]]
                              for key in ("lower", "upper", "center", "scale"))
        print(f"fold {index}: train {len(y_tr):,}, valid {len(y_va):,}, "
              f"history={len(history_names)}; streaming causal history", flush=True)
        if args.design_cache_dir:
            history_tr, history_va = stream_history_range_blocks(
                Path(args.data_root), history_names, history_stats, args.history_window,
                (int(train_ids[0]), int(train_ids[-1])),
                (int(valid_ids[0]), int(valid_ids[-1])))
            if len(history_tr[0]) != len(y_tr) or len(history_va[0]) != len(y_va):
                raise AssertionError("range history is not aligned with fold rows")
        else:
            all_history = stream_history_blocks(
                Path(args.data_root), args.sample_modulo, args.sampling, history_names,
                history_stats, args.history_window)
            history_tr = [block[tr] for block in all_history]
            history_va = [block[va] for block in all_history]
            del all_history

        design_dir = Path(args.design_cache_dir) if args.design_cache_dir else None
        xs_train_path = design_dir / f"fold{index}_xs_train.f32" if design_dir else None
        xs_valid_path = design_dir / f"fold{index}_xs_valid.f32" if design_dir else None
        xs_columns = [xs_tr, *history_tr]
        xs_columns += ([phase_ids[tr].astype(np.float32)] if args.phase_feature else [])
        xs_columns += [aid_tr.astype(np.float32)]
        xs_valid_columns = [xs_va, *history_va]
        xs_valid_columns += ([phase_ids[va].astype(np.float32)] if args.phase_feature else [])
        xs_valid_columns += [aid_va.astype(np.float32)]
        d_tr_xs = stack_design(xs_columns, xs_train_path)
        d_va_xs = stack_design(xs_valid_columns, xs_valid_path)
        xs_cats = ([d_tr_xs.shape[1] - 2, d_tr_xs.shape[1] - 1]
                   if args.phase_feature else [d_tr_xs.shape[1] - 1])
        e_pred = fit_predict_lgbm(d_tr_xs, e_tr, w_tr, d_va_xs, args, "cross", XS_SPEC,
                                   num_iteration=cross_rounds, min_data_rows=frozen_rows,
                                   categorical_columns=xs_cats)
        e_lgbm = e_pred - group_mean(e_pred, va_starts, va_counts)
        close_design(d_tr_xs, xs_train_path)
        close_design(d_va_xs, xs_valid_path)
        del d_tr_xs, d_va_xs
        gc.collect()

        market_train_path = design_dir / f"fold{index}_market_train.f32" if design_dir else None
        market_valid_path = design_dir / f"fold{index}_market_valid.f32" if design_dir else None
        market_columns = [transformed_train[:, market_selected], market_dev_tr, *history_tr]
        market_columns += ([phase_ids[tr].astype(np.float32)] if args.phase_feature else [])
        market_columns += [aid_tr.astype(np.float32)]
        market_valid_columns = [transformed_valid[:, market_selected], market_dev_va, *history_va]
        market_valid_columns += ([phase_ids[va].astype(np.float32)] if args.phase_feature else [])
        market_valid_columns += [aid_va.astype(np.float32)]
        d_tr_market = stack_design(market_columns, market_train_path)
        d_va_market = stack_design(market_valid_columns, market_valid_path)
        market_cats = ([d_tr_market.shape[1] - 2, d_tr_market.shape[1] - 1]
                      if args.phase_feature else [d_tr_market.shape[1] - 1])
        # ⚠️ 内存：market 设计矩阵是全流程最大的一次分配（fold 越大越夸张）。
        # 到这里 transformed_*/xs_*/history_* 都已经并进设计矩阵、不再被引用，
        # 但原本要到折末才 del —— 于是它们在**峰值时刻**白占约 6 GB。
        # 训练前先放掉，峰值直接降一大截（swap=0，没有缓冲，必须省）。
        del (train_features, valid_features, transformed_train, transformed_valid,
             xs_tr, xs_va, market_dev_tr, market_dev_va, wide_dev_tr, wide_dev_va,
             history_tr, history_va)
        gc.collect()
        market_preds = fit_predict_lgbm_checkpoints(
            d_tr_market, y_tr, None, d_va_market, args, "market", MARKET_SPEC,
            MARKET_MIN_DATA_SCALE, checkpoints, min_data_rows=frozen_rows,
            categorical_columns=market_cats)
        checkpoint_components: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for checkpoint, market_pred_at_checkpoint in market_preds.items():
            market_lgbm_at_checkpoint = group_mean(market_pred_at_checkpoint, va_starts, va_counts)
            market_at_checkpoint = ((1.0 - MARKET_LAMBDA) * market_ridge
                                    + MARKET_LAMBDA * market_lgbm_at_checkpoint)
            raw_at_checkpoint = market_at_checkpoint + e_lgbm
            checkpoint_components[checkpoint] = (market_lgbm_at_checkpoint,
                                                 market_at_checkpoint, raw_at_checkpoint)
        close_design(d_tr_market, market_train_path)
        close_design(d_va_market, market_valid_path)
        del d_tr_market, d_va_market
        gc.collect()
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
        del (stats, e_tr, e_pred, market_preds, prediction)
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
            "expanding_train", "expanding_cap", "train_time_id_max",
            "phase_feature", "fold_grid", "disk_cache",
            "design_cache_dir", "fixed_production_features")},
        "rounds": {"cross": int(cross_rounds), "market": int(market_rounds),
                   "market_checkpoints": checkpoints},
        "architecture": {
            "ridge_market": True, "xs_lgbm_weighted": True, "market_lgbm_weighted": False,
            "ridge_feature_count": args.ridge_feature_count,
            "xs_feature_count": args.xs_feature_count,
            "market_feature_count": args.market_feature_count,
            "history_count": HISTORY_COUNT,
            # 逐折的 history 原始列名 —— 用来断言「换宽度只动了 xs/market 块，history 没变」
            "history_names_per_fold": history_names_per_fold,
            "history_window": HISTORY_WINDOW, "blend_weight": 1.0,
            "phase_feature": bool(args.phase_feature),
            "fixed_production_features": args.fixed_production_features,
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
