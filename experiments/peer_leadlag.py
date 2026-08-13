"""2G：跨资产 lead-lag + ①类分量配比重验

## 臂：把**别的资产**的滞后信息喂给树

现在的 history 块**严格只看资产自己**（`previous/difference/rolling_mean/rolling_deviation`
全是该资产自身的历史）。「资产 j 上一期的动静能不能预测资产 i 这一期」——
这条轴项目从没碰过。

### ⚠️ 先说清楚它怎么绕开 2D 的失败模式

2D 里 `e200_xsm`（显式截面均值）**−12.35%**，机制是：那些列在同一个 time_id 内是常数，
等于给树一枚「这是哪个时段」的指纹，ΔB 爆到 +44.93%。

**peer 块同样是 time_id 内常数**，所以这个风险是真的。区别在于喂的是什么：

| | 2D `xsm` | 本实验 peer 块 |
|---|---|---|
| 内容 | 当期特征的截面**均值**（水平量） | 上一期截面**偏差**（相对量） |
| regime 漂移时 | 整体平移 ⟹ **就是 regime 指示器** | 截面去均值把平移消掉 ⟹ 不随 regime 走 |

⭐ 关键推导：若喂的是**去均值后**的 peer 滞后量，「同伴均值」不含新信息 ——
`mean_{j≠i}(dev_j) = −dev_i/(n−1)`（因为 dev 跨资产和为 0），只是自身量的重标度。
⟹ **peer 块唯一能带来的新东西就是「哪个资产」这个非对称结构**，
必须靠它与 `asset_id` 的交互才起作用。这正是 lead-lag 的形状，
也正是「不是 regime 指纹」的理由。

### 预注册

对每个采样 time_id `t`，取**上一次观测**（严格滞后）的全 15 个资产 × k 个特征，
按该折统计量变换后**跨资产去均值**，摊平成 `15k` 列广播给 `t` 的每一行。

| 臂 | k | peer 列数 | 设计总列 |
|---|--:|--:|--:|
| `baseline` | — | 0 | 361 |
| **`peer_k5`（预注册候选）** | 5 | 75 | 436 |
| `peer_k10` | 10 | 150 | 511 |

只改截面块 `ê`；市场模型与 ridge 固定为 08-13 已上线那版。
（k 是容量维度、属②类，本地量级不可信 ⟹ **候选锁 k=5**，k=10 只作梯度诊断。）

## 顺带：①类分量配比重验（纯算术，0 成本）

2F 的 `huber` 露出线索 —— 截面块自己 +4.7% 却因共用全局 scale 导致整体 −0.39%。
ROADMAP 0c 当年算出「放开配比只值 +0.02%」，但那是在**旧架构**上。
本脚本对每折直接解：

    单一 scale：  peak = (A_m+A_e)² / (B_m+B_e+2C)
    放开配比：    peak = aᵀ M⁻¹ a,  a=(A_m,A_e), M=[[B_m,C],[C,B_e]]

⚠️ 这是**本地**数，只用来判断「值不值得为它花公榜点」，不能拿本地解出的
`(c_m,c_e)` 直接上线（①类要靠公榜点解析求解，ROADMAP §5）。

用法：
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python experiments/peer_leadlag.py
输出：outputs/experiments/peer_leadlag.{json,md}
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.validation import rolling_time_folds
from features import apply_robust_transform, cross_sectional_deviation
from train import robust_transform_fit, select_features
from mt_predictability import group_starts
from ridge_data_ladder import row_level_peak
from lgbm_xs import load_rows
from history_peak import (FEATURE_COUNT, HISTORY_FEATURE_COUNT, HISTORY_WINDOW,
                          LGBM_MIN_DATA_FRAC, LGBM_ROUNDS, LGBM_SPEC, TRAIN_WINDOW,
                          build_lag_cache, fit_ridge, history_blocks, paired_stats,
                          ridge_designs, transform_with, verdict)

LAMBDA = 0.5
N_ASSETS = 15
PEER_K = {"peer_k5": 5, "peer_k10": 10}     # 预注册候选是 k=5；k=10 只作梯度诊断
ARMS = ("baseline", "peer_k5", "peer_k10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-asset lead-lag + component-ratio re-check.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="peer_leadlag")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=TRAIN_WINDOW)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=10)
    p.add_argument("--sampling", default="periodic", choices=["periodic", "phase_balanced"])
    p.add_argument("--feature-count", type=int, default=FEATURE_COUNT)
    p.add_argument("--history-count", type=int, default=HISTORY_FEATURE_COUNT)
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    p.add_argument("--lgbm-rounds", type=int, default=LGBM_ROUNDS)
    p.add_argument("--lgbm-seeds", type=int, default=1)
    p.add_argument("--lgbm-seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_mean(values, starts, counts):
    return np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))


def build_peer_cache(files: list[Path], peer_columns: np.ndarray, sample_modulo: int,
                     sampling: str) -> dict[str, np.ndarray]:
    """对每个**采样到的** time_id，快照「此刻为止每个资产的最近一次观测」（**原始值**）。

    严格滞后：快照在处理该 time_id 自己的行**之前**取 ⟹ 绝不含当期信息。
    缓存原始值（不含任何拟合量）⟹ 无泄漏，逐折再套该折的变换统计量。

    某个资产若从未出现过，用 NaN 占位；`apply_robust_transform` 会把它映成 0
    （与「无历史即 0」同口径）。
    """
    names = [FEATURE_COLUMNS[i] for i in peer_columns]
    k = len(peer_columns)
    last = np.full((N_ASSETS, k), np.nan, dtype=np.float32)
    snapshots: list[np.ndarray] = []
    kept_time_ids: list[int] = []
    previous_max = -1

    for path in files:
        started = time.perf_counter()
        table = pq.ParquetFile(path).read(columns=["time_id", "asset_id", *names]).to_pandas()
        tid = table["time_id"].to_numpy(dtype=np.int64)
        aid = table["asset_id"].to_numpy(dtype=np.int64)
        values = table.loc[:, names].to_numpy(dtype=np.float32, copy=True)
        del table
        assert int(tid[0]) > previous_max, "time_id 跨分区了 —— 快照会重复取，必须先处理"
        previous_max = int(tid[-1])
        starts = group_starts(tid)
        stops = np.r_[starts[1:], len(tid)]
        keep = time_sample_mask(tid[starts], sample_modulo, sampling=sampling)
        for a, b, wanted in zip(starts, stops, keep):
            if wanted:                                  # ⚠️ 先快照、后更新 = 严格滞后
                snapshots.append(last.copy())
                kept_time_ids.append(int(tid[a]))
            last[aid[a:b]] = values[a:b]
        print(f"  peer cache {path.name}: 累计 {len(snapshots):,} 个 time_id "
              f"({time.perf_counter()-started:.0f}s)", flush=True)
        del tid, aid, values
        gc.collect()
    return {"snapshot": np.stack(snapshots), "time_id": np.asarray(kept_time_ids, dtype=np.int64)}


def peer_block(snapshots: np.ndarray, stats: dict[str, np.ndarray], peer_columns: np.ndarray,
               counts: np.ndarray) -> np.ndarray:
    """(n_time_ids, 15, k) 原始快照 → 变换 → 跨资产去均值 → 摊平 → 按行广播。"""
    n_time_ids, n_assets, k = snapshots.shape
    flat = snapshots.reshape(-1, k).copy()
    apply_robust_transform(flat, *(stats[key][peer_columns] for key in
                                   ("lower", "upper", "center", "scale")))
    block = flat.reshape(n_time_ids, n_assets, k)
    block -= block.mean(axis=1, keepdims=True)          # ⭐ 跨资产去均值：抹掉 regime 水平
    return np.repeat(block.reshape(n_time_ids, n_assets * k), counts.astype(int), axis=0)


def fit_predict(design_tr, label, weight, design_va, args) -> np.ndarray:
    import lightgbm as lgb

    cat = design_tr.shape[1] - 1
    min_data = max(20, int(round(LGBM_MIN_DATA_FRAC * len(design_tr))))
    out = np.zeros(len(design_va), dtype=np.float64)
    for s in range(args.lgbm_seeds):
        seed = args.lgbm_seed + s
        params = {**LGBM_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                  "num_threads": args.num_threads, "min_data_in_leaf": min_data,
                  "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                  "force_row_wise": True, "feature_pre_filter": False,
                  "seed": seed, "bagging_seed": seed + 1000,
                  "feature_fraction_seed": seed + 2000}
        ds = lgb.Dataset(design_tr, label=label, weight=weight, params=params,
                         categorical_feature=[cat], free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=args.lgbm_rounds)
        out += booster.predict(design_va, num_iteration=args.lgbm_rounds).astype(np.float64)
        del ds, booster
    return out / args.lgbm_seeds


def _moments(y, m_hat, e_hat, w) -> tuple[float, ...]:
    denominator = float(np.dot(w, y * y))
    return (float(np.dot(w, y * m_hat)) / denominator,      # A_m
            float(np.dot(w, y * e_hat)) / denominator,      # A_e
            float(np.dot(w, m_hat * m_hat)) / denominator,  # B_m
            float(np.dot(w, e_hat * e_hat)) / denominator,  # B_e
            float(np.dot(w, m_hat * e_hat)) / denominator)  # C


def component_split(y, m_hat, e_hat, w, time_ids) -> dict[str, float]:
    """「放开分量配比」相对「单一 scale」值多少 —— **样本外**口径。

    ⚠️ 早先那版是在验证折**自己**身上解最优 `(c_m,c_e)`：那是样本内最优、
    按构造恒 ≥ 单一 scale，是上界不是收益。而且 `m̂` 在 time_id 内是常数 ⟹
    市场分量的有效样本数是 **time_id 数**（几千量级），零假设下多一个自由参数
    就能涨约 `1/n_time_ids ÷ peak` ≈ 10% 量级 —— 那个数完全落在噪声里。

    改成：验证折按时间切两半，**前半解系数、后半评分**。
    同时报样本内上界作对照，两者差多少就是那次的乐观量。
    """
    unique = np.unique(time_ids)
    cut = unique[len(unique) // 2]
    first, second = time_ids < cut, time_ids >= cut

    def solve(mask):
        a_m, a_e, b_m, b_e, cross = _moments(y[mask], m_hat[mask], e_hat[mask], w[mask])
        gram = np.array([[b_m, cross], [cross, b_e]], dtype=np.float64)
        vector = np.array([a_m, a_e], dtype=np.float64)
        free = np.linalg.solve(gram, vector)
        single = (a_m + a_e) / (b_m + b_e + 2 * cross)
        return free, float(single)

    def score(mask, c_m, c_e):
        a_m, a_e, b_m, b_e, cross = _moments(y[mask], m_hat[mask], e_hat[mask], w[mask])
        return float(2 * (c_m * a_m + c_e * a_e)
                     - (c_m * c_m * b_m + c_e * c_e * b_e + 2 * c_m * c_e * cross))

    free_fit, single_fit = solve(first)
    out_free = score(second, float(free_fit[0]), float(free_fit[1]))
    out_single = score(second, single_fit, single_fit)
    free_all, single_all = solve(None if False else np.ones(len(y), bool))
    a_m, a_e, b_m, b_e, cross = _moments(y, m_hat, e_hat, w)
    in_sample_free = float(np.array([a_m, a_e]) @ free_all)
    in_sample_single = (a_m + a_e) ** 2 / (b_m + b_e + 2 * cross)
    return {"A_m": a_m, "A_e": a_e, "B_m": b_m, "B_e": b_e, "cross": cross,
            "c_market": float(free_all[0]), "c_cross_section": float(free_all[1]),
            "in_sample_single": in_sample_single, "in_sample_free": in_sample_free,
            "in_sample_gain": in_sample_free / in_sample_single - 1.0,
            "holdout_single": out_single, "holdout_free": out_free,
            "holdout_gain": (out_free / out_single - 1.0) if out_single > 0 else float("nan"),
            "c_market_first_half": float(free_fit[0]),
            "c_cross_section_first_half": float(free_fit[1])}


def run_fold(index, data, cache, peer, train_ids, valid_ids, hist_cols, peer_cols,
             per_arm_cols, args) -> dict[str, Any]:
    started = time.perf_counter()
    tr = np.isin(data["time_id"], train_ids)
    va = np.isin(data["time_id"], valid_ids)

    t_train, stats = robust_transform_fit(data["features"][tr].copy())
    t_valid = transform_with(data["features"][va], stats)
    y_tr = data["target"][tr].astype(np.float64)
    y_va = data["target"][va].astype(np.float64)
    w_tr = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
    w_va = np.maximum(data["weight"][va].astype(np.float64), 0.0)
    tid_tr, tid_va = data["time_id"][tr], data["time_id"][va]
    aid_tr, aid_va = data["asset_id"][tr], data["asset_id"][va]
    tr_starts, va_starts = group_starts(tid_tr), group_starts(tid_va)
    tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)
    va_counts = np.diff(np.r_[va_starts, len(tid_va)]).astype(np.float64)

    fold_alpha = args.ridge_alpha * len(train_ids) / 39_480
    ridge_selected = select_features(t_train, y_tr, w_tr, args.feature_count)
    est = fit_ridge(ridge_designs(t_train, tid_tr, ridge_selected, None), y_tr, w_tr, fold_alpha)
    m_ridge = group_mean((est.intercept_ + ridge_designs(t_valid, tid_va, ridge_selected, None)
                          @ est.coef_).astype(np.float64), va_starts, va_counts)
    del est

    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    lgbm_selected = select_features(t_train, e_tr, np.ones_like(e_tr), args.feature_count)
    xs_tr = cross_sectional_deviation(t_train[:, lgbm_selected].copy(), tid_tr)
    xs_va = cross_sectional_deviation(t_valid[:, lgbm_selected].copy(), tid_va)
    lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
    hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
    hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

    aid_col_tr = aid_tr.astype(np.float32)[:, None]
    aid_col_va = aid_va.astype(np.float32)[:, None]
    body_tr, body_va = [xs_tr, *hist_tr], [xs_va, *hist_va]

    m_lgbm = group_mean(fit_predict(
        np.ascontiguousarray(np.column_stack([t_train[:, lgbm_selected], *body_tr, aid_col_tr])),
        y_tr, None,
        np.ascontiguousarray(np.column_stack([t_valid[:, lgbm_selected], *body_va, aid_col_va])),
        args), va_starts, va_counts)
    market = (1.0 - LAMBDA) * m_ridge + LAMBDA * m_lgbm

    # peer 块：快照按 time_id 对齐到折内的行
    tr_time_mask = np.isin(peer["time_id"], train_ids)
    va_time_mask = np.isin(peer["time_id"], valid_ids)

    row: dict[str, Any] = {"fold": index, "n_valid_rows": int(va.sum()), "arms": {}}
    for arm in args.arms:
        if arm == "baseline":
            extra_tr, extra_va = [], []
        else:
            columns = per_arm_cols[arm]
            positions = np.searchsorted(peer_cols, columns)   # 映射到缓存的列轴
            extra_tr = [peer_block(peer["snapshot"][tr_time_mask][:, :, positions], stats,
                                   columns, tr_counts)]
            extra_va = [peer_block(peer["snapshot"][va_time_mask][:, :, positions], stats,
                                   columns, va_counts)]
        d_tr = np.ascontiguousarray(np.column_stack([*body_tr, *extra_tr, aid_col_tr]))
        d_va = np.ascontiguousarray(np.column_stack([*body_va, *extra_va, aid_col_va]))
        e_hat = fit_predict(d_tr, e_tr, w_tr, d_va, args)
        e_hat -= group_mean(e_hat, va_starts, va_counts)
        entry: dict[str, Any] = {"design_columns": int(d_tr.shape[1]),
                                 "full": row_level_peak(y_va, market + e_hat, w_va)}
        if arm == "baseline":       # ①类分量配比只在基准臂上量一次
            entry["components"] = component_split(y_va, market, e_hat, w_va, tid_va)
        row["arms"][arm] = entry
        del d_tr, d_va, extra_tr, extra_va
        gc.collect()

    base = row["arms"]["baseline"]["full"]["peak"]
    detail = "  ".join(f"{a} {(row['arms'][a]['full']['peak']/base-1)*100:+.2f}%" for a in args.arms)
    comp = row["arms"]["baseline"]["components"]
    print(f"  fold {index}: base {base:.8f} | {detail} | 放开配比 样本外 "
          f"{comp['holdout_gain']*100:+.3f}% / 样本内 {comp['in_sample_gain']*100:+.3f}%  "
          f"(c_m={comp['c_market']:.3f}, c_e={comp['c_cross_section']:.3f})  "
          f"{time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, body_tr, body_va, hist_tr, hist_va, xs_tr, xs_va
    gc.collect()
    return row


def summarise(rows, args) -> dict[str, Any]:
    baseline = np.array([r["arms"]["baseline"]["full"]["peak"] for r in rows])
    base_a = float(np.mean([r["arms"]["baseline"]["full"]["A"] for r in rows]))
    base_b = float(np.mean([r["arms"]["baseline"]["full"]["B"] for r in rows]))
    out = {}
    for arm in args.arms:
        if arm == "baseline":
            continue
        peaks = np.array([r["arms"][arm]["full"]["peak"] for r in rows])
        stats = paired_stats(peaks - baseline, baseline)
        d_a = float(np.mean([r["arms"][arm]["full"]["A"] for r in rows])) / base_a - 1.0
        d_b = float(np.mean([r["arms"][arm]["full"]["B"] for r in rows])) / base_b - 1.0
        out[arm] = {"stats": stats, "verdict": verdict(stats), "delta_A": d_a, "delta_B": d_b,
                    "mechanism_2dA_gt_dB": 2 * d_a > d_b}
    gains = np.array([r["arms"]["baseline"]["components"]["holdout_gain"] for r in rows])
    in_sample = np.array([r["arms"]["baseline"]["components"]["in_sample_gain"] for r in rows])
    return {"baseline_peak_mean": float(baseline.mean()), "arms": out,
            "component_ratio": {
                "holdout_gain_per_fold": [float(g) for g in gains],
                "holdout_gain_mean": float(np.nanmean(gains)),
                "in_sample_gain_per_fold": [float(g) for g in in_sample],
                "in_sample_gain_mean": float(in_sample.mean()),
                "c_market_per_fold": [r["arms"]["baseline"]["components"]["c_market"] for r in rows],
                "c_cross_section_per_fold":
                    [r["arms"]["baseline"]["components"]["c_cross_section"] for r in rows]}}


def render_report(payload) -> str:
    s = payload["summary"]
    c = s["component_ratio"]
    lines = [
        "# 2G：跨资产 lead-lag + ①类分量配比重验",
        "",
        "peer 块 = 上一期全 15 资产 × k 特征，变换后**跨资产去均值**，摊平广播给该 time_id 每一行。",
        "⭐ 喂**去均值后的相对量**而非水平量，是为了绕开 2D 里 `xsm` 那个 regime 指纹失败模式。",
        "",
        f"- ⭐ 基准强度：peak **{s['baseline_peak_mean']:.8f}**",
        "",
        "| 臂 | 设计列 | Δpeak | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | 判据 |",
        "|---|--:|--:|--:|--:|--:|--:|:--:|:--:|",
    ]
    for arm, entry in s["arms"].items():
        st = entry["stats"]
        columns = payload["folds"][0]["arms"][arm]["design_columns"]
        lines.append(
            f"| `{arm}` | {columns} | **{st['relative_gain']*100:+.2f}%** | "
            f"{st['positive_folds']}/{st['n_folds']} | {st['relative_gain_drop_best']*100:+.2f}% | "
            f"{entry['delta_A']*100:+.2f}% | {entry['delta_B']*100:+.2f}% | "
            f"{'✅' if entry['mechanism_2dA_gt_dB'] else '❌'} | "
            f"{'✅ PASS' if entry['verdict']['pass'] else '❌'} |")
    lines += [
        "",
        "⚠️ 预注册候选是 `peer_k5`；`peer_k10` 是②类容量维度，只作梯度诊断，不作选参依据。",
        "",
        "## ①类分量配比重验（纯算术）",
        "",
        f"**样本外**（前半折解系数、后半折评分）折均 **{c['holdout_gain_mean']*100:+.3f}%**；",
        f"样本内上界折均 {c['in_sample_gain_mean']*100:+.3f}% —— 两者之差就是乐观量。",
        "",
        "| fold | 样本外增益 | 样本内上界 | c_market | c_cross_section |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, (g, ins, cm, cx) in enumerate(zip(c["holdout_gain_per_fold"], c["in_sample_gain_per_fold"],
                                             c["c_market_per_fold"], c["c_cross_section_per_fold"])):
        lines.append(f"| {i} | **{g*100:+.3f}%** | {ins*100:+.3f}% | {cm:.4f} | {cx:.4f} |")
    lines += [
        "",
        "对照：ROADMAP 0c 当年在**旧架构**上用三个公榜点真值算出 **+0.02%**，据此结案。",
        "⚠️ 这是**本地**数，只用来判断值不值得为它花公榜点；",
        "本地解出的 `(c_m,c_e)` **不能直接上线**（①类要靠公榜点解析求解，ROADMAP §5）。",
        "⚠️ 逐折 `c` 摆动大 = 这个旋钮本身不稳，与 0c「全局最优 c 逐折在 0.33~0.90 之间摆」一致。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if "baseline" not in args.arms:
        raise SystemExit("baseline 是配对基准，必须在 --arms 里")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"报告已存在：{json_path} / {md_path}。要覆盖请显式加 --force")

    print(f"loading sampled partitions (modulo {args.sample_modulo}/{args.sampling})...", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)
    print(f"{len(all_time_ids):,} 行 / {len(unique_time_ids):,} 个采样 time_id / {len(folds)} 折",
          flush=True)

    tr0 = np.isin(all_time_ids, folds[0][0])
    scratch, _ = robust_transform_fit(data["features"][tr0].copy())
    y0 = data["target"][tr0].astype(np.float64)
    tid0 = data["time_id"][tr0]
    s0 = group_starts(tid0)
    c0 = np.diff(np.r_[s0, len(tid0)]).astype(np.float64)
    e0 = y0 - np.repeat(np.add.reduceat(y0, s0) / c0, c0.astype(int))
    pool = select_features(scratch, e0, np.ones_like(e0), args.feature_count)
    hist_cols = np.sort(pool[select_features(scratch[:, pool], e0, np.ones_like(e0),
                                             args.history_count)])
    # ⚠️ 每个 k 各选一次：`select_features` 返回的是**按下标排序**的结果，
    # 直接对 k=10 的结果取前 5 个会拿到「下标最小的 5 个」而不是「最强的 5 个」。
    per_arm_cols = {arm: np.sort(pool[select_features(scratch[:, pool], e0, np.ones_like(e0),
                                                      PEER_K[arm])])
                    for arm in args.arms if arm != "baseline"}
    peer_cols = (np.unique(np.concatenate(list(per_arm_cols.values())))
                 if per_arm_cols else np.zeros(0, dtype=np.int64))
    del scratch, y0, e0
    gc.collect()
    for arm, columns in per_arm_cols.items():
        print(f"  {arm} 的 peer 列（{len(columns)}）：{columns.tolist()}", flush=True)
    print(f"history 列 {len(hist_cols)} 个；peer 缓存并集 {len(peer_cols)} 列", flush=True)

    files = train_files(Path(args.data_root))
    print("building lag cache (streams every row)...", flush=True)
    cache = build_lag_cache(files, hist_cols, args.sample_modulo, args.history_window,
                            sampling=args.sampling)
    assert np.array_equal(cache["time_id"], all_time_ids), "lag 缓存与采样矩阵的 time_id 不对齐"
    print("building peer cache (streams every row)...", flush=True)
    peer = build_peer_cache(files, peer_cols, args.sample_modulo, args.sampling)
    assert np.array_equal(peer["time_id"], unique_time_ids), "peer 快照与采样 time_id 不对齐"

    rows = [run_fold(i, data, cache, peer, tr_ids, va_ids, hist_cols, peer_cols,
                     per_arm_cols, args)
            for i, (tr_ids, va_ids) in enumerate(folds)]
    payload = {
        "experiment": "peer_leadlag",
        "config": {k: getattr(args, k) for k in
                   ("n_folds", "train_window", "embargo", "sample_modulo", "sampling",
                    "feature_count", "history_count", "history_window", "ridge_alpha",
                    "lgbm_rounds", "lgbm_seeds", "arms")},
        "history_columns": [int(c) for c in hist_cols],
        "peer_columns": [int(c) for c in peer_cols],
        "folds": rows,
        "summary": summarise(rows, args),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_report(payload), encoding="utf-8")
    print(f"\n写出 {json_path}\n写出 {md_path}")
    print(render_report(payload))


if __name__ == "__main__":
    main()
