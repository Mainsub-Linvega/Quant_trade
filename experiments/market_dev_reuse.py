"""2E：市场模型的**偏差部分**被白扔了 —— 捡回来当第二个 ê

## 观察

08-13 上线的架构里，市场模型跑了 1440 棵树，但我们只取它的**截面均值**：

    pred_y = m̂_lgbm + (pred_y − m̂_lgbm)
             ↑ 用了      ↑ **整个扔掉**

扔掉的那半是一个现成的截面残差估计，而且与 `ê_lgbm` **结构不同**：
标签不同（`y` vs `e`）、设计矩阵不同（多 200 列 raw）。
按 ROADMAP §5 的集成公式 `IC_blend = (ρ₁+ρ₂)/√(2(1+c))`，两个相关度不高的
同强度预测子等权平均就能涨 —— 而这里**推理端零额外成本**，那些预测本来就算出来了。

## 预注册（跑之前定死）

    ê      = (1−μ)·ê_lgbm + μ·(pred_y − 截面均值(pred_y))
    m̂      = (1−λ)·m̂_ridge + λ·截面均值(pred_y)
    f      = m̂ + ê

**候选取 λ=0.5 / μ=0.5**（都是先验、不拟合，ROADMAP §5：「别去拟合混合权重，
拟合单个标量的教训见 ab_scale_auto」）。基准是 λ=0.5 / μ=0（= 08-13 已上线那版）。

⚠️ 报告会打出整个 (λ, μ) 网格，但那是**诊断，不是选参依据**。
两个权重都是①类纯后处理旋钮 —— 真要定它们得靠公榜点解析求解，不是靠本地折上挑最大值。
报告里会显式标出「预注册格」与「本地最大格」，两者不同时**以预注册格为准**。

## 为什么这一跑很便宜

λ 和 μ 都只作用在**已经算好的逐折预测**上 ⟹ 每折只需 2 次 LGBM 拟合
（ê 一次、市场模型一次），整个网格是纯算术。

用法：
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python experiments/market_dev_reuse.py
输出：outputs/experiments/market_dev_reuse.{json,md}
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
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import train_files
from src.validation import rolling_time_folds
from features import cross_sectional_deviation
from train import robust_transform_fit, select_features
from mt_predictability import group_starts
from ridge_data_ladder import row_level_peak
from lgbm_xs import load_rows
from history_peak import (FEATURE_COUNT, HISTORY_FEATURE_COUNT, HISTORY_WINDOW,
                          LGBM_MIN_DATA_FRAC, LGBM_ROUNDS, LGBM_SPEC, TRAIN_WINDOW,
                          build_lag_cache, fit_ridge, history_blocks, paired_stats,
                          ridge_designs, transform_with, verdict)

LAMBDAS = (0.25, 0.5, 0.75)          # 诊断网格
MUS = (0.0, 0.25, 0.5, 0.75, 1.0)
REGISTERED = (0.5, 0.5)              # ⭐ 预注册候选（先验，不拟合）
BASELINE = (0.5, 0.0)                # = 08-13 已上线那版


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reuse the market model's deviation part as a second ê.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="market_dev_reuse")
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


def run_fold(index, data, cache, train_ids, valid_ids, hist_cols, args) -> dict[str, Any]:
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

    tail_tr = [xs_tr, *hist_tr, aid_tr.astype(np.float32)[:, None]]
    tail_va = [xs_va, *hist_va, aid_va.astype(np.float32)[:, None]]
    # ê：08-13 上线那版（带权、361 列）
    e_hat = fit_predict(np.ascontiguousarray(np.column_stack(tail_tr)), e_tr, w_tr,
                        np.ascontiguousarray(np.column_stack(tail_va)), args)
    e_hat -= group_mean(e_hat, va_starts, va_counts)
    # 市场模型：08-13 上线那版（561 列、标签 y、无权）
    pred_y = fit_predict(
        np.ascontiguousarray(np.column_stack([t_train[:, lgbm_selected], *tail_tr])), y_tr, None,
        np.ascontiguousarray(np.column_stack([t_valid[:, lgbm_selected], *tail_va])), args)
    m_lgbm = group_mean(pred_y, va_starts, va_counts)
    e_market = pred_y - m_lgbm                      # ← 现在被扔掉的那一半

    row: dict[str, Any] = {
        "fold": index, "n_valid_rows": int(va.sum()),
        # 两个 ê 有多像？太像就没什么可混的
        "corr_e_hat_e_market": float(np.corrcoef(e_hat, e_market)[0, 1]),
        "peak_e_hat_alone": row_level_peak(y_va, e_hat, w_va)["peak"],
        "peak_e_market_alone": row_level_peak(y_va, e_market, w_va)["peak"],
        "grid": {},
    }
    for lam in LAMBDAS:
        market = (1.0 - lam) * m_ridge + lam * m_lgbm
        for mu in MUS:
            blended = market + (1.0 - mu) * e_hat + mu * e_market
            row["grid"][f"{lam:.2f}|{mu:.2f}"] = row_level_peak(y_va, blended, w_va)

    base = row["grid"][f"{BASELINE[0]:.2f}|{BASELINE[1]:.2f}"]["peak"]
    reg = row["grid"][f"{REGISTERED[0]:.2f}|{REGISTERED[1]:.2f}"]["peak"]
    print(f"  fold {index}: 基准(λ.5,μ0) {base:.8f} → 预注册(λ.5,μ.5) {reg:.8f} "
          f"({(reg/base-1)*100:+.2f}%)  corr(ê,ê_mkt)={row['corr_e_hat_e_market']:.3f}  "
          f"{time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, tail_tr, tail_va, hist_tr, hist_va, xs_tr, xs_va
    gc.collect()
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    key = lambda p: f"{p[0]:.2f}|{p[1]:.2f}"
    baseline = np.array([r["grid"][key(BASELINE)]["peak"] for r in rows])
    grid = {}
    for lam in LAMBDAS:
        for mu in MUS:
            k = f"{lam:.2f}|{mu:.2f}"
            peaks = np.array([r["grid"][k]["peak"] for r in rows])
            stats = paired_stats(peaks - baseline, baseline)
            grid[k] = {"stats": stats, "verdict": verdict(stats),
                       "peak_mean": float(peaks.mean())}
    best = max(grid, key=lambda k: grid[k]["peak_mean"])
    return {"baseline_peak_mean": float(baseline.mean()),
            "registered_cell": key(REGISTERED), "baseline_cell": key(BASELINE),
            "best_local_cell": best, "grid": grid,
            "corr_mean": float(np.mean([r["corr_e_hat_e_market"] for r in rows])),
            "peak_e_hat_alone_mean": float(np.mean([r["peak_e_hat_alone"] for r in rows])),
            "peak_e_market_alone_mean": float(np.mean([r["peak_e_market_alone"] for r in rows]))}


def render_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    reg, base, best = s["registered_cell"], s["baseline_cell"], s["best_local_cell"]
    r = s["grid"][reg]
    lines = [
        "# 2E：捡回市场模型被扔掉的偏差部分，当第二个 ê",
        "",
        "`ê = (1−μ)·ê_lgbm + μ·(pred_y − 截面均值)`，`m̂ = (1−λ)·m̂_ridge + λ·截面均值(pred_y)`。",
        "**推理端零额外成本** —— `pred_y` 本来就算出来了，现在只用了它的均值。",
        "",
        f"- ⭐ 基准（λ=0.5, μ=0 = 08-13 已上线那版）peak **{s['baseline_peak_mean']:.8f}**",
        f"- 两个 ê 的相关 **{s['corr_mean']:.4f}**；单独的 peak：",
        f"  `ê_lgbm` **{s['peak_e_hat_alone_mean']:.8f}** / "
        f"`ê_market` **{s['peak_e_market_alone_mean']:.8f}**",
        "",
        f"## ⭐ 预注册格（λ=0.5, μ=0.5）",
        "",
        f"**Δpeak {r['stats']['relative_gain']*100:+.2f}%**，"
        f"{r['stats']['positive_folds']}/{r['stats']['n_folds']} 折，"
        f"去掉最好一折 {r['stats']['relative_gain_drop_best']*100:+.2f}%，"
        f"判据 {'✅ PASS' if r['verdict']['pass'] else '❌'}",
        "",
        "## 诊断网格（**不是选参依据**）",
        "",
        "| λ \\ μ | " + " | ".join(f"{mu:.2f}" for mu in MUS) + " |",
        "|---|" + "---|" * len(MUS),
    ]
    for lam in LAMBDAS:
        cells = []
        for mu in MUS:
            k = f"{lam:.2f}|{mu:.2f}"
            g = s["grid"][k]["stats"]["relative_gain"] * 100
            mark = " ⭐" if k == reg else (" ←基准" if k == base else ("**" if k == best else ""))
            cells.append(f"{g:+.2f}%{mark}" if mark != "**" else f"**{g:+.2f}%**")
        lines.append(f"| {lam:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"本地最大格 = `{best}`。⚠️ **若它与预注册格 `{reg}` 不同，以预注册格为准** ——",
        "λ、μ 都是①类纯后处理旋钮，本地折上挑最大值正是 ROADMAP §5 禁的那种拟合",
        "（`ab_scale_auto` 的教训：每折重估的最优标量在 10 折上 sd=0.41）。",
        "真要定它们，靠公榜点解析求解，0 次额度。",
        "",
        "判据（`history_peak.verdict`，机器判）：配对 Δ 均值为正 + 去掉最好一折仍为正 + ≥ +1%。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
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
    del scratch, y0, e0
    gc.collect()
    print(f"history 列 {len(hist_cols)} 个", flush=True)

    print("building lag cache (streams every row)...", flush=True)
    cache = build_lag_cache(train_files(Path(args.data_root)), hist_cols,
                            args.sample_modulo, args.history_window, sampling=args.sampling)
    assert np.array_equal(cache["time_id"], all_time_ids), "lag 缓存与采样矩阵的 time_id 不对齐"

    rows = [run_fold(i, data, cache, tr_ids, va_ids, hist_cols, args)
            for i, (tr_ids, va_ids) in enumerate(folds)]
    payload = {
        "experiment": "market_dev_reuse",
        "registered": {"lambda": REGISTERED[0], "mu": REGISTERED[1]},
        "config": {k: getattr(args, k) for k in
                   ("n_folds", "train_window", "embargo", "sample_modulo", "sampling",
                    "feature_count", "history_count", "history_window", "ridge_alpha",
                    "lgbm_rounds", "lgbm_seeds")},
        "history_columns": [int(c) for c in hist_cols],
        "folds": rows,
        "summary": summarise(rows),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_report(payload), encoding="utf-8")
    print(f"\n写出 {json_path}\n写出 {md_path}")
    print(render_report(payload))


if __name__ == "__main__":
    main()
