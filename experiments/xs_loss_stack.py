"""2F：截面块的三条没碰过的③类 —— 岭回归残差堆叠 / 稳健损失 / 标签裁尾

`ê` 之外的一切固定为 08-13 已上线那版（市场模型 561 列无权、λ=0.5、ridge 冻结），
单一变量是**截面块怎么训**。三臂全部预注册，推理端成本都是零。

## 臂 1 `stack_e_ridge`：把岭回归的截面偏差喂给树

岭回归的行级预测拆成 `ridge_raw = m̂_ridge + e_ridge`。现在树完全看不到它 ——
两个模型各算各的，最后才在预测层相加。把 `e_ridge` 当**一列特征**喂进去，
树就能去修线性模型剩下的非线性结构，而不是从零重新发现线性部分。

⚠️ **刻意不喂 `m̂_ridge`**：它在同一个 time_id 内是常数，正是 2D 里
`e200_xsm` 把 ΔB 撑到 +44.93% 的那种「时段指纹」。`e_ridge` 是行级的、
截面零均值的，不带这个风险。

## 臂 2 `huber`：稳健损失

target 是重尾的（E_w[y²]=1.1757、sd 1.0784），L2 会被少数离群点主导。
通常不敢换稳健损失是因为它压低预测幅度、偏离条件均值 ——
⭐ **但 `peak = A²/B` 对预测缩放严格不变**，幅度被压不吃亏，
只留下「对离群点稳健」这一半好处。这个不对称性使得稳健损失在本项目的判据下值得一试。

## 臂 3 `winsor5`：标签裁尾

同一机制的另一种写法：训练标签 `e` 裁到 ±5σ（σ 只用训练段算，无泄漏）。
比换损失函数更粗糙，但完全不动 LightGBM 的目标函数，是干净的对照。

## 判据

`history_peak.verdict` 机器判：配对 Δ 均值为正 + 去掉最好一折仍为正 + ≥ +1%。
另报 `2ΔA > ΔB` 与逐折 Δ。

用法：
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python experiments/xs_loss_stack.py
输出：outputs/experiments/xs_loss_stack.{json,md}
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

LAMBDA = 0.5
WINSOR_SIGMA = 5.0
ARMS = ("baseline", "stack_e_ridge", "huber", "winsor5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ridge-residual stacking / robust loss / winsorised label.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="xs_loss_stack")
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


def fit_predict(design_tr, label, weight, design_va, args, objective="regression") -> np.ndarray:
    import lightgbm as lgb

    cat = design_tr.shape[1] - 1
    min_data = max(20, int(round(LGBM_MIN_DATA_FRAC * len(design_tr))))
    out = np.zeros(len(design_va), dtype=np.float64)
    for s in range(args.lgbm_seeds):
        seed = args.lgbm_seed + s
        params = {**LGBM_SPEC, "objective": objective, "metric": "l2", "verbosity": -1,
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

    # ---- 岭回归：产 m̂_ridge，同时留下 e_ridge 给 stack 臂当特征
    fold_alpha = args.ridge_alpha * len(train_ids) / 39_480
    ridge_selected = select_features(t_train, y_tr, w_tr, args.feature_count)
    est = fit_ridge(ridge_designs(t_train, tid_tr, ridge_selected, None), y_tr, w_tr, fold_alpha)
    raw_tr = (est.intercept_
              + ridge_designs(t_train, tid_tr, ridge_selected, None) @ est.coef_).astype(np.float64)
    raw_va = (est.intercept_
              + ridge_designs(t_valid, tid_va, ridge_selected, None) @ est.coef_).astype(np.float64)
    m_ridge = group_mean(raw_va, va_starts, va_counts)
    # e_ridge：行级、截面零均值 ⟹ 不是「时段指纹」（2D 的教训）
    e_ridge_tr = (raw_tr - group_mean(raw_tr, tr_starts, tr_counts)).astype(np.float32)[:, None]
    e_ridge_va = (raw_va - group_mean(raw_va, va_starts, va_counts)).astype(np.float32)[:, None]
    del est, raw_tr, raw_va

    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    lgbm_selected = select_features(t_train, e_tr, np.ones_like(e_tr), args.feature_count)
    xs_tr = cross_sectional_deviation(t_train[:, lgbm_selected].copy(), tid_tr)
    xs_va = cross_sectional_deviation(t_valid[:, lgbm_selected].copy(), tid_va)
    lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
    hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
    hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

    # ⚠️ asset_id 必须留在最后一列 ⟹ stack 臂的新列插在 asset_id **之前**
    aid_col_tr = aid_tr.astype(np.float32)[:, None]
    aid_col_va = aid_va.astype(np.float32)[:, None]
    body_tr = [xs_tr, *hist_tr]
    body_va = [xs_va, *hist_va]
    d_tr = np.ascontiguousarray(np.column_stack([*body_tr, aid_col_tr]))
    d_va = np.ascontiguousarray(np.column_stack([*body_va, aid_col_va]))

    # ---- 市场模型：08-13 上线那版，全臂共用
    m_lgbm = group_mean(fit_predict(
        np.ascontiguousarray(np.column_stack([t_train[:, lgbm_selected], *body_tr, aid_col_tr])),
        y_tr, None,
        np.ascontiguousarray(np.column_stack([t_valid[:, lgbm_selected], *body_va, aid_col_va])),
        args), va_starts, va_counts)
    market = (1.0 - LAMBDA) * m_ridge + LAMBDA * m_lgbm

    sigma = float(np.std(e_tr))
    row: dict[str, Any] = {"fold": index, "n_valid_rows": int(va.sum()),
                           "e_sigma": sigma, "arms": {}}
    for arm in args.arms:
        if arm == "stack_e_ridge":
            a_tr = np.ascontiguousarray(np.column_stack([*body_tr, e_ridge_tr, aid_col_tr]))
            a_va = np.ascontiguousarray(np.column_stack([*body_va, e_ridge_va, aid_col_va]))
            label, objective = e_tr, "regression"
        else:
            a_tr, a_va = d_tr, d_va
            label = np.clip(e_tr, -WINSOR_SIGMA * sigma, WINSOR_SIGMA * sigma) \
                if arm == "winsor5" else e_tr
            objective = "huber" if arm == "huber" else "regression"
        e_hat = fit_predict(a_tr, label, w_tr, a_va, args, objective=objective)
        e_hat -= group_mean(e_hat, va_starts, va_counts)
        row["arms"][arm] = {"design_columns": int(a_tr.shape[1]),
                            "full": row_level_peak(y_va, market + e_hat, w_va),
                            "peak_e_alone": row_level_peak(y_va, e_hat, w_va)["peak"]}
        if arm == "stack_e_ridge":
            del a_tr, a_va
        gc.collect()

    base = row["arms"]["baseline"]["full"]["peak"]
    detail = "  ".join(f"{a} {(row['arms'][a]['full']['peak']/base-1)*100:+.2f}%" for a in args.arms)
    print(f"  fold {index}: base {base:.8f} | {detail}  {time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, d_tr, d_va, body_tr, body_va, hist_tr, hist_va, xs_tr, xs_va
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
                    "mechanism_2dA_gt_dB": 2 * d_a > d_b,
                    "peak_e_alone_mean": float(np.mean([r["arms"][arm]["peak_e_alone"] for r in rows]))}
    return {"baseline_peak_mean": float(baseline.mean()),
            "baseline_peak_e_alone_mean":
                float(np.mean([r["arms"]["baseline"]["peak_e_alone"] for r in rows])),
            "arms": out}


def render_report(payload) -> str:
    s = payload["summary"]
    lines = [
        "# 2F：截面块的三条③类 —— 岭回归残差堆叠 / 稳健损失 / 标签裁尾",
        "",
        "`ê` 之外一切固定为 08-13 已上线那版（市场模型 561 列无权、λ=0.5、ridge 冻结）。",
        "三臂推理端成本均为零。",
        "",
        f"- 折数 {payload['config']['n_folds']}，modulo {payload['config']['sample_modulo']}"
        f"/{payload['config']['sampling']}，LGBM {payload['config']['lgbm_rounds']} 轮 × "
        f"{payload['config']['lgbm_seeds']} 种子",
        f"- ⭐ 基准强度：整体 peak **{s['baseline_peak_mean']:.8f}**，"
        f"其中 ê 单独 **{s['baseline_peak_e_alone_mean']:.8f}**",
        "",
        "| 臂 | 设计列 | Δpeak | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | ê 单独 peak | 判据 |",
        "|---|--:|--:|--:|--:|--:|--:|:--:|--:|:--:|",
    ]
    for arm, entry in s["arms"].items():
        st = entry["stats"]
        columns = payload["folds"][0]["arms"][arm]["design_columns"]
        lines.append(
            f"| `{arm}` | {columns} | **{st['relative_gain']*100:+.2f}%** | "
            f"{st['positive_folds']}/{st['n_folds']} | {st['relative_gain_drop_best']*100:+.2f}% | "
            f"{entry['delta_A']*100:+.2f}% | {entry['delta_B']*100:+.2f}% | "
            f"{'✅' if entry['mechanism_2dA_gt_dB'] else '❌'} | "
            f"{entry['peak_e_alone_mean']:.8f} | "
            f"{'✅ PASS' if entry['verdict']['pass'] else '❌'} |")
    lines += ["", "判据（`history_peak.verdict`，机器判）：配对 Δ 均值为正 + 去掉最好一折仍为正 + ≥ +1%。"]
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
    del scratch, y0, e0
    gc.collect()

    print("building lag cache (streams every row)...", flush=True)
    cache = build_lag_cache(train_files(Path(args.data_root)), hist_cols,
                            args.sample_modulo, args.history_window, sampling=args.sampling)
    assert np.array_equal(cache["time_id"], all_time_ids), "lag 缓存与采样矩阵的 time_id 不对齐"

    rows = [run_fold(i, data, cache, tr_ids, va_ids, hist_cols, args)
            for i, (tr_ids, va_ids) in enumerate(folds)]
    payload = {
        "experiment": "xs_loss_stack",
        "config": {k: getattr(args, k) for k in
                   ("n_folds", "train_window", "embargo", "sample_modulo", "sampling",
                    "feature_count", "history_count", "history_window", "ridge_alpha",
                    "lgbm_rounds", "lgbm_seeds", "arms")},
        "winsor_sigma": WINSOR_SIGMA,
        "history_columns": [int(c) for c in hist_cols],
        "folds": rows,
        "summary": summarise(rows, args),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_report(payload), encoding="utf-8")
    print(f"\n写出 {json_path}\n写出 {md_path}")
    print(render_report(payload))


if __name__ == "__main__":
    main()
