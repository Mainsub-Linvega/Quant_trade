"""2D：市场模型该看什么特征？—— 列集合 × 显式截面均值 的 2×2

## 假设一：市场模型的选列是错配的

08-13 上线的市场模型复用的是 `lgbm_features` —— **按 `e`（截面残差）的 |corr| 选出的 200 列**。
那是为「预测截面残差」挑的列，不是为「预测市场」挑的。而 `market_model`（08-10）实测过：

> 「对 m_t 做特征预选**有害**：top-50 −27%、top-200 −2.7%，**323 列全上最好**」

⟹ 给市场模型全部 323 列，是同一根轴上没摘的果子。

## 假设二（更尖的一条）：树根本看不见市场状态

**LightGBM 是逐行预测的。** 一次 `predict` 里，每一行只带自己那个资产的
`raw_it`；而市场分量 `m_t` 是**整个截面**的性质。树想估市场水平，只能拿
**单个资产的 raw 当噪声代理** —— 它没有任何办法把 15 个资产平均起来
（树只能在单个特征上切分，做不出跨行的聚合）。

而 `cross_sectional_deviation` 内部**恰好算了 `mean_t`，然后把它扔掉**：

    raw_it = mean_t(x) + dev_it        ← 我们只喂了 raw 和 dev，从没喂过 mean

把 `mean_t(x)` 显式加成一个块，对树是**全新信息**（不是 raw、dev 的任何单特征函数）。
推理端零额外成本：`mean = lraw − ldev`，两者本来就都算好了。因果性没问题 ——
官方 runner 每次恰好喂一个 time_id 的全部行，截面均值是当期可得量。

## 预注册的 2×2（跑之前定死）

`ê` 一律固定为**带权 361 列**那版（08-13 已上线），四臂只改市场模型的设计矩阵：

| 臂 | 市场模型设计矩阵 | 列数 |
|---|---|--:|
| `e200`（基准 = 已上线） | `[raw200 ‖ dev200 ‖ hist160 ‖ aid]` | 561 |
| `all323` | `[raw323 ‖ dev200 ‖ hist160 ‖ aid]` | 684 |
| `e200_xsm` | `[raw200 ‖ **xsmean200** ‖ dev200 ‖ hist160 ‖ aid]` | 761 |
| `all323_xsm` | `[raw323 ‖ **xsmean323** ‖ dev200 ‖ hist160 ‖ aid]` | 1007 |

`dev` 块与 `hist` 块**始终是按 e 选的那 200 列**（与 ê 模型共用，不动）——
这样单一变量就是「市场模型多看了什么」。
`m̂ = 0.5·m̂_ridge + 0.5·m̂_lgbm`，λ 固定 0.5（先验，不拟合）。

## 怎么读

同 2A：收益必须**主要出现在 `Δpeak_m`**；只在整体上出现而 `peak_m` 不动 = 不算数。
读相对增益前先看 `baseline_peak` 绝对值（r = −0.986）。

⚠️ `all323` 会**扩推理输入契约**（现在 meta 只存 200 列的统计量，要扩到 323）。
若它赢而 `xsm` 不赢，接线成本比 `xsm` 高 —— 但那是接线问题，不影响本实验的判据。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/market_features.py
输出：outputs/experiments/market_features.{json,md}
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
# 臂名 -> (raw 块用全部 323 列?, 是否加显式截面均值块)
ARMS: dict[str, tuple[bool, bool]] = {
    "e200":       (False, False),      # 基准 = 08-13 已上线那版
    "all323":     (True,  False),
    "e200_xsm":   (False, True),
    "all323_xsm": (True,  True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="What should the market model look at?")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="market_features")
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
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
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


def run_fold(index: int, data, cache, train_ids, valid_ids, hist_cols, args) -> dict[str, Any]:
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
    ridge_raw = (est.intercept_
                 + ridge_designs(t_valid, tid_va, ridge_selected, None) @ est.coef_).astype(np.float64)
    m_ridge = group_mean(ridge_raw, va_starts, va_counts)
    del est, ridge_raw

    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    lgbm_selected = select_features(t_train, e_tr, np.ones_like(e_tr), args.feature_count)
    xs_tr = cross_sectional_deviation(t_train[:, lgbm_selected].copy(), tid_tr)
    xs_va = cross_sectional_deviation(t_valid[:, lgbm_selected].copy(), tid_va)

    lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
    hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
    hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

    # ---- ê：08-13 已上线那版（带权、361 列），四臂共用
    tail_tr = [xs_tr, *hist_tr, aid_tr.astype(np.float32)[:, None]]
    tail_va = [xs_va, *hist_va, aid_va.astype(np.float32)[:, None]]
    e_hat = fit_predict(np.ascontiguousarray(np.column_stack(tail_tr)), e_tr, w_tr,
                        np.ascontiguousarray(np.column_stack(tail_va)), args)
    e_hat -= group_mean(e_hat, va_starts, va_counts)

    row: dict[str, Any] = {
        "fold": index, "n_train_rows": int(tr.sum()), "n_valid_rows": int(va.sum()),
        "peak_m_ridge": row_level_peak(y_va, m_ridge, w_va)["peak"],
        "no_market_peak": row_level_peak(y_va, m_ridge + e_hat, w_va)["peak"],
        "arms": {},
    }

    for arm in args.arms:
        use_all, use_xsm = ARMS[arm]
        cols = slice(None) if use_all else lgbm_selected
        head_tr: list[np.ndarray] = [t_train[:, cols]]
        head_va: list[np.ndarray] = [t_valid[:, cols]]
        if use_xsm:
            # 显式截面均值：mean = raw − dev。推理端 lraw、ldev 本来就都有，零额外成本。
            head_tr.append(head_tr[0] - cross_sectional_deviation(head_tr[0].copy(), tid_tr))
            head_va.append(head_va[0] - cross_sectional_deviation(head_va[0].copy(), tid_va))
        d_tr = np.ascontiguousarray(np.column_stack([*head_tr, *tail_tr]))
        d_va = np.ascontiguousarray(np.column_stack([*head_va, *tail_va]))
        del head_tr, head_va
        m_lgbm = group_mean(fit_predict(d_tr, y_tr, None, d_va, args), va_starts, va_counts)
        market = (1.0 - LAMBDA) * m_ridge + LAMBDA * m_lgbm
        row["arms"][arm] = {"design_columns": int(d_tr.shape[1]),
                            "peak_m_lgbm_alone": row_level_peak(y_va, m_lgbm, w_va)["peak"],
                            "peak_m": row_level_peak(y_va, market, w_va)["peak"],
                            "full": row_level_peak(y_va, market + e_hat, w_va)}
        del d_tr, d_va
        gc.collect()

    base = row["arms"]["e200"]["full"]["peak"]
    detail = "  ".join(f"{a} {(row['arms'][a]['full']['peak']/base-1)*100:+.2f}%" for a in args.arms)
    print(f"  fold {index}: 无市场模型 {row['no_market_peak']:.8f} | e200 基准 {base:.8f} | "
          f"{detail}  {time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, tail_tr, tail_va, xs_tr, xs_va, hist_tr, hist_va
    gc.collect()
    return row


def summarise(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    baseline = np.array([r["arms"]["e200"]["full"]["peak"] for r in rows])
    base_m = np.array([r["arms"]["e200"]["peak_m"] for r in rows])
    base_a = float(np.mean([r["arms"]["e200"]["full"]["A"] for r in rows]))
    base_b = float(np.mean([r["arms"]["e200"]["full"]["B"] for r in rows]))
    out: dict[str, Any] = {}
    for arm in args.arms:
        if arm == "e200":
            continue
        peaks = np.array([r["arms"][arm]["full"]["peak"] for r in rows])
        mkt = np.array([r["arms"][arm]["peak_m"] for r in rows])
        stats = paired_stats(peaks - baseline, baseline)
        d_a = float(np.mean([r["arms"][arm]["full"]["A"] for r in rows])) / base_a - 1.0
        d_b = float(np.mean([r["arms"][arm]["full"]["B"] for r in rows])) / base_b - 1.0
        out[arm] = {"stats": stats, "verdict": verdict(stats), "delta_A": d_a, "delta_B": d_b,
                    "mechanism_2dA_gt_dB": 2 * d_a > d_b,
                    "market_peak_stats": paired_stats(mkt - base_m, base_m),
                    "peak_m_lgbm_alone_mean":
                        float(np.mean([r["arms"][arm]["peak_m_lgbm_alone"] for r in rows]))}
    return {"baseline_peak_mean": float(baseline.mean()),
            "baseline_peak_m_mean": float(base_m.mean()),
            "no_market_peak_mean": float(np.mean([r["no_market_peak"] for r in rows])),
            "e200_peak_m_lgbm_alone_mean":
                float(np.mean([r["arms"]["e200"]["peak_m_lgbm_alone"] for r in rows])),
            "arms": out}


def render_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# 2D：市场模型该看什么特征？（列集合 × 显式截面均值）",
        "",
        "两个假设：① 市场模型复用「按 `e` 选的 200 列」是错配（`market_model` 说 323 列全上最好）；",
        "② **树逐行预测，看不见市场状态** —— `mean_t(x)` 从没被喂过，而 `cross_sectional_deviation`",
        "内部算了它又扔掉。`ê` 固定为已上线那版（带权 361 列），单一变量是市场模型的设计矩阵。",
        "",
        f"- 折数 {payload['config']['n_folds']}，train_window {payload['config']['train_window']:,}，"
        f"modulo {payload['config']['sample_modulo']}/{payload['config']['sampling']}，"
        f"LGBM {payload['config']['lgbm_rounds']} 轮 × {payload['config']['lgbm_seeds']} 种子",
        f"- ⭐ 基准强度：`e200`（已上线）peak **{s['baseline_peak_mean']:.8f}**、"
        f"其中市场块 **{s['baseline_peak_m_mean']:.8f}**；作为参照，**完全不用市场模型**是 "
        f"**{s['no_market_peak_mean']:.8f}**",
        "",
        "| 臂 | 设计列 | Δpeak | 正折 | 去最好折 | Δpeak_m | ΔA | ΔB | 2ΔA>ΔB | 判据 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|",
    ]
    for arm, entry in s["arms"].items():
        columns = payload["folds"][0]["arms"][arm]["design_columns"]
        st, mk = entry["stats"], entry["market_peak_stats"]
        lines.append(
            f"| `{arm}` | {columns} | **{st['relative_gain']*100:+.2f}%** | "
            f"{st['positive_folds']}/{st['n_folds']} | {st['relative_gain_drop_best']*100:+.2f}% | "
            f"{mk['relative_gain']*100:+.2f}% | {entry['delta_A']*100:+.2f}% | "
            f"{entry['delta_B']*100:+.2f}% | {'✅' if entry['mechanism_2dA_gt_dB'] else '❌'} | "
            f"{'✅ PASS' if entry['verdict']['pass'] else '❌'} |")
    lines += ["", "## 诊断：`m̂_lgbm` 单独的 peak_m（不与 ridge 混）", "",
              "| 臂 | peak_m（单独） |", "|---|--:|",
              f"| `e200`（已上线） | {s['e200_peak_m_lgbm_alone_mean']:.8f} |"]
    for arm, entry in s["arms"].items():
        lines.append(f"| `{arm}` | {entry['peak_m_lgbm_alone_mean']:.8f} |")
    lines += [
        "",
        "⚠️ 收益必须**主要出现在 `Δpeak_m`**；只在整体上出现而市场块不动 = 不算数。",
        "⚠️ `all323` 会扩推理输入契约（meta 的统计量从 200 列扩到 323 列）；",
        "`xsm` 则**零额外推理成本**（`mean = lraw − ldev`，两者本来就都算好了）。",
        "",
        "判据（`history_peak.verdict`，机器判）：配对 Δ 均值为正 + 去掉最好一折仍为正 + ≥ +1%。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if "e200" not in args.arms:
        raise SystemExit("e200 是配对基准（= 已上线那版），必须在 --arms 里")
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
    print(f"history 列 {len(hist_cols)} 个：{[int(c) for c in hist_cols[:10]]} ...", flush=True)

    print("building lag cache (streams every row)...", flush=True)
    cache = build_lag_cache(train_files(Path(args.data_root)), hist_cols,
                            args.sample_modulo, args.history_window, sampling=args.sampling)
    assert np.array_equal(cache["time_id"], all_time_ids), "lag 缓存与采样矩阵的 time_id 不对齐"

    rows = [run_fold(i, data, cache, tr_ids, va_ids, hist_cols, args)
            for i, (tr_ids, va_ids) in enumerate(folds)]
    payload = {
        "experiment": "market_features",
        "lambda": LAMBDA,
        "config": {k: getattr(args, k) for k in
                   ("n_folds", "train_window", "embargo", "sample_modulo", "sampling",
                    "feature_count", "history_count", "history_window", "ridge_alpha",
                    "lgbm_rounds", "lgbm_seeds", "arms")},
        "arm_spec": {a: list(ARMS[a]) for a in args.arms},
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
