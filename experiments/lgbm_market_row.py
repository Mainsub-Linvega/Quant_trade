"""2A：行级 LGBM 直接打 `y` —— 给市场分量 `m̂` 找第二个来源。

## 要验证的假设

**市场分量的线性天花板是「聚合」造成的，不是「模型族」造成的。**

现在 `m̂ = 逐 time_id 对岭回归行级预测取无权截面均值`，占约 **49% 的分**，
而它至今仍是 2026-07 那个线性模型。项目对市场块进攻过三次，全部失败：

| 实验 | 做法 | 结果 |
|---|---|---|
| `lgbm_mt`（两轮） | 树打**聚合后**的 `m_t`（888k 行 × 323 列截面均值） | INCONCLUSIVE，最大 `|t|` 0.71 |
| `market_model` | 线性直接回归 `m_t` | +2.0%，6/10 折，去最好折翻负 |
| `mt_lagged` | 滞后的**特征截面均值**进择时块 | 四臂全部在去掉最好一折后翻负 |

⭐ 但 `market_model` 自己的结论正是本实验的理由：

> 「行级岭回归能从 15 倍的行里榨出市场信号，**先取截面均值再回归把那部分信息
>   在进模型之前就扔了**」

⟹ 三次失败全都发生在**聚合层或线性族**里。**行级 + 非线性 + 带 history** 这个组合
从来没有跑出过分数（`strategies/v2_lgbm` 是这个形态，2026-08-08 因加载路径 OOM 被放弃）。
而 history 特征在**截面块**上刚给了 +23.5%（项目唯一一次两位数跳变）——
同一批时间维度信息对市场块有没有用，是当前最大的空白。

## 预注册（跑之前写死，不看结果再加臂）

两个设计矩阵臂，都用 `label = y`（不是 `e`），产出 `m̂_lgbm = 逐 time_id 无权截面均值`：

| 臂 | 设计矩阵 | 为什么值得单独测 |
|---|---|---|
| `y_xs` | `[xs_dev(200) ‖ history(4×40) ‖ asset_id]` —— **与生产逐列相同** | 唯一变量是标签。`xs_dev` 截面均值恒为 0 ⟹ 市场信号只能从 history 块来，是「时间记忆对市场块有没有用」的干净测量 |
| `y_raw` | `[raw(200) ‖ xs_dev(200) ‖ history ‖ asset_id]` | 行级岭回归的直接对应物（它的基底就是 `[raw ‖ dev]`）。`market_model` 那句话预测这一臂更强 |

组合（**λ 是先验、不拟合**，ROADMAP §5）：

    f = (1−λ)·m̂_ridge + λ·m̂_lgbm + ê_lgbm          λ ∈ {0, 0.5, 1.0}

`λ=0` 就是现在的生产架构，作为配对基准。`ê_lgbm` 三个 λ 完全相同（同一个模型、同一份预测）
⟹ 臂间差异**只来自市场块**，这是配对比较的关键。

## 怎么读结果（写在前面，免得看到数字再找解释）

⚠️ **必须同时看 `peak_m` 和整体 `peak`。** 假设若成立，收益应当**主要出现在 `peak_m`**
（市场块自己变强）。若 `peak_m` 不动而整体 `peak` 涨了，那多半是 `m̂_lgbm` 里混进了
截面信息（`ê` 没投影干净的镜像问题），**不算数**。

⚠️ 读相对增益前先看 `baseline_peak` 的绝对值 —— `Δpeak` 与基准强度实测 r = −0.986，
基准被削弱会把比值吹大（NOTES 记过被它误导三次）。

## 口径

测量机器整套 import 自 `history_peak`（折、判据、lag 缓存、`row_level_peak`），
**不复制粘贴** —— 那套刚在 A1′ 上用过，且判据由 `verdict()` 机器判（伤疤清单 #2）。
LGBM 训练**不带 sample_weight**，与生产 `strategies/v3_hybrid/train.py:287` 一致
（带权训练是另一个独立的臂，见 2C，不在本脚本里混测）。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/lgbm_market_row.py
    # 试水：--n-folds 2 --arms y_xs --label lgbm_market_row_trial
输出：outputs/experiments/lgbm_market_row.{json,md}（同名已存在需 --force）
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
# 测量口径整套复用 A1′ 那套成品
from history_peak import (FEATURE_COUNT, HISTORY_FEATURE_COUNT, HISTORY_WINDOW,
                          LGBM_MIN_DATA_FRAC, LGBM_ROUNDS, LGBM_SPEC, TRAIN_WINDOW,
                          build_lag_cache, fit_ridge, history_blocks, paired_stats,
                          ridge_designs, transform_with, verdict)

ARMS = ("y_xs", "y_raw")
LAMBDAS = (0.0, 0.5, 1.0)          # 先验，不拟合（ROADMAP §5）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Row-level LGBM on y as a second market component.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="lgbm_market_row")
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
    """逐 time_id 的**无权**截面均值，广播回每一行（与生产 main.py 同口径）。"""
    return np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))


def train_lgbm(design: np.ndarray, label: np.ndarray, design_valid: np.ndarray,
               rounds: int, n_seeds: int, seed0: int, num_threads: int) -> np.ndarray:
    """与生产同参数的 LGBM；多种子先平均预测。**不带 sample_weight**（与生产一致）。"""
    import lightgbm as lgb

    cat = design.shape[1] - 1                      # ⚠️ asset_id 恒为最后一列
    min_data = max(20, int(round(LGBM_MIN_DATA_FRAC * len(design))))
    out = np.zeros(len(design_valid), dtype=np.float64)
    for s in range(n_seeds):
        seed = seed0 + s
        params = {**LGBM_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                  "num_threads": num_threads, "min_data_in_leaf": min_data,
                  "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                  "force_row_wise": True, "feature_pre_filter": False,
                  "seed": seed, "bagging_seed": seed + 1000,
                  "feature_fraction_seed": seed + 2000}
        ds = lgb.Dataset(design, label=label, params=params,
                         categorical_feature=[cat], free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=rounds)
        out += booster.predict(design_valid, num_iteration=rounds).astype(np.float64)
        del ds, booster
    return out / n_seeds


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

    # ---- 基准市场块：生产岭回归（选列按 target 加权，与生产一致）
    fold_alpha = args.ridge_alpha * len(train_ids) / 39_480
    ridge_selected = select_features(t_train, y_tr, w_tr, args.feature_count)
    est = fit_ridge(ridge_designs(t_train, tid_tr, ridge_selected, None), y_tr, w_tr, fold_alpha)
    ridge_raw = (est.intercept_
                 + ridge_designs(t_valid, tid_va, ridge_selected, None) @ est.coef_).astype(np.float64)
    m_ridge = group_mean(ridge_raw, va_starts, va_counts)
    del est

    # ---- 截面块：与生产完全一致（选列按无权 e、目标 e、投影成无权零均值）
    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    lgbm_selected = select_features(t_train, e_tr, np.ones_like(e_tr), args.feature_count)
    xs_tr = cross_sectional_deviation(t_train[:, lgbm_selected].copy(), tid_tr)
    xs_va = cross_sectional_deviation(t_valid[:, lgbm_selected].copy(), tid_va)

    lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
    hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
    hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

    def design_of(arm: str, xs, hist, t_features, selected, aid):
        blocks = ([t_features[:, selected]] if arm == "y_raw" else [])
        blocks += [xs, *hist, aid.astype(np.float32)[:, None]]
        return np.ascontiguousarray(np.column_stack(blocks))

    d_tr_xs = design_of("y_xs", xs_tr, hist_tr, t_train, lgbm_selected, aid_tr)
    d_va_xs = design_of("y_xs", xs_va, hist_va, t_valid, lgbm_selected, aid_va)

    # ê 三个 λ 共用同一份 —— 臂间差异只来自市场块
    e_hat = train_lgbm(d_tr_xs, e_tr, d_va_xs, args.lgbm_rounds, args.lgbm_seeds,
                       args.lgbm_seed, args.num_threads)
    e_hat -= group_mean(e_hat, va_starts, va_counts)

    row: dict[str, Any] = {
        "fold": index, "n_train_rows": int(tr.sum()), "n_valid_rows": int(va.sum()),
        "peak_m_ridge": row_level_peak(y_va, m_ridge, w_va)["peak"],
        "baseline": row_level_peak(y_va, m_ridge + e_hat, w_va),
        "arms": {},
    }

    for arm in args.arms:
        if arm == "y_xs":
            d_tr, d_va = d_tr_xs, d_va_xs
        else:
            d_tr = design_of(arm, xs_tr, hist_tr, t_train, lgbm_selected, aid_tr)
            d_va = design_of(arm, xs_va, hist_va, t_valid, lgbm_selected, aid_va)
        pred_y = train_lgbm(d_tr, y_tr, d_va, args.lgbm_rounds, args.lgbm_seeds,
                            args.lgbm_seed, args.num_threads)
        m_lgbm = group_mean(pred_y, va_starts, va_counts)
        entry: dict[str, Any] = {
            "design_columns": int(d_tr.shape[1]),
            "peak_m_lgbm": row_level_peak(y_va, m_lgbm, w_va)["peak"],
            # 两个市场估计有多像？0.99 就没什么可混的
            "corr_m_ridge_lgbm": float(np.corrcoef(m_ridge, m_lgbm)[0, 1]),
            "lambda": {},
        }
        for lam in LAMBDAS:
            market = (1.0 - lam) * m_ridge + lam * m_lgbm
            entry["lambda"][f"{lam:.2f}"] = {
                "peak_m": row_level_peak(y_va, market, w_va)["peak"],
                "full": row_level_peak(y_va, market + e_hat, w_va),
            }
        row["arms"][arm] = entry
        base = row["baseline"]["peak"]
        best = max(entry["lambda"].values(), key=lambda v: v["full"]["peak"])["full"]["peak"]
        print(f"  fold {index} [{arm}]: peak_m ridge {row['peak_m_ridge']:.8f} → "
              f"lgbm {entry['peak_m_lgbm']:.8f} | 整体 {base:.8f} → {best:.8f} "
              f"({(best/base-1)*100:+.2f}%) corr={entry['corr_m_ridge_lgbm']:.3f}", flush=True)
        if arm != "y_xs":
            del d_tr, d_va
        gc.collect()

    print(f"  fold {index} 用时 {time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, d_tr_xs, d_va_xs, xs_tr, xs_va, hist_tr, hist_va
    gc.collect()
    return row


def summarise(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    baseline = np.array([r["baseline"]["peak"] for r in rows])
    base_m = np.array([r["peak_m_ridge"] for r in rows])
    out: dict[str, Any] = {}
    for arm in args.arms:
        per_lambda = {}
        for lam in LAMBDAS:
            key = f"{lam:.2f}"
            full = np.array([r["arms"][arm]["lambda"][key]["full"]["peak"] for r in rows])
            mkt = np.array([r["arms"][arm]["lambda"][key]["peak_m"] for r in rows])
            stats = paired_stats(full - baseline, baseline)
            per_lambda[key] = {
                "full_peak_stats": stats,
                "verdict": verdict(stats),
                "market_peak_stats": paired_stats(mkt - base_m, base_m),
                "delta_A": float(np.mean([r["arms"][arm]["lambda"][key]["full"]["A"] for r in rows])
                                 / np.mean([r["baseline"]["A"] for r in rows]) - 1.0),
                "delta_B": float(np.mean([r["arms"][arm]["lambda"][key]["full"]["B"] for r in rows])
                                 / np.mean([r["baseline"]["B"] for r in rows]) - 1.0),
            }
            per_lambda[key]["mechanism_2dA_gt_dB"] = (
                2 * per_lambda[key]["delta_A"] > per_lambda[key]["delta_B"])
        out[arm] = {
            "per_lambda": per_lambda,
            "peak_m_lgbm_mean": float(np.mean([r["arms"][arm]["peak_m_lgbm"] for r in rows])),
            "corr_m_ridge_lgbm_mean": float(np.mean([r["arms"][arm]["corr_m_ridge_lgbm"] for r in rows])),
        }
    return {"baseline_peak_mean": float(baseline.mean()),
            "baseline_peak_m_mean": float(base_m.mean()), "arms": out}


def render_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# 2A：行级 LGBM 直接打 `y` —— 市场分量的第二个来源",
        "",
        "假设：**市场分量的线性天花板是「聚合」造成的，不是「模型族」造成的。**",
        f"`f = (1−λ)·m̂_ridge + λ·m̂_lgbm + ê_lgbm`，λ 先验不拟合；λ=0 即现生产架构。",
        "",
        f"- 折数 {payload['config']['n_folds']}，train_window {payload['config']['train_window']:,}，"
        f"embargo {payload['config']['embargo']}，modulo {payload['config']['sample_modulo']}"
        f"/{payload['config']['sampling']}",
        f"- LGBM {payload['config']['lgbm_rounds']} 轮 × {payload['config']['lgbm_seeds']} 种子，无 sample_weight（与生产一致）",
        "",
        "⭐ **基准强度**（读相对增益前先看这个，Δpeak 与基准强度实测 r = −0.986）：",
        f"整体 peak **{s['baseline_peak_mean']:.8f}**，其中市场块 peak_m **{s['baseline_peak_m_mean']:.8f}**",
        "",
        "## 结果",
        "",
        "| 臂 | 设计列 | λ | 整体 Δpeak | 正折 | 去最好折 | 市场块 Δpeak_m | ΔA | ΔB | 2ΔA>ΔB | 判据 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|",
    ]
    for arm, block in s["arms"].items():
        columns = payload["folds"][0]["arms"][arm]["design_columns"]
        for lam, entry in block["per_lambda"].items():
            st, mk = entry["full_peak_stats"], entry["market_peak_stats"]
            lines.append(
                f"| `{arm}` | {columns} | {lam} | **{st['relative_gain']*100:+.2f}%** | "
                f"{st['positive_folds']}/{st['n_folds']} | {st['relative_gain_drop_best']*100:+.2f}% | "
                f"{mk['relative_gain']*100:+.2f}% | {entry['delta_A']*100:+.2f}% | "
                f"{entry['delta_B']*100:+.2f}% | {'✅' if entry['mechanism_2dA_gt_dB'] else '❌'} | "
                f"{'✅ PASS' if entry['verdict']['pass'] else '❌'} |")
    lines += [
        "",
        "## 诊断",
        "",
        "| 臂 | `m̂_lgbm` 单独的 peak_m | 与 `m̂_ridge` 的相关 |",
        "|---|--:|--:|",
    ]
    for arm, block in s["arms"].items():
        lines.append(f"| `{arm}` | {block['peak_m_lgbm_mean']:.8f} | "
                     f"{block['corr_m_ridge_lgbm_mean']:.4f} |")
    lines += [
        "",
        "⚠️ 判读规则（写在实验之前）：收益必须**主要出现在 `Δpeak_m`** 上。",
        "若 `Δpeak_m` ≈ 0 而整体 `Δpeak` 为正，多半是 `m̂_lgbm` 混进了截面信息，不算数。",
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

    # ---- history 列：与 A1′ / 生产同口径，第 0 折训练窗上预注册一次
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
    assert len(cache["lags"]) == len(all_time_ids), "lag 缓存与采样矩阵行数不一致 —— 两条读取路径口径不同"
    assert np.array_equal(cache["time_id"], all_time_ids), "lag 缓存与采样矩阵的 time_id 不对齐"

    rows = [run_fold(i, data, cache, tr_ids, va_ids, hist_cols, args)
            for i, (tr_ids, va_ids) in enumerate(folds)]
    payload = {
        "experiment": "lgbm_market_row",
        "hypothesis": "市场分量的线性天花板来自聚合，不是模型族；行级非线性 + history 应抬高 peak_m",
        "config": {k: getattr(args, k) for k in
                   ("n_folds", "train_window", "embargo", "sample_modulo", "sampling",
                    "feature_count", "history_count", "history_window", "ridge_alpha",
                    "lgbm_rounds", "lgbm_seeds", "arms")},
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
