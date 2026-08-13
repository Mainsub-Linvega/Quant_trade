"""2A × 2C 组合臂：行级市场模型 与 带权训练 能不能叠加？

## 为什么要单独跑这一支

08-13 两个改动各自过了判据，但它们**作用在不同分量上**：

| 改动 | 动哪一块 | 本地 Δpeak | 判据 |
|---|---|--:|:--:|
| 2A `y_raw` λ=0.5（`lgbm_market_row`） | 市场块 `m̂` | +15.18% | 5/5 ✅ |
| 2C `weighted`（`lgbm_weight_select`） | 截面块 `ê` | +4.06% | 5/5 ✅ |

「若独立则约 +19.9%」是**推算，不是实测**。而且还有一个从没测过的格子：
**市场模型自己要不要也带权训练**（2A 那一跑是无权的，与生产 `ê` 口径一致）。

公榜额度是按「一个模型一次」花的 ⟹ **先在本地把组合定死，一次公榜点买到的信息才最多。**
本脚本 0 次额度、约 20 分钟。

## 预注册的六个臂（跑之前定死）

`f = (1−λ)·m̂_ridge + λ·m̂_lgbm + ê_lgbm`，λ=0.5 固定（先验，ROADMAP §5）。

| 臂 | `ê` 带权 | 市场模型 | 市场模型带权 | 作用 |
|---|:--:|---|:--:|---|
| `baseline` | ❌ | 无（λ=0） | — | 现生产架构，配对基准 |
| `w_e` | ✅ | 无 | — | 复现 2C |
| `mkt` | ❌ | `y_raw` | ❌ | 复现 2A |
| `mkt_we` | ✅ | `y_raw` | ❌ | 组合 |
| `mkt_wm` | ❌ | `y_raw` | ✅ | 新格子 |
| `mkt_both` | ✅ | `y_raw` | ✅ | 组合 |

⚠️ **`mkt` 与 `w_e` 必须复现出各自原实验的量级**，否则说明这一跑的口径与前两跑不一致，
后面的组合数就没法和它们对读 —— 这是本脚本的**阴性对照**。

选列一律走 `raw` 基底（`select_on_dev` 08-13 已被否决：重合 191~193/200、Δpeak −0.26%）。

## 怎么读

报告会同时给「实测组合增益」与「若两项独立的预测值 `(1+Δa)(1+Δb)−1`」。
- 实测 ≈ 预测 ⟹ 两项独立，可以分别接线；
- 实测 < 预测 ⟹ 有重叠，得挑一个或重新定权重。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/combo_market_weight.py
输出：outputs/experiments/combo_market_weight.{json,md}
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

LAMBDA = 0.5          # 先验，不拟合（ROADMAP §5）；2A 已验证它优于 1.0
# 臂名 -> (ê 带权, 用市场模型, 市场模型带权)
ARMS: dict[str, tuple[bool, bool, bool]] = {
    "baseline": (False, False, False),
    "w_e":      (True,  False, False),
    "mkt":      (False, True,  False),
    "mkt_we":   (True,  True,  False),
    "mkt_wm":   (False, True,  True),
    "mkt_both": (True,  True,  True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Do the market model and weighted training compose?")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="combo_market_weight")
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

    cat = design_tr.shape[1] - 1                   # ⚠️ asset_id 恒为最后一列
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

    # ---- 市场基准：生产岭回归（全臂共用）
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

    # 截面块设计（生产口径，361 列）
    d_tr_xs = np.ascontiguousarray(np.column_stack([xs_tr, *hist_tr, aid_tr.astype(np.float32)[:, None]]))
    d_va_xs = np.ascontiguousarray(np.column_stack([xs_va, *hist_va, aid_va.astype(np.float32)[:, None]]))
    needed = {ARMS[a] for a in args.arms}

    # ---- 四个模型最多各训一次，臂之间复用
    e_hat: dict[bool, np.ndarray] = {}
    for weighted in sorted({spec[0] for spec in needed}):
        pred = fit_predict(d_tr_xs, e_tr, (w_tr if weighted else None), d_va_xs, args)
        e_hat[weighted] = pred - group_mean(pred, va_starts, va_counts)

    m_lgbm: dict[bool, np.ndarray] = {}
    if any(spec[1] for spec in needed):
        # 市场块设计（2A 的 y_raw 臂，561 列）
        d_tr_y = np.ascontiguousarray(np.column_stack(
            [t_train[:, lgbm_selected], xs_tr, *hist_tr, aid_tr.astype(np.float32)[:, None]]))
        d_va_y = np.ascontiguousarray(np.column_stack(
            [t_valid[:, lgbm_selected], xs_va, *hist_va, aid_va.astype(np.float32)[:, None]]))
        for weighted in sorted({spec[2] for spec in needed if spec[1]}):
            pred = fit_predict(d_tr_y, y_tr, (w_tr if weighted else None), d_va_y, args)
            m_lgbm[weighted] = group_mean(pred, va_starts, va_counts)
        del d_tr_y, d_va_y
        gc.collect()

    row: dict[str, Any] = {"fold": index, "n_train_rows": int(tr.sum()),
                           "n_valid_rows": int(va.sum()),
                           "peak_m_ridge": row_level_peak(y_va, m_ridge, w_va)["peak"], "arms": {}}
    for arm in args.arms:
        we, use_mkt, wm = ARMS[arm]
        market = ((1.0 - LAMBDA) * m_ridge + LAMBDA * m_lgbm[wm]) if use_mkt else m_ridge
        row["arms"][arm] = {"full": row_level_peak(y_va, market + e_hat[we], w_va),
                            "peak_m": row_level_peak(y_va, market, w_va)["peak"]}

    base = row["arms"]["baseline"]["full"]["peak"]
    detail = "  ".join(f"{a} {(row['arms'][a]['full']['peak']/base-1)*100:+.2f}%" for a in args.arms)
    print(f"  fold {index}: base {base:.8f} | {detail}  {time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, d_tr_xs, d_va_xs, xs_tr, xs_va, hist_tr, hist_va
    gc.collect()
    return row


def summarise(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    baseline = np.array([r["arms"]["baseline"]["full"]["peak"] for r in rows])
    base_m = np.array([r["peak_m_ridge"] for r in rows])
    base_a = float(np.mean([r["arms"]["baseline"]["full"]["A"] for r in rows]))
    base_b = float(np.mean([r["arms"]["baseline"]["full"]["B"] for r in rows]))
    out: dict[str, Any] = {}
    for arm in args.arms:
        if arm == "baseline":
            continue
        peaks = np.array([r["arms"][arm]["full"]["peak"] for r in rows])
        mkt = np.array([r["arms"][arm]["peak_m"] for r in rows])
        stats = paired_stats(peaks - baseline, baseline)
        d_a = float(np.mean([r["arms"][arm]["full"]["A"] for r in rows])) / base_a - 1.0
        d_b = float(np.mean([r["arms"][arm]["full"]["B"] for r in rows])) / base_b - 1.0
        out[arm] = {"stats": stats, "verdict": verdict(stats), "delta_A": d_a, "delta_B": d_b,
                    "mechanism_2dA_gt_dB": 2 * d_a > d_b,
                    "market_peak_stats": paired_stats(mkt - base_m, base_m)}
    # 独立性检验：组合臂的实测 vs (1+Δmkt)(1+Δw_e)−1
    independence = {}
    for combo, parts in (("mkt_we", ("mkt", "w_e")), ("mkt_both", ("mkt_wm", "w_e"))):
        if combo in out and all(p in out for p in parts):
            predicted = (1 + out[parts[0]]["stats"]["relative_gain"]) * \
                        (1 + out[parts[1]]["stats"]["relative_gain"]) - 1
            found = out[combo]["stats"]["relative_gain"]
            independence[combo] = {"parts": list(parts), "predicted_if_independent": predicted,
                                   "measured": found, "shortfall": found - predicted}
    return {"baseline_peak_mean": float(baseline.mean()),
            "baseline_peak_m_mean": float(base_m.mean()),
            "arms": out, "independence": independence}


def render_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# 2A × 2C 组合臂：行级市场模型 与 带权训练 能不能叠加？",
        "",
        f"`f = 0.5·m̂_ridge + 0.5·m̂_lgbm + ê_lgbm`（λ 先验固定）。六臂共用同一批折、"
        "同一个岭回归 `m̂`、同一份 lag 缓存。",
        "",
        f"- 折数 {payload['config']['n_folds']}，train_window {payload['config']['train_window']:,}，"
        f"modulo {payload['config']['sample_modulo']}/{payload['config']['sampling']}，"
        f"LGBM {payload['config']['lgbm_rounds']} 轮 × {payload['config']['lgbm_seeds']} 种子",
        f"- ⭐ 基准强度：整体 peak **{s['baseline_peak_mean']:.8f}**，"
        f"市场块 peak_m **{s['baseline_peak_m_mean']:.8f}**",
        "",
        "| 臂 | ê 带权 | 市场模型 | 市场模型带权 | Δpeak | 正折 | 去最好折 | Δpeak_m | ΔA | ΔB | 2ΔA>ΔB | 判据 |",
        "|---|:--:|:--:|:--:|--:|--:|--:|--:|--:|--:|:--:|:--:|",
    ]
    for arm, entry in s["arms"].items():
        we, use_mkt, wm = payload["arm_spec"][arm]
        st, mk = entry["stats"], entry["market_peak_stats"]
        lines.append(
            f"| `{arm}` | {'✅' if we else '—'} | {'✅' if use_mkt else '—'} | "
            f"{'✅' if wm else '—'} | **{st['relative_gain']*100:+.2f}%** | "
            f"{st['positive_folds']}/{st['n_folds']} | {st['relative_gain_drop_best']*100:+.2f}% | "
            f"{mk['relative_gain']*100:+.2f}% | {entry['delta_A']*100:+.2f}% | "
            f"{entry['delta_B']*100:+.2f}% | {'✅' if entry['mechanism_2dA_gt_dB'] else '❌'} | "
            f"{'✅ PASS' if entry['verdict']['pass'] else '❌'} |")
    lines += ["", "## 两项能不能叠加", "",
              "| 组合 | 由哪两项 | 若独立应有 | 实测 | 差额 |", "|---|---|--:|--:|--:|"]
    for combo, entry in s["independence"].items():
        lines.append(f"| `{combo}` | `{'` + `'.join(entry['parts'])}` | "
                     f"{entry['predicted_if_independent']*100:+.2f}% | "
                     f"**{entry['measured']*100:+.2f}%** | {entry['shortfall']*100:+.2f}% |")
    lines += [
        "",
        "⚠️ **阴性对照**：`mkt` 应复现 `lgbm_market_row` 的 +15.18%、`w_e` 应复现",
        "`lgbm_weight_select` 的 +4.06%。对不上说明这一跑的口径与前两跑不一致，组合数不可对读。",
        "",
        "判据（`history_peak.verdict`，机器判）：配对 Δ 均值为正 + 去掉最好一折仍为正 + ≥ +1%。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if "baseline" not in args.arms:
        raise SystemExit("baseline 臂是配对基准，必须在 --arms 里")
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

    # history 列：与 A1′ / 2A 逐位同口径（第 0 折训练窗预注册，raw 基底）
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
        "experiment": "combo_market_weight",
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
