"""2C：两个口径口子 —— LGBM 训练权重 / 截面块的选列分母。

## 口子 1：训练权重

比赛指标是 `Score = 1 − Σw(y−ŷ)² / Σwy²`，**加权**。而：

| 位置 | 是否带 `weight` |
|---|---|
| 生产 ridge `v1_ridge/train.py:187` | ✅ `sample_weight=w` |
| 当初选 LGBM 超参的 `lgbm_xs.py:297` / `lgbm_blend.py:286` | ✅ `weight=w` |
| **生产 LGBM `v3_hybrid/train.py:287`** | ❌ 无 |
| 出 A1′ 判据的 `history_peak.py:605` | ❌ 无 |

生产与判据一致（都无权）⟹ **不是 bug**。但「带权训练」这条**从来没有被量过**，
而它才是与指标对齐的那个损失。`weight` 只在训练端要（推理端拿不到），所以完全合法。

## 口子 2：选列的分母用错了矩阵

`v1_ridge/train.py:104` 的排序量是 `cov(x_j, e) / sd(x_j)`。对截面块而言：

- **分子已经精确等于 `cov(dev_j, e)`** —— 特征的截面均值部分在每个 time_id 内是常数，
  而 `e` 在每个 time_id 内无权和恒为 0，两者协方差恒等于 0；
- **分母却是 `sd(raw_j)`**，而喂给树的是 `dev_j`。

实测（p007+p008，modulo 50/phase_balanced，55,659 行）`var(dev_j)/var(raw_j)`
中位 **0.428**、10%/90% 分位 0.277/0.773 —— 市场共同分量占比在特征之间差好几倍，
于是这个分母系统性地压低了「市场分量大」的特征。生产在**内层 40 列**已经按 `dev` 选了
（`v3_hybrid/train.py:256`），**外层 200 列却按 raw**（`:240`），自相矛盾。

⚠️ 顺带记一条口径分歧（本脚本会打印，但它不是被测的臂）：出 +10.10% 判据的
`history_peak.py:672` 内层是在 **raw** 上选的，生产 `train.py:256` 在 **dev** 上选 ——
实测内层 40 列只重合 33/40，**上线模型里有 7 个 history 列从没进过那个通过判据的臂**。

## 预注册的臂（2×2，跑之前定死）

| 臂 | 训练权重 | 外层 200 列选在哪 |
|---|---|---|
| `baseline` | 无（现生产） | `raw`（现生产） |
| `weighted` | **有** | raw |
| `select_on_dev` | 无 | **dev** |
| `both` | 有 | dev |

四臂**共用同一批折、同一次流式装载、同一个岭回归 `m̂`** ⟹ 差异只来自截面块，
噪声地板按「只共享 fold 切分」那一档算（NOTES 记的 2.16e-05）。

## 口径

测量机器整套 import 自 `history_peak`，不复制粘贴。判据由 `verdict()` 机器判。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/lgbm_weight_select.py
输出：outputs/experiments/lgbm_weight_select.{json,md}
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

# (臂名, 训练带权, 外层选列基底)
ARMS: dict[str, tuple[bool, str]] = {
    "baseline": (False, "raw"),
    "weighted": (True, "raw"),
    "select_on_dev": (False, "dev"),
    "both": (True, "dev"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LGBM training weights and selection basis.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="lgbm_weight_select")
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


def run_fold(index: int, data, cache, train_ids, valid_ids, hist_cols, args) -> dict[str, Any]:
    import lightgbm as lgb

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

    # ---- 市场块：四臂共用（与生产一致，选列按 target 加权）
    fold_alpha = args.ridge_alpha * len(train_ids) / 39_480
    ridge_selected = select_features(t_train, y_tr, w_tr, args.feature_count)
    est = fit_ridge(ridge_designs(t_train, tid_tr, ridge_selected, None), y_tr, w_tr, fold_alpha)
    ridge_raw = (est.intercept_
                 + ridge_designs(t_valid, tid_va, ridge_selected, None) @ est.coef_).astype(np.float64)
    m_hat = group_mean(ridge_raw, va_starts, va_counts)
    del est, ridge_raw

    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    # 两套外层选列。`dev_all` 只为选列用一次，选完就放掉（323 列 dev 很占内存）
    selected = {"raw": select_features(t_train, e_tr, np.ones_like(e_tr), args.feature_count)}
    dev_all = cross_sectional_deviation(t_train.copy(), tid_tr)
    selected["dev"] = select_features(dev_all, e_tr, np.ones_like(e_tr), args.feature_count)
    del dev_all
    gc.collect()
    overlap = int(len(np.intersect1d(selected["raw"], selected["dev"])))

    lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
    hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
    hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

    designs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for basis, cols in selected.items():
        if basis not in {b for _, b in (ARMS[a] for a in args.arms)}:
            continue
        d_tr = np.ascontiguousarray(np.column_stack(
            [cross_sectional_deviation(t_train[:, cols].copy(), tid_tr), *hist_tr,
             aid_tr.astype(np.float32)[:, None]]))
        d_va = np.ascontiguousarray(np.column_stack(
            [cross_sectional_deviation(t_valid[:, cols].copy(), tid_va), *hist_va,
             aid_va.astype(np.float32)[:, None]]))
        designs[basis] = (d_tr, d_va)

    row: dict[str, Any] = {"fold": index, "n_train_rows": int(tr.sum()),
                           "n_valid_rows": int(va.sum()),
                           "outer_200_overlap_raw_vs_dev": overlap, "arms": {}}
    min_data = max(20, int(round(LGBM_MIN_DATA_FRAC * int(tr.sum()))))
    for arm in args.arms:
        weighted, basis = ARMS[arm]
        d_tr, d_va = designs[basis]
        cat = d_tr.shape[1] - 1
        e_hat = np.zeros(len(d_va), dtype=np.float64)
        for s in range(args.lgbm_seeds):
            seed = args.lgbm_seed + s
            params = {**LGBM_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                      "num_threads": args.num_threads, "min_data_in_leaf": min_data,
                      "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                      "force_row_wise": True, "feature_pre_filter": False,
                      "seed": seed, "bagging_seed": seed + 1000,
                      "feature_fraction_seed": seed + 2000}
            ds = lgb.Dataset(d_tr, label=e_tr, weight=(w_tr if weighted else None),
                             params=params, categorical_feature=[cat], free_raw_data=False)
            booster = lgb.train(params, ds, num_boost_round=args.lgbm_rounds)
            e_hat += booster.predict(d_va, num_iteration=args.lgbm_rounds).astype(np.float64)
            del ds, booster
        e_hat /= args.lgbm_seeds
        e_hat -= group_mean(e_hat, va_starts, va_counts)
        row["arms"][arm] = {"weighted": weighted, "select_basis": basis,
                            "full": row_level_peak(y_va, m_hat + e_hat, w_va)}
        gc.collect()

    base = row["arms"]["baseline"]["full"]["peak"] if "baseline" in row["arms"] else float("nan")
    detail = "  ".join(f"{a} {row['arms'][a]['full']['peak']:.8f}"
                       f"({(row['arms'][a]['full']['peak']/base-1)*100:+.2f}%)" for a in args.arms)
    print(f"  fold {index}: {detail}  [外层 200 列 raw∩dev = {overlap}]  "
          f"{time.perf_counter()-started:.0f}s", flush=True)
    del t_train, t_valid, designs, hist_tr, hist_va
    gc.collect()
    return row


def summarise(rows: list[dict[str, Any]], args) -> dict[str, Any]:
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
    return {"baseline_peak_mean": float(baseline.mean()),
            "outer_200_overlap_raw_vs_dev": [r["outer_200_overlap_raw_vs_dev"] for r in rows],
            "arms": out}


def render_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# 2C：LGBM 训练权重 / 截面块选列分母",
        "",
        "两个口子，四臂 2×2，共用同一批折与同一个岭回归 `m̂` ⟹ 差异只来自截面块。",
        "",
        f"- 折数 {payload['config']['n_folds']}，train_window {payload['config']['train_window']:,}，"
        f"modulo {payload['config']['sample_modulo']}/{payload['config']['sampling']}，"
        f"LGBM {payload['config']['lgbm_rounds']} 轮 × {payload['config']['lgbm_seeds']} 种子",
        f"- ⭐ 基准强度：peak **{s['baseline_peak_mean']:.8f}**（读相对增益前先看这个）",
        f"- 外层 200 列 `raw` 与 `dev` 两种选法的逐折重合数：{s['outer_200_overlap_raw_vs_dev']}",
        "",
        "| 臂 | 训练带权 | 外层选列 | Δpeak | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | 判据 |",
        "|---|:--:|:--:|--:|--:|--:|--:|--:|:--:|:--:|",
    ]
    for arm, entry in s["arms"].items():
        weighted, basis = payload["arm_spec"][arm]
        st = entry["stats"]
        lines.append(
            f"| `{arm}` | {'✅' if weighted else '—'} | `{basis}` | "
            f"**{st['relative_gain']*100:+.2f}%** | {st['positive_folds']}/{st['n_folds']} | "
            f"{st['relative_gain_drop_best']*100:+.2f}% | {entry['delta_A']*100:+.2f}% | "
            f"{entry['delta_B']*100:+.2f}% | {'✅' if entry['mechanism_2dA_gt_dB'] else '❌'} | "
            f"{'✅ PASS' if entry['verdict']['pass'] else '❌'} |")
    lines += ["", "判据（`history_peak.verdict`，机器判）：配对 Δ 均值为正 + 去掉最好一折仍为正 + ≥ +1%。"]
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

    tr0 = np.isin(all_time_ids, folds[0][0])
    scratch, _ = robust_transform_fit(data["features"][tr0].copy())
    y0 = data["target"][tr0].astype(np.float64)
    tid0 = data["time_id"][tr0]
    s0 = group_starts(tid0)
    c0 = np.diff(np.r_[s0, len(tid0)]).astype(np.float64)
    e0 = y0 - np.repeat(np.add.reduceat(y0, s0) / c0, c0.astype(int))
    pool = select_features(scratch, e0, np.ones_like(e0), args.feature_count)
    # ⚠️ 与生产 v3_hybrid/train.py:256 同口径：内层在 **dev** 上选（history_peak 是在 raw 上选的）
    pool_dev = cross_sectional_deviation(scratch[:, pool].copy(), tid0)
    hist_cols = np.sort(pool[select_features(pool_dev, e0, np.ones_like(e0), args.history_count)])
    hist_cols_raw = np.sort(pool[select_features(scratch[:, pool], e0, np.ones_like(e0),
                                                 args.history_count)])
    print(f"history 列 {len(hist_cols)} 个（生产口径 dev）；"
          f"与 history_peak 的 raw 口径重合 {len(np.intersect1d(hist_cols, hist_cols_raw))}/{len(hist_cols)}",
          flush=True)
    del scratch, pool_dev, y0, e0
    gc.collect()

    print("building lag cache (streams every row)...", flush=True)
    cache = build_lag_cache(train_files(Path(args.data_root)), hist_cols,
                            args.sample_modulo, args.history_window, sampling=args.sampling)
    assert np.array_equal(cache["time_id"], all_time_ids), "lag 缓存与采样矩阵的 time_id 不对齐"

    rows = [run_fold(i, data, cache, tr_ids, va_ids, hist_cols, args)
            for i, (tr_ids, va_ids) in enumerate(folds)]
    payload = {
        "experiment": "lgbm_weight_select",
        "config": {k: getattr(args, k) for k in
                   ("n_folds", "train_window", "embargo", "sample_modulo", "sampling",
                    "feature_count", "history_count", "history_window", "ridge_alpha",
                    "lgbm_rounds", "lgbm_seeds", "arms")},
        "arm_spec": {a: list(ARMS[a]) for a in args.arms},
        "history_columns": [int(c) for c in hist_cols],
        "history_columns_overlap_with_screening_raw":
            int(len(np.intersect1d(hist_cols, hist_cols_raw))),
        "folds": rows,
        "summary": summarise(rows, args),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_report(payload), encoding="utf-8")
    print(f"\n写出 {json_path}\n写出 {md_path}")
    print(render_report(payload))


if __name__ == "__main__":
    main()
