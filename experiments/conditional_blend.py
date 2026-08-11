"""A4：条件性组合器 —— `m̂` 与 `ê` 的相对权重随状态量变化，值不值得做。

## 要验证的假设

现在的组合是 `f = m̂ + ê_lgbm`（`blend_weight = 1.0`，退化成没有任何条件性）。
若让 `ê` 的权重随一个**推理端拿得到**的状态量 `z` 变化，样本外 `peak` 提高。

状态量取 **逐 time_id 的 `ê` 截面标准差**（推理端每个 time_id 都算得出，合法）。
直觉：截面离散度大的时候，截面信号更可信、该给 `ê` 更大权重；反之收缩。

## ⭐ 基准必须取「最优**全局**权重」，不是 c = 1

`f = m̂ + c·ê` 里的全局 `c` 是**第①类纯后处理**、解析可解、0 次额度。
若拿 `c = 1` 当基准，A4 会把①类本来就免费的收益算成自己的功劳。
所以这里的基准是**在训练段上解出的最优全局 c**，A4 只赚「条件性」那一份。

## 防泄漏（照搬 A0 `multi` 臂的教训）

系数 `c_b` 在**训练段内**的内层留出段上解；且**用同一批内层模型去预测外层** ——
A0 第一版把内层标定的系数搬到全训练段重训的模型上，尺度失配，5 折从 −41% 摆到 +41%。
按外层验证段的统计量归一化 = 用到验证信息 = 泄漏，不能那样修。
分箱边界也只用内层留出段的分位数，绝不看外层。

## ⚠️ 先验低，写在前面

0c 已证明**线性**组合到顶（+0.02%）；它是被拟合对象、吃②类的病。
本脚本的价值主要是**结案**。

用法：
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python experiments/conditional_blend.py
输出：outputs/experiments/conditional_blend.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

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
from market_model import sign_test_p
from ridge_data_ladder import row_level_peak
from walk_forward_rolling import PROD_SAMPLED_WINDOW
from lgbm_xs import load_rows
from history_peak import (LGBM_MIN_DATA_FRAC, LGBM_SPEC, build_lag_cache, fit_ridge,
                          history_blocks, ridge_designs, transform_with)

N_BINS = 4          # 预注册：状态量分 4 箱（更多箱 = 更容易过拟合，4 是保守值）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A4: conditional blend of m-hat and e-hat")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="conditional_blend")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--train-window", type=int, default=39_480)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=10)
    p.add_argument("--sampling", default="periodic")
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--history-count", type=int, default=40)
    p.add_argument("--history-window", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    p.add_argument("--lgbm-rounds", type=int, default=160)
    p.add_argument("--lgbm-seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--inner-fraction", type=float, default=0.25)
    p.add_argument("--n-bins", type=int, default=N_BINS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_std(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """逐 time_id 的截面标准差，广播回每一行。推理端每个 time_id 都算得出 ⟹ 合法状态量。"""
    mean = np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))
    var = np.add.reduceat((values - mean) ** 2, starts) / counts
    return np.repeat(np.sqrt(np.maximum(var, 0.0)), counts.astype(int))


def optimal_weight(residual: np.ndarray, e_hat: np.ndarray, weight: np.ndarray) -> float:
    """min Σw(residual − c·ê)² 的闭式解。residual = y − m̂。"""
    denominator = float(np.dot(weight, e_hat * e_hat))
    return float(np.dot(weight, residual * e_hat) / denominator) if denominator > 0 else 1.0


def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"{args.label}.md"
    if report.exists() and not args.force:
        raise SystemExit(f"{report} 已存在；要覆盖请加 --force")

    files = train_files(Path(args.data_root))
    print("loading...", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    all_time_ids = data["time_id"]
    folds = rolling_time_folds(np.unique(all_time_ids), args.n_folds,
                               args.train_window, args.embargo)

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
    del scratch
    gc.collect()
    cache = build_lag_cache(files, hist_cols, args.sample_modulo, args.history_window,
                            sampling=args.sampling, verbose=False)

    fold_rows = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        tr = np.isin(all_time_ids, train_ids)
        va = np.isin(all_time_ids, valid_ids)
        fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW

        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        t_valid = transform_with(data["features"][va], stats)
        y_tr = data["target"][tr].astype(np.float64)
        w_tr = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
        tid_tr, aid_tr = data["time_id"][tr], data["asset_id"][tr]
        y_va = data["target"][va].astype(np.float64)
        w_va = np.maximum(data["weight"][va].astype(np.float64), 0.0)
        tid_va, aid_va = data["time_id"][va], data["asset_id"][va]

        # 内层：训练段末尾切一刀（留 embargo），系数与分箱边界只在这里标定
        cut = int(len(train_ids) * (1.0 - args.inner_fraction))
        i_tr = np.isin(tid_tr, train_ids[:max(1, cut - args.embargo)])
        i_va = np.isin(tid_tr, train_ids[cut:])

        sel = select_features(t_train[i_tr], y_tr[i_tr], w_tr[i_tr], args.feature_count)
        est = fit_ridge(ridge_designs(t_train[i_tr], tid_tr[i_tr], sel, None),
                        y_tr[i_tr], w_tr[i_tr], fold_alpha)

        def market(t_feat, tid):
            raw = (est.intercept_ + ridge_designs(t_feat, tid, sel, None) @ est.coef_).astype(np.float64)
            st = group_starts(tid)
            ct = np.diff(np.r_[st, len(tid)]).astype(np.float64)
            return np.repeat(np.add.reduceat(raw, st) / ct, ct.astype(int)), st, ct

        m_inner, si, ci = market(t_train[i_va], tid_tr[i_va])
        m_outer, so, co = market(t_valid, tid_va)

        # LGBM：只在内层训练段上训一次，**同一个模型**同时预测内层留出段与外层验证段
        e_target = y_tr[i_tr] - np.repeat(
            np.add.reduceat(y_tr[i_tr], group_starts(tid_tr[i_tr]))
            / np.diff(np.r_[group_starts(tid_tr[i_tr]), int(i_tr.sum())]).astype(np.float64),
            np.diff(np.r_[group_starts(tid_tr[i_tr]), int(i_tr.sum())]).astype(int))
        lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))

        def design_of(t_feat, tid, aid, mask_for_cache):
            dev = cross_sectional_deviation(t_feat[:, sel].copy(), tid)
            hist = history_blocks(cache["lags"][mask_for_cache], cache["count"][mask_for_cache],
                                  t_feat[:, hist_cols], lo, hi, ce, sc)
            return np.ascontiguousarray(np.column_stack([dev, *hist, aid.astype(np.float32)]))

        tr_idx = np.flatnonzero(tr)
        d_itr = design_of(t_train[i_tr], tid_tr[i_tr], aid_tr[i_tr], tr_idx[i_tr])
        d_iva = design_of(t_train[i_va], tid_tr[i_va], aid_tr[i_va], tr_idx[i_va])
        d_out = design_of(t_valid, tid_va, aid_va, va)
        cat = d_itr.shape[1] - 1
        params = {**LGBM_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                  "num_threads": args.num_threads,
                  "min_data_in_leaf": max(20, int(round(LGBM_MIN_DATA_FRAC * len(d_itr)))),
                  "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                  "force_row_wise": True, "feature_pre_filter": False,
                  "seed": args.lgbm_seed, "bagging_seed": args.lgbm_seed + 1000,
                  "feature_fraction_seed": args.lgbm_seed + 2000}
        booster = lgb.train(params, lgb.Dataset(d_itr, label=e_target, params=params,
                                                categorical_feature=[cat], free_raw_data=False),
                            num_boost_round=args.lgbm_rounds)

        def cross_section(design, starts, counts):
            v = booster.predict(design, num_iteration=args.lgbm_rounds).astype(np.float64)
            return v - np.repeat(np.add.reduceat(v, starts) / counts, counts.astype(int))

        e_inner = cross_section(d_iva, si, ci)
        e_outer = cross_section(d_out, so, co)
        del d_itr, d_iva, d_out, booster
        gc.collect()

        # ---- 基准：训练段内解出的**最优全局** c（第①类，本来就免费）
        r_inner = y_tr[i_va] - m_inner
        c_global = optimal_weight(r_inner, e_inner, w_tr[i_va])

        # ---- A4：按 ê 的截面离散度分箱，每箱一个 c_b（边界只用内层分位数）
        z_inner = group_std(e_inner, si, ci)
        edges = np.quantile(z_inner, np.linspace(0, 1, args.n_bins + 1)[1:-1])
        bin_inner = np.digitize(z_inner, edges)
        weights = []
        for b in range(args.n_bins):
            sel_b = bin_inner == b
            weights.append(optimal_weight(r_inner[sel_b], e_inner[sel_b], w_tr[i_va][sel_b])
                           if sel_b.sum() > 100 else c_global)

        z_outer = group_std(e_outer, so, co)
        c_outer = np.asarray(weights, dtype=np.float64)[np.digitize(z_outer, edges)]

        base = row_level_peak(y_va, m_outer + c_global * e_outer, w_va)
        cond = row_level_peak(y_va, m_outer + c_outer * e_outer, w_va)
        fold_rows.append({"fold": index, "c_global": c_global,
                          "c_bins": [float(v) for v in weights],
                          "bin_spread": float(max(weights) / max(min(weights), 1e-9)),
                          "global": base, "conditional": cond})
        print(f"  fold {index}: 全局最优 c={c_global:.4f} | 分箱 c="
              f"{[round(v,3) for v in weights]} | peak {base['peak']:.8f} → {cond['peak']:.8f} "
              f"({(cond['peak']/base['peak']-1)*100:+.2f}%) [{time.perf_counter()-started:.0f}s]",
              flush=True)
        del t_train, t_valid
        gc.collect()

    b = np.array([r["global"]["peak"] for r in fold_rows])
    c = np.array([r["conditional"]["peak"] for r in fold_rows])
    d = c - b
    drop_best = np.delete(d, int(np.argmax(d))) if len(d) > 1 else d
    checks = {"1_paired_delta_positive": bool(d.mean() > 0),
              "2_survives_drop_best_fold": bool(drop_best.mean() > 0),
              "3_relative_gain_at_least_1pct": bool(d.mean() / b.mean() >= 0.01)}
    payload = {
        "question": "让 ê 的权重随截面离散度变化，比最优全局权重更好吗？",
        "baseline_note": "基准是**训练段内解出的最优全局 c**（第①类纯后处理，本来就免费），"
                         "不是 c=1 —— 否则会把①类的收益算成 A4 的功劳",
        "leakage_guard": "系数与分箱边界只用训练段内层留出段；**同一批内层模型**预测外层"
                         "（A0 multi 臂尺度失配的教训）",
        "state_variable": "逐 time_id 的 ê 截面标准差（推理端可得）",
        "configuration": vars(args),
        "folds": fold_rows,
        "summary": {"baseline_peak_mean": float(b.mean()), "conditional_peak_mean": float(c.mean()),
                    "mean_delta": float(d.mean()), "relative_gain": float(d.mean() / b.mean()),
                    "relative_gain_drop_best": float(drop_best.mean() / b.mean()),
                    "positive_folds": int((d > 0).sum()), "n_folds": int(len(d)),
                    "sign_test_p": sign_test_p(int((d > 0).sum()), len(d))},
        "verdict": {"checks": checks, "pass": all(checks.values())},
    }
    (out / f"{args.label}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    s = payload["summary"]
    lines = ["# A4：条件性组合器（`conditional_blend`，2026-08-11）", "",
             f"**{payload['question']}**", "",
             f"- 基准：{payload['baseline_note']}",
             f"- 状态量：{payload['state_variable']}",
             f"- 防泄漏：{payload['leakage_guard']}", "",
             "| 折 | 全局最优 c | 分箱 c | 基准 peak | 条件 peak | Δ |",
             "|---:|---:|---|---:|---:|---:|"]
    for r in fold_rows:
        lines.append(f"| {r['fold']} | {r['c_global']:.4f} | "
                     f"{[round(v,3) for v in r['c_bins']]} | {r['global']['peak']:.8f} | "
                     f"{r['conditional']['peak']:.8f} | "
                     f"{(r['conditional']['peak']/r['global']['peak']-1)*100:+.2f}% |")
    lines += ["", f"折均 **{s['relative_gain']*100:+.2f}%**，{s['positive_folds']}/{s['n_folds']} 折为正，"
              f"去掉最好一折 {s['relative_gain_drop_best']*100:+.2f}%，符号检验 p={s['sign_test_p']:.3f}", "",
              "## 判据（由代码判，不是报告里的评语）", ""]
    lines += [f"- {'✅' if ok else '❌'} {k}" for k, ok in checks.items()]
    lines += ["", f"**{'✅ PASS' if payload['verdict']['pass'] else '❌ 不过'}**", ""]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    print(f"报告：{report}")


if __name__ == "__main__":
    main()
