"""把 LightGBM 的截面分量装进**完整预测**，整体量一次。

## 这一步在回答什么

`lgbm_xs` 测出 LightGBM 预测截面块 `e` 比岭回归强（t=+6.4，5/5 折）。但那是
**在试验台上单独测那一块**，而比较对象是「只做截面块的 ridge」——
**生产模型是一个 ridge 同时做两块，赢了前者不等于赢了后者。**

这里把它装进车里：拿生产口径 ridge 的完整预测 `ŷ`，拆成
市场分量 `m̂`（逐 time_id 加权截面均值）与截面分量 `ê_ridge = ŷ − m̂`，
然后把 `ê` 换成（或混入）LightGBM 的，**用比赛真正的评分口径整体量一次**。

任何预测都能这么拆，不需要模型本身支持。

## 三件顺带验掉的事

1. **ê 必须投影成截面加权零均值**：`Δscore = 0.28·ΔR²_e` 的恒等式要求预测能干净拆成
   市场 + 截面两块。LGBM 的输出不保证 `Σw·ê = 0`，不为 0 的部分会污染市场分量。
   投影只赚不亏（`Σw·e = 0` 且 `Σw·ê_dev = 0` ⟹
   `Σw(e−ê)² = Σw(e−ê_dev)² + ê_mean²Σw ≥ Σw(e−ê_dev)²`），
   这里**实测 ê_mean 到底多大** —— `lgbm_xs` 里唯一没测的量。
2. **折扣规则的排序翻转对不对**：`xs_loose` 本地分最高但 ΔB 涨 28.1%，
   折后不如 ΔB 只涨 8.6% 的 `xs_shrunk`。两组都跑，直接看装车后谁赢。
3. **替换 vs 混合**：`blend50` 用 0.5/0.5 而**不拟合权重** —— ROADMAP §5 的规矩，
   拟合单个标量的教训见 `ab_scale_auto`（每折 a* 从 0.34 到 1.78，测不出收益）。

## 为什么比峰值 A²/B 而不是原始分

`A = Σw·y·f/Σw·y²`、`B = Σw·f²/Σw·y²`，`Score(a) = 2aA − a²B`，峰值 `= A²/B`
**与 scale 无关**。scale 是纯后处理旋钮，各臂的最优 scale 本来就不同，
比原始分等于比谁的 scale 碰得准。峰值是唯一能公平比较不同模型的量。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/lgbm_blend.py
    # 试水：--sample-modulo 20 --n-folds 3 --candidates xs_shrunk --n-seeds 1 --report lgbm_blend_trial
输出：outputs/experiments/lgbm_blend.{json,md}（同名已存在需 --force）
"""

from __future__ import annotations

import argparse
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

from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import robust_transform_fit, select_features
from features import apply_robust_transform, cross_sectional_deviation
from mt_lagged import weighted_r2
from lgbm_mt import (accumulate, ridge_predictions, ab_terms,
                     paired_stats, verdict_of, self_check)
from lgbm_xs import load_rows

PUBLIC_REFERENCE = 0.00187232      # 严格模型 @ scale 1.13 的公榜实测（2026-08-08）

# 生产口径设计矩阵是 [raw(200) ‖ dev(200)] = 400 列。本实验 modulo 5 下内层训练段
# 约 106 万行，与生产的 1,146,659 行几乎同量级 → 最优 alpha 应在生产的 2e6 附近。
# 网格向两侧各留两档，并在报告里检查最优值是否落在**内部**（落端点 = 基准被压弱）。
# 烟测（modulo 20，内层训练仅 3.3 万行）时最优 alpha 落到 1e4 端点触发了告警 ——
# 最优 alpha 随样本量走，小样本本来就该更强正则。正式跑是 106 万行，预期在 1e6 附近。
# gram 只累积一次、多几档 alpha 是白送，所以两端各留足。
ALPHAS_PROD = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

# 候选沿用 lgbm_xs，但 min_data_in_leaf 改成**训练行数的比例** ——
# lgbm_xs 的内层训练段约 350 万行，这里 modulo 5 只有约 106 万行，
# 沿用绝对值会让树被过度约束、候选的有效容量整个变掉。
CANDIDATES: dict[str, dict[str, Any]] = {
    "xs_shrunk": {"num_leaves": 15, "min_data_frac": 100000 / 3_500_000,   # ≈2.86%
                  "learning_rate": 0.02, "feature_fraction": 0.4, "lambda_l2": 30.0},
    "xs_loose":  {"num_leaves": 63, "min_data_frac": 12000 / 3_500_000,    # ≈0.34%
                  "learning_rate": 0.03, "feature_fraction": 0.7, "lambda_l2": 1.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Does the LGBM cross-sectional part improve the FULL score?")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--report", default="lgbm_blend_unweighted")
    p.add_argument("--force", action="store_true")
    # 默认无权 —— 推理端拿不到 weight（见 split_market 的 docstring）。
    # weighted 只用于对读首轮那份加权结果。
    p.add_argument("--split-weighting", choices=["unweighted", "weighted"], default="unweighted")
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced")
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=None)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--inner-frac", type=float, default=0.10)
    p.add_argument("--max-rounds", type=int, default=1500)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--candidates", nargs="*", default=None)
    return p.parse_args()


# ------------------------------------------------------------------ 截面工具

def group_starts(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    return starts, np.diff(np.r_[starts, len(time_id)])


def split_market(pred: np.ndarray, weight: np.ndarray | None, starts: np.ndarray,
                 counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把任意预测拆成 (市场分量 m̂ 广播回行, 截面分量 pred − m̂)。

    `weight=None` → 无权均值 `m̂_t = mean(pred)`；否则加权 `Σwᵢpredᵢ/Σwᵢ`。

    ⚠️ **推理时只能用无权。** `data/test/*.parquet` 没有 `weight` 列，
    而且 `timeseries_api/runner.py:75` 的 `forbidden = {"weight", "target", ...}`
    会在交给 `predict` 之前把它剥掉 —— 即使跑训练集也拿不到。
    权重也重建不出来：实测每个 asset 约 10 万个不同权重值（几乎每 time_id 一个）、
    组内变异系数 0.15~0.60，且各资产均值跨分区大幅漂移。

    所以默认走无权。加权分支只保留用于对读 ——
    两者之差 = 「用掉推理拿不到的信息」值多少分。
    """
    if weight is None:
        mean = np.add.reduceat(pred, starts) / counts
    else:
        mean = np.add.reduceat(weight * pred, starts) / np.add.reduceat(weight, starts)
    broadcast = np.repeat(mean, counts)
    return broadcast, pred - broadcast


def market_residual(target: np.ndarray, weight: np.ndarray | None,
                    time_id: np.ndarray) -> np.ndarray:
    """训练目标 `e = y − m_t`。与 split_market 同口径（无权 / 加权）。

    不复用 `lgbm_xs.cross_sectional_target` —— 那个写死了加权，
    而 lgbm_xs 是已出结论的实验，不去改它。
    """
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    _, residual = split_market(target, weight, starts, counts)
    return residual


def peak_of(actual: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """峰值 = A²/B，与 scale 无关；顺带回最优 scale。"""
    a, b = ab_terms(actual, pred, weight)
    return {"A": a, "B": b,
            "peak": (a * a / b) if b > 0 else 0.0,
            "best_scale": (a / b) if b > 0 else 0.0}


# ---------------------------------------------------------------------- 主流程

def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.report}.json"
    md_path = output_dir / f"{args.report}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"报告已存在：{json_path} / {md_path}。要覆盖请显式加 --force")

    names = list(args.candidates) if args.candidates else list(CANDIDATES)
    if unknown := [n for n in names if n not in CANDIDATES]:
        raise SystemExit(f"未知候选：{unknown}，可选 {list(CANDIDATES)}")

    self_check()

    print(f"加载训练数据（modulo {args.sample_modulo} / {args.sampling}）…", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features, y, w, tid, aid = (data["features"], data["target"], data["weight"],
                                data["time_id"], data["asset_id"])
    del data
    assert np.all(np.diff(tid) >= 0), "行未按 time_id 排序，截面聚合会算错"
    # 截面均值的口径：无权（推理可做）还是加权（推理做不到，只为对读）
    split_w = w if args.split_weighting == "weighted" else None
    e = market_residual(y, split_w, tid)
    uniq_tid = tid[np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]]
    print(f"{len(tid):,} 行 / {len(uniq_tid):,} 个 time_id；"
          f"截面均值口径 = **{args.split_weighting}**"
          + ("（推理端拿不到 weight，这才是能落地的口径）" if split_w is None
             else "  ⚠️ 推理端做不到，仅供对读"), flush=True)

    train_window = args.train_window or int(len(uniq_tid) * 4 / 9)
    folds = rolling_time_folds(uniq_tid, args.n_folds, train_window, args.embargo)
    n_assets = int(aid.max()) + 1
    arm_names = ["baseline"] + [f"{k}_{c}" for c in names for k in ("replace", "blend50")]

    results: list[dict[str, Any]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()

        def rows_of(ids: np.ndarray) -> np.ndarray:
            return np.arange(int(np.searchsorted(tid, ids[0], "left")),
                             int(np.searchsorted(tid, ids[-1], "right")))

        n_tr = len(train_ids)
        n_inner = max(1, int(n_tr * args.inner_frac))
        inner_valid_ids = train_ids[n_tr - n_inner:]
        inner_train_ids = train_ids[: n_tr - n_inner - args.embargo]
        it_rows, iv_rows = rows_of(inner_train_ids), rows_of(inner_valid_ids)
        va_rows = rows_of(valid_ids)
        assert int(tid[iv_rows[-1]]) < int(valid_ids[0]), "内层早停集越过了外层验证段"

        fold: dict[str, Any] = {
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "inner_train_time_range": [int(inner_train_ids[0]), int(inner_train_ids[-1])],
            "inner_valid_time_range": [int(inner_valid_ids[0]), int(inner_valid_ids[-1])],
            "rows": {"inner_train": len(it_rows), "inner_valid": len(iv_rows),
                     "outer_valid": len(va_rows)},
            "arms": {},
        }

        # ---- 预处理与选列：只在内层训练段拟合（生产口径）
        scratch, stats = robust_transform_fit(features[it_rows].copy())
        selected = select_features(scratch, y[it_rows], w[it_rows], args.feature_count)
        del scratch
        raw200 = features[:, selected].copy()
        apply_robust_transform(raw200, stats["lower"][selected], stats["upper"][selected],
                               stats["center"][selected], stats["scale"][selected])
        dev200 = cross_sectional_deviation(raw200.copy(), tid)

        y_va, w_va, e_va = y[va_rows], w[va_rows], e[va_rows]
        starts_va, counts_va = group_starts(tid[va_rows])

        # ---- baseline：生产口径 ridge，[raw ‖ dev] 400 列 → 直接预测 y
        prod_it = np.column_stack([raw200[it_rows], dev200[it_rows]])
        acc = accumulate(prod_it, y[it_rows], w[it_rows])
        del prod_it
        prod_iv = np.column_stack([raw200[iv_rows], dev200[iv_rows]])
        inner_preds = ridge_predictions(acc, prod_iv, ALPHAS_PROD)
        knob = max(inner_preds, key=lambda k: weighted_r2(y[iv_rows], inner_preds[k], w[iv_rows]))
        del prod_iv, inner_preds
        prod_va = np.column_stack([raw200[va_rows], dev200[va_rows]])
        yhat_base = ridge_predictions(acc, prod_va, [float(knob)])[knob]
        del prod_va, acc, raw200

        # 拆解与投影一律走 split_w（默认 None = 无权），与推理端能做的事对齐
        w_va_split = None if split_w is None else w_va
        m_hat, e_ridge = split_market(yhat_base, w_va_split, starts_va, counts_va)
        fold["arms"]["baseline"] = {"knob": knob, **peak_of(y_va, yhat_base, w_va)}
        # 交叉验证：weighted_zero_mean_r2 与 2aA−a²B 必须一致
        pk = fold["arms"]["baseline"]
        for a_test in (0.5, 1.0, pk["best_scale"]):
            lhs = weighted_zero_mean_r2(y_va, a_test * yhat_base, w_va)
            rhs = 2 * a_test * pk["A"] - a_test * a_test * pk["B"]
            assert abs(lhs - rhs) < 1e-12, f"评分口径不一致：{lhs} vs {rhs}"

        # ---- LightGBM 臂
        x_it = np.ascontiguousarray(np.column_stack([dev200[it_rows], aid[it_rows].astype(np.float32)]))
        x_iv = np.ascontiguousarray(np.column_stack([dev200[iv_rows], aid[iv_rows].astype(np.float32)]))
        x_va = np.ascontiguousarray(np.column_stack([dev200[va_rows], aid[va_rows].astype(np.float32)]))
        del dev200
        cat_index = x_it.shape[1] - 1

        for name in names:
            spec = dict(CANDIDATES[name])
            min_data = max(20, int(round(spec.pop("min_data_frac") * len(it_rows))))
            patience = max(args.early_stopping, int(round(1.0 / spec["learning_rate"])))
            t0 = time.perf_counter()
            acc_pred, best_iters = np.zeros(len(va_rows)), []
            for s in range(args.n_seeds):
                params = {
                    "objective": "regression", "metric": "l2", "verbosity": -1,
                    "num_threads": args.num_threads, "seed": args.seed + s,
                    "bagging_seed": args.seed + 1000 + s,
                    "feature_fraction_seed": args.seed + 2000 + s,
                    "bagging_fraction": 0.7, "bagging_freq": 1,
                    "deterministic": True, "force_row_wise": True,
                    "feature_pre_filter": False,       # 见 lgbm_mt 的坑：跨候选复用会被预筛选污染
                    "min_data_in_leaf": min_data, **spec,
                }
                ds_tr = lgb.Dataset(x_it, label=e[it_rows], weight=w[it_rows], params=params,
                                    categorical_feature=[cat_index], free_raw_data=False)
                ds_va = lgb.Dataset(x_iv, label=e[iv_rows], weight=w[iv_rows], reference=ds_tr,
                                    params=params, categorical_feature=[cat_index],
                                    free_raw_data=False)
                booster = lgb.train(params, ds_tr, num_boost_round=args.max_rounds,
                                    valid_sets=[ds_va], valid_names=["inner"],
                                    callbacks=[lgb.early_stopping(patience, verbose=False)])
                best = int(booster.best_iteration or booster.current_iteration())
                best_iters.append(best)
                acc_pred += booster.predict(x_va, num_iteration=best)
                del booster, ds_tr, ds_va
            e_raw = acc_pred / float(args.n_seeds)

            # **投影成逐 time_id 加权零均值** —— 恒等式成立的前提
            e_mean_bc, e_lgbm = split_market(e_raw, w_va_split, starts_va, counts_va)
            resid = float(np.abs(np.add.reduceat(
                (w_va if w_va_split is not None else np.ones_like(e_lgbm)) * e_lgbm,
                starts_va)).max())
            assert resid < 1e-8, f"投影后逐 time_id 的 ê 之和不为 0（{resid:.2e}）"

            arms = {
                f"replace_{name}": m_hat + e_lgbm,
                f"blend50_{name}": m_hat + 0.5 * e_ridge + 0.5 * e_lgbm,
            }
            for arm, pred in arms.items():
                fold["arms"][arm] = {"knob": "/".join(map(str, best_iters)),
                                     "min_data_in_leaf": min_data,
                                     **peak_of(y_va, pred, w_va)}
            # ê_mean 有多大 + 投影捡回来多少（lgbm_xs 里唯一没测的量）
            fold["arms"][f"replace_{name}"].update({
                "e_mean_rms": float(np.sqrt(np.average(e_mean_bc ** 2, weights=w_va))),
                "e_lgbm_rms": float(np.sqrt(np.average(e_lgbm ** 2, weights=w_va))),
                "r2_e_before_projection": weighted_r2(e_va, e_raw, w_va),
                "r2_e_after_projection": weighted_r2(e_va, e_lgbm, w_va),
                "r2_e_ridge": weighted_r2(e_va, e_ridge, w_va),
                "fit_seconds": float(time.perf_counter() - t0),
            })
            print(f"  fold {index:2d} {name:11s} best_iter={best_iters} min_data={min_data:,} "
                  f"replace峰值={fold['arms'][f'replace_{name}']['peak']:.6f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

        del x_it, x_iv, x_va
        fold["elapsed_seconds"] = float(time.perf_counter() - started)
        results.append(fold)
        print(f"fold {index:2d} 完成：baseline 峰值 {fold['arms']['baseline']['peak']:.6f} "
              f"(α={knob}, {fold['elapsed_seconds']:.0f}s)", flush=True)

    # ------------------------------------------------------------- 汇总
    knobs = [f["arms"]["baseline"]["knob"] for f in results]
    alpha_at_edge = any(k in (f"{ALPHAS_PROD[0]:.0e}", f"{ALPHAS_PROD[-1]:.0e}") for k in knobs)
    if alpha_at_edge:
        print(f"\n⚠️⚠️ baseline 的 alpha 有落在网格端点的（{sorted(set(knobs))}）—— "
              f"基准被人为压弱，本次比较**不可信**，请先扩 ALPHAS_PROD 重跑。\n", flush=True)

    summary = {arm: {
        "mean_peak": float(np.mean([f["arms"][arm]["peak"] for f in results])),
        "mean_best_scale": float(np.mean([f["arms"][arm]["best_scale"] for f in results])),
        "A": float(np.mean([f["arms"][arm]["A"] for f in results])),
        "B": float(np.mean([f["arms"][arm]["B"] for f in results])),
        "per_fold_peak": [f["arms"][arm]["peak"] for f in results],
    } for arm in arm_names}

    base = "baseline"
    comparisons: dict[str, Any] = {}
    for arm in arm_names:
        if arm == base:
            continue
        deltas = [a - b for a, b in zip(summary[arm]["per_fold_peak"],
                                        summary[base]["per_fold_peak"])]
        st = paired_stats(deltas)
        da = (summary[arm]["A"] - summary[base]["A"]) / abs(summary[base]["A"])
        db = (summary[arm]["B"] - summary[base]["B"]) / abs(summary[base]["B"])
        comparisons[arm] = {
            **st,
            "relative_pct": st["mean"] / summary[base]["mean_peak"] * 100.0,
            "delta_A_pct": da * 100.0, "delta_B_pct": db * 100.0,
            "discounted_mechanism_ok": bool(2.0 * (da / 2.2) > db),
            "discounted_peak_pct": ((1 + da / 2.2) ** 2 / (1 + db) - 1) * 100.0,
            "detection_floor": None if st["se"] is None else float(2.0 * st["se"]),
            "detection_floor_pct": None if st["se"] is None else float(
                2.0 * st["se"] / summary[base]["mean_peak"] * 100.0),
            "verdict": verdict_of(st, da, db),
        }

    proj = {n: {
        "e_mean_rms": [f["arms"][f"replace_{n}"]["e_mean_rms"] for f in results],
        "e_lgbm_rms": [f["arms"][f"replace_{n}"]["e_lgbm_rms"] for f in results],
        "r2_e_before": [f["arms"][f"replace_{n}"]["r2_e_before_projection"] for f in results],
        "r2_e_after": [f["arms"][f"replace_{n}"]["r2_e_after_projection"] for f in results],
        "r2_e_ridge": [f["arms"][f"replace_{n}"]["r2_e_ridge"] for f in results],
    } for n in names}

    payload = {
        "question": "把 LightGBM 的截面分量装进完整预测，比赛评分口径下还赢吗？",
        "metric": "峰值 = A²/B（与 scale 无关，唯一能公平比较不同模型的量）",
        "baseline": "生产口径 ridge：[raw(200) ‖ dev(200)] → y，alpha 在内层验证段上选",
        "decision_rule": ("PASS 需 t ≥ 2 且去掉最好一折后 t ≥ 1 且 2·ΔA/2.2 > ΔB。"
                          "由 verdict_of() 计算写入本文件，不靠人在 md 里下判断。"),
        "alpha_at_grid_edge": bool(alpha_at_edge),
        "split_weighting": args.split_weighting,
        "leakage_controls": {
            "inner_split": f"训练段尾部 {args.inner_frac:.0%} 个 time_id + embargo {args.embargo}",
            "outer_valid_never_seen": True,
            "preprocessing_selection_alpha_rounds_all_from_inner_only": True,
            "market_split_uses_prediction_not_label":
                "m̂ 来自 baseline 预测自身的截面均值，不是真实 m_t —— 推理时可算，因果成立",
            "inference_available_information": (
                "unweighted：截面均值全部无权，与推理端一致（test 无 weight 列，"
                "且 runner.py:75 的 forbidden 会剥掉它）"
                if args.split_weighting == "unweighted" else
                "⚠️ weighted：用了推理端拿不到的 weight，**结论不可落地**，仅供对读"),
            "scoring_uses_weights": "算分仍用 src/metric.py 的加权口径 —— 那是比赛评分本身",
        },
        "configuration": {
            "sample_modulo": args.sample_modulo, "sampling": args.sampling,
            "rows": int(len(tid)), "time_ids": int(len(uniq_tid)),
            "feature_count": args.feature_count, "n_assets": n_assets,
            "n_folds": args.n_folds, "train_window": train_window, "embargo": args.embargo,
            "alphas_prod": ALPHAS_PROD, "max_rounds": args.max_rounds,
            "n_seeds": args.n_seeds, "seed": args.seed,
            "public_reference_score": PUBLIC_REFERENCE,
            "candidates": {n: CANDIDATES[n] for n in names},
        },
        "summary": summary, "comparisons": comparisons, "projection_diagnostics": proj,
        "folds": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- markdown
    def fmt(v, spec=".2f"):
        return "n/a" if v is None else format(v, spec)

    lines = [
        "# LightGBM 的截面分量装进完整预测 —— 整体量一次",
        "",
        f"{len(tid):,} 行 / {len(uniq_tid):,} 个 time_id"
        f"（modulo {args.sample_modulo} / {args.sampling}），{args.n_folds} 折 + embargo {args.embargo}。",
        "",
        (f"**截面均值口径 = `{args.split_weighting}`（推理端可落地）** —— "
         "`data/test/*.parquet` 没有 `weight` 列，且 `timeseries_api/runner.py:75` 的 "
         "`forbidden = {\"weight\", \"target\", ...}` 会在交给 `predict` 之前剥掉它。"
         "权重也重建不出来（每 asset 约 10 万个不同值、组内 CV 0.15~0.60、跨分区漂移），"
         "所以拆解与投影只能用无权均值。**算分仍用加权** —— 那是比赛评分本身。"
         if args.split_weighting == "unweighted" else
         f"⚠️ **截面均值口径 = `{args.split_weighting}`，用了推理端拿不到的 weight，"
         "本报告结论不可落地，仅供与无权版对读。**"),
        "",
        "**这一轮回答的问题**：`lgbm_xs` 比的是「只做截面块的 ridge」，"
        "而生产模型是一个 ridge 同时做两块 —— **赢了前者不等于赢了后者**。"
        "这里把 LGBM 的截面分量装进完整预测，用比赛真正的评分口径整体量。",
        "",
        "**比峰值 `A²/B` 不比原始分**：scale 是纯后处理旋钮，各臂最优 scale 本就不同，"
        "比原始分等于比谁的 scale 碰得准。峰值与 scale 无关。",
        "",
    ] + ([f"> ⚠️ **baseline 的 alpha 落在网格端点**（{sorted(set(knobs))}）—— "
          "基准被人为压弱，本报告结论**不可信**。", ""] if alpha_at_edge else []) + [
        "## 各臂峰值",
        "",
        "| 臂 | 平均峰值 | 平均最优 scale | 逐折峰值 |",
        "|---|---:|---:|---|",
    ]
    for arm in arm_names:
        s = summary[arm]
        lines.append(f"| `{arm}` | {s['mean_peak']:.6f} | {s['mean_best_scale']:.3f} | "
                     + " ".join(f"{v:.5f}" for v in s["per_fold_peak"]) + " |")

    lines += ["", f"## 相对 `{base}`（生产口径）的配对 Δ —— **判据是 t**", "",
              "| 臂 | 判定 | mean(Δ峰值) | 相对 | SE | **t** | 正折 | 去掉最好一折 t | ΔA | ΔB | 折后峰值 | 机制 |",
              "|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|"]
    for arm, c in comparisons.items():
        badge = {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
                 "INSUFFICIENT_FOLDS": "🚫 折数不足"}.get(c["verdict"], "⚠️ 未定")
        lines.append(
            f"| `{arm}` | {badge} | {c['mean']:+.6f} | {c['relative_pct']:+.1f}% | "
            f"{fmt(c['se'], '.6f')} | **{fmt(c['t'], '+.2f')}** | "
            f"{c['positive_folds']}/{c['n_folds']} | {fmt(c['drop_best_t'], '+.2f')} | "
            f"{c['delta_A_pct']:+.1f}% | {c['delta_B_pct']:+.1f}% | "
            f"{c['discounted_peak_pct']:+.1f}% | "
            f"{'✅' if c['discounted_mechanism_ok'] else '❌'} |")

    lines += ["", "## 投影诊断（ê 的截面均值到底多大）", "",
              "| 候选 | ê_mean RMS | ê_lgbm RMS | 占比 | R²_e 投影前 | 投影后 | ridge 的 R²_e |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        p = proj[n]
        mm, ll = float(np.mean(p["e_mean_rms"])), float(np.mean(p["e_lgbm_rms"]))
        lines.append(f"| `{n}` | {mm:.6f} | {ll:.6f} | {mm/max(ll,1e-12):.1%} | "
                     f"{np.mean(p['r2_e_before']):+.5f} | {np.mean(p['r2_e_after']):+.5f} | "
                     f"{np.mean(p['r2_e_ridge']):+.5f} |")

    floors = [c["detection_floor_pct"] for c in comparisons.values()
              if c["detection_floor_pct"] is not None]
    lines += ["",
              (f"**检出下限**：t=2 需要峰值相对提升 ≥ **{min(floors):.1f}%**。"
               "INCONCLUSIVE 是「测不出来」，不是「没效果」。")
              if floors else "**折数不足以定义标准误，本报告不能用于下结论。**",
              "",
              "## 怎么读",
              "",
              "- **这才是能回答「要不要花公榜额度」的数** —— `lgbm_xs` 的 t=6.4 是在"
              "「截面块孤立测量」口径下取得的，装车后可能缩水甚至消失。",
              "- `replace` vs `blend50`：0.5/0.5 是先验、不会过拟合（ROADMAP §5）；"
              "`replace` 更激进。两者都赢才说明信号稳。",
              "- `xs_shrunk` vs `xs_loose`：直接检验 2.2 折扣规则的排序翻转对不对 —— "
              "`lgbm_xs` 里本地最好的是 `xs_loose`，折后应该是 `xs_shrunk` 更好。",
              "- **仍然是本地数**。容量改动的本地结论在本项目已被证明不可信，公榜才是裁判。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 各臂峰值 ===")
    for arm in arm_names:
        print(f"  {arm:20s} {summary[arm]['mean_peak']:.6f}  "
              f"(最优 scale {summary[arm]['mean_best_scale']:.3f})")
    print(f"\n=== 相对 {base}（生产口径）===")
    for arm, c in comparisons.items():
        print(f"  {arm:20s} {c['verdict']:<18s} Δ={c['mean']:+.6f} ({c['relative_pct']:+.1f}%)  "
              f"t={fmt(c['t'], '+.2f')}  正折 {c['positive_folds']}/{c['n_folds']}  "
              f"去最好一折 t={fmt(c['drop_best_t'], '+.2f')}  "
              f"折后{c['discounted_peak_pct']:+.1f}%  机制{'✅' if c['discounted_mechanism_ok'] else '❌'}")
    print("\n=== 投影诊断 ===")
    for n in names:
        p = proj[n]
        mm, ll = float(np.mean(p["e_mean_rms"])), float(np.mean(p["e_lgbm_rms"]))
        print(f"  {n:11s} ê_mean RMS={mm:.6f} 占 ê 的 {mm/max(ll,1e-12):.1%}；"
              f"R²_e {np.mean(p['r2_e_before']):+.5f} → {np.mean(p['r2_e_after']):+.5f}"
              f"（ridge {np.mean(p['r2_e_ridge']):+.5f}）")
    if floors:
        print(f"\n检出下限：峰值相对提升 ≥ {min(floors):.1f}%")
    print(f"\n产物：{json_path}\n     {md_path}")


if __name__ == "__main__":
    main()
