"""市场模型（`m̂`）能不能重建得更好 —— **在可交付口径下**重测一遍。

## 为什么要重测：`mt_predictability` 那个「13% 余量」是不可交付的、而且比得不公平

`replace` 上线后岭回归唯一的作用就是产 `m̂`（占预测方差 63%），所以「专门回归 m_t
比现在隐含的好 13%」这条一下子从可选优化变成主线。但那个数有两处硬伤：

1. **设计矩阵用的是「加权」特征截面均值** —— 推理端拿不到 weight
   （test 无该列，且 runner 的 `forbidden` 会剥掉），那个数**没法交付**。
   生产的 `m̂` 是**无权**截面均值。
2. **两个臂的缩放不同**：基准的 `r2_market` 是在 `prediction_scale=0.5` 下算的
   （预测被砍半 → R² 被压低），而直接回归是最小二乘、**自带最优缩放**。
   拿一个被人为压扁的基准去比一个最优缩放的候选，差距里混进了纯粹的 scale 效应。

本脚本把这两处都修掉：

- 每个臂都出**无权**版（可交付）与**加权**版（与历史口径对读）
- 主指标改成**尺度无关**的 `peak_m`，沿用项目已有的 A/B 框架：

      A = ΣW·m·m̂ / ΣW·m²      B = ΣW·m̂² / ΣW·m²      peak = A²/B

  `peak` 就是该臂**在各自最优 scale 下**的 R²_m。比 peak 才是比「模型有多准」，
  比固定 scale 下的 R² 是在比「谁的幅度碰巧对」。

## 臂

| 臂 | 设计矩阵 | 说明 |
|---|---|---|
| `implied_u` | —— | **基准（可交付）**：生产口径岭回归预测的**无权**截面均值 |
| `implied_w` | —— | 基准（历史口径）：同上但取加权截面均值，与 `mt_predictability` 对读 |
| `direct_w`  | 加权 X̄，323 列 | **对照**：复现 `mt_predictability` 的 B 部分 |
| `direct_u`  | 无权 X̄，323 列 | **候选（可交付）** |
| `direct_u_selK` | 无权 X̄，按训练段 |corr| 预选 top-K | K 预注册，不搜 |

目标一律是 `m_w`（逐 time_id 的**加权**截面均值 —— 比赛度量是加权的），
样本权重 `W_t = Σᵢwᵢ`。⚠️ 目标用加权、设计矩阵用无权，这不是笔误：
target 侧我们离线有 weight，设计矩阵侧推理时没有。

## 判据（跑之前写死，由 `verdict()` 判，不靠报告文字）

1. 逐折配对 Δpeak(`direct_u` − `implied_u`) 均值为正，且 ≥ 8/10 折为正
2. 去掉最好的一折之后仍为正
3. **整条 α 阶梯上都为正** —— α 是第②类「拟合紧密度」旋钮，本地量反，
   只在最优 α 上赢不算数
4. 换算成总分的增量 `share_m × Δpeak` ≥ 现有总分的 5%

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/market_model.py
输出：outputs/experiments/market_model.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import fit_model, predict_array, robust_transform_fit
from features import apply_robust_transform
# 复用已有实现，不另写一份（口径唯一性）
from mt_predictability import group_starts, weighted_group_mean, decompose_score
from walk_forward_rolling import PROD_SAMPLED_WINDOW, load_all_sampled

ALPHAS = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)   # 与 mt_predictability 同一条阶梯，便于对读
SELECT_K = (50, 200)                       # 预注册，不搜
BASELINE_ARM = "implied_u"                 # 配对 Δ 的基准
CANDIDATE_ARM = "direct_u"                 # 主候选


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the market component under a deployable calibration.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="market_model")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--train-window", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def unweighted_group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """逐 time_id 的**无权**均值 —— 生产推理端唯一能算的那个（拿不到 weight）。"""
    if values.ndim == 1:
        return np.add.reduceat(values, starts) / counts
    return np.add.reduceat(values, starts, axis=0) / counts[:, None]


def group_means_both(
    features: np.ndarray, weight: np.ndarray, starts: np.ndarray, counts: np.ndarray,
    chunk: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """一次扫出加权与无权两套截面均值，全程 float64 累积但**分块**做。

    直接 `features.astype(np.float64)` 会在 60 万行 × 323 列上多占 1.5 GB；
    这里按 time_id 分块（与 features.py:cross_sectional_deviation 同一手法），
    峰值只有一块的大小。
    """
    n_groups, width = len(starts), features.shape[1]
    weighted = np.empty((n_groups, width), dtype=np.float64)
    unweighted = np.empty((n_groups, width), dtype=np.float64)
    total_w = np.add.reduceat(weight, starts)
    for begin in range(0, n_groups, chunk):
        end = min(begin + chunk, n_groups)
        row_start = int(starts[begin])
        row_stop = int(starts[end]) if end < n_groups else len(features)
        block = features[row_start:row_stop].astype(np.float64)
        local = starts[begin:end] - row_start
        unweighted[begin:end] = np.add.reduceat(block, local, axis=0) / counts[begin:end, None]
        block *= weight[row_start:row_stop, None]
        weighted[begin:end] = np.add.reduceat(block, local, axis=0) / total_w[begin:end, None]
        del block
    return weighted, unweighted


def ab_peak(m: np.ndarray, fitted: np.ndarray, group_weight: np.ndarray) -> dict[str, float]:
    """尺度无关的三件套。peak = 该臂在自己最优 scale 下的 R²_m。

    Score(a) = 2aA − a²B 在 a* = A/B 处取到 A²/B，这与项目在公榜上用的口径同源。
    """
    denominator = float(np.dot(group_weight, m * m))
    a = float(np.dot(group_weight, m * fitted)) / denominator
    b = float(np.dot(group_weight, fitted * fitted)) / denominator
    peak = a * a / b if b > 0 else 0.0
    return {"A": a, "B": b, "peak": peak, "optimal_scale": (a / b) if b > 0 else float("nan"),
            "r2_at_unit_scale": 1.0 - float(np.dot(group_weight, (m - fitted) ** 2)) / denominator}


def weighted_correlations(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """加权单变量相关，用于对 m_t 做预选。全 float64，323 列直接算。"""
    total = float(weight.sum())
    x_mean = (design * weight[:, None]).sum(axis=0) / total
    y_mean = float(np.dot(weight, target)) / total
    centred_y = target - y_mean
    covariance = ((design - x_mean) * weight[:, None] * centred_y[:, None]).sum(axis=0)
    x_variance = (((design - x_mean) ** 2) * weight[:, None]).sum(axis=0)
    y_variance = float(np.dot(weight, centred_y * centred_y))
    return covariance / np.sqrt(np.maximum(x_variance * y_variance, 1e-300))


def ridge_ladder(
    x_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    x_valid: np.ndarray, m_valid: np.ndarray, w_valid: np.ndarray,
) -> dict[str, dict[str, float]]:
    """整条 α 阶梯上的 A/B/peak。列数只有几百，直接用 sklearn 逐 α 拟合。"""
    out: dict[str, dict[str, float]] = {}
    for alpha in ALPHAS:
        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=2000, fit_intercept=True)
        estimator.fit(x_train, y_train, sample_weight=w_train)
        out[f"{alpha:.0e}"] = ab_peak(m_valid, estimator.predict(x_valid), w_valid)
    return out


def sign_test_p(positive: int, total: int) -> float:
    """双尾符号检验（p=0.5）。总折数很小，直接枚举二项分布。"""
    from math import comb
    negative = total - positive
    extreme = max(positive, negative)
    tail = sum(comb(total, k) for k in range(extreme, total + 1))
    return min(1.0, 2.0 * tail / 2**total)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.label}.md"
    if report_path.exists() and not args.force:
        raise SystemExit(f"{report_path} 已存在；要覆盖请加 --force")

    print("loading all partitions...", flush=True)
    data = load_all_sampled(train_files(Path(args.data_root)), args.sample_modulo)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    train_window = args.train_window or int(len(unique_time_ids) * 4 / 9)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, train_window, args.embargo)
    fold_alpha = args.ridge_alpha * train_window / PROD_SAMPLED_WINDOW
    print(f"{len(all_time_ids):,} rows, {len(unique_time_ids):,} time_ids, "
          f"{len(folds)} folds, fold_alpha={fold_alpha:.3e}", flush=True)

    fold_results: list[dict[str, object]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        train_set = np.isin(all_time_ids, train_ids)
        valid_set = np.isin(all_time_ids, valid_ids)

        v_time = data["time_id"][valid_set]
        v_target = data["target"][valid_set].astype(np.float64)
        v_weight = np.maximum(data["weight"][valid_set].astype(np.float64), 0.0)
        v_starts = group_starts(v_time)
        v_counts = np.diff(np.r_[v_starts, len(v_time)]).astype(np.float64)
        group_w = np.add.reduceat(v_weight, v_starts)
        m_valid = weighted_group_mean(v_target, v_weight, v_starts)   # 目标：加权截面均值

        arms: dict[str, dict[str, dict[str, float]]] = {}

        # ---------- 基准：生产口径岭回归的截面均值 ----------
        # scale=1 / clip 关掉：peak 是尺度无关的，任何固定缩放都不该影响结论，
        # 但限幅会破坏线性 → 这里必须关掉，否则 A/B 不再是二次式的系数。
        artifact, selected = fit_model(
            data["features"][train_set], data["target"][train_set],
            data["weight"][train_set], data["time_id"][train_set],
            args.feature_count, fold_alpha,
        )
        v_features = data["features"][valid_set]
        ridge_raw = predict_array(artifact, v_features, v_time, selected,
                                  prediction_scale=1.0, prediction_clip=1e9).astype(np.float64)
        arms["implied_u"] = {"—": ab_peak(m_valid, unweighted_group_mean(ridge_raw, v_starts, v_counts), group_w)}
        arms["implied_w"] = {"—": ab_peak(m_valid, weighted_group_mean(ridge_raw, v_weight, v_starts), group_w)}

        # 顺带留一份生产口径的分数拆解，作为「share_m 是多少」的现场证据
        production = predict_array(artifact, v_features, v_time, selected, 1.13, args.prediction_clip)
        parts = decompose_score(v_target, production, v_weight, v_starts)
        score = weighted_zero_mean_r2(v_target, production, v_weight)
        del artifact, selected, ridge_raw, production
        gc.collect()

        # ---------- 候选：直接回归 m_t ----------
        # 预处理统计量只在训练段拟合（与主模型同规矩）；先做行级稳健变换再取截面均值 ——
        # 生产的 m̂ 就是 mean(transform(x))·coef，顺序反了就不是同一个东西。
        t_features = data["features"][train_set]
        t_features, preprocessing = robust_transform_fit(t_features)
        t_time = data["time_id"][train_set]
        t_weight = np.maximum(data["weight"][train_set].astype(np.float64), 0.0)
        t_starts = group_starts(t_time)
        t_counts = np.diff(np.r_[t_starts, len(t_time)]).astype(np.float64)
        x_train_w, x_train_u = group_means_both(t_features, t_weight, t_starts, t_counts)
        m_train = weighted_group_mean(data["target"][train_set].astype(np.float64), t_weight, t_starts)
        w_train = np.add.reduceat(t_weight, t_starts)
        del t_features
        gc.collect()

        v_feat = apply_robust_transform(
            v_features.copy(), preprocessing["lower"], preprocessing["upper"],
            preprocessing["center"], preprocessing["scale"],
        )
        x_valid_w, x_valid_u = group_means_both(v_feat, v_weight, v_starts, v_counts)
        del v_feat, v_features
        gc.collect()

        arms["direct_w"] = ridge_ladder(x_train_w, m_train, w_train, x_valid_w, m_valid, group_w)
        arms["direct_u"] = ridge_ladder(x_train_u, m_train, w_train, x_valid_u, m_valid, group_w)

        correlations = np.abs(weighted_correlations(x_train_u, m_train, w_train))
        for k in SELECT_K:
            keep = np.sort(np.argsort(correlations, kind="stable")[-k:])
            arms[f"direct_u_sel{k}"] = ridge_ladder(
                x_train_u[:, keep], m_train, w_train, x_valid_u[:, keep], m_valid, group_w)

        fold_results.append({
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "n_train_time_ids": int(len(m_train)),
            "n_valid_time_ids": int(len(m_valid)),
            "production_score": float(score),
            "share_market": float(parts["share_market"]),
            "r2_market_at_scale113": float(parts["r2_market"]),
            "arms": arms,
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        base = arms[BASELINE_ARM]["—"]["peak"]
        best_direct = max(v["peak"] for v in arms[CANDIDATE_ARM].values())
        print(f"fold {index:2d}: implied_u peak={base:+.6f} (a*={arms[BASELINE_ARM]['—']['optimal_scale']:.3f}) | "
              f"implied_w peak={arms['implied_w']['—']['peak']:+.6f} | "
              f"direct_u best peak={best_direct:+.6f} | "
              f"direct_w best peak={max(v['peak'] for v in arms['direct_w'].values()):+.6f} | "
              f"share_m={parts['share_market']:.3f} ({fold_results[-1]['elapsed_seconds']:.0f}s)",
              flush=True)
        del train_set, valid_set, x_train_w, x_train_u, x_valid_w, x_valid_u
        gc.collect()

    # ------------------------------------------------------------------ 汇总与判据
    def peaks(arm: str, alpha: str | None) -> np.ndarray:
        key = "—" if alpha is None else alpha
        return np.array([f["arms"][arm][key]["peak"] for f in fold_results])

    baseline_peaks = peaks(BASELINE_ARM, None)
    share_m = float(np.mean([f["share_market"] for f in fold_results]))
    production_score = float(np.mean([f["production_score"] for f in fold_results]))

    comparisons: dict[str, dict[str, object]] = {}
    for arm in [a for a in fold_results[0]["arms"] if a.startswith("direct")]:
        per_alpha = {}
        for alpha in fold_results[0]["arms"][arm]:
            delta = peaks(arm, alpha) - baseline_peaks
            positive = int((delta > 0).sum())
            without_best = np.delete(delta, int(np.argmax(delta)))
            per_alpha[alpha] = {
                "mean_peak": float(peaks(arm, alpha).mean()),
                "mean_delta": float(delta.mean()),
                "positive_folds": positive,
                "sign_test_p": sign_test_p(positive, len(delta)),
                "mean_delta_drop_best": float(without_best.mean()),
                "relative_gain": float(delta.mean() / baseline_peaks.mean()),
                "score_gain_share": float(share_m * delta.mean() / production_score),
            }
        best_alpha = max(per_alpha, key=lambda a: per_alpha[a]["mean_peak"])
        comparisons[arm] = {"by_alpha": per_alpha, "best_alpha": best_alpha}

    def verdict(arm: str) -> dict[str, object]:
        """四条判据由代码判 —— 报告里的文字不得与这里不一致（伤疤清单 #2）。"""
        by_alpha = comparisons[arm]["by_alpha"]
        best = by_alpha[comparisons[arm]["best_alpha"]]
        checks = {
            "1_paired_delta_positive_8of10": best["mean_delta"] > 0 and best["positive_folds"] >= 8,
            "2_survives_drop_best_fold": best["mean_delta_drop_best"] > 0,
            "3_positive_across_alpha_ladder": all(v["mean_delta"] > 0 for v in by_alpha.values()),
            "4_score_gain_at_least_5pct": best["score_gain_share"] >= 0.05,
        }
        return {"checks": checks, "pass": all(checks.values()),
                "evaluated_at_alpha": comparisons[arm]["best_alpha"]}

    verdicts = {arm: verdict(arm) for arm in comparisons}

    payload = {
        "question": "在可交付口径（无权截面均值 + 尺度无关的 peak）下，专门回归 m_t 是否胜过现有隐含的 m̂？",
        "why": "replace 之后岭回归唯一的作用就是产 m̂（占预测方差 63%），"
               "而 mt_predictability 那个 +13% 用的是加权 X（推理端拿不到 weight）"
               "且基准被 prediction_scale=0.5 压扁，两处都不可信",
        "metric": "peak = A²/B，A=ΣW·m·m̂/ΣW·m²，B=ΣW·m̂²/ΣW·m² —— 各臂在自己最优 scale 下的 R²_m",
        "configuration": {
            "n_folds": args.n_folds, "train_window": train_window, "embargo": args.embargo,
            "sample_modulo": args.sample_modulo, "feature_count": args.feature_count,
            "fold_alpha": fold_alpha, "alphas": list(ALPHAS), "select_k": list(SELECT_K),
            "baseline_arm": BASELINE_ARM,
        },
        "summary": {
            "share_market_mean": share_m,
            "production_score_mean": production_score,
            "baseline_peak_mean": float(baseline_peaks.mean()),
            "implied_w_peak_mean": float(peaks("implied_w", None).mean()),
        },
        "comparisons": comparisons,
        "verdicts": verdicts,
        "folds": fold_results,
    }
    (output_dir / f"{args.label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 报告
    lines = [
        f"# 市场模型重建 —— 可交付口径下的重测（`{args.label}`）",
        "",
        "主指标是**尺度无关**的 `peak = A²/B`（各臂在自己最优 scale 下的 R²_m）。",
        "换成它是因为 `mt_predictability` 的基准是在 `prediction_scale=0.5` 下算的，",
        "而直接回归是最小二乘、自带最优缩放 —— 固定 scale 下的 R² 比的是「谁幅度碰巧对」。",
        "",
        f"折数 {args.n_folds}，embargo {args.embargo}，sample_modulo {args.sample_modulo}，"
        f"训练窗 {train_window:,} 个 time_id，fold_alpha {fold_alpha:.3e}。",
        "",
        "## 基准：现有模型的 `m̂`",
        "",
        "| 口径 | peak_m | 说明 |",
        "|---|---:|---|",
        f"| `implied_u`（**无权**） | {baseline_peaks.mean():+.6f} | **可交付** —— 生产推理端算的就是这个 |",
        f"| `implied_w`（加权） | {peaks('implied_w', None).mean():+.6f} | 历史口径，与 `mt_predictability` 对读 |",
        "",
        f"生产口径分数均值 {production_score:.8f}，share_m 均值 {share_m:.3f}。",
        "",
        "## 候选 vs 基准（逐折配对 Δpeak）",
        "",
        "| 臂 | 最优 α | peak_m | Δ vs implied_u | 相对 | 正折 | 符号 p | 去掉最好一折 | 换算总分 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, comparison in comparisons.items():
        alpha = comparison["best_alpha"]
        row = comparison["by_alpha"][alpha]
        lines.append(
            f"| `{arm}` | {alpha} | {row['mean_peak']:+.6f} | {row['mean_delta']:+.6f} | "
            f"{row['relative_gain']:+.1%} | {row['positive_folds']}/{len(fold_results)} | "
            f"{row['sign_test_p']:.3f} | {row['mean_delta_drop_best']:+.6f} | "
            f"{row['score_gain_share']:+.1%} |")

    lines += ["", "## 判据（由 `verdict()` 判，不是我写的评语）", ""]
    for arm, answer in verdicts.items():
        mark = "✅ PASS" if answer["pass"] else "❌ 未通过"
        lines.append(f"- `{arm}` @ α={answer['evaluated_at_alpha']}：**{mark}**")
        for name, ok in answer["checks"].items():
            lines.append(f"  - {'✅' if ok else '❌'} {name}")

    lines += ["", "## 整条 α 阶梯（第③条判据看的就是这张表）", "",
              "| 臂 | " + " | ".join(f"α={a:.0e}" for a in ALPHAS) + " |",
              "|---|" + "---:|" * len(ALPHAS)]
    for arm, comparison in comparisons.items():
        lines.append(f"| `{arm}` Δ | " + " | ".join(
            f"{comparison['by_alpha'][f'{a:.0e}']['mean_delta']:+.6f}" for a in ALPHAS) + " |")

    lines += ["", "## 逐折", "",
              "| Fold | 验证段 | implied_u | implied_w | direct_u(最优α) | Δ | share_m |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    best_alpha_u = comparisons[CANDIDATE_ARM]["best_alpha"]
    for f in fold_results:
        base = f["arms"][BASELINE_ARM]["—"]["peak"]
        cand = f["arms"][CANDIDATE_ARM][best_alpha_u]["peak"]
        lines.append(
            f"| {f['fold']} | {f['valid_time_range'][0]}–{f['valid_time_range'][1]} | "
            f"{base:+.6f} | {f['arms']['implied_w']['—']['peak']:+.6f} | {cand:+.6f} | "
            f"{cand - base:+.6f} | {f['share_market']:.3f} |")

    (output_dir / f"{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + json.dumps({
        "baseline_peak_mean": float(baseline_peaks.mean()),
        "verdicts": {arm: answer["pass"] for arm, answer in verdicts.items()},
        "report": str(output_dir / f"{args.label}.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
