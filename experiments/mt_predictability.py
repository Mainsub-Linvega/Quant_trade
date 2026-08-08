"""P2-2：市场共同分量 m_t 能不能被预测？—— 决定 68% 那块蛋糕吃不吃得到。

分数可以精确拆成两块。记 m_t = 逐 time_id 的加权截面均值，e = y − m_t，
因为每个 time_id 内 Σw·e = 0，所以交叉项消失：

    Σw·y² = Σw·m² + Σw·e²
    Score = share_m · R²_m  +  share_e · R²_e

    share_m = Σw·m² / Σw·y²        （mt_diagnostics 实测 0.684）
    R²_m    = 1 − Σw(m−m̂)² / Σw·m²
    R²_e    = 1 − Σw(e−ê)² / Σw·e²

本脚本两部分：
  A. 把现有 baseline 的分数按上式拆开 —— 现在这 0.00134 到底从哪来的
  B. 直接拿各特征的加权截面均值去回归 m_t，测样本外 R²_m —— 68% 够不够得着

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/mt_predictability.py
输出：outputs/experiments/mt_predictability.{json,md}
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
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import fit_model, predict_array, robust_transform_fit
from features import apply_robust_transform
from walk_forward_rolling import PROD_SAMPLED_WINDOW, load_all_sampled  # 复用同一个加载器

MT_ALPHAS = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Can cross-sectional feature means predict m_t?")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--train-window", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=0.5)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    return parser.parse_args()


def group_starts(time_ids: np.ndarray) -> np.ndarray:
    """行已按 time_id 排序（分区文件天然满足），返回每个 time_id 的起始下标。"""
    return np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]


def weighted_group_mean(values: np.ndarray, weight: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """逐 time_id 的加权均值。values 可以是 1 维（target）或 2 维（特征矩阵）。"""
    total_w = np.add.reduceat(weight, starts)
    if values.ndim == 1:
        return np.add.reduceat(weight * values, starts) / total_w
    return np.add.reduceat(values * weight[:, None], starts, axis=0) / total_w[:, None]


def decompose_score(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray, starts: np.ndarray
) -> dict[str, float]:
    """把 Score 拆成择时块与截面块。两块之和必须等于 weighted_zero_mean_r2。"""
    w64 = np.maximum(weight.astype(np.float64), 0.0)
    y64, p64 = target.astype(np.float64), prediction.astype(np.float64)
    counts = np.diff(np.r_[starts, len(y64)])

    m_y = weighted_group_mean(y64, w64, starts)
    m_p = weighted_group_mean(p64, w64, starts)
    group_w = np.add.reduceat(w64, starts)

    e_y = y64 - np.repeat(m_y, counts)
    e_p = p64 - np.repeat(m_p, counts)

    total = float(np.dot(w64, y64 * y64))
    market_total = float(np.dot(group_w, m_y * m_y))
    cross_total = float(np.dot(w64, e_y * e_y))

    market_residual = float(np.dot(group_w, (m_y - m_p) ** 2))
    cross_residual = float(np.dot(w64, (e_y - e_p) ** 2))

    share_m = market_total / total
    r2_m = 1.0 - market_residual / market_total
    r2_e = 1.0 - cross_residual / cross_total
    return {
        "share_market": share_m,
        "share_cross": cross_total / total,
        "r2_market": r2_m,
        "r2_cross": r2_e,
        "contribution_market": share_m * r2_m,
        "contribution_cross": (cross_total / total) * r2_e,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading all partitions...", flush=True)
    data = load_all_sampled(train_files(Path(args.data_root)), args.sample_modulo)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    train_window = args.train_window or int(len(unique_time_ids) * 4 / 9)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, train_window, args.embargo)
    fold_alpha = args.ridge_alpha * train_window / PROD_SAMPLED_WINDOW
    print(f"{len(all_time_ids):,} rows, {len(unique_time_ids):,} time_ids, {len(folds)} folds", flush=True)

    fold_results = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        train_set = np.isin(all_time_ids, train_ids)
        valid_set = np.isin(all_time_ids, valid_ids)

        v_time = data["time_id"][valid_set]
        v_target = data["target"][valid_set]
        v_weight = data["weight"][valid_set]
        v_starts = group_starts(v_time)

        # ---------- A. 现有 baseline 的分数拆解 ----------
        artifact, selected = fit_model(
            data["features"][train_set], data["target"][train_set],
            data["weight"][train_set], data["time_id"][train_set],
            args.feature_count, fold_alpha,
        )
        v_features = data["features"][valid_set]
        prediction = predict_array(
            artifact, v_features, v_time, selected, args.prediction_scale, args.prediction_clip
        )
        score = weighted_zero_mean_r2(v_target, prediction, v_weight)
        parts = decompose_score(v_target, prediction, v_weight, v_starts)
        identity_gap = abs(parts["contribution_market"] + parts["contribution_cross"] - score)
        del artifact, selected, prediction
        gc.collect()

        # ---------- B. 直接预测 m_t ----------
        # 预处理统计量只在训练段拟合（与主模型同规矩），先做行级稳健变换再取截面均值，
        # 这样喂给 m_t 回归的输入和模型看到的是同一套数值。
        t_features = data["features"][train_set]
        t_features, preprocessing = robust_transform_fit(t_features)
        t_time = data["time_id"][train_set]
        t_weight = data["weight"][train_set].astype(np.float64)
        t_starts = group_starts(t_time)
        market_x_train = weighted_group_mean(t_features.astype(np.float64), t_weight, t_starts)
        market_y_train = weighted_group_mean(data["target"][train_set].astype(np.float64), t_weight, t_starts)
        market_w_train = np.add.reduceat(t_weight, t_starts)
        del t_features
        gc.collect()

        v_feat = apply_robust_transform(
            v_features.copy(), preprocessing["lower"], preprocessing["upper"],
            preprocessing["center"], preprocessing["scale"],
        )
        v_weight64 = v_weight.astype(np.float64)
        market_x_valid = weighted_group_mean(v_feat.astype(np.float64), v_weight64, v_starts)
        market_y_valid = weighted_group_mean(v_target.astype(np.float64), v_weight64, v_starts)
        market_w_valid = np.add.reduceat(v_weight64, v_starts)
        del v_feat, v_features
        gc.collect()

        market_denominator = float(np.dot(market_w_valid, market_y_valid**2))
        alpha_scores = {}
        for alpha in MT_ALPHAS:
            estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=2000, fit_intercept=True)
            estimator.fit(market_x_train, market_y_train, sample_weight=market_w_train)
            fitted = estimator.predict(market_x_valid)
            residual = float(np.dot(market_w_valid, (market_y_valid - fitted) ** 2))
            alpha_scores[f"{alpha:.0e}"] = 1.0 - residual / market_denominator

        # 单变量 IC：训练段拟合、验证段复核，看有没有任何一列在时序上和 m_t 相关
        ic_train = np.array([np.corrcoef(market_x_train[:, j], market_y_train)[0, 1]
                             for j in range(market_x_train.shape[1])])
        ic_valid = np.array([np.corrcoef(market_x_valid[:, j], market_y_valid)[0, 1]
                             for j in range(market_x_valid.shape[1])])
        ic_train = np.nan_to_num(ic_train)
        ic_valid = np.nan_to_num(ic_valid)
        top = int(np.argmax(np.abs(ic_train)))

        fold_results.append({
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "score": float(score),
            "identity_gap": identity_gap,
            **{k: float(v) for k, v in parts.items()},
            "mt_r2_by_alpha": {k: float(v) for k, v in alpha_scores.items()},
            "mt_best_alpha": max(alpha_scores, key=alpha_scores.get),
            "mt_best_r2": float(max(alpha_scores.values())),
            "univariate_ic": {
                "top_feature": top,
                "ic_train": float(ic_train[top]),
                "ic_valid_same_feature": float(ic_valid[top]),
                "max_abs_ic_train": float(np.abs(ic_train).max()),
                "corr_ic_train_valid": float(np.corrcoef(ic_train, ic_valid)[0, 1]),
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        f = fold_results[-1]
        print(
            f"fold {index:2d}: score={score:.8f} = 择时 {f['contribution_market']:+.2e} "
            f"+ 截面 {f['contribution_cross']:+.2e} (恒等式残差 {identity_gap:.1e}) | "
            f"share_m={f['share_market']:.3f} R²_m={f['r2_market']:+.5f} R²_e={f['r2_cross']:+.5f} | "
            f"m_t 直接预测 R²={f['mt_best_r2']:+.5f}@{f['mt_best_alpha']} "
            f"| IC 训练/验证相关 {f['univariate_ic']['corr_ic_train_valid']:+.3f} "
            f"({f['elapsed_seconds']:.0f}s)",
            flush=True,
        )
        del train_set, valid_set, market_x_train, market_x_valid
        gc.collect()

    def col(name):
        return np.array([f[name] for f in fold_results])

    summary = {
        name: {"mean": float(col(name).mean()), "min": float(col(name).min()), "max": float(col(name).max())}
        for name in ["score", "share_market", "r2_market", "r2_cross",
                     "contribution_market", "contribution_cross", "mt_best_r2"]
    }
    summary["max_identity_gap"] = float(col("identity_gap").max())
    summary["mt_r2_positive_folds"] = int((col("mt_best_r2") > 0).sum())
    summary["ic_train_valid_corr_mean"] = float(
        np.mean([f["univariate_ic"]["corr_ic_train_valid"] for f in fold_results])
    )

    payload = {
        "question": "市场共同分量 m_t 能否被特征的截面均值预测？",
        "decomposition": "Score = share_m·R²_m + share_e·R²_e",
        "configuration": {
            "n_folds": args.n_folds, "train_window": train_window, "embargo": args.embargo,
            "sample_modulo": args.sample_modulo, "feature_count": args.feature_count,
            "ridge_alpha": fold_alpha, "prediction_scale": args.prediction_scale,
            "mt_alphas": MT_ALPHAS,
        },
        "summary": summary,
        "folds": fold_results,
    }
    (output_dir / "mt_predictability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# P2-2：市场共同分量 m_t 的可预测性",
        "",
        "`Score = share_m·R²_m + share_e·R²_e`（m = 逐 time_id 加权截面均值，e = y − m）",
        "",
        f"恒等式最大残差 {summary['max_identity_gap']:.2e}（应为 0，是拆解正确性的自检）",
        "",
        "## A. 现有 baseline 的分数从哪来",
        "",
        "| Fold | Score | share_m | R²_m | R²_e | 择时贡献 | 截面贡献 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in fold_results:
        lines.append(
            f"| {f['fold']} | {f['score']:.8f} | {f['share_market']:.3f} | {f['r2_market']:+.6f} | "
            f"{f['r2_cross']:+.6f} | {f['contribution_market']:+.3e} | {f['contribution_cross']:+.3e} |"
        )
    s = summary
    lines += [
        f"| **mean** | {s['score']['mean']:.8f} | {s['share_market']['mean']:.3f} | "
        f"{s['r2_market']['mean']:+.6f} | {s['r2_cross']['mean']:+.6f} | "
        f"**{s['contribution_market']['mean']:+.3e}** | **{s['contribution_cross']['mean']:+.3e}** |",
        "",
        "## B. 直接用特征截面均值回归 m_t（样本外 R²_m）",
        "",
        "| Fold | " + " | ".join(f"α={a:.0e}" for a in MT_ALPHAS) + " | 最好 |",
        "|---:|" + "---:|" * (len(MT_ALPHAS) + 1),
    ]
    for f in fold_results:
        lines.append(
            f"| {f['fold']} | "
            + " | ".join(f"{f['mt_r2_by_alpha'][f'{a:.0e}']:+.5f}" for a in MT_ALPHAS)
            + f" | {f['mt_best_r2']:+.5f} |"
        )
    lines += [
        "",
        f"样本外 R²_m 为正的 fold：{summary['mt_r2_positive_folds']}/{len(fold_results)}，"
        f"均值 {s['mt_best_r2']['mean']:+.5f}",
        "",
        f"单变量 IC 在训练段与验证段之间的相关性（均值）：{summary['ic_train_valid_corr_mean']:+.3f}"
        " —— 接近 0 说明训练段找到的相关性换一段就不成立。",
        "",
        "## 怎么读",
        "",
        f"- share_m ≈ {s['share_market']['mean']:.3f}：择时那块占 target 方差的比例（复现 mt_diagnostics）",
        "- 每提高 R²_m 一个百分点，总分就加 `share_m × 0.01` ≈ "
        f"{s['share_market']['mean'] * 0.01:.4f} —— 是现在总分的 "
        f"{s['share_market']['mean'] * 0.01 / max(s['score']['mean'], 1e-12):.0f} 倍",
        "- 若 B 部分的样本外 R²_m 稳定 ≤ 0，则这 68% 对任何模型都吃不到，"
        "所有精力都该压在截面那部分（换 LightGBM 也救不了）",
    ]
    (output_dir / "mt_predictability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "contribution_market_mean": s["contribution_market"]["mean"],
        "contribution_cross_mean": s["contribution_cross"]["mean"],
        "mt_best_r2_mean": s["mt_best_r2"]["mean"],
        "mt_r2_positive_folds": summary["mt_r2_positive_folds"],
        "report": str(output_dir / "mt_predictability.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
