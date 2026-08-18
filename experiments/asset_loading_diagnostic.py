"""per-asset 完整载荷诊断：15 个资产该不该有各自的线性系数？

## 问题

现在整个模型对 15 个资产用**同一套系数**。那 200 列截面去均值只是剔掉共同分量，
**不允许不同资产对同一个特征有不同载荷**。资产之间到底同不同质？

采样数据 2,645,530 行 / 15 资产 ≈ 每资产 17.6 万行，对 200 个系数识别得很好；
15 × 200 = 3000 列的 `XᵀX` 也完全放得下。这是干净的③类（结构/信息），
不是本地不可靠的「拟合紧密度」轴。

## ⚠️ 先摆出已有的不利证据

| 已做过的 | 结果 |
|---|---|
| 每资产**标量** scale（`fit_asset_slopes`） | 本地 +1.99%/3-of-4 → **公榜 Δ=−6.9e-06，不可辨别** |
| 稀疏 asset×feature **残差**交互（k=4/8/16） | 全负（`v3_sparse_asset_feature_residual`） |
| residual atlas 的逐资产桶 | 有逐折稳定性，但只是分桶报分，不是载荷诊断 |

⟹ 「从没测过 per-asset」不准确；准确的说法是**「每资产完整线性载荷」没测过**，
而相邻的两个弱版本都失败了。先验偏负，本脚本的主要价值是**关掉或打开这条轴**。

## 设计

目标是**截面分量** `e = y − 逐 time_id 无权均值`（与生产的截面块同口径），
设计矩阵是截面去均值后的 200 列（`cross_sectional_deviation`）。三个臂，同一批验证行：

1. `shared`   ——全体共用一套 ridge 系数（基准）
2. `per_asset`——每个资产各拟一套
3. `shrunk`   ——`β_a(κ) = (Xᵀ_a W X_a + κ·G_shared/15 + αI)⁻¹(…)` 向共享解收缩，κ 阶梯；
                纯 per_asset 必然过拟合，收缩版才是有意义的上界。κ→∞ 必须逐位退化为 shared。

指标：尺度无关 `peak = A²/B`（在 target 上评，与项目口径一致），逐折配对。
另报**逐资产 peak 离散度**与**资产间系数向量两两相关** —— 这两个才是「异质性有多大」的直读。

判据：折均 >0、≥4/5 折、去最好折 >0、相对 ≥1%、配对 bootstrap CI 下界 >0。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/asset_loading_diagnostic.py
输出：outputs/experiments/<label>.{json,md}
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
# ⚠️ 顺序与 v3_production_oof.py 一致：v1_ridge 与 v3_hybrid 都有 train.py / features.py，
# 只有 v1_ridge 的 train 提供 robust_transform_fit / select_features（伤疤规则 §4）。
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import apply_robust_transform, cross_sectional_deviation  # noqa: E402
from lgbm_xs import load_rows  # noqa: E402
from market_model import sign_test_p  # noqa: E402
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402

FEATURE_COUNT = 200
SAMPLE_MODULO, SAMPLING = 5, "phase_balanced"
TRAIN_WINDOW, EMBARGO, N_FOLDS = 78_960, 6, 5
RIDGE_ALPHA = 2_000_000.0
SHRINKAGE = [0.0, 1.0, 10.0, 100.0, 1000.0, float("inf")]
MIN_RELATIVE_GAIN = 0.01
MIN_POSITIVE_FOLDS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="asset_loading_diagnostic")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))


def normal_equations(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xw = X * w[:, None]
    return Xw.T @ X, Xw.T @ y


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    print(f"loading sampled data: modulo {SAMPLE_MODULO}/{SAMPLING}", flush=True)
    data = load_rows(Path(args.data_root), SAMPLE_MODULO, SAMPLING)
    features = data["features"]
    target = data["target"].astype(np.float64, copy=False)
    weight = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    time_ids, asset_ids = data["time_id"], data["asset_id"]
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(unique_time_ids, N_FOLDS, TRAIN_WINDOW, EMBARGO)
    assets = np.unique(asset_ids)
    print(f"{len(target):,} 行 / {len(assets)} 个资产 "
          f"（每资产约 {len(target)//len(assets):,} 行）/ {len(folds)} 折", flush=True)

    kappa_keys = ["inf" if not np.isfinite(k) else f"{k:g}" for k in SHRINKAGE]
    fold_peaks: dict[str, list[float]] = {"shared": [], **{f"shrunk_{k}": [] for k in kappa_keys}}
    fold_moments: dict[str, list[np.ndarray]] = {name: [] for name in fold_peaks}
    per_asset_report: list[dict[str, Any]] = []
    coefficient_similarity: list[float] = []

    for index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        tr = np.isin(time_ids, train_ids)
        va = np.isin(time_ids, valid_ids)
        transformed_train, stats = robust_transform_fit(features[tr].copy())
        transformed_valid = features[va].copy()
        apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                               stats["center"], stats["scale"])
        y_tr, w_tr, tid_tr, aid_tr = target[tr], weight[tr], time_ids[tr], asset_ids[tr]
        y_va, w_va, tid_va, aid_va = target[va], weight[va], time_ids[va], asset_ids[va]

        s_tr = np.r_[0, np.flatnonzero(tid_tr[1:] != tid_tr[:-1]) + 1]
        c_tr = np.diff(np.r_[s_tr, len(tid_tr)]).astype(np.float64)
        e_tr = y_tr - group_mean(y_tr, s_tr, c_tr)
        selected = select_features(transformed_train, e_tr, np.ones_like(e_tr), FEATURE_COUNT)
        X_tr = cross_sectional_deviation(transformed_train[:, selected].copy(), tid_tr).astype(np.float64)
        X_va = cross_sectional_deviation(transformed_valid[:, selected].copy(), tid_va).astype(np.float64)
        del transformed_train, transformed_valid
        gc.collect()

        fold_alpha = RIDGE_ALPHA * len(train_ids) / TRAIN_WINDOW
        G_all, b_all = normal_equations(X_tr, e_tr, w_tr)
        ridge = np.eye(X_tr.shape[1]) * fold_alpha
        beta_shared = np.linalg.solve(G_all + ridge, b_all)

        # 逐资产法方程；shrunk 由它们与共享法方程的凸组合解出 ⟹ κ→∞ 精确退化为 shared
        per_asset: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for a in assets:
            m = aid_tr == a
            per_asset[int(a)] = normal_equations(X_tr[m], e_tr[m], w_tr[m])

        prior_G, prior_b = G_all / len(assets), b_all / len(assets)
        predictions: dict[str, np.ndarray] = {"shared": X_va @ beta_shared}
        betas_at_zero: dict[int, np.ndarray] = {}
        for kappa, key in zip(SHRINKAGE, kappa_keys):
            pred = np.empty(len(y_va), dtype=np.float64)
            for a in assets:
                Ga, ba = per_asset[int(a)]
                if not np.isfinite(kappa):
                    beta = beta_shared
                else:
                    beta = np.linalg.solve(Ga + kappa * prior_G + ridge, ba + kappa * prior_b)
                    if kappa == 0.0:
                        betas_at_zero[int(a)] = beta
                mask = aid_va == a
                pred[mask] = X_va[mask] @ beta
            predictions[f"shrunk_{key}"] = pred
        # κ→∞ 必须逐位等于 shared
        gap = float(np.abs(predictions["shrunk_inf"] - predictions["shared"]).max())
        if gap > 1e-9:
            raise AssertionError(f"kappa=inf does not reduce to shared (max {gap:.3e})")

        # 资产间系数向量的两两相关（纯 per-asset 解，κ=0）
        matrix = np.array([betas_at_zero[int(a)] for a in assets])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        similarity = (matrix / np.maximum(norms, 1e-30)) @ (matrix / np.maximum(norms, 1e-30)).T
        upper = similarity[np.triu_indices(len(assets), k=1)]
        coefficient_similarity.append(float(np.mean(upper)))

        # 逐折 peak（在 target 上评，尺度无关）与逐 time_id 矩（给配对 bootstrap）
        denom = float(np.dot(w_va, y_va * y_va))
        s_va = np.r_[0, np.flatnonzero(tid_va[1:] != tid_va[:-1]) + 1]
        gidx = np.repeat(np.arange(len(s_va)), np.diff(np.r_[s_va, len(tid_va)]))
        for name, pred in predictions.items():
            A = float(np.dot(w_va, y_va * pred)) / denom
            B = float(np.dot(w_va, pred * pred)) / denom
            fold_peaks[name].append(A * A / B if B > 0 else 0.0)
            fold_moments[name].append(np.column_stack([
                np.bincount(gidx, weights=w_va * y_va * y_va, minlength=len(s_va)),
                np.bincount(gidx, weights=w_va * y_va * pred, minlength=len(s_va)),
                np.bincount(gidx, weights=w_va * pred * pred, minlength=len(s_va))]))

        # 逐资产 peak（共享系数下），看资产之间难度差多少
        rows = []
        for a in assets:
            m = aid_va == a
            d = float(np.dot(w_va[m], y_va[m] * y_va[m]))
            A = float(np.dot(w_va[m], y_va[m] * predictions["shared"][m])) / d
            B = float(np.dot(w_va[m], predictions["shared"][m] ** 2)) / d
            rows.append({"asset": int(a), "peak": A * A / B if B > 0 else 0.0, "rows": int(m.sum())})
        per_asset_report.append({"fold": index, "assets": rows})
        print(f"  fold {index}: shared peak={fold_peaks['shared'][-1]:.8f}, "
              f"per_asset(κ=0)={fold_peaks['shrunk_0'][-1]:.8f}, "
              f"资产间系数相关均值={coefficient_similarity[-1]:+.3f} "
              f"[{time.perf_counter()-fold_started:.0f}s]", flush=True)
        del X_tr, X_va, per_asset, predictions
        gc.collect()

    base = np.array(fold_peaks["shared"])
    rng = np.random.default_rng(args.boot_seed)
    results: dict[str, Any] = {}
    for key in kappa_keys:
        name = f"shrunk_{key}"
        arm = np.array(fold_peaks[name])
        delta = arm - base
        drop_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
        positive = int((delta > 0).sum())
        samples = []
        for _ in range(args.n_boot):
            total_a = total_b = 0.0
            for f in range(len(folds)):
                rows_b, rows_a = fold_moments["shared"][f], fold_moments[name][f]
                n = len(rows_b)
                nb = int(np.ceil(n / args.block_size))
                st = rng.integers(0, max(n - args.block_size, 0) + 1, size=nb)
                sp = np.minimum(st + args.block_size, n)
                idx = np.concatenate([np.arange(a, b) for a, b in zip(st, sp)])
                for src, acc in ((rows_b, "b"), (rows_a, "a")):
                    t = src[idx].sum(axis=0)
                    peak = (t[1] / t[0]) ** 2 / (t[2] / t[0]) if t[2] > 0 else 0.0
                    if acc == "b":
                        total_b += peak
                    else:
                        total_a += peak
            samples.append((total_a - total_b) / len(folds))
        boot = np.asarray(samples, dtype=float)
        ci = {k: float(np.percentile(boot, q)) for k, q in (("p2.5", 2.5), ("p50", 50.0), ("p97.5", 97.5))}
        checks = {
            "1_mean_delta_positive": bool(delta.mean() > 0),
            "2_at_least_4_of_5_folds_positive": bool(positive >= MIN_POSITIVE_FOLDS),
            "3_survives_drop_best_fold": bool(drop_best.mean() > 0),
            "4_relative_gain_at_least_1pct": bool(delta.mean() / base.mean() >= MIN_RELATIVE_GAIN),
            "5_paired_bootstrap_ci_lower_bound_positive": bool(ci["p2.5"] > 0),
        }
        results[name] = {"kappa": key, "per_fold_peak": arm.tolist(),
                         "mean_delta": float(delta.mean()),
                         "mean_delta_drop_best": float(drop_best.mean()),
                         "relative": float(delta.mean() / base.mean()),
                         "positive_folds": positive, "n_folds": len(delta),
                         "sign_test_p": sign_test_p(positive, len(delta)),
                         "paired_bootstrap": ci, "checks": checks, "pass": all(checks.values())}
        print(f"  {name:16s} Δ折均 {delta.mean():+.3e}（{delta.mean()/base.mean()*100:+.2f}%）"
              f" 正折 {positive}/{len(delta)} {'PASS' if all(checks.values()) else 'FAIL'}", flush=True)

    asset_peaks = np.array([[r["peak"] for r in f["assets"]] for f in per_asset_report])
    payload = {
        "experiment": "asset_loading_diagnostic",
        "question": "15 个资产该不该有各自的线性载荷？",
        "prior_evidence_against": [
            "每资产标量 scale：本地 +1.99%/3-of-4 → 公榜 Δ=−6.9e-06，不可辨别",
            "稀疏 asset×feature 残差交互 k=4/8/16：全负",
        ],
        "caliber": {"n_folds": N_FOLDS, "sample_modulo": SAMPLE_MODULO, "sampling": SAMPLING,
                    "train_window": TRAIN_WINDOW, "embargo": EMBARGO,
                    "feature_count": FEATURE_COUNT, "ridge_alpha": RIDGE_ALPHA,
                    "target": "截面分量 e = y − 逐 time_id 无权均值",
                    "metric": "peak = A²/B，在 target 上评"},
        "shared_per_fold_peak": base.tolist(), "shared_mean_peak": float(base.mean()),
        "shrinkage_grid": kappa_keys,
        "arms": results,
        "heterogeneity": {
            "mean_pairwise_coefficient_similarity": float(np.mean(coefficient_similarity)),
            "per_fold_similarity": coefficient_similarity,
            "asset_peak_mean": asset_peaks.mean(axis=0).tolist(),
            "asset_peak_spread_ratio": float(asset_peaks.mean(axis=0).max()
                                             / max(asset_peaks.mean(axis=0).min(), 1e-12)),
            "assets": [int(a) for a in assets],
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    het = payload["heterogeneity"]
    passed = [n for n, r in results.items() if r["pass"]]
    lines = ["# per-asset 完整载荷诊断", "",
             f"口径：{N_FOLDS} 折 / modulo {SAMPLE_MODULO} `{SAMPLING}` / 训练窗 {TRAIN_WINDOW:,} / "
             f"{FEATURE_COUNT} 列截面去均值；目标是截面分量 `e`；指标 `peak = A²/B`。", "",
             f"`shared` 基准逐折 peak 均值 **{base.mean():.8f}**。", "",
             "> ⚠️ 先验偏负：每资产**标量** scale 公榜实测不可辨别（Δ=−6.9e-06）；"
             "稀疏 asset×feature 残差交互 k=4/8/16 全负。本诊断补的是**完整线性载荷**这一格。", "",
             "| κ（向共享收缩） | Δ折均 | 相对 | 正折 | 去最好折 | 配对 CI | 判定 |",
             "|---|---:|---:|---:|---:|---|:--:|"]
    for key in kappa_keys:
        r = results[f"shrunk_{key}"]
        ci = r["paired_bootstrap"]
        lines.append(f"| {key} | {r['mean_delta']:+.3e} | {r['relative']*100:+.2f}% | "
                     f"{r['positive_folds']}/{r['n_folds']} | {r['mean_delta_drop_best']:+.3e} | "
                     f"[{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] | "
                     f"{'✅' if r['pass'] else '❌'} |")
    lines += ["", "## 异质性有多大（这两个才是直读）", "",
              f"- 资产间**系数向量两两余弦相关**均值：**{het['mean_pairwise_coefficient_similarity']:+.3f}**"
              "（接近 1 ⟹ 各资产载荷几乎一样，异质性无从谈起）",
              f"- 逐资产 peak（共享系数下）最大/最小 = **{het['asset_peak_spread_ratio']:.2f}×**"
              "（资产之间难度差多少，与载荷是否该分开是两件事）", "",
              f"## 判定：{'✅ ' + ', '.join(passed) if passed else '❌ 全部不通过'}", ""]
    if not passed:
        lines += ["整条 κ 阶梯都不过门槛。结合资产间系数相关 "
                  f"{het['mean_pairwise_coefficient_similarity']:+.3f}，读作**资产在线性载荷上近似同质**；"
                  "这也回过头解释了为什么每资产标量 scale 在公榜上不可辨别。", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
