"""同口径复测：在**今天的口径**下，直接回归 `m_t` 还是不是输给今天的两分量 `m̂`？

## 背景：这个问题仓库里已经答过一次，但那次的基准和口径都是旧的

`mt_predictability` 曾给出「专门回归 m_t 比隐含的 m̂ 好 13%」。`market_model.py` 把它推翻了，
指出两处硬伤：(1) 设计矩阵用**加权**特征截面均值，而推理端拿不到 `weight`，那个数不可交付；
(2) 基准在 `prediction_scale=0.5` 下算 R²，而直接回归自带最优缩放 —— 比的是「谁幅度碰巧对」。
改成无权口径 + 尺度无关 `peak = A²/B` 后，`direct_u` 只有 +2.0%、6/10 折、符号 p=0.754，
**四条判据全否**，整条 α 阶梯只有 1e6 一格为正。

但 `market_model` 用的是**旧口径**（10 折 / modulo 10 / 训练窗 39,480）和**旧基准**
（`implied_u` = 岭回归**单独**产的 m̂）。今天的基准是 `0.5·ridge + 0.5·行级 LGBM` 的两分量
blend，口径是 5 折 / modulo 5 / phase_balanced / 训练窗 78,960。所以剩一个窄问题：

> 换到今天的口径、对今天的两分量 m̂，直接回归 `m_t` 是不是仍然输？

**预注册预期：仍然输。** 若结果相反，那是推翻一条已结案结论 —— 必须单独确认，不得直接采信。

## 口径

- `m_t` 用**无权**截面均值（= 生产分解与训练目标口径，也是推理端唯一算得出的那个）。
- 指标是尺度无关的 `peak = A²/B`，A/B 以 `Σ_t W_t·m_t²` 为分母，`W_t = Σ_i w_i`
  —— 与逐行 `Σw·m·m̂` 逐位等价（m 在 time_id 内是常数）。
- fold 划分**直接从 OOF cache 读**，保证与基准逐折对齐，不重新推导。
- 候选臂只在**采样**训练段上拟合（与生产的数据食谱一致），预处理/选列也只用训练段。

判据逐字沿用 `market_model.py:verdict`（折数 10→5，正折门槛按同一比例 8/10 → 4/5）。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/market_direct_recheck.py
输出：outputs/experiments/<label>.{json,md}；缓存 outputs/cache/xbar_unweighted_m5pb.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files  # noqa: E402
from features import apply_robust_transform  # noqa: E402
from history_features import iter_complete_time_batches  # noqa: E402  跨 batch 的 time_id 切分
from market_model import ab_peak, sign_test_p  # noqa: E402  指标与符号检验都复用既有实现
from train import robust_transform_fit  # noqa: E402

DEFAULT_OOF = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"
SAMPLE_MODULO, SAMPLING = 5, "phase_balanced"
ALPHAS = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
POSITIVE_FOLD_FRACTION = 0.8      # market_model 是 8/10；这里 5 折 ⟹ 4/5
MIN_SCORE_GAIN_SHARE = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(DEFAULT_OOF))
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--cache", default=str(_REPO_ROOT / "outputs" / "cache" / "xbar_unweighted_m5pb.npz"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="market_direct_recheck_3s480")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_unweighted_aggregates(data_root: Path) -> dict[str, np.ndarray]:
    """逐 time_id 的**无权**特征截面均值、无权 m_t、以及权重和 W_t（只留采样 time_id）。"""
    columns = ["time_id", "weight", *FEATURE_COLUMNS, "target"]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("time_id", "xbar", "m", "W")}
    for path in train_files(data_root):
        started, kept = time.perf_counter(), 0
        for frame in iter_complete_time_batches(path, columns, batch_size=120_000):
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            mask = time_sample_mask(tid, SAMPLE_MODULO, sampling=SAMPLING)
            if not mask.any():
                continue
            frame = frame.loc[mask]
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(tid)]).astype(np.float64)
            features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
            np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            target = frame["target"].to_numpy(dtype=np.float64, copy=False)
            weight = np.maximum(frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0)
            parts["time_id"].append(tid[starts])
            parts["xbar"].append((np.add.reduceat(features, starts, axis=0)
                                  / counts[:, None]).astype(np.float32))
            parts["m"].append(np.add.reduceat(target, starts) / counts)
            parts["W"].append(np.add.reduceat(weight, starts))
            kept += len(starts)
        print(f"  {path.name}: {kept:,} sampled time_ids ({time.perf_counter()-started:.0f}s)",
              flush=True)
    return {k: np.concatenate(v) for k, v in parts.items()}


def paired(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    delta = candidate - baseline
    drop_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    positive = int((delta > 0).sum())
    return {"mean_peak": float(candidate.mean()), "mean_delta": float(delta.mean()),
            "mean_delta_drop_best": float(drop_best.mean()),
            "relative": float(delta.mean() / baseline.mean()) if baseline.mean() else float("nan"),
            "positive_folds": positive, "n_folds": int(len(delta)),
            "sign_test_p": sign_test_p(positive, len(delta)),
            "per_fold_delta": [float(v) for v in delta]}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"loading unweighted aggregates from {cache_path}", flush=True)
        with np.load(cache_path, allow_pickle=False) as c:
            agg = {k: c[k] for k in ("time_id", "xbar", "m", "W")}
    else:
        print("building unweighted cross-sectional aggregates (one full scan)", flush=True)
        agg = build_unweighted_aggregates(Path(args.data_root))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **agg)
        print(f"wrote {cache_path}", flush=True)

    order = np.argsort(agg["time_id"], kind="stable")
    tid_all = agg["time_id"][order]
    xbar_all = agg["xbar"][order]
    m_all = agg["m"][order].astype(np.float64)
    W_all = agg["W"][order].astype(np.float64)

    # ---- 基准：今天的两分量 m̂，fold 划分直接从 OOF cache 读（不重新推导）
    with np.load(args.oof, allow_pickle=False) as d:
        fold_row = d["fold"].astype(np.int16)
        keep = fold_row >= 0
        tid_row = d["time_id"].astype(np.int64)[keep]
        market_row = d["market"].astype(np.float64)[keep]
        target_row = d["target"].astype(np.float64)[keep]
        weight_row = np.maximum(d["weight"].astype(np.float64)[keep], 0.0)
        raw_row = d["prediction_raw"].astype(np.float64)[keep]
        fold_row = fold_row[keep]
    starts = np.r_[0, np.flatnonzero(tid_row[1:] != tid_row[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(tid_row)]).astype(np.float64)
    oof_tid = tid_row[starts]
    oof_fold = fold_row[starts]
    oof_mhat = market_row[starts]                       # market 在 time_id 内恒定
    oof_m = np.add.reduceat(target_row, starts) / counts  # 无权截面均值

    index = np.searchsorted(tid_all, oof_tid)
    if not np.array_equal(tid_all[index], oof_tid):
        raise AssertionError("OOF time_ids are not a subset of the aggregate cache")
    if float(np.abs(m_all[index] - oof_m).max()) > 1e-9:
        raise AssertionError("unweighted m_t from the cache disagrees with the OOF cache")
    W_oof = W_all[index]

    folds = sorted(np.unique(oof_fold))
    train_ranges: dict[int, np.ndarray] = {}
    for f in folds:
        valid_tid = oof_tid[oof_fold == f]
        # 训练段 = 验证段之前、隔 embargo 的那 78,960 个采样 time_id（与 v3_production_oof 同构）
        cut = int(np.searchsorted(tid_all, valid_tid[0]))
        train_ranges[int(f)] = np.arange(max(0, cut - 6 - 78_960), max(0, cut - 6))
        if len(train_ranges[int(f)]) < 1000:
            raise AssertionError(f"fold {f} has too few training time_ids")

    baseline_peaks, per_fold_baseline = [], {}
    for f in folds:
        sel = oof_fold == f
        stats = ab_peak(oof_m[sel], oof_mhat[sel], W_oof[sel])
        baseline_peaks.append(stats["peak"])
        per_fold_baseline[int(f)] = stats
    baseline_peaks = np.asarray(baseline_peaks)
    print(f"基准（今天的两分量 m̂）逐折 peak_m: "
          f"{[round(v, 8) for v in baseline_peaks]}  均值 {baseline_peaks.mean():.8f}", flush=True)

    # ---- 候选：无权截面均值 → Ridge → m_t
    by_alpha: dict[str, Any] = {}
    for alpha in ALPHAS:
        peaks = []
        for f in folds:
            rows = train_ranges[int(f)]
            design_train = xbar_all[rows].copy()
            transformed, stats = robust_transform_fit(design_train)
            valid_index = index[oof_fold == f]
            design_valid = xbar_all[valid_index].copy()
            apply_robust_transform(design_valid, stats["lower"], stats["upper"],
                                   stats["center"], stats["scale"])
            model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-8, max_iter=2000)
            model.fit(transformed, m_all[rows], sample_weight=W_all[rows])
            fitted = model.predict(design_valid).astype(np.float64)
            sel = oof_fold == f
            peaks.append(ab_peak(oof_m[sel], fitted, W_oof[sel])["peak"])
            del transformed, design_train, design_valid
        by_alpha[f"{alpha:g}"] = paired(np.asarray(peaks), baseline_peaks)
        row = by_alpha[f"{alpha:g}"]
        print(f"  α={alpha:g}: peak {row['mean_peak']:.8f}  Δ {row['mean_delta']:+.8f} "
              f"({row['relative']*100:+.2f}%)  正折 {row['positive_folds']}/{row['n_folds']}",
              flush=True)

    best_alpha = max(by_alpha, key=lambda a: by_alpha[a]["mean_delta"])
    best = by_alpha[best_alpha]
    # 换算到总分：Δscore ≈ w_m · Δpeak_m。w_m 与当前 peak 都**就地实测**，不抄常数。
    target_energy = float(np.dot(weight_row, target_row * target_row))
    share_m = float(np.dot(W_oof, oof_m * oof_m) / target_energy)
    current_peak = float(np.dot(weight_row, target_row * raw_row) ** 2
                         / (target_energy * np.dot(weight_row, raw_row * raw_row)))
    score_gain_share = best["mean_delta"] * share_m / current_peak
    checks = {
        "1_paired_delta_positive_4of5":
            best["mean_delta"] > 0 and best["positive_folds"] >= int(np.ceil(
                POSITIVE_FOLD_FRACTION * best["n_folds"])),
        "2_survives_drop_best_fold": best["mean_delta_drop_best"] > 0,
        "3_positive_across_alpha_ladder": all(v["mean_delta"] > 0 for v in by_alpha.values()),
        "4_score_gain_at_least_5pct": score_gain_share >= MIN_SCORE_GAIN_SHARE,
    }
    checks = {k: bool(v) for k, v in checks.items()}

    payload = {
        "experiment": "market_direct_recheck",
        "question": "换到今天的口径、对今天的两分量 m̂，直接回归 m_t 是不是仍然输？",
        "preregistered_expectation": "仍然输；若相反则是推翻已结案结论，必须单独确认",
        "caliber": {"n_folds": len(folds), "sample_modulo": SAMPLE_MODULO, "sampling": SAMPLING,
                    "train_window_sampled_time_ids": 78_960, "embargo": 6,
                    "cross_sectional_mean": "unweighted",
                    "metric": "peak = A²/B（各臂在自己最优 scale 下的 R²_m）",
                    "fold_source": "直接读 OOF cache 的 fold 标签"},
        "prior_result": {"source": "outputs/experiments/market_model.md",
                         "direct_u": "+2.0%、6/10 折、符号 p=0.754、四条判据全否（旧口径、旧基准）"},
        "baseline_peaks": [float(v) for v in baseline_peaks],
        "baseline_mean_peak": float(baseline_peaks.mean()),
        "by_alpha": by_alpha, "best_alpha": best_alpha,
        "measured": {"share_m": share_m, "current_peak": current_peak},
        "score_gain_share": score_gain_share,
        "verdict": {"checks": checks, "pass": all(checks.values())},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 同口径复测：直接回归 m_t vs 今天的两分量 m̂", "",
             f"口径：{len(folds)} 折 / modulo {SAMPLE_MODULO} `{SAMPLING}` / 训练窗 78,960 个采样 "
             f"time_id / embargo 6；**无权**截面均值；指标 `peak = A²/B`。", "",
             f"预注册预期：**{payload['preregistered_expectation']}**", "",
             f"旧结论（`market_model.md`，旧口径旧基准）：{payload['prior_result']['direct_u']}", "",
             f"## 基准：今天的两分量 m̂，逐折 peak_m 均值 **{baseline_peaks.mean():.8f}**", "",
             "| α | 候选 peak_m | Δ vs 基准 | 相对 | 正折 | 符号 p | 去最好折 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for alpha, row in by_alpha.items():
        mark = " ←最好" if alpha == best_alpha else ""
        lines.append(f"| {alpha}{mark} | {row['mean_peak']:.8f} | {row['mean_delta']:+.8f} | "
                     f"{row['relative']*100:+.2f}% | {row['positive_folds']}/{row['n_folds']} | "
                     f"{row['sign_test_p']:.3f} | {row['mean_delta_drop_best']:+.8f} |")
    lines += ["", f"换算到总分的份额（`w_m·Δpeak / 当前 peak`）：**{score_gain_share*100:+.2f}%**", "",
              "## 判据（由代码判）", ""]
    lines += [f"- {'✅' if ok else '❌'} {k}" for k, ok in checks.items()]
    lines += ["", f"**{'✅ PASS —— 推翻了已结案结论，需单独确认' if all(checks.values()) else '❌ 不通过 —— 与旧口径结论一致'}**", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n判定：{'PASS' if all(checks.values()) else 'FAIL'}（best α={best_alpha}）")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
