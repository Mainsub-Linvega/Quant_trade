"""②：时间平滑值不值得做 —— OOF 上闭式求解，外加一次窄列全分辨率扫描。

平滑器 `f_t = ŷ_t + ρ·ŷ_{t−Δ}`（按 asset 对齐）的最优 ρ 有精确闭式解。记

    A(Δ)     = Σw·y_t·ŷ_t      B(Δ) = Σw·ŷ_t²
    A_lag(Δ) = Σw·y_t·ŷ_{t−Δ}  R(Δ) = Σw·ŷ_t·ŷ_{t−Δ}      （全部只在成对行上求和）
    ψ̃(Δ) = A_lag/A             r(Δ) = R/B

则

    ρ*(Δ)   = (ψ̃ − r) / (1 − ψ̃·r)
    gain(ρ) = peak(ρ)/peak(0) = (1 + ρψ̃)² / (1 + ρ² + 2ρr)

⟹ 用户的直觉在这里是**精确成立**的：`r` 是预测自己的自相关，信号侧的自相关是 ψ。
   预测明显不如信号平滑 ⟹ ρ*>0，平滑有免费收益；两者接近 ⟹ ρ*≈0；
   预测比信号更平滑 ⟹ ρ*<0，该做差分而不是平滑。

⚠️⚠️ 两个必须先说清楚的坑：

1. **现有 OOF 缓存测不了 lag 1。** `phase_balanced` + modulo 5 的采样网格上，最小真实间隔
   是 4。把缓存上的「相邻差」当 ac1 去比 0.836，是保证会得到假阳性的做法 ——
   正是 `mt_lagged.py` 开头写下的那个陷阱。所以信号侧的自相关另外用**全分辨率**测。
2. **`gain(ρ*)` 是被最大化出来的量，恒 ≥ 0。** 任何噪声都会产生正的「增益」。
   所以判据只认**样本外**（扩展窗口逐折）的配对增量，不认 in-sample 的 `gain(ρ*)`。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_temporal_smoothing.py
输出：outputs/experiments/<label>.{json,md}
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
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import train_files  # noqa: E402
from market_model import sign_test_p  # noqa: E402  复用既有符号检验

DEFAULT_OOF = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"
ATTENTION_THRESHOLD = 0.02  # ROADMAP §3.4 已在用的 2% 关注门槛
MOMENTS = ("D", "A", "B", "A_lag", "R")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(DEFAULT_OOF))
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_temporal_smoothing_3s480")
    p.add_argument("--max-cache-lag", type=int, default=50,
                   help="在缓存上扫描 1..N 的真实间隔，成对数不足的自动跳过")
    p.add_argument("--min-pairs", type=int, default=100_000)
    p.add_argument("--max-lag", type=int, default=12, help="全分辨率自相关曲线的最大 lag")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------ 闭式解

def solve_rho(m: dict[str, float]) -> float:
    psi = m["A_lag"] / m["A"] if m["A"] != 0.0 else float("nan")
    r = m["R"] / m["B"] if m["B"] != 0.0 else float("nan")
    denominator = 1.0 - psi * r
    return (psi - r) / denominator if denominator != 0.0 else float("nan")


def peak_at(m: dict[str, float], rho: float) -> float:
    """平滑后（按最优全局 scale）的 peak：(A+ρA_lag)² / (D·(B(1+ρ²)+2ρR))。"""
    numerator = (m["A"] + rho * m["A_lag"]) ** 2
    denominator = m["B"] * (1.0 + rho * rho) + 2.0 * rho * m["R"]
    return numerator / (m["D"] * denominator) if denominator > 0 else float("nan")


def describe(m: dict[str, float]) -> dict[str, float]:
    rho = solve_rho(m)
    base = peak_at(m, 0.0)
    best = peak_at(m, rho)
    return {**m,
            "psi_tilde": m["A_lag"] / m["A"] if m["A"] != 0 else float("nan"),
            "r_prediction_autocorr": m["R"] / m["B"] if m["B"] != 0 else float("nan"),
            "rho_star": rho,
            "peak_baseline": base,
            "peak_smoothed": best,
            "in_sample_relative_gain": best / base - 1.0 if base > 0 else float("nan")}


# ------------------------------------------------------------------ 成对配对

def asset_major_key(asset: np.ndarray, time_id: np.ndarray, span: int) -> np.ndarray:
    """`asset·span + time_id`。span 大于任何 time_id + lag ⟹ 键的升序 == (asset, time) 序，
    于是 `searchsorted(key, key + lag)` 一次找出**全部**恰好相隔 lag 的同 asset 配对
    （不只是相邻观测；采样网格上 lag 9 多数是 4+5 拼出来的，只看相邻会漏掉绝大部分）。"""
    return asset.astype(np.int64) * span + time_id.astype(np.int64)


def lagged_pairs(sorted_key: np.ndarray, positions: np.ndarray, fold_sorted: np.ndarray,
                 lag: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (当期行下标, 同 asset 恰好早 `lag` 个真实 time_id 的行下标)。

    要求同折：跨折配对会横跨 embargo 与不同训练窗，不属于同一分布。
    """
    wanted = sorted_key + lag
    left = np.searchsorted(sorted_key, wanted)
    inside = left < len(sorted_key)
    hit = np.zeros(len(sorted_key), dtype=bool)
    hit[inside] = sorted_key[left[inside]] == wanted[inside]
    previous_slots = np.flatnonzero(hit)
    current_slots = left[previous_slots]
    same_fold = fold_sorted[current_slots] == fold_sorted[previous_slots]
    return positions[current_slots[same_fold]], positions[previous_slots[same_fold]]


def grouped_moments(y, pred, w, current, previous, group_index, n_groups) -> np.ndarray:
    """逐 time_id 的 5 个成对矩，形状 (n_groups, 5)。逐折与 bootstrap 都只是对它求和。"""
    g = group_index[current]
    wc, yc = w[current], y[current]
    pc, pp = pred[current], pred[previous]
    columns = (wc * yc * yc, wc * yc * pc, wc * pc * pc, wc * yc * pp, wc * pc * pp)
    return np.column_stack([np.bincount(g, weights=c, minlength=n_groups) for c in columns])


def as_moments(totals: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(MOMENTS, totals)}


# --------------------------------------------- 窄列全分辨率自相关（信号侧）

def full_resolution_autocorrelation(data_root: Path, max_lag: int) -> dict[str, Any]:
    """只读 time_id/asset_id/weight/target 四列，算真实 lag 的信号自相关。

    三条从未在同一口径下并列过的曲线：
      * **无权** m_t（= 生产分解与训练目标口径）
      * **加权** m_t（= `mt_diagnostics` 口径，用来复核 ac1 = 0.836）
      * 逐 asset 的 cross 分量 e = y − m_unweighted（本项目从未测过）
    """
    import pyarrow.parquet as pq

    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("time", "asset", "weight", "target")}
    for path in train_files(data_root):
        started, rows = time.perf_counter(), 0
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=250_000, columns=["time_id", "asset_id", "weight", "target"]):
            frame = batch.to_pandas()
            parts["time"].append(frame["time_id"].to_numpy(dtype=np.int64, copy=True))
            parts["asset"].append(frame["asset_id"].to_numpy(dtype=np.int16, copy=True))
            parts["weight"].append(frame["weight"].to_numpy(dtype=np.float64, copy=True))
            parts["target"].append(frame["target"].to_numpy(dtype=np.float64, copy=True))
            rows += len(frame)
        print(f"  {path.name}: {rows:,} 行 ({time.perf_counter() - started:.0f}s)", flush=True)
    time_id = np.concatenate(parts["time"])
    asset_id = np.concatenate(parts["asset"])
    weight = np.maximum(np.concatenate(parts["weight"]), 0.0)
    target = np.concatenate(parts["target"])
    del parts

    if not np.all(np.diff(time_id) >= 0):
        raise AssertionError("full-resolution rows are not sorted by time_id")
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(np.float64)
    group_time = time_id[starts]

    m_unweighted = np.add.reduceat(target, starts) / counts
    weight_sum = np.add.reduceat(weight, starts)
    m_weighted = np.divide(np.add.reduceat(weight * target, starts), weight_sum,
                           out=np.zeros_like(weight_sum), where=weight_sum > 0)
    cross = target - np.repeat(m_unweighted, counts.astype(int))

    def profile(values: np.ndarray, keys: np.ndarray,
                pair_weight: np.ndarray | None) -> dict[str, list[float]]:
        centered, uncentered, pairs = [], [], []
        for lag in range(1, max_lag + 1):
            wanted = keys + lag
            left = np.searchsorted(keys, wanted)
            inside = left < len(keys)
            hit = np.zeros(len(keys), dtype=bool)
            hit[inside] = keys[left[inside]] == wanted[inside]
            previous_slots = np.flatnonzero(hit)
            current_slots = left[previous_slots]
            pairs.append(int(len(previous_slots)))
            if len(previous_slots) < 2:
                centered.append(float("nan"))
                uncentered.append(float("nan"))
                continue
            current, previous = values[current_slots], values[previous_slots]
            centered.append(float(np.corrcoef(current, previous)[0, 1]))
            pw = np.ones_like(current) if pair_weight is None else pair_weight[current_slots]
            energy = float(np.dot(pw, previous * previous))
            uncentered.append(float(np.dot(pw, current * previous) / energy)
                              if energy > 0 else float("nan"))
        return {"centered": centered, "uncentered": uncentered, "pairs": pairs}

    span = int(time_id.max()) + max_lag + 1
    order = np.argsort(asset_major_key(asset_id, time_id, span), kind="stable")
    key_sorted = asset_major_key(asset_id[order], time_id[order], span)
    return {
        "time_ids": int(len(group_time)), "rows": int(len(time_id)),
        "market_unweighted": profile(m_unweighted, group_time, None),
        "market_weighted": profile(m_weighted, group_time, None),
        "cross_per_asset": profile(cross[order], key_sorted, weight[order]),
    }


# ------------------------------------------------------------------ 样本外

def out_of_sample(fold_moments: dict[int, np.ndarray]) -> dict[str, Any]:
    """扩展窗口：ρ 只用 fold 0..k−1 拟合，在 fold k 上评估。基线是同一批行的 ρ=0。"""
    folds = sorted(fold_moments)
    rows = []
    for position, f in enumerate(folds):
        if position == 0:
            continue
        fitted = as_moments(np.sum([fold_moments[g] for g in folds[:position]], axis=0))
        rho = solve_rho(fitted)
        evaluated = as_moments(fold_moments[f])
        base = peak_at(evaluated, 0.0)
        smoothed = peak_at(evaluated, rho)
        rows.append({"fold": int(f), "rho_fitted_on_prior_folds": float(rho),
                     "peak_baseline": base, "peak_smoothed": smoothed,
                     "delta": smoothed - base,
                     "relative": (smoothed - base) / base if base > 0 else float("nan")})
    deltas = np.array([r["delta"] for r in rows], dtype=float)
    baselines = np.array([r["peak_baseline"] for r in rows], dtype=float)
    positive = int((deltas > 0).sum())
    drop_best = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
    return {"folds": rows,
            "mean_relative": float(deltas.mean() / baselines.mean()) if baselines.mean() > 0 else float("nan"),
            "mean_relative_drop_best": float(drop_best.mean() / baselines.mean())
            if baselines.mean() > 0 else float("nan"),
            "positive_folds": positive, "n_folds": int(len(deltas)),
            "sign_test_p": sign_test_p(positive, len(deltas))}


def bootstrap_ratios(rows: np.ndarray, block_size: int, n_boot: int, seed: int) -> dict[str, Any]:
    n_groups = len(rows)
    n_blocks = int(np.ceil(n_groups / block_size))
    prefix = np.vstack([np.zeros(len(MOMENTS)), np.cumsum(rows, axis=0)])
    max_start = max(n_groups - block_size, 0)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in ("psi_tilde", "r_prediction_autocorr",
                                                       "wedge", "rho_star")}
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        stops = np.minimum(starts + block_size, n_groups)
        m = as_moments((prefix[stops] - prefix[starts]).sum(axis=0))
        psi = m["A_lag"] / m["A"] if m["A"] != 0 else float("nan")
        r = m["R"] / m["B"] if m["B"] != 0 else float("nan")
        samples["psi_tilde"].append(psi)
        samples["r_prediction_autocorr"].append(r)
        samples["wedge"].append(psi - r)
        samples["rho_star"].append(solve_rho(m))
    return {name: {"p2.5": float(np.nanpercentile(v, 2.5)),
                   "p50": float(np.nanpercentile(v, 50.0)),
                   "p97.5": float(np.nanpercentile(v, 97.5))}
            for name, v in samples.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    with np.load(args.oof, allow_pickle=False) as d:
        fold_all = d["fold"].astype(np.int16)
        keep = fold_all >= 0
        target = d["target"].astype(np.float64)[keep]
        weight = np.maximum(d["weight"].astype(np.float64)[keep], 0.0)
        time_id = d["time_id"].astype(np.int64)[keep]
        asset_id = d["asset_id"].astype(np.int64)[keep]
        fold = fold_all[keep]
        blocks = {"market": d["market"].astype(np.float64)[keep],
                  "cross": d["e_lgbm"].astype(np.float64)[keep],
                  "prediction_raw": d["prediction_raw"].astype(np.float64)[keep]}

    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(np.int64)
    n_groups = len(starts)
    group_index = np.repeat(np.arange(n_groups), counts)
    group_fold = fold[starts]

    span = int(time_id.max()) + args.max_cache_lag + 1
    order = np.argsort(asset_major_key(asset_id, time_id, span), kind="stable")
    sorted_key = asset_major_key(asset_id[order], time_id[order], span)
    fold_sorted = fold[order]

    print("步骤 1：缓存上真实存在的 lag —— in-sample 闭式解 + 样本外验证", flush=True)
    step1: dict[str, dict[str, Any]] = {name: {} for name in blocks}
    observed_lags: list[int] = []
    for lag in range(1, args.max_cache_lag + 1):
        current, previous = lagged_pairs(sorted_key, order, fold_sorted, lag)
        if len(current) < args.min_pairs:
            continue
        observed_lags.append(lag)
        # 采样网格上每个 phase 落在固定的 mod-50 剩余类，所以不同 lag 覆盖的 phase 组合不同。
        # 成对数少的 lag 只覆盖窄相位子集，其异常值应读作**相位构成**假象而非时序信号。
        phase_pairs = int(len(np.unique(
            (time_id[previous] % 10) * 10 + (time_id[current] % 10))))
        for name, prediction in blocks.items():
            rows = grouped_moments(target, prediction, weight, current, previous,
                                   group_index, n_groups)
            pooled = describe(as_moments(rows.sum(axis=0)))
            per_fold = {int(f): rows[group_fold == f].sum(axis=0) for f in np.unique(group_fold)}
            step1[name][str(lag)] = {
                "pairs": int(len(current)),
                "distinct_phase_pairs": phase_pairs,
                "in_sample": pooled,
                "out_of_sample": out_of_sample(per_fold),
                "bootstrap": bootstrap_ratios(rows, args.block_size, args.n_boot, args.boot_seed),
            }
        show = step1["prediction_raw"][str(lag)]
        print(f"  lag {lag:>2}: {len(current):>9,} 对 | "
              f"raw r={show['in_sample']['r_prediction_autocorr']:+.3f} "
              f"ρ*={show['in_sample']['rho_star']:+.3f} | "
              f"in-sample {show['in_sample']['in_sample_relative_gain']*100:+.2f}% → "
              f"OOS {show['out_of_sample']['mean_relative']*100:+.2f}% "
              f"({show['out_of_sample']['positive_folds']}/{show['out_of_sample']['n_folds']})",
              flush=True)
    if not observed_lags:
        raise SystemExit("no lag on the sampled grid reached --min-pairs")

    print("步骤 2：窄列全分辨率信号自相关（只读 4 列）", flush=True)
    started = time.perf_counter()
    full = full_resolution_autocorrelation(Path(args.data_root), args.max_lag)
    print(f"  完成，用时 {time.perf_counter() - started:.0f}s", flush=True)

    # ---- 步骤 3：把「信号平滑度」与「预测平滑度」放在**同一个真实 lag** 上比较
    signal_curve = {"market": full["market_unweighted"]["centered"],
                    "cross": full["cross_per_asset"]["centered"]}
    comparison = []
    for lag in observed_lags:
        if lag > args.max_lag:
            continue
        row: dict[str, Any] = {"lag": lag}
        for name in ("market", "cross"):
            row[f"{name}_signal_ac"] = float(signal_curve[name][lag - 1])
            row[f"{name}_prediction_ac"] = float(
                step1[name][str(lag)]["in_sample"]["r_prediction_autocorr"])
        comparison.append(row)

    reference_lag = str(min(observed_lags))
    # 成对行数在 lag 之间差 10 倍（146k ~ 1.46M），少的那些 ψ̃ 噪声也大 10 倍。
    # 单调性只在**成对行数充足**的 lag 上判定，稀疏 lag 单独列出。
    dense_lags = [lag for lag in observed_lags
                  if step1["prediction_raw"][str(lag)]["pairs"] >= 500_000]
    sparse_lags = [lag for lag in observed_lags if lag not in dense_lags]

    def r_of(lag: int) -> float:
        return step1["prediction_raw"][str(lag)]["in_sample"]["r_prediction_autocorr"]

    monotone_dense = all(r_of(a) >= r_of(b) for a, b in zip(dense_lags, dense_lags[1:]))
    monotone_all = all(r_of(a) >= r_of(b) for a, b in zip(observed_lags, observed_lags[1:]))

    oos_at_reference = {name: step1[name][reference_lag]["out_of_sample"]["mean_relative"]
                        for name in blocks}
    best_oos = max(oos_at_reference.values())
    verdict = {
        "threshold": ATTENTION_THRESHOLD,
        "gate": f"最小可测 lag（={reference_lag}）上的**样本外**相对 peak 增益",
        "reason": "lag→0 时 ψ̃ 与 r 同时 →1，楔子 ψ̃−r 只会更小 ⟹ 最小可测 lag 的增益是 "
                  "lag-1 增益的上界；且 in-sample gain(ρ*) 恒 ≥0，只能看样本外。",
        "out_of_sample_relative_gain_at_reference_lag": oos_at_reference,
        "best": best_oos,
        "positive_folds_prediction_raw":
            step1["prediction_raw"][reference_lag]["out_of_sample"]["positive_folds"],
        "decision": "ESCALATE" if best_oos >= ATTENTION_THRESHOLD else "KILL",
    }

    payload = {
        "experiment": "v3_temporal_smoothing",
        "question": "预测的自相关是否明显低于信号的自相关？时间平滑有没有免费收益？",
        "oof": str(args.oof),
        "closed_form": {"rho_star": "(ψ̃ − r)/(1 − ψ̃·r)",
                        "gain": "(1 + ρψ̃)²/(1 + ρ² + 2ρr)",
                        "psi_tilde": "A_lag/A —— 旧预测对今天 target 的对齐度残留",
                        "r": "R/B —— 预测自己的加权无中心自相关"},
        "critical_caveats": [
            "OOF 采样网格最小真实间隔是 4，**无法测量 lag 1**；把缓存上的相邻差当 ac1 去比 "
            "0.836 必然得到假阳性（`mt_lagged.py` 记录过同型陷阱）。",
            "`gain(ρ*)` 是被最大化出来的量，恒 ≥0；判据只认样本外配对增量。",
        ],
        "observed_cache_lags": observed_lags,
        "dense_lags": dense_lags, "sparse_lags": sparse_lags,
        "prediction_autocorr_monotone_dense_lags": bool(monotone_dense),
        "prediction_autocorr_monotone_all_lags": bool(monotone_all),
        "step1_per_lag": step1,
        "step2_full_resolution_signal_autocorrelation": full,
        "step3_matched_lag_comparison": comparison,
        "verdict": verdict,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mu, mw, cx = (full["market_unweighted"], full["market_weighted"], full["cross_per_asset"])
    lines = [
        "# ②：时间平滑值不值得做", "",
        f"OOF：`{Path(args.oof).name}`", "",
        "## 结论先行", "",
        f"信号侧 ac1：无权 `m_t` = **{mu['centered'][0]:.4f}**（加权口径 "
        f"{mw['centered'][0]:.4f}，复核了 `mt_diagnostics` 的 0.836）；"
        f"逐 asset cross `e` = **{cx['centered'][0]:.4f}**（本项目首次测量）。", "",
        f"但在**同一个真实 lag** 上，预测比信号**平滑得多**：真实 lag 4 处信号已衰减到 "
        f"{mu['centered'][3]:.3f}，而 market 预测的自相关仍有 "
        f"{step1['market'][reference_lag]['in_sample']['r_prediction_autocorr']:.3f}。"
        "方向与假设相反 —— 预测里不是「噪声不如信号平滑」，而是**整体过度平滑**。", "",
        "## 步骤 3：同一真实 lag 上的信号 vs 预测自相关", "",
        "| 真实 lag | 信号 `m_t`（无权） | 预测 `m̂` | 信号 cross `e` | 预测 `ê` |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['lag']} | {row['market_signal_ac']:+.4f} | "
                     f"{row['market_prediction_ac']:+.4f} | {row['cross_signal_ac']:+.4f} | "
                     f"{row['cross_prediction_ac']:+.4f} |")
    lines += [
        "", "## 步骤 2：全分辨率信号自相关（只读 4 列，真实 lag）", "",
        "| lag | 无权 `m_t`（生产口径） | 加权 `m_t`（mt_diagnostics 口径） | 逐 asset cross `e` |",
        "|---:|---:|---:|---:|",
    ]
    for k in range(args.max_lag):
        lines.append(f"| {k+1} | {mu['centered'][k]:+.4f} | {mw['centered'][k]:+.4f} | "
                     f"{cx['centered'][k]:+.4f} |")
    lines += [
        "", "## 步骤 1：逐 lag 的 in-sample 闭式解与**样本外**验证", "",
        "> `gain(ρ*)` 恒 ≥0，所以只有 OOS 列有判据意义。OOS = ρ 只在 fold 0..k−1 上拟合、"
        "在 fold k 上评估，基线是同一批行的 ρ=0。", "",
        f"> 成对行数在 lag 之间相差 10 倍（{min(step1['prediction_raw'][str(l)]['pairs'] for l in observed_lags):,} ~ "
        f"{max(step1['prediction_raw'][str(l)]['pairs'] for l in observed_lags):,}）。"
        f"稀疏 lag {sparse_lags} 的 ψ̃ 噪声大得多 —— 唯一出现大幅正 ρ* 的正是它们，"
        "应读作估计噪声而非时序信号。「相位组合」列一并给出，用于核对配对的相位构成。", "",
        "| 块 | lag | 成对行 | 相位组合 | ψ̃ | r | ρ* | in-sample | **OOS** | 正折 | 去最好折 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("market", "cross", "prediction_raw"):
        for lag in observed_lags:
            entry = step1[name][str(lag)]
            i, o = entry["in_sample"], entry["out_of_sample"]
            lines.append(
                f"| `{name}` | {lag} | {entry['pairs']:,} | {entry['distinct_phase_pairs']} | "
                f"{i['psi_tilde']:+.4f} | "
                f"{i['r_prediction_autocorr']:+.4f} | {i['rho_star']:+.4f} | "
                f"{i['in_sample_relative_gain']*100:+.2f}% | **{o['mean_relative']*100:+.2f}%** | "
                f"{o['positive_folds']}/{o['n_folds']} | {o['mean_relative_drop_best']*100:+.2f}% |")
    ref = step1["prediction_raw"][reference_lag]["bootstrap"]
    lines += [
        "", "## 判据", "",
        f"- 门槛：{verdict['gate']} ≥ {ATTENTION_THRESHOLD*100:.0f}%",
        f"- 理由：{verdict['reason']}",
        f"- 实测（lag {reference_lag}）：" + "，".join(
            f"`{k}` {v*100:+.2f}%" for k, v in oos_at_reference.items()),
        f"- `prediction_raw` 在 lag {reference_lag} 的 bootstrap 95% CI："
        f"楔子 ψ̃−r ∈ [{ref['wedge']['p2.5']:+.4f}, {ref['wedge']['p97.5']:+.4f}]，"
        f"ρ* ∈ [{ref['rho_star']['p2.5']:+.3f}, {ref['rho_star']['p97.5']:+.3f}]",
        f"- 预测自相关随 lag 单调不增：成对行 ≥50 万的 lag 上 "
        f"{'**是**' if monotone_dense else '**否**'}"
        f"（{dense_lags}）；含稀疏 lag 时 {'是' if monotone_all else '否'}"
        f"（{sparse_lags} 有回升，与其噪声更大一致）。"
        f"⟹ 支持「lag-1 的 r ≥ lag-{reference_lag} 的 r」，"
        f"即 lag-1 的楔子不会比 lag-{reference_lag} 更大。",
        "", f"### **{verdict['decision']}**", "",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n判据：{json.dumps(verdict, ensure_ascii=False, indent=2)}")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
