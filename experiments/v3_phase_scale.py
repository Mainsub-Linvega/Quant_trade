"""③：最优 scale 该不该分 phase —— OOF 上解析求解，零公榜配额。

`phase = time_id % 10` 在推理端**确定已知**（`main.py` 拿得到 `time_id`），
所以它和被否掉的 asset×regime / asset×magnitude 有本质区别：那些条件在**估计出来**的量上，
噪声大还会漂移；phase 是一口钟。且它是纯①类后处理参数，不需要重训。

Score 在未触 clip 时是精确二次式，于是分 phase 的最优系数有闭式解：

  主臂（分 phase scale）  Score = Σ_p (2·a_p·Σ_p w·y·ŷ − a_p²·Σ_p w·ŷ²) / D
                          a_p(κ) = (Σ_p w·y·ŷ + κ·总/10) / (Σ_p w·ŷ² + κ·总/10)
  次臂（分 phase 两系数）  f = c_m,p·m̂ + c_e,p·ê，  c_p(κ) = (G_p + κ·G/10)⁻¹ (v_p + κ·v/10)

κ = 0 是纯分 phase，κ → ∞ 精确退化为全局解 —— 退化性由断言检查，不靠肉眼。

## 先写判据，再看结果（CLAUDE.md §5.1）

**基线**：**同一批训练折上解出的全局解**，不是 1.16、也不是 pooled 最优。
否则会把全局重标定的收益记到 phase 臂头上（`conditional_blend.py` 的教训）。
次臂的基线是**全局两系数**解，这样次臂只拿到「分 phase」那部分功劳，
不会把「两系数比单 scale 好」也算进来。

**评估**：扩展窗口。fold k（k≥1）的系数只用 fold 0..k−1 拟合，在 fold k 上评估。
留一折（LOFO）只作稳定性探针，用到未来折，不作判据。

**PASS 门槛（全部满足）**：
  1. OOS 配对 Score 增量折均 > 0
  2. ≥ 3/4 评估折为正
  3. 去掉最好一折仍为正
  4. 相对增益 ≥ 1%
  5. κ ∈ {1,2,5} 连成一段平台都为正（抗 κ 选择的乐观偏差）
  6. block bootstrap 95% CI 下界 > 0

**不做**：不改 `main.py`、不改 `hybrid_meta.json`、不建候选目录。`prediction_scale`
属模型身份（CLAUDE.md §6），上线需要 meta schema 变更 + promotion 全套门禁 + 用户授权。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_phase_scale.py
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from market_model import sign_test_p  # noqa: E402

DEFAULT_OOF = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"
PHASE_PERIOD = 10
SHRINKAGE = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
PLATEAU = (1.0, 2.0, 5.0)          # 门槛 5：这一段必须整段为正
MIN_RELATIVE_GAIN = 0.01           # 门槛 4
MOMENTS = ("yy", "ymh", "yeh", "mhmh", "mheh", "eheh")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(DEFAULT_OOF))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_phase_scale_3s480")
    p.add_argument("--prediction-clip", type=float, default=0.5)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------- 充分统计量

def phase_totals(rows: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """把逐 time_id 的矩按 phase 汇总，返回 (PHASE_PERIOD, 6)。"""
    out = np.zeros((PHASE_PERIOD, len(MOMENTS)), dtype=np.float64)
    for index in range(len(MOMENTS)):
        out[:, index] = np.bincount(phase, weights=rows[:, index], minlength=PHASE_PERIOD)
    return out


def _column(totals: np.ndarray, name: str) -> np.ndarray:
    return totals[:, MOMENTS.index(name)]


def scale_numerator(totals: np.ndarray) -> np.ndarray:
    """Σ w·y·ŷ，ŷ = m̂ + ê。"""
    return _column(totals, "ymh") + _column(totals, "yeh")


def scale_denominator(totals: np.ndarray) -> np.ndarray:
    """Σ w·ŷ²。"""
    return (_column(totals, "mhmh") + 2.0 * _column(totals, "mheh")
            + _column(totals, "eheh"))


def fit_scales(totals: np.ndarray, kappa: float) -> np.ndarray:
    """主臂：分 phase scale，向全局收缩 κ。"""
    numerator, denominator = scale_numerator(totals), scale_denominator(totals)
    if not np.isfinite(kappa):                      # κ → ∞：精确的全局解
        return np.full(PHASE_PERIOD, numerator.sum() / denominator.sum())
    prior_n, prior_d = numerator.sum() / PHASE_PERIOD, denominator.sum() / PHASE_PERIOD
    return (numerator + kappa * prior_n) / (denominator + kappa * prior_d)


def gram_and_vector(totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """次臂：每个 phase 的 2×2 Gram 与右端项。"""
    G = np.empty((PHASE_PERIOD, 2, 2), dtype=np.float64)
    G[:, 0, 0] = _column(totals, "mhmh")
    G[:, 0, 1] = G[:, 1, 0] = _column(totals, "mheh")
    G[:, 1, 1] = _column(totals, "eheh")
    v = np.column_stack([_column(totals, "ymh"), _column(totals, "yeh")])
    return G, v


def fit_blend(totals: np.ndarray, kappa: float) -> np.ndarray:
    """次臂：分 phase 两系数 (c_m,p, c_e,p)，向全局两系数解收缩 κ。"""
    G, v = gram_and_vector(totals)
    if not np.isfinite(kappa):
        c = np.linalg.solve(G.sum(axis=0), v.sum(axis=0))
        return np.tile(c, (PHASE_PERIOD, 1))
    prior_G, prior_v = G.sum(axis=0) / PHASE_PERIOD, v.sum(axis=0) / PHASE_PERIOD
    # numpy 2 的 solve 把 (10, 2) 当成一个 10×2 矩阵，所以要显式加一维再压回去
    return np.linalg.solve(G + kappa * prior_G, (v + kappa * prior_v)[..., None])[..., 0]


def score_scales(totals: np.ndarray, scales: np.ndarray) -> float:
    """把一组固定的 a_p 施加到给定的矩上，得到比赛 Score。"""
    numerator, denominator = scale_numerator(totals), scale_denominator(totals)
    energy = _column(totals, "yy").sum()
    return float((2.0 * scales * numerator - scales ** 2 * denominator).sum() / energy)


def score_blend(totals: np.ndarray, coefficients: np.ndarray) -> float:
    G, v = gram_and_vector(totals)
    energy = _column(totals, "yy").sum()
    quadratic = np.einsum("pi,pij,pj->p", coefficients, G, coefficients)
    linear = np.einsum("pi,pi->p", coefficients, v)
    return float((2.0 * linear - quadratic).sum() / energy)


ARMS = {
    "phase_scale": {"fit": fit_scales, "score": score_scales,
                    "label": "主臂：分 phase scale a_p = A_p/B_p",
                    "baseline": "同一批训练折上解出的**全局单 scale**"},
    "phase_blend": {"fit": fit_blend, "score": score_blend,
                    "label": "次臂：分 phase 两系数 c_m,p·m̂ + c_e,p·ê",
                    "baseline": "同一批训练折上解出的**全局两系数**解"},
}


# --------------------------------------------------------------- 评估协议

def expanding_window(arm: str, fold_phase_totals: dict[int, np.ndarray],
                     kappa: float) -> dict[str, Any]:
    """fold k 的系数只用 fold 0..k−1 拟合；基线是同一臂的 κ=∞（全局）解。"""
    fit, score = ARMS[arm]["fit"], ARMS[arm]["score"]
    folds = sorted(fold_phase_totals)
    rows = []
    for position, f in enumerate(folds):
        if position == 0:
            continue
        trained = np.sum([fold_phase_totals[g] for g in folds[:position]], axis=0)
        evaluated = fold_phase_totals[f]
        candidate = score(evaluated, fit(trained, kappa))
        baseline = score(evaluated, fit(trained, float("inf")))
        rows.append({"fold": int(f), "baseline": baseline, "candidate": candidate,
                     "delta": candidate - baseline})
    deltas = np.array([r["delta"] for r in rows], dtype=float)
    baselines = np.array([r["baseline"] for r in rows], dtype=float)
    drop_best = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
    reference = float(np.abs(baselines).mean())
    positive = int((deltas > 0).sum())
    return {"folds": rows, "mean_delta": float(deltas.mean()),
            "mean_delta_drop_best": float(drop_best.mean()),
            "mean_baseline": float(baselines.mean()),
            "relative": float(deltas.mean() / reference) if reference > 0 else float("nan"),
            "relative_drop_best": float(drop_best.mean() / reference)
            if reference > 0 else float("nan"),
            "positive_folds": positive, "n_folds": int(len(deltas)),
            "sign_test_p": sign_test_p(positive, len(deltas))}


def leave_one_fold_out(arm: str, fold_phase_totals: dict[int, np.ndarray],
                       kappa: float) -> dict[str, Any]:
    """诊断用：用到未来折，只看 phase 模式稳不稳，不作判据。"""
    fit, score = ARMS[arm]["fit"], ARMS[arm]["score"]
    folds = sorted(fold_phase_totals)
    deltas = []
    for f in folds:
        trained = np.sum([fold_phase_totals[g] for g in folds if g != f], axis=0)
        evaluated = fold_phase_totals[f]
        deltas.append(score(evaluated, fit(trained, kappa))
                      - score(evaluated, fit(trained, float("inf"))))
    array = np.array(deltas, dtype=float)
    return {"mean_delta": float(array.mean()), "positive_folds": int((array > 0).sum()),
            "n_folds": int(len(array))}


def bootstrap_oos(arm: str, rows: np.ndarray, phase: np.ndarray, group_fold: np.ndarray,
                  fold_phase_totals: dict[int, np.ndarray], kappa: float,
                  block_size: int, n_boot: int, seed: int) -> dict[str, float]:
    """系数按扩展窗口固定不动，只对**评估折的 time_id** 做连续块重采样。"""
    fit, score = ARMS[arm]["fit"], ARMS[arm]["score"]
    folds = sorted(fold_phase_totals)
    rng = np.random.default_rng(seed)
    per_fold_coefficients = {}
    for position, f in enumerate(folds):
        if position == 0:
            continue
        trained = np.sum([fold_phase_totals[g] for g in folds[:position]], axis=0)
        per_fold_coefficients[f] = (fit(trained, kappa), fit(trained, float("inf")))

    samples = []
    for _ in range(n_boot):
        total_delta, total_energy = 0.0, 0.0
        for f, (candidate_coefficients, baseline_coefficients) in per_fold_coefficients.items():
            selected = np.flatnonzero(group_fold == f)
            n_groups = len(selected)
            n_blocks = int(np.ceil(n_groups / block_size))
            starts = rng.integers(0, max(n_groups - block_size, 0) + 1, size=n_blocks)
            index = np.concatenate([selected[s:min(s + block_size, n_groups)] for s in starts])
            resampled = phase_totals(rows[index], phase[index])
            energy = _column(resampled, "yy").sum()
            total_delta += (score(resampled, candidate_coefficients)
                            - score(resampled, baseline_coefficients)) * energy
            total_energy += energy
        samples.append(total_delta / total_energy if total_energy > 0 else np.nan)
    array = np.asarray(samples, dtype=float)
    return {"p2.5": float(np.nanpercentile(array, 2.5)),
            "p50": float(np.nanpercentile(array, 50.0)),
            "p97.5": float(np.nanpercentile(array, 97.5))}


def heterogeneity_test(rows: np.ndarray, phase: np.ndarray, block_size: int,
                       n_boot: int, seed: int) -> dict[str, Any]:
    """phase 之间到底有没有真异质性？还是纯估计噪声？

    方差分量：`观测到的 A_p 跨 phase 方差` 里，有一部分只是每个 A_p 自己的抽样方差。
    用 block bootstrap 量出后者，相减得到**超额方差**。超额 ≤ 0 ⟹ 观测到的离散度
    完全能由噪声解释，不能说「phase 有异质性但被淹没」，只能说「测不出来」。
    """
    n_groups = len(rows)
    n_blocks = int(np.ceil(n_groups / block_size))
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_boot, PHASE_PERIOD), dtype=np.float64)
    for index in range(n_boot):
        starts = rng.integers(0, max(n_groups - block_size, 0) + 1, size=n_blocks)
        select = np.concatenate([np.arange(s, min(s + block_size, n_groups)) for s in starts])
        totals = phase_totals(rows[select], phase[select])
        replicates[index] = scale_numerator(totals) / _column(totals, "yy").sum()

    observed = scale_numerator(phase_totals(rows, phase)) / _column(phase_totals(rows, phase), "yy").sum()
    observed_variance = float(np.var(observed, ddof=1))
    sampling_variance = float(np.mean(np.var(replicates, axis=0, ddof=1)))
    # 每个 replicate 自己的跨 phase 方差 ⟹ 给超额方差一个 bootstrap 分布
    replicate_spread = np.var(replicates, axis=1, ddof=1) - sampling_variance
    return {
        "observed_cross_phase_variance_of_A": observed_variance,
        "bootstrap_sampling_variance_of_A": sampling_variance,
        "excess_variance": observed_variance - sampling_variance,
        "excess_variance_ci": {"p2.5": float(np.percentile(replicate_spread, 2.5)),
                               "p97.5": float(np.percentile(replicate_spread, 97.5))},
        "observed_over_sampling_ratio": observed_variance / sampling_variance
        if sampling_variance > 0 else float("nan"),
        "heterogeneity_detected": bool(observed_variance - sampling_variance > 0
                                       and np.percentile(replicate_spread, 2.5) > 0),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


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
        fold = fold_all[keep]
        m_hat = d["market"].astype(np.float64)[keep]
        e_hat = d["e_lgbm"].astype(np.float64)[keep]
        raw = d["prediction_raw"].astype(np.float64)[keep]

    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    columns = {"yy": target * target, "ymh": target * m_hat, "yeh": target * e_hat,
               "mhmh": m_hat * m_hat, "mheh": m_hat * e_hat, "eheh": e_hat * e_hat}
    rows = np.column_stack([np.add.reduceat(weight * columns[name], starts) for name in MOMENTS])
    phase = (time_id[starts] % PHASE_PERIOD).astype(np.int64)
    group_fold = fold[starts]

    pooled = phase_totals(rows, phase)
    if (_column(pooled, "yy") <= 0).any():
        raise AssertionError("some phase has zero target energy in the OOF")

    # ---- 退化断言：κ→∞ 必须逐位复原全局解；能量必须复原总能量
    global_scale = scale_numerator(pooled).sum() / scale_denominator(pooled).sum()
    if not np.allclose(fit_scales(pooled, float("inf")), global_scale, rtol=0, atol=1e-15):
        raise AssertionError("kappa=inf scale arm does not reduce to the global scale")
    global_blend = np.linalg.solve(*(x.sum(axis=0) for x in gram_and_vector(pooled)))
    if not np.allclose(fit_blend(pooled, float("inf")), global_blend, rtol=0, atol=1e-15):
        raise AssertionError("kappa=inf blend arm does not reduce to the global solution")
    if abs(_column(pooled, "yy").sum() - float(np.dot(weight, target * target))) > 1e-6:
        raise AssertionError("phase-partitioned target energy does not recover the total")

    fold_phase_totals = {int(f): phase_totals(rows[group_fold == f], phase[group_fold == f])
                         for f in np.unique(group_fold)}

    results: dict[str, Any] = {}
    for arm in ARMS:
        by_kappa = {}
        for kappa in SHRINKAGE:
            key = "inf" if not np.isfinite(kappa) else f"{kappa:g}"
            by_kappa[key] = {
                "expanding_window": expanding_window(arm, fold_phase_totals, kappa),
                "leave_one_fold_out": leave_one_fold_out(arm, fold_phase_totals, kappa),
            }
        finite = [k for k in SHRINKAGE if np.isfinite(k)]
        plateau_ok = all(
            by_kappa[f"{k:g}"]["expanding_window"]["mean_delta"] > 0 for k in PLATEAU)
        best_key = max((f"{k:g}" for k in finite),
                       key=lambda k: by_kappa[k]["expanding_window"]["mean_delta"])
        primary = by_kappa[best_key]["expanding_window"]
        boot = bootstrap_oos(arm, rows, phase, group_fold, fold_phase_totals,
                             float(best_key), args.block_size, args.n_boot, args.boot_seed)
        checks = {
            "1_oos_mean_delta_positive": bool(primary["mean_delta"] > 0),
            "2_at_least_3_of_4_folds_positive": bool(primary["positive_folds"] >= 3),
            "3_survives_drop_best_fold": bool(primary["mean_delta_drop_best"] > 0),
            "4_relative_gain_at_least_1pct": bool(primary["relative"] >= MIN_RELATIVE_GAIN),
            "5_kappa_plateau_1_2_5_all_positive": plateau_ok,
            "6_bootstrap_ci_lower_bound_positive": bool(boot["p2.5"] > 0),
        }
        results[arm] = {
            "description": ARMS[arm]["label"], "baseline": ARMS[arm]["baseline"],
            "by_kappa": by_kappa, "selected_kappa": best_key,
            "bootstrap": {"block_size": args.block_size, "n_boot": args.n_boot, **boot},
            "checks": checks, "pass": all(checks.values()),
        }

    # ---- 诊断：phase 异质性到底在 A 还是 B；逐折 a_p 的秩稳定性
    A_p = scale_numerator(pooled) / _column(pooled, "yy").sum()
    B_p = scale_denominator(pooled) / _column(pooled, "yy").sum()
    fold_scales = {int(f): fit_scales(fold_phase_totals[int(f)], 0.0).tolist()
                   for f in sorted(fold_phase_totals)}
    fold_keys = sorted(fold_scales)
    rank_stability = [
        {"folds": [a, b], "spearman": spearman(np.array(fold_scales[a]), np.array(fold_scales[b]))}
        for i, a in enumerate(fold_keys) for b in fold_keys[i + 1:]]
    spearman_values = np.array([r["spearman"] for r in rank_stability], dtype=float)

    pooled_scales = fit_scales(pooled, 0.0)
    clip_headroom = float(np.abs(raw).max() * pooled_scales.max())

    payload = {
        "experiment": "v3_phase_scale",
        "question": "最优 scale 分 phase 比全局常数更好吗？（phase = time_id % 10，线上确定已知）",
        "oof": str(args.oof), "rows": int(len(target)), "time_ids": int(len(starts)),
        "preregistered": {
            "shrinkage_grid": [("inf" if not np.isfinite(k) else k) for k in SHRINKAGE],
            "plateau_required_positive": list(PLATEAU),
            "min_relative_gain": MIN_RELATIVE_GAIN,
            "protocol": "扩展窗口：fold k 的系数只用 fold 0..k−1 拟合；"
                        "基线是同一批训练折上解出的全局解（不是 1.16、不是 pooled 最优）",
        },
        "phase_diagnostics": {
            "A_per_phase": A_p.tolist(), "B_per_phase": B_p.tolist(),
            "optimal_scale_per_phase": pooled_scales.tolist(),
            "global_scale": float(global_scale),
            "A_relative_spread": float(A_p.max() / A_p.min() - 1.0),
            "B_relative_spread": float(B_p.max() / B_p.min() - 1.0),
            "scale_relative_spread": float(pooled_scales.max() / pooled_scales.min() - 1.0),
            "fold_optimal_scales": fold_scales,
            "fold_rank_stability_spearman": rank_stability,
            "mean_pairwise_spearman": float(np.nanmean(spearman_values)),
            "variance_components": heterogeneity_test(rows, phase, args.block_size,
                                                      args.n_boot, args.boot_seed),
        },
        "clip_check": {"prediction_clip": args.prediction_clip,
                       "max_abs_emitted_at_largest_phase_scale": clip_headroom,
                       "clip_binds": bool(clip_headroom > args.prediction_clip)},
        "arms": results,
        "replication_semantics": [
            "换一份 OOF cache（不同种子数/轮数）重跑，用的是**同一批行、同一套 fold**。"
            "所以两份 cache 的 a_p 高度一致只能排除「模型专属假象」，"
            "**不能**证明 phase 模式可外推 —— 数据侧的噪声实现在两份 cache 里是同一个。",
            "能回答外推的是两件事：逐折 a_p 的 Spearman 秩相关（跨时间是否稳定），"
            "以及方差分量里的超额方差（离散度是否超出抽样噪声）。判据只看这两个和 OOS。",
        ],
        "deployment_note":
            "即使通过，也**不能**把 OOF 的绝对 a_p 搬上线：OOF 全局最优 scale 与公榜标定的 "
            "1.16 相差很大，属本项目已知的本地/公榜尺子分歧。可部署形式是对已标定全局值做"
            "乘性修正 scale_p = 1.16 × (a_p / a_global)，只搬相对模式。"
            "且 prediction_scale 属模型身份，上线需 meta schema 变更 + promotion 全套门禁 + 用户授权。",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    diag = payload["phase_diagnostics"]
    vc = diag["variance_components"]
    lines = [
        "# ③：最优 scale 该不该分 phase", "",
        f"OOF：`{Path(args.oof).name}`；{len(target):,} 行 / {len(starts):,} 个 time_id。",
        "",
        "## 预注册（先写判据，再看结果）", "",
        f"- 收缩网格 κ = {payload['preregistered']['shrinkage_grid']}",
        f"- 协议：{payload['preregistered']['protocol']}",
        f"- 门槛：折均 >0、≥3/4 折为正、去最好折 >0、相对 ≥{MIN_RELATIVE_GAIN*100:.0f}%、"
        f"κ∈{list(PLATEAU)} 整段为正、bootstrap 95% CI 下界 >0", "",
        "## 分 phase 的 A、B、最优 scale（pooled）", "",
        "| phase | A_p | B_p | a_p = A_p/B_p |", "|---:|---:|---:|---:|",
    ]
    for p in range(PHASE_PERIOD):
        lines.append(f"| {p} | {diag['A_per_phase'][p]:.8f} | {diag['B_per_phase'][p]:.8f} | "
                     f"{diag['optimal_scale_per_phase'][p]:.4f} |")
    lines += [
        f"| **全局** | — | — | **{diag['global_scale']:.4f}** |", "",
        f"离散度：A_p 相对跨度 {diag['A_relative_spread']*100:.1f}%，"
        f"B_p {diag['B_relative_spread']*100:.1f}%，a_p {diag['scale_relative_spread']*100:.1f}%。"
        f"⟹ 异质性主要在 **{'A（信号）' if diag['A_relative_spread'] > diag['B_relative_spread'] else 'B（方差）'}**。",
        "",
        f"逐折 a_p 的两两 Spearman 秩相关均值 = **{diag['mean_pairwise_spearman']:+.3f}**"
        "（接近 0 表示 phase 排序在折间不稳定）。", "",
        "### 方差分量：这些离散度里有多少是真的？", "",
        "| 量 | 值 |", "|---|---:|",
        f"| 观测到的 A_p 跨 phase 方差 | {vc['observed_cross_phase_variance_of_A']:.4e} |",
        f"| bootstrap 测到的 A_p 抽样方差 | {vc['bootstrap_sampling_variance_of_A']:.4e} |",
        f"| 比值（观测 / 抽样） | {vc['observed_over_sampling_ratio']:.2f}× |",
        f"| **超额方差** | **{vc['excess_variance']:+.4e}** "
        f"（95% CI [{vc['excess_variance_ci']['p2.5']:+.2e}, "
        f"{vc['excess_variance_ci']['p97.5']:+.2e}]） |",
        "",
        f"⟹ **{'检出真异质性' if vc['heterogeneity_detected'] else '测不出真异质性'}**："
        f"观测到的 A_p 离散度{'超出' if vc['heterogeneity_detected'] else '没有超出'}"
        "单个 A_p 自身的抽样噪声。"
        + ("" if vc["heterogeneity_detected"] else
           "按 CLAUDE.md 的口径，这里只能写「测不出来」，不能写「phase 之间没有差别」。"), "",
        f"clip 核对：最大 |raw|×最大 a_p = {clip_headroom:.4f}，"
        f"clip = {args.prediction_clip} ⟹ "
        f"{'⚠️ 会触限，二次式不再精确' if payload['clip_check']['clip_binds'] else '不触限，二次式精确成立'}。",
        "", "## 两条臂的样本外结果", "",
    ]
    for arm, result in results.items():
        lines += [f"### {result['description']}", "",
                  f"基线：{result['baseline']}", "",
                  "| κ | OOS 折均 Δ | 相对 | 正折 | 去最好折 | LOFO 折均 Δ | LOFO 正折 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for kappa in SHRINKAGE:
            key = "inf" if not np.isfinite(kappa) else f"{kappa:g}"
            e = result["by_kappa"][key]["expanding_window"]
            l = result["by_kappa"][key]["leave_one_fold_out"]
            mark = " ←选中" if key == result["selected_kappa"] else ""
            lines.append(f"| {key}{mark} | {e['mean_delta']:+.3e} | {e['relative']*100:+.2f}% | "
                         f"{e['positive_folds']}/{e['n_folds']} | "
                         f"{e['mean_delta_drop_best']:+.3e} | {l['mean_delta']:+.3e} | "
                         f"{l['positive_folds']}/{l['n_folds']} |")
        b = result["bootstrap"]
        lines += ["", f"选中 κ={result['selected_kappa']} 的 block bootstrap 95% CI："
                  f"[{b['p2.5']:+.3e}, {b['p97.5']:+.3e}]（中位数 {b['p50']:+.3e}）", "",
                  "| 门槛 | 结果 |", "|---|---|"]
        lines += [f"| {k} | {'✅' if ok else '❌'} |" for k, ok in result["checks"].items()]
        lines += ["", f"**{'✅ PASS' if result['pass'] else '❌ 不通过'}**", ""]
    lines += ["## 限制：哪种复核能证明什么", "",
              *[f"- {item}" for item in payload["replication_semantics"]], "",
              "## 部署说明", "", payload["deployment_note"], ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for arm, result in results.items():
        print(f"{arm}: {'PASS' if result['pass'] else 'FAIL'} "
              f"(κ={result['selected_kappa']}, "
              f"OOS {result['by_kappa'][result['selected_kappa']]['expanding_window']['relative']*100:+.2f}%)")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
