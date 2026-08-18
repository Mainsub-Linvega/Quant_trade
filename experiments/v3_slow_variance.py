"""P1：预测里的「死方差」—— 慢分量该不该降权。纯 OOF 后处理，不训练、不写模型。

## 机制

`v3_temporal_smoothing` 测到：真实信号 `m_t` 的自相关在 lag 5 就归零
（0.837 → 0.612 → 0.397 → 0.183 → 0.000，≈MA(6) 重叠窗），而预测 `m̂` 在 lag 50 仍有 0.324。
target 在 5 步以后没有记忆 ⟹ 预测里任何**持续远超 6 步**的成分几乎不可能与 target 协变，
只能抬高 `B`。而 `peak = A²/B`，把它降权是纯增益。

## 做法

逐 asset 在折内做**因果** trailing mean（只用当期之前的 K 个观测），把每一块拆成 slow/fast：

    m̂ = m̂_slow + m̂_fast      ê = ê_slow + ê_fast

四个分量张成一个线性空间，于是下面四个模型全是同一个 Gram 的**投影**，闭式可解且严格嵌套：

| 模型 | 系数 | 回答什么 |
|---|---|---|
| `M0` | 1 | 全局单 scale = **基线** |
| `M1` | 2 | 两块各自 scale（复核 08-10「分量配比」在当前架构下还值不值） |
| `M2` | 2 | slow/fast（死方差假设本身） |
| `M3` | 4 | 块 × slow/fast（死方差在哪一块） |

嵌套 ⟹ 增量可归因，不会把 M1 的收益记到 M2 头上（`conditional_blend.py` 的教训）。

## 判据先行（CLAUDE.md §5.1）

基线一律是**同一批训练折上解出的 M0**，不是 1.16、不是 pooled 最优。
评估用扩展窗口：fold k 的系数只用 fold 0..k−1 拟合。6 条门槛见 `CHECK_NAMES`。

⚠️ 诚实声明：K ∈ {10,25,50,100,200} 我在写这个脚本**之前**扫过一遍并看到 K 越大越好，
所以这里的 K 阶梯不是干净的预注册。抗选择的保护有三层：K 连续段门槛、
第二份 cache 复现、以及最后的全分辨率口径核对（那个才是真正独立的检验）。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_slow_variance.py
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

from src.metric import scale_invariant_score  # noqa: E402  唯一指标实现
from market_model import sign_test_p  # noqa: E402
from v3_residual_adapters import apply_asset_slopes, fit_asset_slopes  # noqa: E402

DEFAULT_OOF = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"

COMPONENTS = ("m_slow", "m_fast", "e_slow", "e_fast")
MODELS: dict[str, np.ndarray] = {
    "M0_global_scale": np.array([[1.0, 1.0, 1.0, 1.0]]),
    "M1_block_scales": np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
    "M2_slow_fast": np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
    "M3_block_x_slow_fast": np.eye(4),
}
BASELINE_MODEL = "M0_global_scale"
K_LADDER: list[float] = [25, 50, 100, 200, 400, 800, 1600, float("inf")]
PLATEAU_K = (100, 200, 400)      # 门槛 5：这一段必须整段为正
MIN_RELATIVE_GAIN = 0.01         # 门槛 4
ASSET_SHRINK_PER_FOLD = 100.0    # 候选用 5 折 shrink=500，即每折 100
CHECK_NAMES = ("1_oos_mean_delta_positive", "2_at_least_3_of_4_folds_positive",
               "3_survives_drop_best_fold", "4_relative_gain_at_least_1pct",
               "5_K_plateau_all_positive", "6_bootstrap_ci_lower_bound_positive")

# 15 个矩：D、v(4)、G 上三角(10)
_GRAM_PAIRS = [(i, j) for i in range(4) for j in range(i, 4)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(DEFAULT_OOF))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_slow_variance_3s480")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------ 慢分量

def causal_trailing_mean(values: np.ndarray, segment_starts: np.ndarray,
                         segment_ends: np.ndarray, window: float) -> np.ndarray:
    """每个段内、**只用当期之前**的观测求均值。

    `window=inf` 退化为 expanding mean（= 因果版的「静态每资产偏置」）。
    段首没有历史 ⟹ 取自身，于是 fast=0，不制造假信号。
    """
    out = np.empty_like(values)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    for start, end in zip(segment_starts, segment_ends):
        index = np.arange(start, end)
        low = np.full(len(index), start) if np.isinf(window) \
            else np.maximum(start, index - int(window))
        count = np.maximum(index - low, 1)
        out[index] = (cumulative[index] - cumulative[low]) / count
        out[start] = values[start]
    return out


def group_moments(components: np.ndarray, target: np.ndarray, weight: np.ndarray,
                  group_index: np.ndarray, n_groups: int) -> np.ndarray:
    """逐 time_id 的 15 个加权矩。之后逐折与 bootstrap 都只是对它求和。"""
    columns = [np.bincount(group_index, weights=weight * target * target, minlength=n_groups)]
    for i in range(4):
        columns.append(np.bincount(group_index, weights=weight * target * components[:, i],
                                   minlength=n_groups))
    for i, j in _GRAM_PAIRS:
        columns.append(np.bincount(group_index,
                                   weights=weight * components[:, i] * components[:, j],
                                   minlength=n_groups))
    return np.column_stack(columns)


def unpack(totals: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    denominator = float(totals[0])
    v = totals[1:5].astype(np.float64)
    G = np.empty((4, 4), dtype=np.float64)
    for slot, (i, j) in enumerate(_GRAM_PAIRS):
        G[i, j] = G[j, i] = totals[5 + slot]
    return denominator, v, G


def fit(totals: np.ndarray, model: str) -> np.ndarray:
    _, v, G = unpack(totals)
    T = MODELS[model]
    return np.linalg.solve(T @ G @ T.T, T @ v)


def score(totals: np.ndarray, model: str, coefficients: np.ndarray) -> float:
    denominator, v, G = unpack(totals)
    T = MODELS[model]
    Gm, vm = T @ G @ T.T, T @ v
    return float((2.0 * coefficients @ vm - coefficients @ Gm @ coefficients) / denominator)


def ab_of(totals: np.ndarray, model: str, coefficients: np.ndarray) -> tuple[float, float]:
    """把多系数解等价成一个预测 f，报告它的 A 与 B —— 机制核对要看是减 B 还是加 A。"""
    denominator, v, G = unpack(totals)
    T = MODELS[model]
    u = T.T @ coefficients                      # f = Σ u_i · component_i
    return float(u @ v / denominator), float(u @ G @ u / denominator)


# --------------------------------------------------------------- 评估协议

def expanding_window(fold_totals: dict[int, np.ndarray], model: str) -> dict[str, Any]:
    folds = sorted(fold_totals)
    rows = []
    for position, f in enumerate(folds):
        if position == 0:
            continue
        trained = np.sum([fold_totals[g] for g in folds[:position]], axis=0)
        evaluated = fold_totals[f]
        c_model, c_base = fit(trained, model), fit(trained, BASELINE_MODEL)
        candidate = score(evaluated, model, c_model)
        baseline = score(evaluated, BASELINE_MODEL, c_base)
        A_c, B_c = ab_of(evaluated, model, c_model)
        A_b, B_b = ab_of(evaluated, BASELINE_MODEL, c_base)
        rows.append({"fold": int(f), "baseline": baseline, "candidate": candidate,
                     "delta": candidate - baseline,
                     "coefficients": [float(x) for x in c_model],
                     "baseline_scale": float(c_base[0]),
                     "A_baseline": A_b, "B_baseline": B_b, "A_candidate": A_c, "B_candidate": B_c})
    deltas = np.array([r["delta"] for r in rows], dtype=float)
    baselines = np.array([r["baseline"] for r in rows], dtype=float)
    reference = float(np.abs(baselines).mean())
    drop_best = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
    positive = int((deltas > 0).sum())
    dA = float(np.mean([r["A_candidate"] - r["A_baseline"] for r in rows]))
    dB = float(np.mean([r["B_candidate"] - r["B_baseline"] for r in rows]))
    meanA = float(np.mean([r["A_baseline"] for r in rows]))
    meanB = float(np.mean([r["B_baseline"] for r in rows]))
    return {"folds": rows, "mean_delta": float(deltas.mean()),
            "mean_delta_drop_best": float(drop_best.mean()),
            "relative": float(deltas.mean() / reference) if reference > 0 else float("nan"),
            "relative_drop_best": float(drop_best.mean() / reference)
            if reference > 0 else float("nan"),
            "positive_folds": positive, "n_folds": int(len(deltas)),
            "sign_test_p": sign_test_p(positive, len(deltas)),
            "mechanism": {"relative_delta_A": dA / meanA if meanA else float("nan"),
                          "relative_delta_B": dB / meanB if meanB else float("nan")}}


def bootstrap(rows: np.ndarray, group_fold: np.ndarray, fold_totals: dict[int, np.ndarray],
              model: str, block_size: int, n_boot: int, seed: int) -> dict[str, float]:
    """系数按扩展窗口固定不动，只对**评估折的 time_id** 做连续块重采样。"""
    folds = sorted(fold_totals)
    frozen = {}
    for position, f in enumerate(folds):
        if position == 0:
            continue
        trained = np.sum([fold_totals[g] for g in folds[:position]], axis=0)
        frozen[f] = (fit(trained, model), fit(trained, BASELINE_MODEL))
    prefix = np.vstack([np.zeros(rows.shape[1]), np.cumsum(rows, axis=0)])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        total_delta, total_energy = 0.0, 0.0
        for f, (candidate_coefficients, baseline_coefficients) in frozen.items():
            selected = np.flatnonzero(group_fold == f)
            low, high = int(selected[0]), int(selected[-1]) + 1
            n_groups = high - low
            n_blocks = int(np.ceil(n_groups / block_size))
            starts = rng.integers(low, max(high - block_size, low) + 1, size=n_blocks)
            stops = np.minimum(starts + block_size, high)
            totals = (prefix[stops] - prefix[starts]).sum(axis=0)
            energy = float(totals[0])
            total_delta += (score(totals, model, candidate_coefficients)
                            - score(totals, BASELINE_MODEL, baseline_coefficients)) * energy
            total_energy += energy
        samples.append(total_delta / total_energy if total_energy > 0 else np.nan)
    array = np.asarray(samples, dtype=float)
    return {"p2.5": float(np.nanpercentile(array, 2.5)),
            "p50": float(np.nanpercentile(array, 50.0)),
            "p97.5": float(np.nanpercentile(array, 97.5))}


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
        m_hat = d["market"].astype(np.float64)[keep]
        e_hat = d["e_lgbm"].astype(np.float64)[keep]
        raw = d["prediction_raw"].astype(np.float64)[keep]

    # 逐 asset 的时间序列必须在**折内**连续：按 (fold, asset, time) 排一次序
    order = np.lexsort((time_id, asset_id, fold))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    y_s, w_s, t_s, a_s, f_s = (target[order], weight[order], time_id[order],
                               asset_id[order], fold[order])
    m_s, e_s = m_hat[order], e_hat[order]
    segment = f_s.astype(np.int64) * 1000 + a_s
    segment_starts = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1]
    segment_ends = np.r_[segment_starts[1:], len(segment)]

    # 分组索引按**原始时间序**（time_id 连续块），bootstrap 才能取连续时间块
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(np.int64)
    n_groups = len(starts)
    group_index_original = np.repeat(np.arange(n_groups), counts)
    group_index = group_index_original[order]
    group_fold = fold[starts]

    if abs(float(np.abs(raw - (m_hat + e_hat)).max())) > 1e-12:
        raise AssertionError("prediction_raw != market + e_lgbm")

    # ---- 两条臂：生产 ê，以及**因果重拟**的每资产 scale（候选 v3_asset_cross_*）之上
    arms = {"production": e_s}
    folds_sorted = sorted(np.unique(group_fold))
    # ⚠️ 限制：fold 0 没有可用的过去，无法因果地施加 adapter，只能保持原样。
    # 它不参与评估，但会作为**训练折**参与系数拟合 ⟹ 该臂的拟合集是「未适配 fold 0 +
    # 已适配后续折」的混合。线上也是同一个处境（最初一段没有历史可标定），
    # 所以这是诚实的因果设置，但它让该臂的系数比 production 臂多一层噪声 —— 读结论时要记着。
    adapted = e_s.copy()
    for position, f in enumerate(folds_sorted):
        if position == 0:
            continue
        mask = f_s == f
        prior = np.isin(f_s, folds_sorted[:position])
        slopes = fit_asset_slopes(t_s[prior], y_s[prior], w_s[prior], a_s[prior], e_s[prior],
                                  ASSET_SHRINK_PER_FOLD * position)
        adapted[mask] = apply_asset_slopes(t_s[mask], a_s[mask], e_s[mask], slopes)
    arms["asset_adapter"] = adapted

    results: dict[str, Any] = {}
    for arm_name, e_arm in arms.items():
        by_k: dict[str, Any] = {}
        for K in K_LADDER:
            key = "inf" if np.isinf(K) else f"{int(K)}"
            m_slow = causal_trailing_mean(m_s, segment_starts, segment_ends, K)
            e_slow = causal_trailing_mean(e_arm, segment_starts, segment_ends, K)
            components = np.column_stack([m_slow, m_s - m_slow, e_slow, e_arm - e_slow])
            if float(np.abs(components[:, 0] + components[:, 1] - m_s).max()) > 1e-12:
                raise AssertionError("slow + fast does not reconstruct the market block")
            if float(np.abs(components[:, 2] + components[:, 3] - e_arm).max()) > 1e-12:
                raise AssertionError("slow + fast does not reconstruct the cross block")
            rows = group_moments(components, y_s, w_s, group_index, n_groups)
            fold_totals = {int(f): rows[group_fold == f].sum(axis=0) for f in folds_sorted}
            pooled = rows.sum(axis=0)

            if arm_name == "production" and key == "200":
                reference = scale_invariant_score(target, raw, weight)
                fitted = fit(pooled, BASELINE_MODEL)
                if abs(float(fitted[0]) - float(reference["optimal_scale"])) > 1e-12:
                    raise AssertionError("M0 does not reproduce src.metric optimal_scale")
                if abs(score(pooled, BASELINE_MODEL, fitted) - float(reference["peak"])) > 1e-14:
                    raise AssertionError("M0 does not reproduce src.metric peak")
            in_sample = {name: score(pooled, name, fit(pooled, name)) for name in MODELS}
            for wider, narrower in (("M1_block_scales", "M0_global_scale"),
                                    ("M2_slow_fast", "M0_global_scale"),
                                    ("M3_block_x_slow_fast", "M1_block_scales"),
                                    ("M3_block_x_slow_fast", "M2_slow_fast")):
                if in_sample[wider] < in_sample[narrower] - 1e-12:
                    raise AssertionError(f"nesting violated: {wider} < {narrower}")

            by_k[key] = {"in_sample_peak": in_sample,
                         "pooled_coefficients": {name: [float(x) for x in fit(pooled, name)]
                                                 for name in MODELS},
                         "out_of_sample": {name: expanding_window(fold_totals, name)
                                           for name in MODELS if name != BASELINE_MODEL},
                         "_rows": rows, "_fold_totals": fold_totals}

        arm_result: dict[str, Any] = {"by_K": {}}
        for model in MODELS:
            if model == BASELINE_MODEL:
                continue
            finite = [k for k in K_LADDER if not np.isinf(k)]
            best_key = max((f"{int(k)}" for k in finite),
                           key=lambda k: by_k[k]["out_of_sample"][model]["mean_delta"])
            primary = by_k[best_key]["out_of_sample"][model]
            boot = bootstrap(by_k[best_key]["_rows"], group_fold, by_k[best_key]["_fold_totals"],
                             model, args.block_size, args.n_boot, args.boot_seed)
            plateau = all(by_k[f"{k}"]["out_of_sample"][model]["mean_delta"] > 0 for k in PLATEAU_K)
            checks = dict(zip(CHECK_NAMES, (
                primary["mean_delta"] > 0,
                primary["positive_folds"] >= 3,
                primary["mean_delta_drop_best"] > 0,
                primary["relative"] >= MIN_RELATIVE_GAIN,
                plateau,
                boot["p2.5"] > 0)))
            checks = {k: bool(v) for k, v in checks.items()}
            arm_result[model] = {"selected_K": best_key, "bootstrap": boot,
                                 "checks": checks, "pass": all(checks.values())}
        for key, entry in by_k.items():
            arm_result["by_K"][key] = {"in_sample_peak": entry["in_sample_peak"],
                                       "pooled_coefficients": entry["pooled_coefficients"],
                                       "out_of_sample": entry["out_of_sample"]}
        results[arm_name] = arm_result

    payload = {
        "experiment": "v3_slow_variance",
        "question": "预测里持续远超 target 记忆长度（~6 步）的慢分量，是不是只抬高 B 的死方差？",
        "oof": str(args.oof), "rows": int(len(target)), "time_ids": int(n_groups),
        "mechanism": "信号 m_t 的自相关在真实 lag 5 归零，而预测 m̂ 在 lag 50 仍有 0.324"
                     "（证据 outputs/experiments/v3_temporal_smoothing_3s480.md）",
        "preregistered": {
            "K_ladder_sampled_steps": ["inf" if np.isinf(k) else int(k) for k in K_LADDER],
            "plateau_required_positive": list(PLATEAU_K),
            "min_relative_gain": MIN_RELATIVE_GAIN,
            "baseline": "同一批训练折上解出的 M0 全局单 scale（不是 1.16、不是 pooled 最优）",
            "protocol": "扩展窗口：fold k 的系数只用 fold 0..k−1 拟合",
            "honesty_note": "K ∈ {10,25,50,100,200} 在写本脚本前已被探索过并看到 K 越大越好，"
                            "所以 K 阶梯不是干净的预注册。抗选择保护 = K 连续段门槛 + "
                            "第二份 cache 复现 + 全分辨率口径核对。",
        },
        "asset_adapter_arm_limitation":
            "fold 0 无法因果适配（没有可用的过去），它不参与评估但参与系数拟合 ⟹ "
            "该臂拟合集是「未适配 fold 0 + 已适配后续折」的混合。线上同样如此，"
            "所以是诚实的因果设置，但该臂系数比 production 臂多一层噪声。",
        "caliber_warning":
            "本脚本的 trailing mean 走**采样格**（每 ~5 个真实 time_id 一个点），"
            "生产走全分辨率 ⟹ K=200 采样步 ≈ 1000 真实步。两者估计同一个慢分量，"
            "全分辨率点更多、估计更准（方向有利），但必须由全分辨率核对确认。",
        "arms": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# P1：预测里的「死方差」—— 慢分量该不该降权", "",
             f"OOF：`{Path(args.oof).name}`；{len(target):,} 行 / {n_groups:,} 个 time_id。", "",
             f"**机制**：{payload['mechanism']}", "",
             f"⚠️ **口径**：{payload['caliber_warning']}", "",
             f"⚠️ **诚实声明**：{payload['preregistered']['honesty_note']}", "",
             "## 模型阶梯（同一个 Gram 的投影，严格嵌套）", "",
             "| 模型 | 系数 | 回答什么 |", "|---|---:|---|",
             "| `M0_global_scale` | 1 | 全局单 scale = **基线** |",
             "| `M1_block_scales` | 2 | 两块各自 scale |",
             "| `M2_slow_fast` | 2 | 死方差假设本身 |",
             "| `M3_block_x_slow_fast` | 4 | 死方差在哪一块 |", ""]
    for arm_name, arm_result in results.items():
        lines += [f"## 臂：`{arm_name}`", ""]
        if arm_name == "asset_adapter":
            lines += [f"> ⚠️ {payload['asset_adapter_arm_limitation']}", ""]
        lines += [
                  "| 模型 | K | OOS 折均 Δ | 相对 | 正折 | 去最好折 | ΔA | ΔB |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            if model == BASELINE_MODEL:
                continue
            for K in K_LADDER:
                key = "inf" if np.isinf(K) else f"{int(K)}"
                o = arm_result["by_K"][key]["out_of_sample"][model]
                mark = " ←选中" if key == arm_result[model]["selected_K"] else ""
                lines.append(
                    f"| `{model}`{mark} | {key} | {o['mean_delta']:+.3e} | "
                    f"{o['relative']*100:+.2f}% | {o['positive_folds']}/{o['n_folds']} | "
                    f"{o['relative_drop_best']*100:+.2f}% | "
                    f"{o['mechanism']['relative_delta_A']*100:+.2f}% | "
                    f"{o['mechanism']['relative_delta_B']*100:+.2f}% |")
        lines.append("")
        for model in MODELS:
            if model == BASELINE_MODEL:
                continue
            r = arm_result[model]
            b = r["bootstrap"]
            entry = arm_result["by_K"][r["selected_K"]]
            names = {"M1_block_scales": ("m̂", "ê"),
                     "M2_slow_fast": ("slow", "fast"),
                     "M3_block_x_slow_fast": COMPONENTS}[model]
            coefficients = entry["pooled_coefficients"][model]
            base_scale = entry["pooled_coefficients"][BASELINE_MODEL][0]
            lines += [f"### `{model}`（选中 K={r['selected_K']}）", "",
                      "pooled 系数（对照基线单 scale "
                      f"{base_scale:.4f}）：" + "、".join(
                          f"`{n}` = {c:.4f}" for n, c in zip(names, coefficients)), "",
                      f"block bootstrap 95% CI：[{b['p2.5']:+.3e}, {b['p97.5']:+.3e}]"
                      f"（中位数 {b['p50']:+.3e}）", "",
                      "| 门槛 | 结果 |", "|---|---|"]
            lines += [f"| {k} | {'✅' if ok else '❌'} |" for k, ok in r["checks"].items()]
            lines += ["", f"**{'✅ PASS' if r['pass'] else '❌ 不通过'}**", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for arm_name, arm_result in results.items():
        for model in MODELS:
            if model == BASELINE_MODEL:
                continue
            r = arm_result[model]
            o = arm_result["by_K"][r["selected_K"]]["out_of_sample"][model]
            print(f"{arm_name:14} {model:22} {'PASS' if r['pass'] else 'FAIL'} "
                  f"(K={r['selected_K']}, OOS {o['relative']*100:+.2f}%, "
                  f"{o['positive_folds']}/{o['n_folds']})")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
