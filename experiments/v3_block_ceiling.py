"""①：market / cross 两个块各自还剩多少 —— 纯 OOF 后处理，不训练、不写模型。

用 oracle 替换回答「精力该往哪块投」：

    把 m̂ 换成真实 m，保留 ê   → market 完美时的 Score
    把 ê 换成真实 e，保留 m̂   → cross  完美时的 Score

⚠️ 这两个数**本身**没有决策价值：它们只是复述了方差拆分（m 占 target 加权能量约 71%，
e 约 29%），所以两个「天花板」必然分别约等于 0.71 和 0.29。真正能指导分配精力的是
项目里已有的同一套框架（`mt_predictability` 的 `Score ≈ 0.72·R²_m + 0.28·R²_e`）：

    Score ≈ w_m·R²_m + w_e·R²_e        w_m = Σw·m²/D，w_e = Σw·e²/D
    R²_m  = peak(m, m̂)（= 加权无中心相关² ），R²_e = peak(e, ê)

`w_m / w_e` 是**同样一点 R² 改进值多少分**的兑换率；`R²_m / R²_e` 是**两块各自收割了多少**。
两者一起才回答「剩下的 IC 藏在哪一块」。

所有统计量都是逐行加权二阶矩之和 ⟹ 先按 time_id 预聚合成 15 个矩，之后逐折与 block
bootstrap 都只是对这些矩求和，代价可以忽略。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_block_ceiling.py
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

DEFAULT_OOF = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"

# 15 个二阶矩的顺序（对称对只存一次）。名字里 h 后缀 = hat（模型预测）。
MOMENTS = ("yy", "ym", "ye", "ymh", "yeh",
           "mm", "me", "mmh", "meh",
           "ee", "emh", "eeh",
           "mhmh", "mheh", "eheh")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(DEFAULT_OOF))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_block_ceiling_3s480")
    p.add_argument("--block-size", type=int, default=500, help="block bootstrap 的块长（time_id 数）")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_starts(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)]).astype(np.int64)
    return starts, counts


def group_moments(y, m, e, mh, eh, w, starts) -> np.ndarray:
    """逐 time_id 的 15 个加权二阶矩，形状 (n_groups, 15)。"""
    columns = {
        "yy": y * y, "ym": y * m, "ye": y * e, "ymh": y * mh, "yeh": y * eh,
        "mm": m * m, "me": m * e, "mmh": m * mh, "meh": m * eh,
        "ee": e * e, "emh": e * mh, "eeh": e * eh,
        "mhmh": mh * mh, "mheh": mh * eh, "eheh": eh * eh,
    }
    return np.column_stack([np.add.reduceat(w * columns[name], starts) for name in MOMENTS])


def as_dict(totals: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(MOMENTS, totals)}


def _peak(a_num: float, b_num: float, denominator: float) -> dict[str, float]:
    """由未归一化的 Σw·y·p 与 Σw·p² 得到 A/B/peak/IC/最优 scale。"""
    A = a_num / denominator
    B = b_num / denominator
    peak = A * A / B if B > 0.0 else 0.0
    return {"A": A, "B": B, "peak": peak,
            "IC": float(np.sqrt(peak)) if peak > 0 else 0.0,
            "optimal_scale": A / B if B > 0.0 else float("nan")}


def statistics(totals: np.ndarray) -> dict[str, Any]:
    """全部结论都只是这 15 个矩的函数 —— 逐折与 bootstrap 复用同一个函数。"""
    M = as_dict(totals)
    D = M["yy"]

    components = {
        # 名字 → (Σw·y·p, Σw·p²)
        "y_perfect": (D, D),
        "m_true": (M["ym"], M["mm"]),
        "e_true": (M["ye"], M["ee"]),
        "m_hat": (M["ymh"], M["mhmh"]),
        "e_hat": (M["yeh"], M["eheh"]),
        "m_hat+e_hat": (M["ymh"] + M["yeh"], M["mhmh"] + 2 * M["mheh"] + M["eheh"]),
        "m_true+e_hat": (M["ym"] + M["yeh"], M["mm"] + 2 * M["meh"] + M["eheh"]),
        "m_hat+e_true": (M["ymh"] + M["ye"], M["mhmh"] + 2 * M["emh"] + M["ee"]),
    }
    table = {name: _peak(a, b, D) for name, (a, b) in components.items()}

    # 能量拆分：w_m + w_e + 交叉项 ≡ 1（无权截面均值口径下交叉项不为 0）
    w_m, w_e = M["mm"] / D, M["ee"] / D
    w_cross = 2.0 * M["me"] / D

    # 各块的收割率：把块自己当 target，peak 就是加权无中心相关²
    r2_m = M["mmh"] ** 2 / (M["mm"] * M["mhmh"]) if M["mm"] > 0 and M["mhmh"] > 0 else 0.0
    r2_e = M["eeh"] ** 2 / (M["ee"] * M["eheh"]) if M["ee"] > 0 and M["eheh"] > 0 else 0.0

    # 两系数最优组合器 f = c_m·m̂ + c_e·ê（闭式，Gram 求逆）
    G = np.array([[M["mhmh"], M["mheh"]], [M["mheh"], M["eheh"]]], dtype=np.float64) / D
    v = np.array([M["ymh"], M["yeh"]], dtype=np.float64) / D
    try:
        c = np.linalg.solve(G, v)
    except np.linalg.LinAlgError:
        c = np.full(2, np.nan)
    peak_two = float(v @ c)
    share_m = float(v[0] * c[0] / peak_two) if peak_two > 0 else float("nan")

    base = table["m_hat+e_hat"]["peak"]
    return {
        "components": table,
        "energy_split": {"w_market": w_m, "w_cross": w_e, "w_cross_term": w_cross,
                         "sum": w_m + w_e + w_cross,
                         "exchange_rate_market_over_cross": w_m / w_e if w_e > 0 else float("nan")},
        "harvest": {
            "r2_market": r2_m, "r2_cross": r2_e,
            "ic_market": float(np.sqrt(r2_m)), "ic_cross": float(np.sqrt(r2_e)),
            "r2_ratio_cross_over_market": r2_e / r2_m if r2_m > 0 else float("nan"),
            "ic_ratio_cross_over_market": float(np.sqrt(r2_e / r2_m)) if r2_m > 0 else float("nan"),
            # 与 mt_predictability 同框架的近似分解（交叉项让它不是恒等式，故一并报告残差）
            "approx_score_market": w_m * r2_m,
            "approx_score_cross": w_e * r2_e,
            "approx_sum": w_m * r2_m + w_e * r2_e,
            "approx_vs_two_coefficient_peak": w_m * r2_m + w_e * r2_e - peak_two,
        },
        "two_coefficient": {"c_market": float(c[0]), "c_cross": float(c[1]),
                            "peak": peak_two, "share_market": share_m,
                            "share_cross": 1.0 - share_m if np.isfinite(share_m) else float("nan"),
                            "gram_offdiagonal_over_D": M["mheh"] / D},
        "oracle_gain": {
            "baseline_peak": base,
            "market_perfect_peak": table["m_true+e_hat"]["peak"],
            "cross_perfect_peak": table["m_hat+e_true"]["peak"],
            "market_perfect_delta": table["m_true+e_hat"]["peak"] - base,
            "cross_perfect_delta": table["m_hat+e_true"]["peak"] - base,
        },
    }


def bootstrap(group_moment_rows: np.ndarray, block_size: int, n_boot: int,
              seed: int) -> dict[str, dict[str, float]]:
    """按 time_id 连续块重采样。所有关心的标量都从同一批 replicate 里取分位数。"""
    n_groups = len(group_moment_rows)
    n_blocks = int(np.ceil(n_groups / block_size))
    # 前缀和 ⟹ 每个块的矩之和 O(1) 取出
    prefix = np.vstack([np.zeros(len(MOMENTS)), np.cumsum(group_moment_rows, axis=0)])
    max_start = max(n_groups - block_size, 0)
    rng = np.random.default_rng(seed)
    keys = ("r2_market", "r2_cross", "r2_ratio_cross_over_market")
    share_key = "share_market"
    samples: dict[str, list[float]] = {k: [] for k in (*keys, share_key, "w_market", "w_cross")}
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        stops = np.minimum(starts + block_size, n_groups)
        totals = (prefix[stops] - prefix[starts]).sum(axis=0)
        stats = statistics(totals)
        for key in keys:
            samples[key].append(stats["harvest"][key])
        samples[share_key].append(stats["two_coefficient"][share_key])
        samples["w_market"].append(stats["energy_split"]["w_market"])
        samples["w_cross"].append(stats["energy_split"]["w_cross"])
    return {name: {"p2.5": float(np.percentile(values, 2.5)),
                   "p50": float(np.percentile(values, 50.0)),
                   "p97.5": float(np.percentile(values, 97.5))}
            for name, values in samples.items()}


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

    starts, counts = group_starts(time_id)
    m_true = np.repeat(np.add.reduceat(target, starts) / counts, counts)  # 无权截面均值 = 生产口径
    e_true = target - m_true

    # ---- 结构断言：报告的全部前提，失败即停，不靠肉眼核对
    max_gap = float(np.abs(raw - (m_hat + e_hat)).max())
    if max_gap > 1e-12:
        raise AssertionError(f"prediction_raw != market + e_lgbm (max {max_gap:.3e})")
    m_hat_spread = float(np.abs(m_hat - np.repeat(np.add.reduceat(m_hat, starts) / counts, counts)).max())
    if m_hat_spread > 1e-12:
        raise AssertionError(f"market is not constant within time_id (max {m_hat_spread:.3e})")
    e_hat_mean = float(np.abs(np.add.reduceat(e_hat, starts) / counts).max())
    if e_hat_mean > 1e-9:
        raise AssertionError(f"e_lgbm is not zero-mean within time_id (max {e_hat_mean:.3e})")

    rows = group_moments(target, m_true, e_true, m_hat, e_hat, weight, starts)
    pooled = statistics(rows.sum(axis=0))

    # ---- 自洽性检查：与唯一指标实现对拍，且能量拆分必须复原 1
    reference = scale_invariant_score(target, raw, weight)
    got = pooled["components"]["m_hat+e_hat"]
    for key in ("A", "B", "peak"):
        if abs(got[key] - float(reference[key])) > 1e-12 * max(1.0, abs(float(reference[key]))):
            raise AssertionError(f"moment-based {key} disagrees with src.metric: "
                                 f"{got[key]!r} vs {reference[key]!r}")
    if abs(pooled["energy_split"]["sum"] - 1.0) > 1e-12:
        raise AssertionError(f"energy split does not sum to 1: {pooled['energy_split']['sum']!r}")
    if abs(pooled["components"]["y_perfect"]["peak"] - 1.0) > 1e-12:
        raise AssertionError("peak(y, y) must be exactly 1")

    group_fold = fold[starts]
    per_fold = {}
    for f in np.unique(group_fold):
        per_fold[int(f)] = statistics(rows[group_fold == f].sum(axis=0))

    boot = bootstrap(rows, args.block_size, args.n_boot, args.boot_seed)

    fold_r2m = np.array([per_fold[f]["harvest"]["r2_market"] for f in sorted(per_fold)])
    fold_r2e = np.array([per_fold[f]["harvest"]["r2_cross"] for f in sorted(per_fold)])
    fold_share = np.array([per_fold[f]["two_coefficient"]["share_market"] for f in sorted(per_fold)])

    payload = {
        "experiment": "v3_block_ceiling",
        "question": "market 块和 cross 块各自还剩多少？剩下的 IC 藏在哪一块？",
        "oof": str(args.oof),
        "rows": int(len(target)), "time_ids": int(len(starts)),
        "definitions": {
            "m_true": "逐 time_id 的**无权**截面均值（= 生产分解与训练目标口径）",
            "e_true": "y − m_true",
            "m_hat": "OOF `market` = (1−λ)·ridge + λ·行级 LGBM 的逐 time_id 无权截面均值",
            "e_hat": "OOF `e_lgbm` = 截面 LGBM 预测再逐 time_id 无权去均值",
            "peak": "A²/B，即最优全局 scale 下的 Score（= 加权无中心相关²）",
            "R2_market": "peak(m_true, m_hat)；R2_cross = peak(e_true, e_hat)",
            "w_market": "Σw·m² / Σw·y²，即同样一点 R² 改进能换来多少 Score",
        },
        "caveats": [
            "两个 oracle Score 分别≈ w_market 与 w_cross，本身只是方差拆分的复述；"
            "可比较的量是 R2 收割率与 w 兑换率。",
            "这是 OOF 尺子。本项目已多次出现 OOF 与公榜量反，占分比不等于公榜上的占分比。",
        ],
        "pooled": pooled,
        "per_fold": per_fold,
        "fold_dispersion": {
            "r2_market": {"mean": float(fold_r2m.mean()), "min": float(fold_r2m.min()),
                          "max": float(fold_r2m.max())},
            "r2_cross": {"mean": float(fold_r2e.mean()), "min": float(fold_r2e.min()),
                         "max": float(fold_r2e.max())},
            "share_market": {"mean": float(fold_share.mean()), "min": float(fold_share.min()),
                             "max": float(fold_share.max())},
            "folds_where_cross_harvest_exceeds_market": int((fold_r2e > fold_r2m).sum()),
            "n_folds": int(len(fold_r2m)),
        },
        "bootstrap": {"block_size": args.block_size, "n_boot": args.n_boot,
                      "seed": args.boot_seed, "intervals": boot},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    h, o, es, tc = (pooled["harvest"], pooled["oracle_gain"],
                    pooled["energy_split"], pooled["two_coefficient"])
    lines = [
        "# ①：market / cross 两个块各自还剩多少", "",
        f"OOF：`{Path(args.oof).name}`；{len(target):,} 行 / {len(starts):,} 个 time_id。", "",
        "## 用户点名的两个 Score（oracle 替换）", "",
        "| 组合 | Score(peak) | IC | 相对基线 |", "|---|---:|---:|---:|",
        f"| `m̂ + ê`（当前） | {o['baseline_peak']:.8f} | "
        f"{pooled['components']['m_hat+e_hat']['IC']:.5f} | — |",
        f"| `m + ê`（market 完美） | {o['market_perfect_peak']:.8f} | "
        f"{pooled['components']['m_true+e_hat']['IC']:.5f} | {o['market_perfect_delta']:+.6f} |",
        f"| `m̂ + e`（cross 完美） | {o['cross_perfect_peak']:.8f} | "
        f"{pooled['components']['m_hat+e_true']['IC']:.5f} | {o['cross_perfect_delta']:+.6f} |",
        "",
        "⚠️ 这两个数分别≈ `w_market` 与 `w_cross`，**只是方差拆分的复述**，不能直接当"
        "「哪块还剩得多」的答案。下面两张表才是。", "",
        "## 能量兑换率：同样一点 R² 改进值多少分", "",
        "| 量 | 值 |", "|---|---:|",
        f"| `w_market = Σw·m²/D` | {es['w_market']:.6f} |",
        f"| `w_cross  = Σw·e²/D` | {es['w_cross']:.6f} |",
        f"| 交叉项 `2Σw·m·e/D` | {es['w_cross_term']:+.6f} |",
        f"| **兑换率 w_market / w_cross** | **{es['exchange_rate_market_over_cross']:.3f}×** |",
        "",
        "## 收割率：每块自己已经拿到多少", "",
        "| 块 | R²（= 相关²） | IC | bootstrap 95% CI (R²) |", "|---|---:|---:|---|",
        f"| market | {h['r2_market']:.8f} | {h['ic_market']:.5f} | "
        f"[{boot['r2_market']['p2.5']:.2e}, {boot['r2_market']['p97.5']:.2e}] |",
        f"| cross | {h['r2_cross']:.8f} | {h['ic_cross']:.5f} | "
        f"[{boot['r2_cross']['p2.5']:.2e}, {boot['r2_cross']['p97.5']:.2e}] |",
        "",
        f"cross / market 的 R² 收割率之比 = **{h['r2_ratio_cross_over_market']:.2f}×**"
        f"（IC 空间 {h['ic_ratio_cross_over_market']:.2f}×），"
        f"bootstrap 95% CI ["
        f"{boot['r2_ratio_cross_over_market']['p2.5']:.2f}, "
        f"{boot['r2_ratio_cross_over_market']['p97.5']:.2f}]；"
        f"逐折 {payload['fold_dispersion']['folds_where_cross_harvest_exceeds_market']}"
        f"/{payload['fold_dispersion']['n_folds']} 折 cross 收割率更高。", "",
        "## 当前占分拆解（两系数最优组合器，Gram 闭式解）", "",
        f"`f = {tc['c_market']:.4f}·m̂ + {tc['c_cross']:.4f}·ê`，peak = {tc['peak']:.8f}", "",
        "| 块 | 占分 | bootstrap 95% CI |", "|---|---:|---|",
        f"| market | {tc['share_market']*100:.1f}% | "
        f"[{boot['share_market']['p2.5']*100:.1f}%, {boot['share_market']['p97.5']*100:.1f}%] |",
        f"| cross | {tc['share_cross']*100:.1f}% | "
        f"[{(1-boot['share_market']['p97.5'])*100:.1f}%, "
        f"{(1-boot['share_market']['p2.5'])*100:.1f}%] |",
        "",
        f"实测 `⟨m̂,ê⟩_w/D = {tc['gram_offdiagonal_over_D']:.3e}`（**不为 0**，"
        "旧笔记的 `⟨m̂,ê⟩ ≡ 0` 只是近似）。", "",
        "## 逐折", "",
        "| 折 | R²_market | R²_cross | market 占分 |", "|---:|---:|---:|---:|",
    ]
    for f in sorted(per_fold):
        r = per_fold[f]
        lines.append(f"| {f} | {r['harvest']['r2_market']:.8f} | {r['harvest']['r2_cross']:.8f} | "
                     f"{r['two_coefficient']['share_market']*100:.1f}% |")
    lines += ["", "## 限制", ""]
    lines += [f"- {c}" for c in payload["caveats"]]
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
