"""三分量 `f = c_m·m̂ + c_r·ê_ridge + c_l·ê_lgbm` 的最优配比 —— 闭式解。

## 为什么这是免费的

`Score(c) = 2·aᵀc − cᵀMc`，其中 `a = (A_m, A_er, A_el)ᵀ`、`M` 是三个分量的二阶矩阵。
这是个凹二次型，最优解与最优值都有闭式：

    c* = M⁻¹a          peak = aᵀM⁻¹a

`M` **已经全部已知**（`m̂` 从 replace 的 CSV 直接算，两个 `ê` 从
`public_csv_fingerprints.json` 的 Gram 矩阵反解，且 `⟨m̂,ê⟩ ≡ 0`）。
`a` 里只缺 `A_m` —— 交一次「只有 m̂」的 CSV 就有了（见 `market_submission.py`）。

所以：**一次公榜额度 → 整个最优配比。** 配比本身是纯后处理（参数三分类的第①类），
不花额度。

## 现在的模型不在最优点上

生产是 `scale·(m̂ + (1−w)·ê_r + w·ê_l)` —— 2 个自由度，隐含约束 `c_r + c_l = c_m`。
放开这个约束就是本脚本要算的东西。而且 `replace`（w=1）把 `ê_ridge` 整个丢了，
但 `corr(ê_r, ê_l) = 0.618` —— **丢掉的那部分里有独立信息**。

## 没拿到 A_m 之前也能用

不带 `--market-score` 时，本脚本把 `A_m/A(1)` 在一个区间上扫一遍，
给出各种情形下的最优配比与峰值 —— 用来判断「这一次额度值不值得花」。

用法：
    .venv/bin/python experiments/component_optimum.py                    # 扫描
    .venv/bin/python experiments/component_optimum.py \\
        --market-score 0.0021 --market-scale 1.3                        # 精确解
输出：outputs/experiments/component_optimum.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = _REPO_ROOT / "outputs" / "experiments"

# 公榜两点法/一点法解出的精确值（与 ab_decomposition.PUBLIC_MODELS 同源）
A_RIDGE, B_RIDGE = 0.00164960, 0.00145335        # strict_ridge = m̂ + ê_ridge
A_HYBRID = 0.00188239                             # v3_hybrid, blend_weight 0.5
A_REPLACE = A_RIDGE + 2 * (A_HYBRID - A_RIDGE)    # blend_weight 1.0 → m̂ + ê_lgbm（精确）
REPLACE_POINT = (1.16, 0.0024872338)              # ledger 2026-08-09
B_REPLACE = (2 * REPLACE_POINT[0] * A_REPLACE - REPLACE_POINT[1]) / REPLACE_POINT[0] ** 2

COMPONENTS = ("market", "ridge_dev", "lgbm_dev")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-form optimal component weights.")
    parser.add_argument("--market-score", type=float, default=None,
                        help="「只有 m̂」那份 CSV 的公榜分；给了就精确解，不给就扫描")
    parser.add_argument("--market-scale", type=float, default=None)
    parser.add_argument("--source", default=str(_REPO_ROOT / "outputs" / "submission_replace_s116.csv"))
    parser.add_argument("--source-scale", type=float, default=1.16)
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    return parser.parse_args()


def second_moments(args: argparse.Namespace) -> dict[str, Any]:
    """三个分量的二阶矩阵 M（B 的口径）。

    CSV 还在就现算并缓存；删了就读上一次缓存 —— 那 1.1 GB 的 CSV 迟早要清掉。
    """
    cache = EXPERIMENTS / "component_optimum.json"
    source = Path(args.source)
    if not source.exists():
        if not cache.exists():
            raise SystemExit(f"{source.name} 不在，也没有缓存 —— 先在 CSV 还在时跑一次")
        print(f"{source.name} 已删，改用 {cache.name} 里缓存的二阶矩", flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))["moments"]

    frame = pd.read_csv(source)
    parts = [pd.read_parquet(path, columns=["row_id", "time_id"])
             for path in sorted((Path(args.data_root) / "test").glob("*.parquet"))]
    merged = frame.merge(pd.concat(parts, ignore_index=True), on="row_id", how="left")
    raw = merged["target"].to_numpy(dtype=np.float64) / args.source_scale
    time_ids = merged["time_id"].to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    market = np.repeat(np.add.reduceat(raw, starts) / counts, counts)

    # k 把「无权二阶矩」标定成 B 的口径。用 replace 的精确 B 标定 → 那一条误差为 0。
    scaling = B_REPLACE / (raw ** 2).mean()
    market_moment = float((market ** 2).mean())

    payload = json.loads((EXPERIMENTS / "public_csv_fingerprints.json").read_text(encoding="utf-8"))
    gram = np.array(payload["gram"])
    names = payload["gram_names"]
    i = names.index("submission_strict_scale113.csv")      # f0 = m̂ + ê_ridge
    j = names.index("submission_hybrid_base0856.csv")      # fh = m̂ + ½ê_ridge + ½ê_lgbm
    # ê_r = f0 − m̂，ê_l = (2fh − f0) − m̂，且 ⟨ê, m̂⟩ = 0
    ridge_ridge = gram[i, i] - market_moment
    lgbm_lgbm = (4 * gram[j, j] - 4 * gram[i, j] + gram[i, i]) - market_moment
    ridge_lgbm = (2 * gram[i, j] - gram[i, i]) - market_moment

    matrix = scaling * np.array([[market_moment, 0.0, 0.0],
                                 [0.0, ridge_ridge, ridge_lgbm],
                                 [0.0, ridge_lgbm, lgbm_lgbm]])
    return {
        "M": matrix.tolist(), "components": list(COMPONENTS),
        "B_market": matrix[0, 0], "B_ridge_dev": matrix[1, 1], "B_lgbm_dev": matrix[2, 2],
        "corr_ridge_lgbm": float(matrix[1, 2] / np.sqrt(matrix[1, 1] * matrix[2, 2])),
        "cross_market_dev": float(np.abs((market * (raw - market)).mean())),
        "reconciliation_B_ridge_model": float(matrix[0, 0] + matrix[1, 1]),
        "reconciliation_target": B_RIDGE,
    }


def solve(matrix: np.ndarray, alignment: np.ndarray) -> dict[str, Any]:
    """c* = M⁻¹a，peak = aᵀM⁻¹a。"""
    weights = np.linalg.solve(matrix, alignment)
    peak = float(alignment @ weights)
    return {"weights": weights.tolist(), "peak": peak, "ic": peak ** 0.5,
            # 换算成生产参数化：scale·(m̂ + w_r·ê_r + w_l·ê_l)
            "as_production": {"prediction_scale": float(weights[0]),
                              "ridge_dev_weight": float(weights[1] / weights[0]),
                              "lgbm_dev_weight": float(weights[2] / weights[0])}}


def main() -> None:
    args = parse_args()
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    moments = second_moments(args)
    matrix = np.array(moments["M"])
    current_peak = A_REPLACE ** 2 / B_REPLACE

    print(f"二阶矩阵 M（B 口径）：")
    for name, row in zip(COMPONENTS, matrix):
        print("  " + name.ljust(10) + "  ".join(f"{v: .8f}" for v in row))
    print(f"  corr(ê_r, ê_l) = {moments['corr_ridge_lgbm']:.4f}")
    print(f"  对账 B_m + B_er = {moments['reconciliation_B_ridge_model']:.8f} vs "
          f"公榜真值 {moments['reconciliation_target']:.8f} "
          f"（差 {moments['reconciliation_B_ridge_model']/moments['reconciliation_target']-1:+.2%}）")
    print(f"\n现在的模型（replace）：峰值 {current_peak:.8f}，IC {current_peak**0.5:.5f}")

    results: list[dict[str, Any]] = []
    if args.market_score is not None:
        if args.market_scale is None:
            raise SystemExit("给了 --market-score 就必须给 --market-scale")
        a_market = (args.market_score + args.market_scale ** 2 * matrix[0, 0]) / (
            2 * args.market_scale)
        fractions = [("实测", a_market)]
    else:
        print("\n没给 --market-score → 扫描 A_m 的可能取值（交一次 m̂ 就能定死）")
        # 本地先验：mt_predictability 量过纯岭回归的分数拆解是 择时 0.001138 / 截面 0.000200。
        # 在最优配比下 peak = A_m²/B_m + A_er²/B_er，于是
        #   A_m/A_er = √(ratio · B_m/B_er)，再配上 A_m + A_er = A(0) 就能反解出 A_m。
        ratio = 0.001138 / 0.000200
        share = np.sqrt(ratio * matrix[0, 0] / matrix[1, 1])
        prior = A_RIDGE * share / (share + 1.0)
        fractions = [(f"A_m = {f:.0%}·A(1)", f * A_REPLACE) for f in (0.6, 0.7, 0.8, 0.9)]
        fractions.append((f"本地先验 ({prior/A_REPLACE:.0%})", prior))
        fractions.sort(key=lambda item: item[1])
        print(f"  （本地先验：mt_predictability 的 85/15 拆解 → A_m ≈ {prior:.8f} "
              f"= {prior/A_REPLACE:.1%}·A(1)）")

    print(f"\n{'情形':<18}{'A_m':>12}{'A_er':>12}{'A_el':>12}{'最优峰值':>13}{'vs replace':>12}")
    for label, a_market in fractions:
        alignment = np.array([a_market, A_RIDGE - a_market, A_REPLACE - a_market])
        answer = solve(matrix, alignment)
        answer.update({"label": label, "A": alignment.tolist(),
                       "vs_replace": answer["peak"] / current_peak - 1.0,
                       "market_only_peak": a_market ** 2 / matrix[0, 0]})
        results.append(answer)
        print(f"{label:<18}{alignment[0]:>12.8f}{alignment[1]:>12.8f}{alignment[2]:>12.8f}"
              f"{answer['peak']:>13.8f}{answer['vs_replace']:>+11.2%}")

    print(f"\n最优配比（换算成 scale·(m̂ + w_r·ê_r + w_l·ê_l)）：")
    for answer in results:
        production = answer["as_production"]
        print(f"  {answer['label']:<18} scale {production['prediction_scale']:.4f}  "
              f"w_ridge {production['ridge_dev_weight']:+.4f}  "
              f"w_lgbm {production['lgbm_dev_weight']:+.4f}   "
              f"（市场块单独的峰值 {answer['market_only_peak']:.8f}）")

    payload = {
        "identity": "Score(c) = 2aᵀc − cᵀMc；c* = M⁻¹a，peak = aᵀM⁻¹a",
        "why_free": "配比是纯后处理（参数三分类第①类）；M 全已知，a 只缺 A_m",
        "exact_inputs": {"A_ridge": A_RIDGE, "B_ridge": B_RIDGE, "A_hybrid": A_HYBRID,
                         "A_replace": A_REPLACE, "B_replace": B_REPLACE,
                         "replace_point": REPLACE_POINT},
        "current_replace_peak": current_peak,
        "moments": moments,
        "results": results,
        "market_score": args.market_score, "market_scale": args.market_scale,
    }
    (EXPERIMENTS / "component_optimum.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 三分量最优配比（闭式解）",
        "",
        "`f = c_m·m̂ + c_r·ê_ridge + c_l·ê_lgbm`，`Score(c) = 2aᵀc − cᵀMc` 是凹二次型 →",
        "`c* = M⁻¹a`、`peak = aᵀM⁻¹a`。**配比是纯后处理，不花公榜额度。**",
        "",
        "## 二阶矩阵 M（已全部已知）",
        "",
        "| | market | ridge_dev | lgbm_dev |",
        "|---|---:|---:|---:|",
    ]
    for name, row in zip(COMPONENTS, matrix):
        lines.append(f"| {name} | " + " | ".join(f"{v:.8f}" for v in row) + " |")
    lines += [
        "",
        f"`corr(ê_r, ê_l) = {moments['corr_ridge_lgbm']:.4f}` —— 只有六成相关，",
        "**`replace` 把 `ê_ridge` 整个丢掉，丢掉的里面有独立信息**。",
        "",
        f"对账：`B_m + B_er = {moments['reconciliation_B_ridge_model']:.8f}` vs 公榜真值 "
        f"`{moments['reconciliation_target']:.8f}`，差 "
        f"{moments['reconciliation_B_ridge_model']/moments['reconciliation_target']-1:+.2%}。",
        "",
        "## 结果",
        "",
        f"现在的 `replace`：峰值 **{current_peak:.8f}**（IC {current_peak**0.5:.5f}）。",
        "",
        "| 情形 | A_m | 最优峰值 | vs replace | 最优 scale | w_ridge | w_lgbm | 市场块单独峰值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for answer in results:
        production = answer["as_production"]
        lines.append(
            f"| {answer['label']} | {answer['A'][0]:.8f} | {answer['peak']:.8f} | "
            f"{answer['vs_replace']:+.2%} | {production['prediction_scale']:.4f} | "
            f"{production['ridge_dev_weight']:+.4f} | {production['lgbm_dev_weight']:+.4f} | "
            f"{answer['market_only_peak']:.8f} |")
    if args.market_score is None:
        prior_row = next((r for r in results if r["label"].startswith("本地先验")), None)
        lines += [
            "",
            "⚠️ 上表是**扫描**，不是结论 —— `A_m` 还没测。",
            "交一次「只有 m̂」的 CSV（`experiments/market_submission.py`，纯算术不用跑模型）就能定死。",
            "",
            "## ⚠️ 老实说：重新配比这件事本身大概率不值钱",
            "",
        ]
        if prior_row is not None:
            lines += [
                f"本地先验（`mt_predictability` 的 85/15 拆解换算过来，`A_m ≈ "
                f"{prior_row['A'][0]:.8f}`）下，最优配比只比现在的 `replace` 高 "
                f"**{prior_row['vs_replace']:+.2%}** —— **约等于零**。",
                "",
                "那些 +22% / +55% 的行要求 `A_m > A(0) = 0.00164960`，也就是 `ê_ridge` 的对齐是**负的**",
                "（岭回归的截面分量在公榜上帮倒忙）。可能，但与本地证据不符，别指望。",
                "",
                "**所以交这一次的价值不在「调配比赚一点」，而在「量清楚分数从哪来」**：",
                "",
                f"- 本地先验说市场块单独的峰值是 `{prior_row['market_only_peak']:.8f}`，"
                f"占 replace 峰值的 {prior_row['market_only_peak']/current_peak:.0%}",
                "- 若属实：把市场模型的 `R²_m` 提 13%（`mt_predictability` 量过的余量）"
                f"≈ 总分 **+{0.13*prior_row['market_only_peak']/current_peak:.1%}**，"
                "是目前识别出的最大一块",
                "- **这一次额度买的是「值不值得花几天重建市场模型」的依据**，不是分数",
            ]
    (EXPERIMENTS / "component_optimum.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告 → {EXPERIMENTS / 'component_optimum.md'}")


if __name__ == "__main__":
    main()
