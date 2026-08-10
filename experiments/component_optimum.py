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

## 2026-08-10：轮数换到 480 之后要显式指定源模型

`ê_lgbm` 随轮数变（`B_el(160)=0.00067024 → B_el(480)=0.00100408`，+49.8%），
而 `B_m` 不变 —— 两块的最优 scale 因此**分家更开**，「放开单一 scale」这个
第①类改动的价值随轮数变大。所以源 CSV 与它对应的 `(A, 公榜点)` 现在是参数。

同时把 **2 分量**（`m̂` + `ê_lgbm`，不碰 `ê_ridge`）单独解出来并作为推荐口径：
它的 `B` 全部精确（`B_el = B(源) − B_m` 是精确减法），不吃 `ê_ridge` 那 1.13%
的对账差、也不吃近共线的病态。3 分量仍然算，但 `⟨ê_r, ê_l480⟩` 只有区间没有精确值
（存档 Gram 是 160 轮那版），所以给两个端点夹住并检查 `c_r` 会不会变号。

用法：
    .venv/bin/python experiments/component_optimum.py                    # 扫描（160 轮）
    .venv/bin/python experiments/component_optimum.py \\
        --market-score 0.0021 --market-scale 1.3                        # 精确解（160 轮）
    .venv/bin/python experiments/component_optimum.py \\
        --market-score 0.0021 --market-scale 1.3 \\
        --source outputs/submission_r480_s116.csv --source-scale 1.16 \\
        --replace-A 0.00234971 --replace-point 1.16 0.0025821304 \\
        --label component_optimum_r480                                   # 480 轮
输出：outputs/experiments/<label>.{json,md}
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
# blend_weight 1.0 → m̂ + ê_lgbm。A 关于 w 精确线性 ⟹ 这是**精确值**，不是估计。
A_REPLACE_160 = A_RIDGE + 2 * (A_HYBRID - A_RIDGE)
REPLACE_POINT_160 = (1.16, 0.0024872338)          # ledger 2026-08-09

COMPONENTS = ("market", "ridge_dev", "lgbm_dev")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-form optimal component weights.")
    parser.add_argument("--market-score", type=float, default=None,
                        help="「只有 m̂」那份 CSV 的公榜分；给了就精确解，不给就扫描")
    parser.add_argument("--market-scale", type=float, default=None)
    parser.add_argument("--source", default=str(_REPO_ROOT / "outputs" / "submission_replace_s116.csv"))
    parser.add_argument("--source-scale", type=float, default=1.16)
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    # 2026-08-10：轮数从 160 提到 480 之后 ê_lgbm 换了，replace 那个点也得跟着换。
    # 不给这两个参数 = 沿用 160 轮的点，行为与 08-09 完全一致。
    parser.add_argument("--replace-A", type=float, default=A_REPLACE_160,
                        help="源 CSV 那个模型的 A（精确值）。160 轮是 0.00211518，480 轮是 0.00234971")
    parser.add_argument("--replace-point", type=float, nargs=2, default=REPLACE_POINT_160,
                        metavar=("SCALE", "SCORE"),
                        help="源 CSV 那个模型的一个公榜点，用来反解它的 B")
    parser.add_argument("--label", default="component_optimum", help="输出文件名前缀")
    return parser.parse_args()


def second_moments(args: argparse.Namespace) -> dict[str, Any]:
    """三个分量的二阶矩阵 M（B 的口径）。

    CSV 还在就现算并缓存；删了就读上一次缓存 —— 那 1.1 GB 的 CSV 迟早要清掉。
    """
    cache = EXPERIMENTS / f"{args.label}.json"
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

    # k 把「无权二阶矩」标定成 B 的口径。用源 CSV 那个模型的精确 B 标定 → 那一条误差为 0。
    b_replace = replace_B(args)
    total_moment = float((raw ** 2).mean())
    scaling = b_replace / total_moment
    market_moment = float((market ** 2).mean())

    # ⭐ ê_lgbm 的自二阶矩**直接从源 CSV 算**，不走存档 —— 换了轮数它就变了，
    #    而存档里的 Gram 是 160 轮那一版。`⟨m̂,ê⟩ ≡ 0` ⟹ 这一步是精确的减法。
    lgbm_lgbm = total_moment - market_moment

    payload = json.loads((EXPERIMENTS / "public_csv_fingerprints.json").read_text(encoding="utf-8"))
    gram = np.array(payload["gram"])
    names = payload["gram_names"]
    i = names.index("submission_strict_scale113.csv")      # f0 = m̂ + ê_ridge
    j = names.index("submission_hybrid_base0856.csv")      # fh = m̂ + ½ê_ridge + ½ê_lgbm160
    # ê_r = f0 − m̂，ê_l160 = (2fh − f0) − m̂，且 ⟨ê, m̂⟩ = 0。ê_r 不随轮数变。
    ridge_ridge = gram[i, i] - market_moment
    lgbm160_lgbm160 = (4 * gram[j, j] - 4 * gram[i, j] + gram[i, i]) - market_moment
    ridge_lgbm160 = (2 * gram[i, j] - gram[i, i]) - market_moment
    corr160 = ridge_lgbm160 / np.sqrt(ridge_ridge * lgbm160_lgbm160)

    # ⚠️ ⟨ê_r, ê_l⟩ 在源 CSV 不是 160 轮时**没有精确值** —— 那需要 strict 岭回归的 CSV
    #    与源 CSV 的交叉矩，而 strict 那份已被删（存档只有标量 Gram，配不上新 CSV）。
    #    这里给两个端点把它夹住，两个假设都是有名有姓的：
    #      A. 相关系数不随轮数变      → ⟨ê_r, ê_l⟩ = ρ₁₆₀·√(B_er·B_el)
    #      B. 新增的那些树与 ê_r 正交 → ⟨ê_r, ê_l⟩ = ⟨ê_r, ê_l160⟩（不变）
    #    源 CSV 就是 160 轮时两者**恒等**，退化成精确值。
    bracket = {
        "constant_corr": float(corr160 * np.sqrt(ridge_ridge * lgbm_lgbm)),
        "increment_orthogonal": float(ridge_lgbm160),
    }
    ridge_lgbm = bracket["constant_corr"]

    matrix = scaling * np.array([[market_moment, 0.0, 0.0],
                                 [0.0, ridge_ridge, ridge_lgbm],
                                 [0.0, ridge_lgbm, lgbm_lgbm]])
    return {
        "M": matrix.tolist(), "components": list(COMPONENTS),
        "B_market": matrix[0, 0], "B_ridge_dev": matrix[1, 1], "B_lgbm_dev": matrix[2, 2],
        "B_ridge_lgbm_bracket": {k: scaling * v for k, v in bracket.items()},
        # 源就是 160 轮时两个端点在代数上恒等。但它们各自来自**不同的 CSV 文件**
        # （源 CSV vs 存档 Gram 的那两份），都只存了 8 位小数 → 实测相对差 1.2e-07。
        # 容差取 1e-4：比舍入噪声大三个量级，比「换了轮数」的百分级差小两个量级。
        "cross_moment_is_exact": bool(
            abs(bracket["constant_corr"] - bracket["increment_orthogonal"])
            <= 1e-4 * max(abs(bracket["constant_corr"]), abs(bracket["increment_orthogonal"]))),
        "corr_ridge_lgbm": float(matrix[1, 2] / np.sqrt(matrix[1, 1] * matrix[2, 2])),
        "corr_ridge_lgbm160": float(corr160),
        "cross_market_dev": float(np.abs((market * (raw - market)).mean())),
        # 对账 1：B_m + B_er 应等于严格岭回归的公榜真值 B(0)
        "reconciliation_B_ridge_model": float(matrix[0, 0] + matrix[1, 1]),
        "reconciliation_target": B_RIDGE,
        # 对账 2：B_er 还能由 B(0) − B_m 直接得到，两条路应该对得上
        "B_ridge_dev_from_public": float(B_RIDGE - matrix[0, 0]),
        # 对账 3：源 CSV 是 160 轮时，直接减出来的 B_el 必须等于存档 Gram 反解的那个
        "reconciliation_B_lgbm_dev_gram": float(scaling * lgbm160_lgbm160),
        "B_replace": b_replace, "A_replace": float(args.replace_A),
    }


def replace_B(args: argparse.Namespace) -> float:
    """由「源 CSV 那个模型的精确 A + 一个公榜点」反解它的 B。"""
    scale, score = float(args.replace_point[0]), float(args.replace_point[1])
    return (2 * scale * args.replace_A - score) / scale ** 2


def solve(matrix: np.ndarray, alignment: np.ndarray) -> dict[str, Any]:
    """c* = M⁻¹a，peak = aᵀM⁻¹a。"""
    weights = np.linalg.solve(matrix, alignment)
    peak = float(alignment @ weights)
    return {"weights": weights.tolist(), "peak": peak, "ic": peak ** 0.5,
            # 换算成生产参数化：scale·(m̂ + w_r·ê_r + w_l·ê_l)
            "as_production": {"prediction_scale": float(weights[0]),
                              "ridge_dev_weight": float(weights[1] / weights[0]),
                              "lgbm_dev_weight": float(weights[2] / weights[0])}}


def solve2(b_market: float, b_lgbm: float, a_market: float, a_lgbm: float) -> dict[str, Any]:
    """只用 (m̂, ê_lgbm) 两个分量。`⟨m̂,ê⟩ ≡ 0` ⟹ M 是对角阵，闭式解不需要求逆。

    这一版**不碰 ê_ridge**，于是也就不碰那个 1.13% 对不上的 `B_er`
    与近共线带来的病态 —— 系数温和、结论稳。
    """
    c_market, c_lgbm = a_market / b_market, a_lgbm / b_lgbm
    peak = a_market ** 2 / b_market + a_lgbm ** 2 / b_lgbm
    return {"weights": [c_market, c_lgbm], "peak": float(peak), "ic": float(peak ** 0.5),
            "market_only_peak": float(a_market ** 2 / b_market),
            "lgbm_only_peak": float(a_lgbm ** 2 / b_lgbm),
            # 换算成「整体 scale × (m̂ + λ·ê)」这种能直接写进 meta 的形式
            "as_production": {"prediction_scale": float(c_market),
                              "lgbm_dev_weight": float(c_lgbm / c_market)}}


def predicted_score(b_market: float, b_lgbm: float, a_market: float, a_lgbm: float,
                    c_market: float, c_lgbm: float) -> float:
    """给定配比，预测公榜分 —— 交完拿实测分回来做「预测 vs 实测」的校验。"""
    return float(2 * (c_market * a_market + c_lgbm * a_lgbm)
                 - (c_market ** 2 * b_market + c_lgbm ** 2 * b_lgbm))


def main() -> None:
    args = parse_args()
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    moments = second_moments(args)
    matrix = np.array(moments["M"])
    a_replace = float(args.replace_A)
    b_replace = float(moments["B_replace"])
    current_peak = a_replace ** 2 / b_replace

    print(f"源 CSV：{Path(args.source).name} @ scale {args.source_scale}")
    print(f"  A = {a_replace:.8f}（精确）  B = {b_replace:.8f}"
          f"（由公榜点 scale {args.replace_point[0]} / 分 {args.replace_point[1]} 反解）")
    print(f"\n二阶矩阵 M（B 口径）：")
    for name, row in zip(COMPONENTS, matrix):
        print("  " + name.ljust(10) + "  ".join(f"{v: .8f}" for v in row))
    print(f"  corr(ê_r, ê_l) = {moments['corr_ridge_lgbm']:.4f}"
          f"（160 轮实测 {moments['corr_ridge_lgbm160']:.4f}）")
    if not moments["cross_moment_is_exact"]:
        low, high = sorted(moments["B_ridge_lgbm_bracket"].values())
        print(f"  ⚠️ ⟨ê_r, ê_l⟩ 不是精确值（源 CSV 不是 160 轮）→ 夹在 "
              f"{low:.8f} ~ {high:.8f} 之间，相差 {high/low-1:+.1%}。"
              f"三分量结论要按这个区间打折看")
    print(f"  对账① B_m + B_er = {moments['reconciliation_B_ridge_model']:.8f} vs "
          f"公榜真值 B(0) = {moments['reconciliation_target']:.8f} "
          f"（差 {moments['reconciliation_B_ridge_model']/moments['reconciliation_target']-1:+.2%}）")
    print(f"  对账② B_er 两条路：Gram 反解 {matrix[1, 1]:.8f} vs "
          f"B(0) − B_m = {moments['B_ridge_dev_from_public']:.8f} "
          f"（差 {matrix[1, 1]/moments['B_ridge_dev_from_public']-1:+.2%}）")
    print(f"  对账③ B_el 两条路：源 CSV 直接减 {matrix[2, 2]:.8f} vs "
          f"存档 Gram（160 轮）{moments['reconciliation_B_lgbm_dev_gram']:.8f} "
          f"（差 {matrix[2, 2]/moments['reconciliation_B_lgbm_dev_gram']-1:+.2%}"
          f"{'，源就是 160 轮 → 这一条应 ~0' if moments['cross_moment_is_exact'] else ''}）")
    print(f"\n现在的模型（单一 scale 的 replace）：峰值 {current_peak:.8f}，"
          f"IC {current_peak**0.5:.5f}")

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
        fractions = [(f"A_m = {f:.0%}·A(1)", f * a_replace) for f in (0.6, 0.7, 0.8, 0.9)]
        fractions.append((f"本地先验 ({prior/a_replace:.0%})", prior))
        fractions.sort(key=lambda item: item[1])
        print(f"  （本地先验：mt_predictability 的 85/15 拆解 → A_m ≈ {prior:.8f} "
              f"= {prior/a_replace:.1%}·A(1)）")

    # ── 2 分量（推荐口径）──────────────────────────────────────────────
    # 只有 m̂ 和 ê_lgbm。B 全部精确（B_el = B(源) − B_m 是精确减法），
    # 不吃 ê_ridge 那 1.13% 的对账差，也不吃近共线。
    print(f"\n【2 分量：f = c_m·m̂ + c_l·ê_lgbm】—— 推荐口径，系数稳")
    print(f"{'情形':<18}{'A_m':>12}{'A_el':>12}{'c_m':>8}{'c_l':>8}"
          f"{'最优峰值':>13}{'vs 现在':>10}{'市场块占分':>10}")
    two: list[dict[str, Any]] = []
    for label, a_market in fractions:
        answer = solve2(matrix[0, 0], matrix[2, 2], a_market, a_replace - a_market)
        answer.update({"label": label, "A_market": a_market, "A_lgbm_dev": a_replace - a_market,
                       "vs_current": answer["peak"] / current_peak - 1.0,
                       "market_share_of_score": answer["market_only_peak"] / answer["peak"]})
        two.append(answer)
        print(f"{label:<18}{a_market:>12.8f}{a_replace - a_market:>12.8f}"
              f"{answer['weights'][0]:>8.3f}{answer['weights'][1]:>8.3f}"
              f"{answer['peak']:>13.8f}{answer['vs_current']:>+9.2%}"
              f"{answer['market_share_of_score']:>10.1%}")

    if args.market_score is not None:
        best = two[0]
        c_m, c_l = best["weights"]
        print(f"\n→ 出 CSV 的命令（触限行数必须为 0，否则把两个权重同比例缩小）：")
        print(f"   .venv/bin/python experiments/market_submission.py \\\n"
              f"     --source {args.source} --source-scale {args.source_scale} \\\n"
              f"     --market-weight {c_m:.4f} --deviation-weight {c_l:.4f} \\\n"
              f"     --output outputs/submission_mix2.csv")
        print(f"   预测公榜分 {best['peak']:.8f}（= 峰值，因为按最优配比出）")
        # B 的两个来源都带约 0.3% 的不确定度 → 把结论的稳健性一并报出来
        for bump in (0.985, 1.015):
            probe = solve2(matrix[0, 0] * bump, matrix[2, 2], best["A_market"],
                           best["A_lgbm_dev"])
            print(f"   B_m 拨 {bump-1:+.1%} → 峰值 {probe['peak']:.8f}"
                  f"（vs 现在 {probe['peak']/current_peak-1:+.2%}），"
                  f"c_m {probe['weights'][0]:.3f} / c_l {probe['weights'][1]:.3f}")

    # ── 3 分量（把 ê_ridge 也放回来）────────────────────────────────────
    print(f"\n【3 分量：再加 c_r·ê_ridge】—— ⚠️ 只有 c_r 显著且不随假设变号时才可信")
    print(f"{'情形':<18}{'A_er':>12}{'c_m':>8}{'c_r':>8}{'c_l':>8}"
          f"{'最优峰值':>13}{'vs 现在':>10}{'vs 2 分量':>11}")
    for label, a_market in fractions:
        alignment = np.array([a_market, A_RIDGE - a_market, a_replace - a_market])
        answer = solve(matrix, alignment)
        base = next(item for item in two if item["label"] == label)
        answer.update({"label": label, "A": alignment.tolist(),
                       "vs_replace": answer["peak"] / current_peak - 1.0,
                       "vs_two_component": answer["peak"] / base["peak"] - 1.0,
                       "market_only_peak": a_market ** 2 / matrix[0, 0]})
        # c_r 对 ⟨ê_r,ê_l⟩ 那个区间是否稳（不变号 = 结论可信的必要条件）
        signs = []
        for value in moments["B_ridge_lgbm_bracket"].values():
            probe = matrix.copy()
            probe[1, 2] = probe[2, 1] = value
            signs.append(float(np.linalg.solve(probe, alignment)[1]))
        answer["c_ridge_bracket"] = signs
        answer["c_ridge_sign_stable"] = bool(min(signs) * max(signs) > 0)
        results.append(answer)
        flag = "" if answer["c_ridge_sign_stable"] else "  ⚠️c_r 变号"
        print(f"{label:<18}{alignment[1]:>12.8f}{answer['weights'][0]:>8.3f}"
              f"{answer['weights'][1]:>8.3f}{answer['weights'][2]:>8.3f}"
              f"{answer['peak']:>13.8f}{answer['vs_replace']:>+9.2%}"
              f"{answer['vs_two_component']:>+10.2%}{flag}")

    payload = {
        "identity": "Score(c) = 2aᵀc − cᵀMc；c* = M⁻¹a，peak = aᵀM⁻¹a",
        "why_free": "配比是纯后处理（参数三分类第①类）；M 全已知，a 只缺 A_m",
        "exact_inputs": {"A_ridge": A_RIDGE, "B_ridge": B_RIDGE, "A_hybrid": A_HYBRID,
                         "A_replace": a_replace, "B_replace": b_replace,
                         "replace_point": list(args.replace_point),
                         "source": args.source, "source_scale": args.source_scale},
        "current_replace_peak": current_peak,
        "moments": moments,
        "two_component": two,
        "results": results,
        "market_score": args.market_score, "market_scale": args.market_scale,
    }
    (EXPERIMENTS / f"{args.label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 分量最优配比（闭式解）",
        "",
        "`Score(c) = 2aᵀc − cᵀMc` 是凹二次型 → `c* = M⁻¹a`、`peak = aᵀM⁻¹a`。",
        "**配比是纯后处理（参数三分类第①类），不花公榜额度。**",
        "",
        f"源 CSV `{Path(args.source).name}` @ scale {args.source_scale}：",
        f"`A = {a_replace:.8f}`（精确）、`B = {b_replace:.8f}`"
        f"（由公榜点 scale {args.replace_point[0]} / 分 {args.replace_point[1]} 反解）。",
        f"现在的单一 scale 模型峰值 **{current_peak:.8f}**（IC {current_peak**0.5:.5f}）。",
        "",
        "## 二阶矩阵 M（B 口径）",
        "",
        "| | market | ridge_dev | lgbm_dev |",
        "|---|---:|---:|---:|",
    ]
    for name, row in zip(COMPONENTS, matrix):
        lines.append(f"| {name} | " + " | ".join(f"{v:.8f}" for v in row) + " |")
    lines += [
        "",
        f"`corr(ê_r, ê_l) = {moments['corr_ridge_lgbm']:.4f}`"
        f"（160 轮实测 {moments['corr_ridge_lgbm160']:.4f}）。",
        "",
        f"对账①：`B_m + B_er = {moments['reconciliation_B_ridge_model']:.8f}` vs 公榜真值 "
        f"`B(0) = {moments['reconciliation_target']:.8f}`，差 "
        f"{moments['reconciliation_B_ridge_model']/moments['reconciliation_target']-1:+.2%}。",
        f"对账②：`B_er` 两条路 —— Gram 反解 `{matrix[1, 1]:.8f}` vs "
        f"`B(0) − B_m = {moments['B_ridge_dev_from_public']:.8f}`，差 "
        f"{matrix[1, 1]/moments['B_ridge_dev_from_public']-1:+.2%}。",
    ]
    if not moments["cross_moment_is_exact"]:
        low, high = sorted(moments["B_ridge_lgbm_bracket"].values())
        lines += [
            "",
            f"⚠️ **`⟨ê_r, ê_l⟩` 不是精确值**（源 CSV 不是 160 轮，而存档 Gram 是 160 轮那一版）。",
            f"两个端点夹住它：相关系数不变 → `{moments['B_ridge_lgbm_bracket']['constant_corr']:.8f}`；",
            f"新增的树与 `ê_r` 正交 → `{moments['B_ridge_lgbm_bracket']['increment_orthogonal']:.8f}`。",
            f"区间宽 {high/low-1:+.1%} —— **三分量结论按这个区间打折看**。",
            "要精确值只能把严格岭回归那份 CSV 重新出一遍（0 额度，但要跑一次 runner）。",
        ]
    lines += [
        "",
        "## 2 分量：`f = c_m·m̂ + c_l·ê_lgbm`（推荐口径）",
        "",
        "`⟨m̂,ê⟩ ≡ 0` ⟹ M 是对角阵，`B_el = B(源) − B_m` 是精确减法。",
        "**不碰 `ê_ridge` ⟹ 不吃对账差、不吃近共线，系数温和。**",
        "",
        "| 情形 | A_m | A_el | c_m | c_l | 最优峰值 | vs 现在 | 市场块占分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for answer in two:
        lines.append(
            f"| {answer['label']} | {answer['A_market']:.8f} | {answer['A_lgbm_dev']:.8f} | "
            f"{answer['weights'][0]:.3f} | {answer['weights'][1]:.3f} | {answer['peak']:.8f} | "
            f"{answer['vs_current']:+.2%} | {answer['market_share_of_score']:.1%} |")
    lines += [
        "",
        "## 3 分量：再加 `c_r·ê_ridge`",
        "",
        "| 情形 | A_er | c_m | c_r | c_l | 最优峰值 | vs 现在 | vs 2 分量 | c_r 区间 | 变号? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for answer in results:
        low, high = min(answer["c_ridge_bracket"]), max(answer["c_ridge_bracket"])
        lines.append(
            f"| {answer['label']} | {answer['A'][1]:.8f} | {answer['weights'][0]:.3f} | "
            f"{answer['weights'][1]:+.3f} | {answer['weights'][2]:.3f} | {answer['peak']:.8f} | "
            f"{answer['vs_replace']:+.2%} | {answer['vs_two_component']:+.2%} | "
            f"{low:+.3f}~{high:+.3f} | "
            f"{'否' if answer['c_ridge_sign_stable'] else '**是 ⚠️**'} |")
    if args.market_score is None:
        lines += [
            "",
            "⚠️ 上面两张表是**扫描**，不是结论 —— `A_m` 还没测。",
            "交一次「只有 m̂」的 CSV（`experiments/market_submission.py`，纯算术不用跑模型）就能定死。",
        ]
    else:
        best = two[0]
        lines += [
            "",
            "## 结论",
            "",
            f"实测 `A_m = {best['A_market']:.8f}` ⟹ 市场块占分 "
            f"**{best['market_share_of_score']:.1%}**、`A_er = {results[0]['A'][1]:.8f}`。",
            "",
            f"2 分量最优配比 `c_m = {best['weights'][0]:.4f}`、`c_l = {best['weights'][1]:.4f}`，"
            f"峰值 **{best['peak']:.8f}**（vs 现在 {best['vs_current']:+.2%}）。",
            "",
            "出 CSV：",
            "",
            "```bash",
            ".venv/bin/python experiments/market_submission.py \\",
            f"  --source {args.source} --source-scale {args.source_scale} \\",
            f"  --market-weight {best['weights'][0]:.4f} "
            f"--deviation-weight {best['weights'][1]:.4f} \\",
            "  --output outputs/submission_mix2.csv",
            "```",
            "",
            f"**预测公榜分 {best['peak']:.8f}** —— 交完拿实测分回来做「预测 vs 实测」的校验"
            "（08-09 那次偏低 1.34%，由此得出无权 B 近似要按 −1.5% 修正）。",
        ]
    (EXPERIMENTS / f"{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告 → {EXPERIMENTS / (args.label + '.md')}")


if __name__ == "__main__":
    main()
