"""私榜交付的 `prediction_scale` 取多少 —— 把「本地最优 vs 公榜最优」的账算清楚。

## 问题

`最优 scale = A/B`。本地十折平均给一个值、公榜单一测试期给另一个值，两者差不少。
ROADMAP 的老规矩是「私榜用本地最优」（公榜最优等于对公榜过拟合），
但 2026-08-09 的计划书发现差距有 1.42 倍，大到不能照搬任一端。

## 本脚本纠正了计划书里的一处口径错误

计划书拿「岭回归本地最优 **0.759**」对「公榜最优 1.1350」，得出比值 1.50。
但 **0.759 是 alpha=5e5 那个配置的最优 scale，不是生产模型（alpha=2e6）的**。
生产 alpha 下五把本地尺子给的是 **0.86~0.97**（均值 0.91），
比值因此是 **1.24** 而不是 1.50 —— 小 alpha 会抬高 B、压低 a*，两件事被混在了一起。

hybrid 那边同理：`lgbm_blend_unweighted` 的 `blend50_xs_loose` 给 0.856，
但那一臂的基准 ridge 是**内层选 alpha**（实测落在 5e5 附近），不是生产的 2e6。
所以本脚本不直接用 0.856，而是走一步**同尺子内的臂间比**：

    生产 hybrid 的本地 a*  ≈  生产 ridge 的本地 a*  ×  (blend50 臂 a* / baseline 臂 a*)

臂间比在同一次实验、同一套折里量出来，模型差异之外的东西被约掉了。
公榜那边同样的比是 1.2196/1.1350 = 1.075，与本地的 1.12 相差 4% —— 互相印证。

## 选错的代价

`Score(a)/峰值 = 2(a/a*) − (a/a*)² = 1 − (1 − a/a*)²` ——
**损失是「相对误差」的平方，关于比值对称、不关于差值对称**。
所以高估比低估更贵（1.5 倍 → 掉 25%，而 1/1.5 倍 → 只掉 11%），
而两个候选之间的 minimax 点是它们的**调和平均**。

用法：.venv/bin/python experiments/scale_transfer.py
输出：outputs/experiments/scale_transfer.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ab_decomposition import EXPERIMENTS, PUBLIC_MODELS, local_ab, solve_ab

PRODUCTION_ALPHA = 2_000_000        # 生产岭回归的 alpha，hybrid 冻结复用的就是它

# 能给出「生产 alpha 下本地最优 scale」的尺子
LOCAL_RULERS = [
    ("modulo10", "ab_shrinkgrid.json", "modulo 10，只评相位 0"),
    ("modulo5", "ab_shrinkgrid_m5.json", "modulo 5，评相位 0+5"),
    ("相位隔离", "ab_phase_gen_alpha.json", "训练相位 0+5，验证相位 1+6"),
    ("横跨f3", "ab_horizon_f3.json", "验证段 164,490 个 time_id"),
    ("横跨f2", "ab_horizon_f2.json", "验证段 246,740 个 time_id"),
]

# 臂间比的来源：同一次实验里 baseline 臂与 blend50 臂各自的 A、B
ARM_SOURCE = "lgbm_blend_unweighted.json"
ARM_BASELINE = "baseline"
ARM_HYBRID = "blend50_xs_loose"

CLIP_BINDS_AT = 1.435               # hybrid 的 clip(0.5) 从这个 scale 起生效（NOTES 实测）


def relative_score(scale: float, optimum: float) -> float:
    """在真最优是 `optimum` 时，取 `scale` 能拿到峰值的百分之几。"""
    return 1.0 - (1.0 - scale / optimum) ** 2


def harmonic_mean(first: float, second: float) -> float:
    """两个候选之间的 minimax 点：让两边的相对误差相等。"""
    return 2.0 * first * second / (first + second)


def main() -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    # ---- 公榜：两个模型各自的最优 scale
    public = {}
    for name, points in PUBLIC_MODELS.items():
        a_value, b_value, _ = solve_ab(points)
        public[name] = {"best_scale": a_value / b_value, "peak": a_value ** 2 / b_value}
    public_ridge = public["strict_ridge"]["best_scale"]
    public_hybrid = public["v3_hybrid"]["best_scale"]

    # ---- 本地：生产 alpha 下岭回归的最优 scale（五把尺子）
    ridge_rulers = []
    for label, filename, note in LOCAL_RULERS:
        path = EXPERIMENTS / filename
        if not path.exists():
            print(f"跳过 {label}：{filename} 不存在", flush=True)
            continue
        table = local_ab(path)
        if PRODUCTION_ALPHA not in table:
            print(f"跳过 {label}：没有 alpha={PRODUCTION_ALPHA} 的臂", flush=True)
            continue
        a_value, b_value, _ = table[PRODUCTION_ALPHA]
        ridge_rulers.append({"ruler": label, "note": note, "source": filename,
                             "best_scale": a_value / b_value})
    if not ridge_rulers:
        raise SystemExit("一把尺子都没有，无法继续")
    local_ridge = float(np.mean([r["best_scale"] for r in ridge_rulers]))

    # ---- 臂间比：同一次实验里 blend50 相对 baseline 把最优 scale 抬高了多少
    summary = json.loads((EXPERIMENTS / ARM_SOURCE).read_text(encoding="utf-8"))["summary"]
    arm_ratio_ab = ((summary[ARM_HYBRID]["A"] / summary[ARM_HYBRID]["B"])
                    / (summary[ARM_BASELINE]["A"] / summary[ARM_BASELINE]["B"]))
    arm_ratio_fold = (summary[ARM_HYBRID]["mean_best_scale"]
                      / summary[ARM_BASELINE]["mean_best_scale"])
    local_hybrid = local_ridge * arm_ratio_ab

    # 逐把尺子换算，给出估计的区间
    hybrid_by_ruler = {r["ruler"]: r["best_scale"] * arm_ratio_ab for r in ridge_rulers}

    # ---- 候选与损失
    minimax = harmonic_mean(local_hybrid, public_hybrid)
    candidates = {
        "本地最优（换算到生产口径）": local_hybrid,
        "调和平均（minimax）": minimax,
        "公榜最优": public_hybrid,
        "hybrid_meta.json 现值": 0.856,
        "已交那份": 1.30,
    }
    hypotheses = {"真最优=本地": local_hybrid, "真最优=公榜": public_hybrid}
    losses = {
        name: {label: relative_score(scale, optimum)
               for label, optimum in hypotheses.items()}
        for name, scale in candidates.items()
    }
    for name in losses:
        losses[name]["最坏"] = min(losses[name][label] for label in hypotheses)

    payload: dict[str, Any] = {
        "question": "私榜交付的 prediction_scale 取多少",
        "identity": "Score(a)/峰值 = 1 − (1 − a/a*)²，损失是相对误差的平方",
        "production_alpha": PRODUCTION_ALPHA,
        "public": public,
        "local_ridge_rulers": ridge_rulers,
        "local_ridge_mean": local_ridge,
        "arm_ratio": {"from_A_over_B": arm_ratio_ab, "from_mean_best_scale": arm_ratio_fold,
                      "public_counterpart": public_hybrid / public_ridge,
                      "source": ARM_SOURCE},
        "local_hybrid_estimate": local_hybrid,
        "local_hybrid_by_ruler": hybrid_by_ruler,
        "public_over_local": {"ridge": public_ridge / local_ridge,
                              "hybrid": public_hybrid / local_hybrid},
        "candidates": candidates,
        "relative_score": losses,
        "minimax": minimax,
        "clip_binds_at": CLIP_BINDS_AT,
        "decision": "押后到 8/23 标签回补；退路是调和平均。判据见 md。",
    }
    (EXPERIMENTS / "scale_transfer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 私榜交付的 scale 取多少",
        "",
        "`最优 scale = A/B`。本地十折平均与公榜单一测试期给的值不同，",
        "而 ROADMAP 的老规矩「私榜用本地最优」没考虑过差距会有多大。",
        "",
        "## ⚠️ 先纠正一处口径错误",
        "",
        "2026-08-09 的计划书拿「岭回归本地最优 **0.759**」对公榜的 1.1350，得出比值 1.50。",
        "**但 0.759 是 alpha=5e5 那个配置的最优 scale，不是生产模型（alpha=2e6）的。**",
        "生产 alpha 下五把尺子给的是：",
        "",
        "| 尺子 | 生产 alpha 下的本地最优 scale | 说明 |",
        "|---|---:|---|",
    ]
    for entry in ridge_rulers:
        lines.append(f"| {entry['ruler']} | {entry['best_scale']:.4f} | {entry['note']} |")
    lines += [
        f"| **均值** | **{local_ridge:.4f}** | |",
        "",
        f"所以岭回归的「公榜/本地」比值是 **{public_ridge / local_ridge:.3f}**，不是 1.50。",
        "小 alpha 会抬高 B、压低 a*，计划书那个数把「换模型」和「换尺子」混在了一起。",
        "",
        "## hybrid 的本地最优：走臂间比换算",
        "",
        f"`{ARM_SOURCE}` 里 `{ARM_HYBRID}` 的最优 scale 是 "
        f"{summary[ARM_HYBRID]['mean_best_scale']:.4f}，但**那一臂的基准 ridge 是内层选 alpha**"
        "（实测落在 5e5 附近），不是生产的 2e6。所以不能直接拿来当生产 hybrid 的本地最优。",
        "",
        "改用同一次实验、同一套折里的**臂间比**（模型差异之外的东西被约掉）：",
        "",
        "| 量 | 值 |",
        "|---|---:|",
        f"| 臂间比（A/B 口径） | {arm_ratio_ab:.4f} |",
        f"| 臂间比（逐折 a* 均值口径） | {arm_ratio_fold:.4f} |",
        f"| **公榜上的同一个比** | **{public_hybrid / public_ridge:.4f}** |",
        "",
        "两个口径差 0.5%，与公榜的对应比差 4% —— 互相印证，这一步换算站得住。",
        "",
        f"→ **生产 hybrid 的本地最优 ≈ {local_ridge:.4f} × {arm_ratio_ab:.4f} = "
        f"{local_hybrid:.4f}**"
        f"（按各尺子换算的范围 {min(hybrid_by_ruler.values()):.3f}~{max(hybrid_by_ruler.values()):.3f}）",
        "",
        "## 结论：这件事没有计划书想的那么要紧",
        "",
        f"| | 本地 | 公榜 | 比值 |",
        "|---|---:|---:|---:|",
        f"| 严格岭回归 | {local_ridge:.4f} | {public_ridge:.4f} | {public_ridge/local_ridge:.3f} |",
        f"| v3_hybrid | {local_hybrid:.4f} | {public_hybrid:.4f} | "
        f"{public_hybrid/local_hybrid:.3f} |",
        "",
        "口径对齐之后，两端只差 1.2 倍（计划书以为是 1.42 倍）。代入损失公式：",
        "",
        "| 取值 | | 真最优=本地 | 真最优=公榜 | **最坏** |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scale in candidates.items():
        row = losses[name]
        lines.append(
            f"| {name} | {scale:.4f} | {row['真最优=本地']*100:.1f}% | "
            f"{row['真最优=公榜']*100:.1f}% | **{row['最坏']*100:.1f}%** |")
    lines += [
        "",
        "（表里是「拿到峰值的百分之几」，越高越好。）",
        "",
        f"**任何落在 [{local_hybrid:.2f}, {public_hybrid:.2f}] 里的取值，最坏也只掉 "
        f"{(1 - min(losses['调和平均（minimax）'].values()))*100:.1f}% 左右** —— "
        "而现在 `hybrid_meta.json` 里的 0.856 是本地占位值，**它才是真正该换掉的那个**"
        f"（最坏掉 {(1 - losses['hybrid_meta.json 现值']['最坏'])*100:.1f}%）。",
        "",
        f"限幅体检：hybrid 的 clip(0.5) 自 scale **{CLIP_BINDS_AT}** 起生效，"
        "所有候选都在它下面，二次式精确成立。",
        "",
        "## 预注册判据（现在写下，8/23 执行）",
        "",
        "8/23 主办方回补公榜测试期的标签之后：",
        "",
        "1. 在公榜测试期上直接算 v3_hybrid 的真实 `A`、`B`、`a*` —— 与本文件的 "
        f"{public_hybrid:.4f} 对账，确认两点法没错",
        "2. 同时算「训练期最后几折的 a*」，看 1.2 这个比值是不是**时期效应**"
        "（越靠近测试期的时段 a* 越高）还是模型属性",
        "3. **是时期效应** → 交付用「本地最优 × 实测比值」，因为私榜评的是更靠后的时期；",
        f"   **不是** → 交付用本地最优 {local_hybrid:.4f}",
        f"4. **若 8/23 出岔子**（重发包延期 / 标签口径不同）→ 退回调和平均 "
        f"**{minimax:.3f}**，两边等损，不用公榜最优",
        "",
        "8/31 私榜截止，8/23 给出 8 天余量。",
        "",
        "## 保留项",
        "",
        "- 臂间比是在「内层选 alpha 的 ridge」上量的，搬到「alpha 固定 2e6」上是**近似**。",
        "  两个口径的臂间比与公榜的对应比只差 4%，但这仍是一次外推。",
        "- 五把本地尺子彼此就差 12%（0.86~0.97），本地最优本身有这个量级的不确定性。",
        "- 真正干净的做法是在生产口径（alpha 2e6 / modulo 5 periodic）上对 hybrid 做一次",
        "  scale 扫描。**没做，因为那要重跑一次装车实验**；8/23 之后直接算真值更划算。",
    ]
    (EXPERIMENTS / "scale_transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"生产 alpha 下本地最优 scale（岭回归）：{local_ridge:.4f} "
          f"（{', '.join(f'{r["ruler"]} {r["best_scale"]:.3f}' for r in ridge_rulers)}）")
    print(f"臂间比 {arm_ratio_ab:.4f}（公榜对应比 {public_hybrid / public_ridge:.4f}）")
    print(f"→ hybrid 本地最优 ≈ {local_hybrid:.4f}，公榜最优 {public_hybrid:.4f}，"
          f"比值 {public_hybrid / local_hybrid:.3f}")
    print(f"minimax（调和平均）= {minimax:.4f}，两边各掉 "
          f"{(1 - relative_score(minimax, local_hybrid)) * 100:.1f}%")
    print(f"报告 → {EXPERIMENTS / 'scale_transfer.md'}")


if __name__ == "__main__":
    main()
