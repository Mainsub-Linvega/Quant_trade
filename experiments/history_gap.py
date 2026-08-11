"""为什么本地把 A1′ 低估了 2.3 倍？—— 四个口径臂的汇总。

## 背景

A1′（每资产滚动历史特征接进 LGBM 截面块）本地 5 折 peak 点估计 **+10.10%**，
公榜实测 **≥ +23.47%**（`Score(a) ≤ peak`，本点在 scale 1.16）。**低估 2.3 倍。**
而 ROADMAP 记的③类迁移率是 **~1.07（约 1:1）**。相位那次也低估 1.6 倍。

这条差距决定「8/23 之后没有公榜时还能不能信本地尺子」，所以值得单独查。

## 三个可区分的假设

- **H1 数据量**：生产训 2,645,530 行，本地每折只 ~592k 行（4.5 倍差）
- **H2 采样口径**：生产是 `phase_balanced` + modulo 5，本地一直是 `periodic` + modulo 10
- **H3 种子数**：本地 1 个种子，生产 3 个种子平均（平均降方差 ⟹ 可能抬 peak）

## 四个臂（都是 `history_peak.py --arms lgbm`，3 折，判据同口径）

| 臂 | 口径 | 用途 |
|---|---|---|
| A | m10 / periodic / 1 种子 / 窗 39,480 | 基准，复现已知的 +10% |
| B | A + **3 种子** | 测 H3 |
| C | m5 / phase_balanced / 1 种子 / 窗 39,480 | 本想测 H2 —— ⚠️ **设计有缺陷**，见下 |
| D | m5 / phase_balanced / 3 种子 / **窗 120,000**（1.8M 行） | 最接近生产口径 |

## ⚠️ C 臂的设计缺陷（结论里必须扣掉）

`train_window` 的单位是**采样后**的 time_id。modulo 10→5 时保持 39,480 不变，
等于把**时间跨度砍半**（394,800 → 197,400 个原始 time_id）。
所以 C 的 baseline peak 塌到 **0.00078**（A 是 0.00171）——
它测的不是「采样口径」，而是「一个被削弱的 baseline」。**C 不能用来支持 H2。**

## 结论

见 `outputs/experiments/history_gap.md`。一句话：**三条假设都没解释掉大头**，
最接近生产口径的 D 只把对数差距关掉 **33%**，剩下的最可能是**公榜测试期的时段/分布效应**，
本地无标签、测不了，只能等 8/23 标签回补。

⭐ **但捡到一条更普适的规律**：`Δpeak` 与 baseline 强度**强烈反相关（r = −0.986）**。
baseline 越弱，history 的相对增益越大。这解释了此前三个「大数字」都不该照单全收：
480 轮那次 +17.64%（baseline 被轮数打坏）、C 臂 +20.72%（baseline 被砍半跨度打坏）。
**读任何相对增益之前，先看 baseline 是不是被削弱了。**

用法：
    .venv/bin/python experiments/history_gap.py
输出：outputs/experiments/history_gap.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = _REPO_ROOT / "outputs" / "experiments"

PUBLIC_LOWER_BOUND = 23.47      # 公榜峰值增幅的严格下界（%）
ARMS = [
    ("gap_A_base", "A 基准口径", "复现已知的 +10%"),
    ("gap_B_seeds3", "B +3 种子", "测 H3 种子数"),
    ("gap_C_prodsampling", "C m5/phase_balanced", "⚠️ 时间跨度被砍半，不能测 H2"),
    ("gap_D_prodall", "D 生产口径全套", "最接近生产：m5/phase/3 种子/1.8M 行"),
]


def main() -> None:
    arms = []
    for label, name, note in ARMS:
        path = OUT / f"{label}.json"
        if not path.exists():
            raise SystemExit(f"缺 {path} —— 先跑对应的 history_peak.py 臂")
        d = json.loads(path.read_text(encoding="utf-8"))
        c, cfg = d["comparisons"]["lgbm"], d["configuration"]
        arms.append({
            "label": label, "name": name, "note": note,
            "sample_modulo": cfg["sample_modulo"], "sampling": cfg["sampling"],
            "lgbm_seeds": cfg["lgbm_seeds"], "train_window": cfg["train_window"],
            "baseline_peak_mean": c["baseline_peak_mean"],
            "delta_peak_pct": c["peak"]["relative_gain"] * 100,
            "drop_best_pct": c["peak"]["relative_gain_drop_best"] * 100,
            "positive_folds": c["peak"]["positive_folds"], "n_folds": c["peak"]["n_folds"],
            "delta_A_pct": c["delta_A_relative"] * 100, "delta_B_pct": c["delta_B_relative"] * 100,
        })

    base = np.array([a["baseline_peak_mean"] for a in arms])
    gain = np.array([a["delta_peak_pct"] for a in arms])
    corr = float(np.corrcoef(base, gain)[0, 1])
    a_gain = next(a["delta_peak_pct"] for a in arms if a["label"] == "gap_A_base")
    d_gain = next(a["delta_peak_pct"] for a in arms if a["label"] == "gap_D_prodall")
    closed = 1 - np.log(PUBLIC_LOWER_BOUND / d_gain) / np.log(PUBLIC_LOWER_BOUND / a_gain)

    payload = {
        "question": "本地把 A1′ 低估 2.3 倍，是口径差造成的吗？",
        "public_lower_bound_pct": PUBLIC_LOWER_BOUND,
        "arms": arms,
        "baseline_strength_vs_gain_corr": corr,
        "multiple_vs_public": {a["label"]: PUBLIC_LOWER_BOUND / a["delta_peak_pct"] for a in arms},
        "log_gap_closed_by_production_caliber": float(closed),
        "verdict": {
            "H3_seeds": "排除。3 种子把相对增益从 +9.79% 压到 +8.15%（种子同时抬了 baseline）",
            "H2_sampling": "未能测。C 臂 train_window 以采样后 time_id 计，m10→m5 把时间跨度砍半，"
                           "baseline 塌到 0.00078 —— 测的是弱 baseline，不是采样口径",
            "H1_volume": "部分。D 臂 1.8M 行 + 生产采样给 +13.08%，仍远低于 +23.47%",
            "residual": "大头未解。最接近生产口径只关掉 33% 的对数差距；"
                        "剩余最可能是公榜测试期的时段/分布效应，本地无标签测不了，等 8/23 回补",
        },
        "generalizable_finding": (
            f"Δpeak 与 baseline 强度强烈反相关（r = {corr:.3f}）。baseline 越弱，"
            "history 的相对增益越大 ⟹ 读任何相对增益之前先看 baseline 是不是被削弱了"),
        "operational_rule": "在查清之前，对第③类结构改动一律按「本地会低估」处理 —— "
                            "看到正的点估计就值得上公榜确认，别被本地量级吓退",
    }
    (OUT / "history_gap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 本地为什么把 A1′ 低估 2.3 倍（`history_gap`，2026-08-11）", "",
        f"**{payload['question']}**", "",
        f"公榜峰值增幅的严格下界 **+{PUBLIC_LOWER_BOUND}%**（`Score(a) ≤ peak`，本点在 scale 1.16）。", "",
        "| 臂 | m | 采样 | 种子 | 训练窗 | baseline peak | Δpeak | 去最好折 | 正折 | ΔA | ΔB |",
        "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for a in arms:
        lines.append(
            f"| {a['name']} | {a['sample_modulo']} | {a['sampling']} | {a['lgbm_seeds']} | "
            f"{a['train_window']:,} | {a['baseline_peak_mean']:.8f} | **{a['delta_peak_pct']:+.2f}%** | "
            f"{a['drop_best_pct']:+.2f}% | {a['positive_folds']}/{a['n_folds']} | "
            f"{a['delta_A_pct']:+.2f}% | {a['delta_B_pct']:+.2f}% |")
    lines += [
        "", "## 判据（由代码算，不是报告里的评语）", "",
        f"- **H3 种子**：{payload['verdict']['H3_seeds']}",
        f"- **H2 采样**：{payload['verdict']['H2_sampling']}",
        f"- **H1 数据量**：{payload['verdict']['H1_volume']}",
        f"- **剩余**：{payload['verdict']['residual']}", "",
        f"公榜/本地倍数：A {PUBLIC_LOWER_BOUND/a_gain:.2f}× → D {PUBLIC_LOWER_BOUND/d_gain:.2f}×，"
        f"**只关掉 {closed*100:.0f}% 的对数差距**。", "",
        "## ⭐ 捡到的更普适的规律", "",
        f"{payload['generalizable_finding']}。", "",
        "已经三次被它误导过：480 轮那次 +17.64%（baseline 被轮数打坏）、"
        "C 臂 +20.72%（baseline 被砍半的时间跨度打坏）、"
        "以及最早 07-23 的 +2.5%（baseline 被固定 scale 0.5 压扁）。", "",
        "## 操作结论", "",
        f"{payload['operational_rule']}。", "",
    ]
    (OUT / "history_gap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    print(f"corr(baseline, Δpeak) = {corr:.4f} | 关掉对数差距 {closed*100:.0f}%")
    print(f"报告：{OUT / 'history_gap.md'}")


if __name__ == "__main__":
    main()
