"""对 `target_mlp_screen` 的重新分析：等权 −54.49% 到底否证了什么。

## 为什么要重算

`outputs/experiments/target_mlp_screen.json` 的结论是「等权集成增益 −54.49%、0/5 折 ⟹
stop MLP search」，并据此关闭了 MLP 方向。但**等权**是一个很强的约束：把一个 peak 只有
基准 24.3% 的分量按 50/50 掺进去，掉分是代数上的必然，与「这个分量有没有独立信息」是两件事。

```text
A₂≈0、B₂≈B₁、C≈0 时：A=½A₁，B=¼(B₁+2C+B₂)=½B₁ ⟹ Peak=½Peak₁ ⟹ −50%
实测 −54.49%，与这个签名只差一点。
```

⟹ 那个数说明的是「等权掺一个弱模型会掉分」，**不是**「这个分量没有独立信息」。

## 怎么不重训就回答正确的问题

`target_mlp_screen.json` 里每折都记了 baseline / mlp / equal_blend 三条臂的 `A` 和 `B`
（口径 `A=⟨y,p⟩_w/D`、`B=⟨p,p⟩_w/D`、`peak=A²/B`）。等权臂满足

    A_e = (A_b + A_m)/2                       ← 可用来校验分解成立
    B_e = (B_b + 2C + B_m)/4                  ⟹  C = 2·B_e − (B_b + B_m)/2

⟹ **交叉项 C 可以从已有 JSON 反解出来**，于是两分量的最优配比 peak 有闭式：

    peak_opt = (A_b²·B_m − 2·A_b·A_m·C + A_m²·B_b) / (B_b·B_m − C²)

不需要任何训练，不需要原始预测向量。

## ⚠️ 这是 oracle 上界，不是可部署增益

最优系数是在**评估折上重解**的。仓库自己量过这道让步：`horizon_auxiliary_cache_probe` 的
`null_frozen_scale` 臂（不加任何分量、只把 scale 冻结在过去折）就已经是
**−2.54%（pure_e）/ −3.84%（full）** ⟹ 冻结系数的诚实做法要在本结果上再打这么多折扣。

另外基准偏弱：`target_mlp_screen` 用的是 1 seed × 160 轮 / `sample_modulo=10` /
`train_window=39,480` / 100 特征，而生产是 3 seeds × 480 轮 / modulo 5 / 78,960 / 200 特征。

⟹ 本脚本的产物只用于**回答「MLP 家族是否携带独立信息」这一个问题**，
不得当作「blend 能提多少分」的估计，也不得据此跳过冻结系数终审。

用法：
    .venv/bin/python experiments/target_mlp_oracle_blend.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# horizon_auxiliary_cache_probe 实测的「冻结系数 vs 评估折重解」让步区间
FROZEN_CONCESSION = (-0.0384, -0.0254)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screen", default=str(_REPO_ROOT / "outputs" / "experiments" /
                                           "target_mlp_screen.json"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="target_mlp_oracle_blend")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def oracle_two_component_peak(a_b: float, b_b: float, a_m: float, b_m: float,
                              cross: float) -> float:
    """两分量 `c1·p_b + c2·p_m` 在最优 (c1,c2) 下的 peak。

    peak = Vᵀ S⁻¹ V，V=(A_b, A_m)，S=[[B_b, C], [C, B_m]]。展开即下式。
    数学上必然 ≥ max(peak_b, peak_m)（两分量至少能退化成单分量）。
    """
    determinant = b_b * b_m - cross * cross
    if determinant <= 0.0:
        raise ValueError(f"Gram 矩阵非正定：det={determinant:.6e}")
    return (a_b * a_b * b_m - 2.0 * a_b * a_m * cross + a_m * a_m * b_b) / determinant


def analyse(screen: dict) -> dict:
    rows = []
    for record in screen["folds"]:
        a_b, b_b = float(record["baseline"]["A"]), float(record["baseline"]["B"])
        a_m, b_m = float(record["mlp"]["A"]), float(record["mlp"]["B"])
        a_e, b_e = float(record["equal_blend"]["A"]), float(record["equal_blend"]["B"])
        # 分解自洽性：等权臂的 A 必须恰好是两者均值，否则「等权」不是我以为的那个组合
        if abs(a_e - 0.5 * (a_b + a_m)) > 1e-15:
            raise SystemExit(f"fold {record['fold']}：A_e 与 (A_b+A_m)/2 不符 ⟹ 分解假设不成立")
        cross = 2.0 * b_e - 0.5 * (b_b + b_m)
        peak_b, peak_m = a_b * a_b / b_b, a_m * a_m / b_m
        peak_opt = oracle_two_component_peak(a_b, b_b, a_m, b_m, cross)
        if peak_opt < max(peak_b, peak_m) - 1e-18:
            raise SystemExit(f"fold {record['fold']}：oracle peak 低于单分量 ⟹ 解错了")
        rows.append({
            "fold": int(record["fold"]),
            "peak_baseline": peak_b,
            "peak_mlp": peak_m,
            "mlp_relative_to_baseline": peak_m / peak_b,
            "cross_term": cross,
            "gram_cosine": cross / float(np.sqrt(b_b * b_m)),
            "prediction_correlation": float(record["prediction_correlation"]),
            "peak_equal_blend": float(record["equal_blend"]["peak"]),
            "relative_equal_blend": float(record["equal_blend"]["peak"]) / peak_b - 1.0,
            "peak_oracle_blend": peak_opt,
            "relative_oracle_blend": peak_opt / peak_b - 1.0,
        })
    gains = np.array([row["relative_oracle_blend"] for row in rows], dtype=np.float64)
    equal = np.array([row["relative_equal_blend"] for row in rows], dtype=np.float64)
    return {
        "folds": rows,
        "oracle_mean": float(gains.mean()),
        "oracle_positive_folds": int((gains > 0).sum()),
        "oracle_drop_best": float(np.sort(gains)[:-1].mean()),
        "equal_mean": float(equal.mean()),
        "equal_positive_folds": int((equal > 0).sum()),
        "honest_range_after_frozen_concession":
            [float(gains.mean() + FROZEN_CONCESSION[0]), float(gains.mean() + FROZEN_CONCESSION[1])],
    }


def render(payload: dict) -> str:
    summary, config = payload["summary"], payload["source_configuration"]
    lines = [
        "# `target_mlp_screen` 重新分析：oracle 最优配比（`target_mlp_oracle_blend`）",
        "",
        "> **一句话**：等权 −54.49% 否证的是「等权掺弱模型」，不是「MLP 没有独立信息」。",
        "> 但 oracle 上界也**不构成**证成 —— 它是评估折上重解系数的上界。",
        "",
        "## 结果（不训练，从 `target_mlp_screen.json` 的逐折 A/B 反解交叉项）",
        "",
        "| fold | 基准 peak | MLP/基准 | Gram 余弦 | 逐行 corr | 等权 | **ORACLE 最优** |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["folds"]:
        lines.append(
            f"| {row['fold']} | {row['peak_baseline']:.8f} | {row['mlp_relative_to_baseline']:.1%} | "
            f"{row['gram_cosine']:.3f} | {row['prediction_correlation']:.3f} | "
            f"{row['relative_equal_blend']:+.2%} | **{row['relative_oracle_blend']:+.2%}** |")
    honest = summary["honest_range_after_frozen_concession"]
    lines += [
        "",
        f"**折均 {summary['oracle_mean']:+.2%}**，正折 **{summary['oracle_positive_folds']}/5**，"
        f"去最好折 {summary['oracle_drop_best']:+.2%}。",
        f"（对照：等权折均 {summary['equal_mean']:+.2%}、正折 {summary['equal_positive_folds']}/5。）",
        "",
        "## 怎么读这个数",
        "",
        "1. ⭐ **等权 −54.49% 不是否证。** 把 peak 只有基准 24.3% 的分量按 50/50 掺进去，",
        "   掉分是代数上的必然（`A₂≈0` 时 `Peak=½Peak₁ ⟹ −50%`，实测 −54.49% 与之只差一点）。",
        "   换成最优配比后 **5/5 折为正** ⟹ MLP 家族确实携带了基准没有的信息。",
        "2. ⚠️ **oracle 上界也不是证成。** 系数在评估折上重解。仓库量过这道让步：",
        f"   `horizon_auxiliary_cache_probe` 的 `null_frozen_scale` 臂本身就是 "
        f"{FROZEN_CONCESSION[0]:+.2%} ~ {FROZEN_CONCESSION[1]:+.2%}。",
        f"   ⟹ 冻结系数的诚实值约 **{honest[0]:+.2%} ~ {honest[1]:+.2%}**，"
        "**恰好卡在 ③类 +3% 门槛上**，不是安全通过。",
        "3. ⚠️ **基准偏弱。** 本 screen 用的是",
        f"   `{config['n_folds']} 折 / sample_modulo={config['sample_modulo']} / "
        f"train_window={config['train_window']:,} / {config['current_feature_count']} 特征 / "
        f"max_iter={config['max_iter']}`，",
        "   而生产是 3 seeds × 480 轮 / modulo 5 / 78,960 / 200 特征 ⟹ 对生产的真实差距只会更大。",
        "4. **折间极不均匀**：MLP 相对强度在 fold 1/0 是 53%/40%，在 fold 4/2 只有 5%/6%；",
        "   增益几乎全部来自前两折 ⟹ 去最好折仍为正，但这条线的稳健性是待验证项，不是已知项。",
        "",
        "## 结论（用途边界）",
        "",
        "本文件**只回答一个问题**：MLP 家族是否携带基准没有的信息 —— **是**。",
        "它**不**回答「blend 能提多少分」，也**不**能替代冻结系数的终审。",
        "⟹ 允许把多任务辅助监督（`experiments/multitask_mlp.py`）作为一次预注册筛选立项；",
        "**不**允许据此跳过 Stage 2 的冻结系数五折门禁。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    screen_path = Path(args.screen)
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    summary = analyse(screen)
    payload = {
        "experiment": "target_mlp_oracle_blend",
        "question": "等权 −54.49% 否证的是「等权掺弱模型」还是「MLP 没有独立信息」？",
        "method": "从 target_mlp_screen 的逐折 A/B 反解交叉项 C，闭式解两分量最优配比 peak",
        "caveat": "ORACLE 上界：系数在评估折上重解；基准也弱于生产。不得当作可部署增益。",
        "source": str(screen_path),
        "source_configuration": screen["configuration"],
        "frozen_concession_reference": {
            "source": "outputs/experiments/horizon_auxiliary_cache_probe.json 的 null_frozen_scale 臂",
            "range": list(FROZEN_CONCESSION),
        },
        "summary": summary,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / f"{args.label}.json", out / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} 或 {md_path} 已存在；要覆盖请加 --force")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(payload), encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}")
    print(f"\noracle 折均 {summary['oracle_mean']:+.2%}，正折 {summary['oracle_positive_folds']}/5，"
          f"去最好折 {summary['oracle_drop_best']:+.2%}")
    print(f"冻结系数折算后约 {summary['honest_range_after_frozen_concession'][0]:+.2%} ~ "
          f"{summary['honest_range_after_frozen_concession'][1]:+.2%}")


if __name__ == "__main__":
    main()
