"""把「本地为什么和公榜不一致」分解到具体的项上。

分数关于 scale 是精确二次式（不触限幅时）：

    Score(a) = 2aA − a²B      A = Σw·y·f / Σw·y²   （信号对齐程度）
                              B = Σw·f² / Σw·y²    （预测方差）

于是 `最优scale a* = A/B`、`峰值 = A²/B`，反过来：

    A = 峰值 / a*        B = 峰值 / a*²

**所以任何已有结果都能反解出 A 和 B，不用重跑模型。** 本脚本只读 json。

这么拆的价值：alpha 从 2e6 降到 5e5 时，五把本地尺子和公榜在 ΔB 上几乎一致，
分歧**完全在 ΔA** —— 本地一致高估「放松正则能让信号变强多少」约 2.4 倍。
由此得到的工作规则见 NOTES.md「A/B 分解」一节。

用法：.venv/bin/python experiments/ab_decomposition.py
输出：outputs/experiments/ab_decomposition.{json,md}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXPERIMENTS = _REPO_ROOT / "outputs" / "experiments"

# 配置名里的 alpha 缩写 → 真实 alpha（与 walk_forward_rolling.CONFIGS 的网格循环一致）
ALPHA_TAGS = {"a8": 250_000, "a4": 500_000, "a2": 1_000_000,
              "a1": 2_000_000, "x2": 4_000_000, "x4": 8_000_000}

# 公榜实测点。每条都对应 experiments/ledger.csv 里的一次提交。
# 同一个模型的多个 scale 点即可解出该模型的 A、B。
PUBLIC_POINTS: dict[int, list[tuple[float, float]]] = {
    2_000_000: [
        (0.5, 0.00128602),                    # ledger 2026-08-08 rollback 行
        (0.6424065227113341, 0.00151886),     # ledger 2026-08-07 keep 行
        (1.13, 0.00186805),                   # ledger 2026-08-08 keep 行（当前上线）
    ],
    500_000: [
        (0.8, 0.00150852),                    # ledger 2026-08-08 rollback 行
        (1.2, 0.0011693833),                  # ledger 2026-08-08 rollback 行（CSV 按比例缩放）
    ],
}

# 本地尺子：标签 → (产物文件, 一句话说明)
LOCAL_RULERS = [
    ("modulo10", "ab_shrinkgrid.json", "modulo 10，只评相位 0"),
    ("modulo5", "ab_shrinkgrid_m5.json", "modulo 5，评相位 0+5"),
    ("相位隔离", "ab_phase_gen_alpha.json", "训练相位 0+5，验证相位 1+6（没见过）"),
    ("横跨f3", "ab_horizon_f3.json", "验证段 164,490 个 time_id（公榜的 77%）"),
    ("横跨f2", "ab_horizon_f2.json", "验证段 246,740 个 time_id（公榜的 115%）"),
]

REFERENCE_ALPHA = 2_000_000
COMPARE_ALPHA = 500_000


def solve_ab(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """多个 (scale, score) 点 → (A, B, 最大残差)。两点即可定，多点用最小二乘。"""
    scales = np.array([p[0] for p in points], dtype=np.float64)
    scores = np.array([p[1] for p in points], dtype=np.float64)
    matrix = np.column_stack([2 * scales, -scales * scales])
    (a_value, b_value), *_ = np.linalg.lstsq(matrix, scores, rcond=None)
    residual = float(np.abs(matrix @ np.array([a_value, b_value]) - scores).max())
    return float(a_value), float(b_value), residual


def local_ab(path: Path, feature_count: int = 200) -> dict[int, tuple[float, float, float]]:
    """从一个 ab_*.json 里，按 alpha 分组、用它的多个 scale 臂解出 A、B。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[int, dict[float, float]] = defaultdict(dict)
    for arm in list(payload["arms"])[1:]:
        parts = arm.split("_")
        if len(parts) != 3 or not parts[0].startswith("g"):
            continue
        count, tag, scale = parts
        if int(count[1:]) != feature_count or tag not in ALPHA_TAGS:
            continue
        grouped[ALPHA_TAGS[tag]][int(scale[1:]) / 100] = float(
            np.mean([fold["scores"][arm] for fold in payload["folds"]])
        )
    return {
        alpha: solve_ab(sorted(points.items()))
        for alpha, points in grouped.items()
        if len(points) >= 2
    }


def main() -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    rulers: list[dict[str, Any]] = []

    a_pub, b_pub, res_pub = solve_ab(PUBLIC_POINTS[REFERENCE_ALPHA])
    a_pub2, b_pub2, res_pub2 = solve_ab(PUBLIC_POINTS[COMPARE_ALPHA])
    rulers.append({
        "ruler": "公榜", "note": "全部 10 个相位，214,538 个 time_id",
        "by_alpha": {
            str(REFERENCE_ALPHA): {"A": a_pub, "B": b_pub, "residual": res_pub},
            str(COMPARE_ALPHA): {"A": a_pub2, "B": b_pub2, "residual": res_pub2},
        },
    })

    for label, name, note in LOCAL_RULERS:
        path = EXPERIMENTS / name
        if not path.exists():
            print(f"跳过 {label}：{name} 不存在", flush=True)
            continue
        table = local_ab(path)
        if REFERENCE_ALPHA not in table or COMPARE_ALPHA not in table:
            print(f"跳过 {label}：缺 alpha={REFERENCE_ALPHA} 或 {COMPARE_ALPHA} 的臂", flush=True)
            continue
        rulers.append({
            "ruler": label, "note": note, "source": name,
            "by_alpha": {
                str(alpha): {"A": values[0], "B": values[1], "residual": values[2]}
                for alpha, values in sorted(table.items())
            },
        })

    def deltas(entry: dict[str, Any]) -> dict[str, float]:
        ref = entry["by_alpha"][str(REFERENCE_ALPHA)]
        cmp_ = entry["by_alpha"][str(COMPARE_ALPHA)]
        return {
            "dA": cmp_["A"] / ref["A"] - 1.0,
            "dB": cmp_["B"] / ref["B"] - 1.0,
            "peak_ratio": (ref["A"] ** 2 / ref["B"]) / (cmp_["A"] ** 2 / cmp_["B"]),
        }

    for entry in rulers:
        entry["delta"] = deltas(entry)

    public = rulers[0]
    locals_ = rulers[1:]
    discount = (
        float(np.mean([e["delta"]["dA"] for e in locals_]) / public["delta"]["dA"])
        if locals_ else float("nan")
    )

    payload = {
        "identity": "Score(a) = 2aA − a²B；A = 峰值/a*，B = 峰值/a*²",
        "meaning": {"A": "Σw·y·f/Σw·y²，信号对齐程度", "B": "Σw·f²/Σw·y²，预测方差"},
        "compared": {"reference_alpha": REFERENCE_ALPHA, "compare_alpha": COMPARE_ALPHA},
        "public_points": {str(k): v for k, v in PUBLIC_POINTS.items()},
        "rulers": rulers,
        "local_over_public_dA": discount,
    }
    (EXPERIMENTS / "ab_decomposition.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# A/B 分解：本地与公榜的分歧到底在哪一项",
        "",
        "`Score(a) = 2aA − a²B`，`A = Σw·y·f/Σw·y²`（信号对齐）、`B = Σw·f²/Σw·y²`（预测方差）。",
        "`最优scale = A/B`、`峰值 = A²/B` ⟹ `A = 峰值/a*`、`B = 峰值/a*²`，",
        "**所以任何已有结果都能反解出 A、B，不用重跑。**",
        "",
        f"下表比较 alpha 从 {REFERENCE_ALPHA:,} 降到 {COMPARE_ALPHA:,} 时两项各自怎么动。",
        "",
        "| 尺子 | A(2e6) | A(5e5) | **ΔA** | B(2e6) | B(5e5) | ΔB | 峰值比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in rulers:
        ref = entry["by_alpha"][str(REFERENCE_ALPHA)]
        cmp_ = entry["by_alpha"][str(COMPARE_ALPHA)]
        d = entry["delta"]
        bold = "**" if entry["ruler"] == "公榜" else ""
        lines.append(
            f"| {bold}{entry['ruler']}{bold} | {ref['A']:.6f} | {cmp_['A']:.6f} | "
            f"{bold}{d['dA']*100:+.1f}%{bold} | {ref['B']:.6f} | {cmp_['B']:.6f} | "
            f"{d['dB']*100:+.1f}% | {d['peak_ratio']:.3f} |"
        )

    lines += [
        "",
        "## 读法",
        "",
        "- **ΔB 五把本地尺子与公榜几乎一致** → 预测方差对正则的响应，本地量得准",
        "- **分歧完全在 ΔA** → 本地一致高估「放松正则能让信号变强多少」，",
        f"  倍数 = 本地 ΔA 均值 / 公榜 ΔA = **{discount:.2f}**",
        "",
        "## 由此得到的工作规则",
        "",
        "> 靠**增加信号对齐（A↑）**起作用的改动，本地测出的收益要打约 "
        f"{discount:.1f} 折再信；",
        "> 靠**降低预测方差（B↓）**起作用的改动，大致 1:1 迁移。",
        ">",
        "> 增加容量（更多列 / 更弱正则 / 更深的树）主要通过 A↑ 起作用 → 一律重罚。",
        "",
        "⚠️ 这个倍数目前只由**两个 alpha、两次公榜双点测量**支撑，是量级参考不是精确系数。",
        "每多一次公榜两点测量都应回填 `PUBLIC_POINTS` 重算。",
        "",
        "## 各尺子说明",
        "",
    ]
    for entry in rulers:
        source = f"（`{entry['source']}`）" if "source" in entry else ""
        lines.append(f"- **{entry['ruler']}**{source}：{entry['note']}")
    lines += [
        "",
        f"抛物线拟合残差最大 {max(v['residual'] for e in rulers for v in e['by_alpha'].values()):.2e}"
        "（应 ~1e-12；大了说明触了限幅，二次式不再精确）。",
    ]
    (EXPERIMENTS / "ab_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'尺子':<10}{'ΔA':>10}{'ΔB':>10}{'峰值比':>9}")
    print("-" * 40)
    for entry in rulers:
        d = entry["delta"]
        print(f"{entry['ruler']:<10}{d['dA']*100:>+9.1f}%{d['dB']*100:>+9.1f}%{d['peak_ratio']:>9.3f}")
    print(f"\n本地 ΔA / 公榜 ΔA = {discount:.2f}")
    print(f"报告 → {EXPERIMENTS / 'ab_decomposition.md'}")


if __name__ == "__main__":
    main()
