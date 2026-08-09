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

⚠️ 2026-08-09 扩写：折扣系数**不是普适常数，它取决于改动的性质**。
本脚本现在算两条轴：

- **正则强度轴**（alpha 2e6→5e5）：原有的公榜 vs 五把本地尺子，ΔA 比值约 2.2
- **换模型轴**（严格岭回归 → v3_hybrid，加一个结构不同的 LGBM 截面分量）：
  公榜两组两点法 vs `lgbm_blend_unweighted` 的 `blend50_xs_loose`，ΔA 比值约 1.1

两条轴差了一倍，所以「本地打 2.2 折」只对第一类改动成立。

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
#
# ⚠️ 下面这两组都是 **legacy 求解器**（LSQR tol 1e-4/100，sha e2bec9a9…）那一版的点，
# 它们定义的是「正则强度轴」。严格求解器 c23a8cfb… 与 v3_hybrid 的点在 PUBLIC_MODELS 里。
PUBLIC_POINTS: dict[int, list[tuple[float, float]]] = {
    2_000_000: [
        (0.5, 0.00128602),                    # ledger 2026-08-08 rollback 行
        (0.6424065227113341, 0.00151886),     # ledger 2026-08-07 keep 行
        (1.13, 0.00186805),                   # ledger 2026-08-08 keep 行
    ],
    500_000: [
        (0.8, 0.00150852),                    # ledger 2026-08-08 rollback 行
        (1.2, 0.0011693833),                  # ledger 2026-08-08 rollback 行（CSV 按比例缩放）
    ],
}

# 「换模型轴」的公榜点，按模型分组。两点即可解出该模型自己的 A、B、最优 scale、峰值。
PUBLIC_MODELS: dict[str, list[tuple[float, float]]] = {
    "strict_ridge": [
        (1.13, 0.00187232),                   # ledger 2026-08-08 keep 行（严格求解器 c23a8cfb）
        (0.92, 0.0018051540),                 # ledger 2026-08-09 keep 行（同模型，CSV 按比例缩放）
    ],
    "v3_hybrid": [
        (0.90, 0.00213810),                   # ledger 2026-08-09 keep 行
        (1.30, 0.0022857726),                 # ledger 2026-08-09 keep 行（自 base 0.856 缩放）
    ],
}

# ⚠️ `replace`（blend_weight=1.0）只有**一个**公榜点，两点法解不了 —— 但它不需要两点。
#
# A 是 f 的线性泛函，而 f 关于 blend_weight 精确线性
# （f(w) = f(0) + 2w·(f(0.5) − f(0))，3750 行真实数据实测恒等式 max|Δ|=3.9e-08）
# ⟹ A(w) 是一次函数。w=0 恰好**逐位**就是 strict_ridge（m̂ + ê_ridge），w=0.5 是 v3_hybrid，
# 两组已花掉的公榜点把整条 A(w) 钉死 ⟹ A(1) 是精确值，一个点就能反解 B。
# 详见 NOTES「整条 blend_weight 曲线是免费解出来的」。
ONE_POINT_MODELS: dict[str, dict[str, Any]] = {
    "v3_hybrid_replace": {
        "blend_weight": 1.0,
        "point": (1.16, 0.0024872338),        # ledger 2026-08-09 keep 行
        "note": "blend_weight 1.0；A 由 strict_ridge 与 v3_hybrid 线性外推，精确",
    },
}

# A(w) = A(0) + 2w·(A(0.5) − A(0))，两个基点
BLEND_WEIGHT_BASE = ("strict_ridge", "v3_hybrid")


def a_of_blend_weight(solved: dict[str, tuple[float, float, float]], weight: float) -> float:
    """由两个基点线性外推出任意 blend_weight 的 A（精确，不是近似）。"""
    base, half = (solved[name] for name in BLEND_WEIGHT_BASE)
    return base[0] + 2.0 * weight * (half[0] - base[0])


def b_from_one_point(scale: float, score: float, a_value: float) -> float:
    """A 已知时，一个 (scale, score) 点反解 B：Score = 2aA − a²B。"""
    return (2.0 * scale * a_value - score) / scale ** 2

# 换模型轴的本地对照：产物文件 → 该臂的 delta_A_pct / delta_B_pct（读产物，不重算）
LOCAL_COMPONENT_SOURCE = ("lgbm_blend_unweighted.json", "blend50_xs_loose")

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


def model_axis() -> dict[str, Any]:
    """「换模型轴」：严格岭回归 → v3_hybrid。公榜两组两点法 vs 本地装车实验。

    公榜那侧两个模型各自解出 A、B；本地那侧直接读 lgbm_blend_unweighted 的 ΔA/ΔB
    （那是同一次实验里 baseline 臂与 blend50 臂的配对结果，不在这里重算）。

    ⚠️ 保留项：本地的 baseline 是逐折拟合的 ridge（内层选 alpha / phase_balanced），
    公榜的 baseline 是生产 ridge（alpha 2e6 / modulo 5 periodic）——**两个基准不是同一个**，
    所以这是近似对读。但 ΔA 比值 1.07 与正则轴的 2.22 差一倍以上，不是口径噪声能解释的。
    """
    solved = {name: solve_ab(points) for name, points in PUBLIC_MODELS.items()}
    # 一点法的模型：A 由 blend_weight 线性外推（精确），B 反解
    for name, spec in ONE_POINT_MODELS.items():
        a_value = a_of_blend_weight(solved, spec["blend_weight"])
        scale, score = spec["point"]
        solved[name] = (a_value, b_from_one_point(scale, score, a_value), 0.0)
    before, after = solved["strict_ridge"], solved["v3_hybrid"]
    public = {
        "before": {"model": "strict_ridge", "A": before[0], "B": before[1],
                   "best_scale": before[0] / before[1], "peak": before[0] ** 2 / before[1],
                   "residual": before[2]},
        "after": {"model": "v3_hybrid", "A": after[0], "B": after[1],
                  "best_scale": after[0] / after[1], "peak": after[0] ** 2 / after[1],
                  "residual": after[2]},
        "dA": after[0] / before[0] - 1.0,
        "dB": after[1] / before[1] - 1.0,
    }
    public["peak_ratio"] = public["after"]["peak"] / public["before"]["peak"]

    name, arm = LOCAL_COMPONENT_SOURCE
    path = EXPERIMENTS / name
    local: dict[str, Any] | None = None
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        comparison = payload["comparisons"][arm]
        local = {
            "source": name, "arm": arm,
            "dA": comparison["delta_A_pct"] / 100.0,
            "dB": comparison["delta_B_pct"] / 100.0,
            "peak_ratio": 1.0 + comparison["relative_pct"] / 100.0,
        }
    else:
        print(f"跳过换模型轴的本地一侧：{name} 不存在", flush=True)

    table = {
        name: {"A": values[0], "B": values[1], "best_scale": values[0] / values[1],
               "peak": values[0] ** 2 / values[1], "ic": (values[0] ** 2 / values[1]) ** 0.5,
               "solved_from": "两点法" if name in PUBLIC_MODELS else "一点法（A 由线性外推精确定出）"}
        for name, values in solved.items()
    }
    return {"public": public, "local": local, "all_models": table,
            "local_over_public_dA": (local["dA"] / public["dA"]) if local else float("nan"),
            "local_over_public_dB": (local["dB"] / public["dB"]) if local else float("nan")}


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
    discount_b = (
        float(np.mean([e["delta"]["dB"] for e in locals_]) / public["delta"]["dB"])
        if locals_ else float("nan")
    )
    component = model_axis()

    payload = {
        "identity": "Score(a) = 2aA − a²B；A = 峰值/a*，B = 峰值/a*²",
        "meaning": {"A": "Σw·y·f/Σw·y²，信号对齐程度", "B": "Σw·f²/Σw·y²，预测方差"},
        "compared": {"reference_alpha": REFERENCE_ALPHA, "compare_alpha": COMPARE_ALPHA},
        "public_points": {str(k): v for k, v in PUBLIC_POINTS.items()},
        "public_models": {k: v for k, v in PUBLIC_MODELS.items()},
        "rulers": rulers,
        "local_over_public_dA": discount,
        "local_over_public_dB": discount_b,
        "change_axes": {
            "regularisation": {
                "name": "调正则强度（alpha 2e6 → 5e5）",
                "public_dA": public["delta"]["dA"], "public_dB": public["delta"]["dB"],
                "public_peak_ratio": 1.0 / public["delta"]["peak_ratio"],
                "local_dA": float(np.mean([e["delta"]["dA"] for e in locals_])) if locals_ else float("nan"),
                "local_dB": float(np.mean([e["delta"]["dB"] for e in locals_])) if locals_ else float("nan"),
                "local_over_public_dA": discount,
                "local_over_public_dB": discount_b,
            },
            "new_component": {
                "name": "加一个结构不同的模型分量（LGBM 截面块 blend50）",
                "public_dA": component["public"]["dA"], "public_dB": component["public"]["dB"],
                "public_peak_ratio": component["public"]["peak_ratio"],
                "local_dA": component["local"]["dA"] if component["local"] else float("nan"),
                "local_dB": component["local"]["dB"] if component["local"] else float("nan"),
                "local_over_public_dA": component["local_over_public_dA"],
                "local_over_public_dB": component["local_over_public_dB"],
            },
        },
        "model_axis": component,
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
        "## ⭐ 但这个倍数不是普适常数 —— 它取决于改动的性质",
        "",
        "2026-08-09 拿到第二条轴：把 LGBM 的截面分量 blend50 进来（严格岭回归 → v3_hybrid），",
        "公榜两组两点法给出的 ΔA 与本地几乎一样 —— **没有 2.2 倍的高估**。",
        "",
        "| 改动性质 | 公榜 ΔA | 本地 ΔA | **ΔA 比值** | 公榜 ΔB | 本地 ΔB | ΔB 比值 | 公榜峰值比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in payload["change_axes"].values():
        lines.append(
            f"| {axis['name']} | {axis['public_dA']*100:+.1f}% | {axis['local_dA']*100:+.1f}% | "
            f"**{axis['local_over_public_dA']:.2f}** | {axis['public_dB']*100:+.1f}% | "
            f"{axis['local_dB']*100:+.1f}% | {axis['local_over_public_dB']:.2f} | "
            f"{axis['public_peak_ratio']:.3f} |"
        )
    comp_pub = component["public"]
    lines += [
        "",
        "## 公榜上解出来的所有模型",
        "",
        "| 模型 | A | B | 最优 scale | 峰值 | IC | 怎么解的 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, entry in sorted(component["all_models"].items(), key=lambda kv: kv[1]["peak"]):
        lines.append(
            f"| {name} | {entry['A']:.8f} | {entry['B']:.8f} | {entry['best_scale']:.4f} | "
            f"{entry['peak']:.8f} | {entry['ic']:.5f} | {entry['solved_from']} |")
    lines += [
        "",
        "⭐ **`v3_hybrid_replace` 只花了一次额度**：A 关于 `blend_weight` 精确线性，",
        "`strict_ridge`（w=0）与 `v3_hybrid`（w=0.5）两组已花掉的点把 A(1) 钉死，一点反解 B 即可。",
        "这条把「每个模型两次额度」的规矩，在同一个线性族里降到了一次。",
    ]
    lines += [
        "",
        f"判据 `2·ΔA > ΔB`：2×{comp_pub['dA']*100:.2f}% = {2*comp_pub['dA']*100:.2f}% "
        f"vs {comp_pub['dB']*100:.2f}% → "
        f"**{'通过' if 2*comp_pub['dA'] > comp_pub['dB'] else '不通过'}**，"
        f"峰值 {comp_pub['peak_ratio']:.4f} 倍（{(comp_pub['peak_ratio']-1)*100:+.1f}%）。",
        "",
        "⚠️ **保留项**：换模型轴的本地基准是逐折拟合的 ridge（内层选 alpha / phase_balanced），",
        "公榜基准是生产 ridge（alpha 2e6 / modulo 5 periodic）——**两个基准不是同一个**，",
        "所以这是近似对读。但 ΔA 比值差一倍以上是量级差异，不是口径噪声能解释的。",
        "",
        "## 由此得到的工作规则（2026-08-09 修订）",
        "",
        "> 靠**增加信号对齐（A↑）**起作用的改动，本地收益要打折再信 —— **折多少看改动性质**：",
        f"> 调正则强度 / 特征数这类「拧紧拧松同一个模型」的，打约 {discount:.1f} 折；",
        f"> 加一个结构不同的模型分量这类，约 {payload['change_axes']['new_component']['local_over_public_dA']:.1f} 折（几乎不打）。",
        ">",
        "> 靠**降低预测方差（B↓）**起作用的改动，大致 1:1 迁移。",
        ">",
        "> 增加容量（更多列 / 更弱正则 / 更深的树）主要通过 A↑ 起作用 → 在第一类里一律重罚。",
        "",
        "⚠️ 每条轴目前都只有**一个**公榜观测点（正则轴两个 alpha、换模型轴两个模型），",
        "是量级参考不是精确系数。每多一次公榜两点测量都应回填 `PUBLIC_POINTS` / `PUBLIC_MODELS` 重算。",
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
    print(f"\n正则强度轴：本地 ΔA / 公榜 ΔA = {discount:.2f}")
    axis = payload["change_axes"]["new_component"]
    print(f"换模型轴  ：本地 ΔA / 公榜 ΔA = {axis['local_over_public_dA']:.2f} "
          f"（公榜 {axis['public_dA']*100:+.1f}% vs 本地 {axis['local_dA']*100:+.1f}%）")
    print(f"           公榜峰值 {comp_pub['before']['peak']:.8f} → {comp_pub['after']['peak']:.8f} "
          f"（{(comp_pub['peak_ratio']-1)*100:+.1f}%）")
    print(f"报告 → {EXPERIMENTS / 'ab_decomposition.md'}")


if __name__ == "__main__":
    main()
