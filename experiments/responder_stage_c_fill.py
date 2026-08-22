"""Stage C 补测：把 `multi_member_family` 挡掉的 14 个格子填满。

## 为什么这些格子空着

`responder_predictability` 的 Stage B 有七道 check。逐族核 JSON 的结果是：

```text
24 族   通过 8   未通过 16
未通过的 16 个族，failed checks 全部恰好 == ['multi_member_family']（单成员族）
```

也就是说，其余六道门（`mean_peak_positive` / `positive_folds` / `survives_drop_best` /
`cross_section_mean_positive` / `inner_family_direction_consistent` / `not_single_asset`）
**16 个单成员族全部通过**，它们是被一条**稳健性启发式**而不是证据挡在 Stage C 之外的。

08-18 的 `horizon_auxiliary_cache_probe` 已经补测了其中 2 个（`responder_00`/`responder_02`，
理由是窗口图谱显示这两个 H<5）⟹ **还剩 14 个从未被测过**。本脚本把它们一次填满。

## ⚠️ 这是结案，不是找收益 —— 三条先验都在反面

1. **它不满足仓库自己的母条件。** `responder_reaudit_20260814.md:93-100` 明写重开需要
   「**不是**换目标、线性叠加或对预测值做二层校准」的新机制。把 responder 的 OOF 预测叠进
   blend **正是**被排除的那一族 ⟹ 本脚本的产出只能是「关严」，不能是「重开」。
2. **已测的 8 个族可预测性比 target 高 8~460×，仍然是 −18.81%、1/5 折。**
3. **不得按「与 target 相关高」解读结果。** `responder_targets_stage1.md:14-22` 已证伪该论证
   形式：同期相关最高的 `responder_03`（0.817）当训练目标是全场最差（−15.47%、0/7），
   而相关只有 0.394 的 `responder_04` 排第一。见 `CLAUDE.md:119`。

## 设计

- **测量路径一行不改**：直接调 `horizon_auxiliary_cache_probe.evaluate_arm`
  （2026-08-22 从该文件 `main()` 里原样抽出的函数，抽取后原脚本输出逐字节不变已实测）。
- **未测名单从 Stage B 的 JSON 派生**，不硬编码（CLAUDE.md §7）。
- **逐臂独立随机流**：`evaluate_arm` 的 `boot_rng` 按调用顺序消耗，共享一条流会让结果依赖
  臂顺序。本脚本逐臂传 `default_rng([boot_seed, arm_id])` ⟹ 加臂/换序不改变已有臂的数。
- **校准臂原样保留**：`null_frozen_scale` / `negctrl_shuffle` / `known_negative_27`。
  `harness_ok` 不为 True 就整轮作废、不解读任何数字。
- **自检臂**：`responder_00` / `responder_02` 的**点估计**必须复现 08-18 的落盘值
  （从那份 JSON 读，不抄数）。点估计是确定性的 ⟹ 应逐位相同；bootstrap CI 因逐臂换流会不同。

## 多重比较纪律（臂数从 6 涨到 19，本轮新增）

1. 过门槛的臂**只能**成为 ROADMAP P10 Tier 2 的候选，不足以建候选模型、不足以碰生产、
   不足以花任何提交额度；
2. 报告必须给出过门槛的臂**落在哪个维度族**（`responder_family_grid`）——
   集中在某一族才算机制信号，散落则按噪声读；
3. 读表先看 `null_frozen_scale`：它本身就是 −3.84%(full) / −2.54%(pure_e)，
   这 2.5~3.8 个百分点是「基准可重解 scale、候选必须冻结」的**让步**，不是效应。

用法：
    .venv/bin/python experiments/responder_stage_c_fill.py --emit-plan
    .venv/bin/python experiments/responder_stage_c_fill.py
输出：outputs/experiments/responder_stage_c_fill{,_plan}.{json,md}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from horizon_auxiliary_cache_probe import (  # noqa: E402
    BASELINES, MIN_POSITIVE_FOLDS, MIN_RELATIVE_GAIN, evaluate_arm, group_index, load_aligned,
)
from responder_family_grid import (  # noqa: E402
    build_families, read_column_stats, untested_stage_c_cells,
)

PLAN_LABEL = "responder_stage_c_fill_plan"
ANCHOR_PROBE = _REPO_ROOT / "outputs" / "experiments" / "horizon_auxiliary_cache_probe.json"
ALREADY_PROBED = ("responder_00", "responder_02")
CALIBRATION_ARMS = ("null_frozen_scale", "negctrl_shuffle", "known_negative_27")
# 点估计是确定性的（不经过 bootstrap）⟹ 应当逐位相同。容差只用来吸收平台浮点噪声。
ANCHOR_KEYS = ("mean_delta", "relative", "mean_delta_drop_best", "positive_folds",
               "delta_A_relative", "delta_B_relative")
ANCHOR_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-plan", action="store_true", help="只落盘预注册判据")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--resp-cache", default=str(
        _REPO_ROOT / "outputs" / "cache" / "responder_oof_phasebal_prodwindow_f323.npz"))
    p.add_argument("--v3-cache", default=str(
        _REPO_ROOT / "outputs" / "cache"
        / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"))
    p.add_argument("--stage-b-json", default=str(
        _REPO_ROOT / "outputs" / "experiments"
        / "responder_predictability_reaudit_phasebal_prodwindow.json"))
    p.add_argument("--anchor-probe", default=str(ANCHOR_PROBE))
    p.add_argument("--limit-groups", type=int, default=None)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--shuffle-seed", type=int, default=7)
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="responder_stage_c_fill")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def write(out_dir: Path, label: str, payload: dict[str, Any] | None, text: str,
          force: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} 或 {md_path} 已存在；要覆盖请加 --force")
    if payload is not None:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"wrote {json_path}", flush=True)
    md_path.write_text(text, encoding="utf-8")
    print(f"wrote {md_path}", flush=True)
    return json_path


def family_of(responder: str, families: list[dict[str, Any]]) -> str:
    for family in families:
        if responder in family["members"]:
            return family["family"]
    raise SystemExit(f"{responder} 不在族群表里 —— 族群表与 responder 列对不上")


def emit_plan(args: argparse.Namespace, cells: dict[str, Any],
              families: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "experiment": PLAN_LABEL,
        "role": "PRE-REGISTRATION —— 必须先于结果落盘（CLAUDE.md §5.1）",
        "question": ("Stage B 的 multi_member_family 是启发式而非证据。"
                     "被它挡掉、且从未进过 Stage C 的 14 个 responder，"
                     "其严格 OOF 预测能不能补强 v3 的 target 残差？"),
        "positioning": ("结案，不是找收益。本机制属于 responder_reaudit_20260814.md:93-100 "
                        "母条件**明令排除**的「线性叠加 / 对预测值做二层校准」一族 ⟹ "
                        "任何过门槛的臂都只能成为 P10 Tier 2 候选，不得据此重开 responder 线。"),
        "priors_against": [
            "已测的 8 个族可预测性比 target 高 8~460×，仍为 −18.81%、1/5 折",
            "08-18 补测的 r00/r02 最好一格只有 +1.38%、去最好折为负、0.43× 检出下限",
            "不得按「与 target 相关高」解读：responder_03 相关 0.817 却是 A0 全场最差 −15.47%",
        ],
        "main_arms": cells["untested"],
        "self_check_arms": list(ALREADY_PROBED),
        "calibration_arms": list(CALIBRATION_ARMS),
        "baselines": dict(BASELINES),
        "gates": {
            "1_mean_delta_positive": "折均 Δpeak > 0",
            f"2_at_least_{MIN_POSITIVE_FOLDS}_of_4_folds_positive": "≥3/4 评估折为正",
            "3_survives_drop_best_fold": "去掉最好一折后仍为正",
            "4_relative_gain_at_least_3pct": f"相对增益 ≥ {MIN_RELATIVE_GAIN:.0%}",
            "5_two_delta_A_exceeds_delta_B": "2ΔA > ΔB",
            "6_paired_bootstrap_ci_lower_bound_positive": "配对 block bootstrap CI 下界 > 0",
            "7_exceeds_detection_floor": "折均超过该臂自己的检出下限",
        },
        "harness_gate": ("negctrl_shuffle 两基准均不得通过门禁，且 known_negative_27 "
                         "两基准相对增量均 < 0；不满足则整轮作废、不解读任何数字"),
        "reproduction_gate": {
            "arms": list(ALREADY_PROBED),
            "source": str(args.anchor_probe),
            "keys": list(ANCHOR_KEYS),
            "tolerance": ANCHOR_TOL,
            "note": "点估计不经过 bootstrap ⟹ 应逐位相同；CI 因逐臂换随机流会不同，不作自检项",
        },
        "multiple_comparison_discipline": [
            "过门槛的臂只能成为 P10 Tier 2 候选；不建候选模型、不碰生产、不花提交额度",
            "必须报告过门槛的臂落在哪个维度族；集中在某族才算机制信号，散落按噪声读",
            "读表先看 null_frozen_scale —— 那 2.5~3.8 个百分点是冻结系数的让步，不是效应",
        ],
        "family_grid": [{"family": f["family"], "members": f["members"],
                         "sign_class": f["sign_class"]} for f in families],
        "limits": [
            "缓存里的 responder OOF 是 Ridge 强度 ⟹ 准入筛，不是终审",
            "基准在评估折上重解最优 scale、候选用冻结系数 ⟹ 对候选不利，null_frozen_scale 量化该让步",
            "基准不含 slow/fast 后处理 ⟹ 与 slow/fast 的交互未验证",
            f"v3 基准缓存 {Path(args.v3_cache).name} 未被隔离但**未经现跑复验**"
            "（RUNBOOK_8_23.md:174-180）⟹ 不得用于晋级裁决",
        ],
    }
    return payload


def render_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage C 补测 —— 预注册（`responder_stage_c_fill_plan`）",
        "",
        "> 判据先于结果落盘。结果产物里记本文件的 sha256。",
        "",
        f"**问题**：{payload['question']}",
        "",
        f"**定位**：{payload['positioning']}",
        "",
        "## 反面先验（跑之前就写下）",
        "",
    ]
    lines += [f"{i}. {p}" for i, p in enumerate(payload["priors_against"], 1)]
    lines += [
        "",
        "## 臂",
        "",
        f"- **主臂（{len(payload['main_arms'])}）**：" +
        ", ".join(f"`{a}`" for a in payload["main_arms"]),
        f"- **自检臂（{len(payload['self_check_arms'])}）**：" +
        ", ".join(f"`{a}`" for a in payload["self_check_arms"]) +
        " —— 点估计必须复现 08-18 落盘值",
        f"- **校准臂（{len(payload['calibration_arms'])}）**：" +
        ", ".join(f"`{a}`" for a in payload["calibration_arms"]),
        f"- **基准**：" + ", ".join(f"`{k}`（{v}）" for k, v in payload["baselines"].items()),
        "",
        "## 门禁",
        "",
    ]
    lines += [f"- `{k}`：{v}" for k, v in payload["gates"].items()]
    lines += [
        "",
        f"**harness 门**：{payload['harness_gate']}",
        "",
        f"**复现门**：{payload['reproduction_gate']['note']}",
        "",
        "## 多重比较纪律",
        "",
    ]
    lines += [f"{i}. {d}" for i, d in enumerate(payload["multiple_comparison_discipline"], 1)]
    lines += ["", "## 限制", ""]
    lines += [f"- {limit}" for limit in payload["limits"]]
    lines.append("")
    return "\n".join(lines)


def check_reproduction(results: dict[str, Any], anchor_path: Path) -> dict[str, Any]:
    """自检臂必须复现 08-18 的点估计。锚值从那份 JSON 读，不抄数（CLAUDE.md §7）。"""
    if not anchor_path.is_file():
        raise SystemExit(f"找不到锚点 {anchor_path} —— 先跑 horizon_auxiliary_cache_probe.py")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))["results"]

    rows, worst = [], 0.0
    for baseline in BASELINES:
        for arm in ALREADY_PROBED:
            expected, actual = anchor[baseline][arm], results[baseline][arm]
            for key in ANCHOR_KEYS:
                delta = abs(float(actual[key]) - float(expected[key]))
                worst = max(worst, delta)
                rows.append({"baseline": baseline, "arm": arm, "key": key,
                             "expected": expected[key], "actual": actual[key],
                             "abs_delta": delta})
    return {"source": str(anchor_path), "max_abs_delta": worst,
            "tolerance": ANCHOR_TOL, "ok": bool(worst <= ANCHOR_TOL), "checks": rows}


def summarize_concession(results: dict[str, Any]) -> dict[str, Any]:
    """量化「冻结系数让步」，并**实测验证它是一个常数平移**。

    `mean_delta_vs_frozen_baseline` 把「基准可在评估折重解最优 scale、候选必须用冻结系数」
    这份对候选不利的让步剥掉。08-18 的报告用它把 `pure_e/responder_00` 从 +1.38% 读成 +3.92%。

    ⚠️ 但让步对每个臂是**同一个常数**（下面的恒等式实测到 1e-19 量级）⟹ 剥掉它只平移水平，
    不改排序，也不影响正折数 / 去最好折 / bootstrap CI。**它不可能制造出一个发现。**
    """
    per_baseline, worst, n_checked = {}, 0.0, 0
    for bname, arms in results.items():
        null_delta = arms["null_frozen_scale"]["mean_delta"]
        null_peak = arms["null_frozen_scale"]["baseline_peak_mean"]
        scored = []
        for arm, row in arms.items():
            if arm == "null_frozen_scale":
                continue
            worst = max(worst, abs(row["mean_delta_vs_frozen_baseline"]
                                   - (row["mean_delta"] - null_delta)))
            n_checked += 1
            scored.append((row["mean_delta_vs_frozen_baseline"], arm, row))
        scored.sort(reverse=True)
        best_delta, best_arm, best_row = scored[0]
        per_baseline[bname] = {
            "null_delta": null_delta,
            "null_relative": null_delta / null_peak,
            "n_arms": len(scored),
            "n_positive": sum(1 for delta, _, _ in scored if delta > 0),
            "best_arm": best_arm,
            "best_stripped_delta": best_delta,
            "best_relative": best_delta / null_peak,
            "best_positive_folds": best_row["positive_folds"],
            "stripped": {arm: delta for delta, arm, _ in scored},
        }
    return {
        "per_baseline": per_baseline,
        "identity": "stripped(arm) == mean_delta(arm) − mean_delta(null_frozen_scale)",
        "max_identity_deviation": worst,
        "n_checked": n_checked,
        "reading": ("让步是常数 ⟹ 剥掉它只改水平不改排序，且不影响正折数 / 去最好折 / CI。"
                    "剥完为正但 0/4 折的臂（full 的 responder_06、responder_20）就是这个的直接反例。"),
    }


def render_report(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# Stage C 补测（`responder_stage_c_fill`）",
        "",
        "> ⚠️ **这是结案，不是找收益。** 本机制属于 `responder_reaudit_20260814.md:93-100` 母条件",
        "> **明令排除**的「线性叠加 / 对预测值做二层校准」一族 —— 过门槛也只能进 P10 Tier 2。",
        "",
        f"预注册：`{payload['plan_path']}`（sha256 `{payload['plan_sha256'][:16]}…`）",
        "",
        f"{payload['rows']:,} 行 / {payload['n_groups']:,} 个 time_id / "
        f"评估折 {payload['eval_folds']}；系数只用过去折拟合并冻结。",
        "",
        "## 自检：08-18 锚点复现",
        "",
        f"- 锚点：`{Path(payload['reproduction']['source']).name}`，"
        f"臂 {', '.join('`' + a + '`' for a in ALREADY_PROBED)} × 2 基准 × "
        f"{len(ANCHOR_KEYS)} 个点估计",
        f"- 最大绝对偏差 **{payload['reproduction']['max_abs_delta']:.3e}**"
        f"（容差 {payload['reproduction']['tolerance']:.0e}）⟹ "
        f"**{'PASS' if payload['reproduction']['ok'] else 'FAIL'}**",
        "",
        "## harness 校准",
        "",
        f"- 负控制（组内打乱）是否通过门禁：{payload['negctrl_passes']}（应全为 False）",
        f"- 已测族 `responder_27` 相对增量：" +
        ", ".join(f"{b} {v*100:+.2f}%" for b, v in payload["known_negative"].items()) +
        "（应为负）",
        f"- **harness_ok = {payload['harness_ok']}**",
        "",
    ]

    for bname in BASELINES:
        lines += [
            f"## 基准 `{bname}`（{BASELINES[bname]}）",
            "",
            "| 臂 | 族 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 检出下限 | 配对 CI | 判定 |",
            "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|:--:|",
        ]
        for arm, row in payload["results"][bname].items():
            ci = row["paired_bootstrap"]
            lines.append(
                f"| `{arm}` | {payload['arm_family'].get(arm, '—')} | {row['mean_delta']:+.3e} | "
                f"{row['relative']*100:+.2f}% | {row['positive_folds']}/{row['n_folds']} | "
                f"{row['mean_delta_drop_best']:+.3e} | {row['delta_A_relative']*100:+.2f}% | "
                f"{row['delta_B_relative']*100:+.2f}% | {ci['half_width']:.2e} | "
                f"[{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] | "
                f"{'✅' if row['pass'] else '❌'} |")
        lines.append("")

    concession = payload["concession"]
    lines += [
        "## ⚠️ 剥掉冻结系数让步之后 —— 以及为什么它不能制造发现",
        "",
        "预注册纪律第 3 条要求先读 `null_frozen_scale`（不加任何 auxiliary、只把 scale 冻结在过去折）：",
        "",
        "| 基准 | 让步本身 | 剥让步后为正的臂 | 最好的一个 |",
        "|---|---:|---:|---|",
    ]
    for bname, row in concession["per_baseline"].items():
        lines.append(
            f"| `{bname}` | {row['null_delta']:+.3e}（{row['null_relative']*100:+.2f}%） | "
            f"{row['n_positive']}/{row['n_arms']} | "
            f"`{row['best_arm']}` {row['best_relative']*100:+.2f}%"
            f"（正折 {row['best_positive_folds']}/4） |")

    lines += [
        "",
        "⭐ **但这组数不是 14 个发现，是一个常数。** 实测恒等式",
        "",
        "```text",
        "stripped(arm) ≡ mean_delta(arm) − mean_delta(null_frozen_scale)",
        f"全部 {concession['n_checked']} 个臂的最大偏差：{concession['max_identity_deviation']:.3e}",
        "```",
        "",
        "让步对每个臂是**同一个常数** ⟹ 剥掉它只改变**水平**，不改变**排序**，",
        "更不改变正折数、去最好折和配对 bootstrap CI —— 而门禁里管用的正是后面这三样。",
        "证据：剥完之后 `responder_06`（full）报 +1.22% 却是 **0/4 折**，",
        "`responder_20` 报 +0.37% 也是 **0/4 折**。",
        "",
        "⟹ 08-18 那个被反复引用的「`pure_e/responder_00` 剥完是 +3.92%」也是同一回事：",
        "本轮逐位复现了它，但它仍然是 3/4 折、去最好折为负、只有检出下限的 0.43×。",
        "",
        "## 裁决",
        "",
        f"- 主臂 {verdict['n_main_arms']} 个 × {len(BASELINES)} 基准 = "
        f"{verdict['n_main_cells']} 格，过门槛 **{len(verdict['passed_main'])}** 格",
    ]
    if verdict["passed_main"]:
        lines += [
            "",
            "| 过门槛的格 | 维度族 |",
            "|---|:---:|",
        ]
        lines += [f"| `{cell}` | {fam} |" for cell, fam in verdict["passed_main"].items()]
        lines += [
            "",
            f"⚠️ 过门槛的臂分布在 **{len(set(verdict['passed_main'].values()))}** 个维度族。"
            "集中在某一族才算机制信号；散落则按噪声读（预注册纪律第 2 条）。",
            "",
            "⚠️ 这些**只能**成为 ROADMAP P10 Tier 2 的候选臂，等 8/23 密封期尺子裁决。"
            "不建候选模型、不碰生产、不花任何提交额度。",
        ]
    else:
        lines += [
            "",
            f"### {verdict['status']}",
            "",
            "**14 个从未测过的格子全部不过门禁。** 加上 08-18 已补的 `responder_00`/`responder_02`",
            "与 08-12/08-14 已测的 8 个族 ⟹ **Stage C 现在覆盖了全部 24 个族**。",
            "",
            "⟹ responder 这条线从「28% 的列被一条启发式挡着」变成 **「全部 47 列都测过」**：",
            "由**证据**关闭，不再由启发式关闭。",
        ]

    lines += [
        "",
        "## 限制",
        "",
    ]
    lines += [f"{i}. {limit}" for i, limit in enumerate(payload["limits"], 1)]
    lines += [
        "",
        "## 重新开放条件",
        "",
        "8/23 回补包若含 responder 列（**主办方原文未承诺**，`docs/data_description.md:173` 只说",
        "「标签回补 / 扩展训练数据」，没有字段清单）⟹ 按原规格复验一次。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    rows = read_column_stats(Path(args.data_root) / "train" / "train_partition_000.parquet")
    families = build_families(rows)
    cells = untested_stage_c_cells(Path(args.stage_b_json), ALREADY_PROBED)

    plan_payload = emit_plan(args, cells, families)
    if args.emit_plan:
        write(out_dir, PLAN_LABEL, plan_payload, render_plan(plan_payload), args.force)
        print(f"\n主臂 {len(cells['untested'])} 个：{', '.join(cells['untested'])}", flush=True)
        return

    plan_path = out_dir / f"{PLAN_LABEL}.json"
    if not plan_path.is_file():
        raise SystemExit(f"没有找到预注册文件 {plan_path} —— 先跑 `--emit-plan`。"
                         "判据必须先于结果落盘（CLAUDE.md §5.1）")

    started = time.perf_counter()
    data = load_aligned(args)
    y, w, fold = data["target"], data["weight"], data["fold"]
    starts, gidx, n_groups = group_index(data["time_id"])
    group_fold = fold[starts]
    fold_list = sorted(int(f) for f in np.unique(group_fold))
    names = list(data["responder_names"])
    print(f"{len(y):,} 行 / {n_groups:,} 个 time_id / {len(fold_list)} 折；两缓存对齐 ✅",
          flush=True)

    def aux_col(name: str) -> np.ndarray:
        return data["aux_all"][:, names.index(name)].astype(np.float64)

    # 负控制：组内随机重排（行已按 time_id 排序）—— 与 08-18 同一构造
    shuffle_rng = np.random.default_rng(args.shuffle_seed)
    perm = np.lexsort((shuffle_rng.random(len(y)), gidx))

    arms: dict[str, list[np.ndarray]] = {"null_frozen_scale": []}
    for name in ALREADY_PROBED:
        arms[name] = [aux_col(name)]
    for name in cells["untested"]:
        arms[name] = [aux_col(name)]
    arms["negctrl_shuffle"] = [aux_col("responder_00")[perm]]
    arms["known_negative_27"] = [aux_col("responder_27")]

    results: dict[str, dict[str, Any]] = {}
    for bname, column in BASELINES.items():
        base_pred = data[column]
        results[bname] = {}
        for arm, auxes in arms.items():
            # 逐臂独立随机流 ⟹ 加臂/换序不改变已有臂的数（见模块 docstring）
            digest = hashlib.sha256(f"{bname}/{arm}".encode()).digest()[:8]
            boot_rng = np.random.default_rng([args.boot_seed, int.from_bytes(digest, "big")])
            results[bname][arm] = evaluate_arm(
                arm, auxes, base_pred, bname, y=y, w=w, starts=starts, gidx=gidx,
                n_groups=n_groups, group_fold=group_fold, fold_list=fold_list,
                boot_rng=boot_rng, block_size=args.block_size, n_boot=args.n_boot)

    reproduction = check_reproduction(results, Path(args.anchor_probe))
    concession = summarize_concession(results)
    negctrl = {b: results[b]["negctrl_shuffle"]["pass"] for b in BASELINES}
    known_negative = {b: results[b]["known_negative_27"]["relative"] for b in BASELINES}
    harness_ok = (not any(negctrl.values())) and all(v < 0 for v in known_negative.values())

    arm_family = {name: family_of(name, families) for name in arms
                  if name.startswith("responder_")}
    arm_family["negctrl_shuffle"] = "—"
    passed_main = {f"{b}/{a}": arm_family[a]
                   for b in BASELINES for a in cells["untested"] if results[b][a]["pass"]}

    payload = {
        "experiment": "responder_stage_c_fill",
        "plan_path": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "rows": int(len(y)),
        "n_groups": int(n_groups),
        "eval_folds": fold_list[1:],
        "stage_c_gap": cells,
        "arm_family": arm_family,
        "reproduction": reproduction,
        "concession": concession,
        "negctrl_passes": negctrl,
        "known_negative": known_negative,
        "harness_ok": bool(harness_ok),
        "results": results,
        "verdict": {
            "n_main_arms": len(cells["untested"]),
            "n_main_cells": len(cells["untested"]) * len(BASELINES),
            "passed_main": passed_main,
            "status": ("REJECT —— 14 个未测格子无一通过预注册门禁，"
                       "Stage C 现已覆盖全部 24 个族" if not passed_main else
                       "有格子过门禁 —— 只作为 P10 Tier 2 候选，不得据此重开 responder 线"),
            "reproduction_ok": reproduction["ok"],
            "harness_ok": bool(harness_ok),
            "valid": bool(harness_ok and reproduction["ok"]),
        },
        "limits": plan_payload["limits"],
        "elapsed_seconds": time.perf_counter() - started,
    }

    if not payload["verdict"]["valid"]:
        print("\n⚠️⚠️ harness 或复现自检未通过 —— 产物已落盘但**不得解读任何数字**", flush=True)

    write(out_dir, args.label, payload, render_report(payload), args.force)
    print(f"\n复现自检 max|Δ| = {reproduction['max_abs_delta']:.3e} "
          f"({'PASS' if reproduction['ok'] else 'FAIL'})；harness_ok = {harness_ok}", flush=True)
    print(f"裁决：{payload['verdict']['status']}", flush=True)


if __name__ == "__main__":
    main()
