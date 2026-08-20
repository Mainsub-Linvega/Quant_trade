"""NN 独立能力阶梯：把「MLP 只有树的 20%」从**预算事实**变成**能力事实**。

## 为什么需要这条

`target_mlp_screen`（08-12）与 `multitask_mlp_stage1`（08-19）都报过 MLP 的**独立** peak：
折均 24.3%（弱基准）、17.4%/20.3%（生产 3s480 基准）。看起来 NN「只有树的 20%」。

但两次都是 `max_iter=12`，且 `tol=0.0` + `n_iter_no_change=max_iter+1`
⟹ **早停是关掉的，12 次不是收敛，是跑完预算被掐断**（JSON 里五折的 `iterations` 全是 12）。
cross 头只有 `(64,32)`、约 2.6 万参数，对面是 6 片森林 × 480 棵树。

⟹ 那个 20% 是关于**一个 12 epoch 冒烟测试**的事实，不是关于 NN 的事实。
本实验只回答一个问题：**再给预算，这条曲线往哪走？**

## 设计：测量路径一行都不改

阶梯 = 用**同一个** `experiments/multitask_mlp.py` 跑四次不同 `--max-iter`。
本脚本只做两件事：跑前落盘判据（`--emit-plan`）、跑后汇总判定（`--summarize`）。

⭐ **白送的自检**：12-epoch 那一档必须复现出 17.42% / 20.28%。对不上就说明环境或数据变了，
整条阶梯作废 —— **阶梯的第一级就是它自己的回归测试**。

## 预注册（跑之前钉死，不得因为看到曲线而修改）

```text
轴          max_iter ∈ {12, 50, 150, 400}     单轴，不同时动容量
读数        独立 MLP peak / 基准 peak         不是混合增益 —— 这次问的是「NN 本身多强」
两个臂      target_only 与 multitask（λ=0.3，08-19 已预注册）
门槛        max(两臂) ≥ 基准的 50%
条件延长    若 400 档相对 150 档仍 ≥ +5%（还在爬），追加一档 1200
```

50% 这个数有仓库数据支撑：旧 screen 里 MLP 到 40.2% 时 oracle 混合增益才刚进两位数
（+10.64%），到 4.8~6.2% 时 ≈ 0，08-19 在 17~20% 时是 +0.026%。

## ⚠️ 结论的适用范围（必须一起写出去）

本阶梯测的是**一个特定 NN 配方**，不是「NN 这个模型族」。三处对 NN 不利且本轮不动：

1. 特征是按 `|corr(feature, e)|` 选的 top-200 —— 那是为**线性/树**挑的判据；
2. `asset_id` 给的是 15 维 one-hot，不是 embedding（树那边是原生 categorical）；
3. sklearn `MLPRegressor`：Adam 定学习率，无调度、无 LayerNorm、无 dropout（只有 L2 `alpha`）。

⟹ 即便曲线在 35% 饱和，也只能否掉这个配方。上面三条正是 v5（8/31 之后）要处理的东西。

用法：
    .venv/bin/python experiments/nn_capacity_ladder.py --emit-plan
    for E in 12 50 150 400; do
      .venv/bin/python experiments/multitask_mlp.py \\
          --folds 0 --blend-mode oracle --max-iter $E --label multitask_mlp_e$E
    done
    .venv/bin/python experiments/nn_capacity_ladder.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- 预注册常量：跑之前钉死，**不得因为看到曲线而修改** ----
LADDER = (12, 50, 150, 400)
GATE_FRACTION = 0.50               # max(两臂) 独立 peak / 基准 peak 的推进门槛
EXTENSION_TRIGGER = 0.05           # 400 相对 150 仍 ≥ +5% ⟹ 曲线还在爬
EXTENSION_RUNG = 1200
ARMS = ("target_only", "multitask")

# 12-epoch 档的复现基准：outputs/experiments/multitask_mlp_stage1.json（2026-08-19）
REPRODUCTION_ANCHOR = {"target_only": 0.17428700119543933, "multitask": 0.20283264080006497}
REPRODUCTION_TOLERANCE = 1e-3

# 除这两项外，四档的 configuration 必须逐项相同 —— 防止「顺手也改了别的」让曲线无法归因
CONFIG_ALLOWED_TO_DIFFER = frozenset({"max_iter", "label"})

PLAN_LABEL = "nn_capacity_ladder_plan"
SUMMARY_LABEL = "nn_capacity_ladder"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-plan", action="store_true", help="落盘预注册判据（默认动作）")
    p.add_argument("--summarize", action="store_true", help="读各档 JSON 并判定")
    p.add_argument("--rungs", type=int, nargs="+", default=list(LADDER))
    p.add_argument("--result-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label-prefix", default="multitask_mlp_e")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def rung_path(result_dir: Path, prefix: str, rung: int) -> Path:
    return result_dir / f"{prefix}{rung}.json"


def read_rung(path: Path) -> dict:
    """取出一档的独立 MLP 强度与配置。只读 `folds[0]` —— 阶梯固定跑 fold 0。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload["folds"]
    if len(folds) != 1:
        raise SystemExit(f"{path.name} 有 {len(folds)} 折；阶梯口径是固定 fold 0，不接受多折")
    fold = folds[0]
    arms = {}
    for arm in ARMS:
        if arm not in fold["arms"]:
            raise SystemExit(f"{path.name} 缺少臂 {arm}")
        arms[arm] = float(fold["arms"][arm]["mlp_relative_to_baseline"])
    return {
        "path": str(path),
        "max_iter": int(payload["configuration"]["max_iter"]),
        "configuration": payload["configuration"],
        "baseline_peak": float(fold["baseline"]["peak"]),
        "arms": arms,
        "best_arm": max(arms, key=arms.get),
        "best_relative": max(arms.values()),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
    }


def assert_single_axis(rungs: list[dict]) -> None:
    """四档除 `max_iter`/`label` 外必须逐项相同 —— 否则曲线无法归因给预算。"""
    reference = rungs[0]["configuration"]
    for rung in rungs[1:]:
        drift = sorted(
            key for key in set(reference) | set(rung["configuration"])
            if key not in CONFIG_ALLOWED_TO_DIFFER
            and reference.get(key) != rung["configuration"].get(key))
        if drift:
            raise SystemExit(
                f"配置漂移：max_iter={rung['max_iter']} 这档与 {rungs[0]['max_iter']} 档在 "
                f"{drift} 上不同 —— 阶梯是**单轴**实验，多动一项曲线就无法归因给预算。")


def assert_reproduces_anchor(rungs: list[dict]) -> dict:
    """12-epoch 档必须复现 08-19 的结果 —— 阶梯的第一级就是它自己的回归测试。"""
    anchor = next((r for r in rungs if r["max_iter"] == 12), None)
    if anchor is None:
        return {"checked": False,
                "note": "阶梯里没有 12 档 ⟹ 跳过自检（不推荐：失去唯一的外部锚点）"}
    drift = {arm: anchor["arms"][arm] - expected
             for arm, expected in REPRODUCTION_ANCHOR.items()}
    worst = max(abs(value) for value in drift.values())
    if worst > REPRODUCTION_TOLERANCE:
        raise SystemExit(
            f"12-epoch 档没有复现 2026-08-19 的结果（最大偏差 {worst:.2e} > "
            f"{REPRODUCTION_TOLERANCE:g}）：{ {k: round(v, 6) for k, v in drift.items()} }\n"
            "  ⟹ 环境或数据变了，整条阶梯作废。先查为什么，不要解读曲线。")
    return {"checked": True, "max_abs_drift": worst,
            "drift": drift, "expected": dict(REPRODUCTION_ANCHOR)}


def judge(rungs: list[dict]) -> dict:
    """按预注册判据给结论。曲线形状与门槛两件事分开报。"""
    ordered = sorted(rungs, key=lambda r: r["max_iter"])
    best = max(ordered, key=lambda r: r["best_relative"])
    peak_rung, final_rung = best, ordered[-1]

    # 条件延长：最后两档之间还在爬？（相对增幅，不是绝对）
    still_climbing = False
    climb_rate = None
    if len(ordered) >= 2:
        previous, last = ordered[-2], ordered[-1]
        if previous["best_relative"] > 0:
            climb_rate = last["best_relative"] / previous["best_relative"] - 1.0
            still_climbing = climb_rate >= EXTENSION_TRIGGER

    passes = best["best_relative"] >= GATE_FRACTION
    turned_over = final_rung["best_relative"] < peak_rung["best_relative"]
    if passes and still_climbing:
        verdict, action = "RESULT / 仍在爬", (
            f"≥{GATE_FRACTION:.0%} 且曲线未饱和 ⟹ **容量/预算才是绑定约束**，"
            "这是 v5 最值得做的信号。v5 列为 8/31 之后的正式候选。")
    elif passes:
        verdict, action = "RESULT / 已饱和", (
            f"≥{GATE_FRACTION:.0%} 但曲线已饱和 ⟹ v5 列为 8/31 之后的正式候选，"
            "并记录达到该水平所需的预算。")
    else:
        verdict, action = "REJECTED", (
            f"曲线在 <{GATE_FRACTION:.0%} 处"
            + ("掉头" if turned_over else "饱和")
            + " ⟹ **sklearn MLPRegressor + 生产特征表示 + 这套预算**下不具竞争力。"
            "⚠️ 这**不是**「NN 不行」—— 见适用范围三条。")
    return {
        "gate_fraction": GATE_FRACTION,
        "best_relative": best["best_relative"],
        "best_at_max_iter": best["max_iter"],
        "best_arm": best["best_arm"],
        "final_relative": final_rung["best_relative"],
        "climb_rate_last_step": climb_rate,
        "still_climbing": still_climbing,
        "turned_over": turned_over,
        "passes_gate": passes,
        "verdict": verdict,
        "action": action,
        "extension_due": bool(still_climbing),
        "extension_rung": EXTENSION_RUNG if still_climbing else None,
    }


def render_plan() -> str:
    return "\n".join([
        "# NN 独立能力阶梯 —— 预注册（`nn_capacity_ladder_plan`）",
        "",
        "> ⚠️ **本文件必须先于任何一档跑起来之前落盘。** 看到曲线之后不得修改门槛或档位。",
        "",
        "## 问题",
        "",
        "`target_mlp_screen`(08-12) 与 `multitask_mlp_stage1`(08-19) 报的 MLP 独立 peak",
        "（24.3% / 17.4%~20.3% of 基准）都是在 `max_iter=12` 下测的，而早停是关掉的",
        "（`tol=0.0`、`n_iter_no_change=max_iter+1`），JSON 里 `iterations` 全部等于 12",
        "⟹ **那是跑完预算被掐断，不是收敛**。本实验只问：再给预算，曲线往哪走？",
        "",
        "## 设计（单轴）",
        "",
        "```text",
        f"轴          max_iter ∈ {{{', '.join(str(r) for r in LADDER)}}}",
        "冻结项      架构 (32,)/(64,32)、batch 4096、lr 1e-3、alpha 1e-3、seed 2026、",
        "            200 特征、modulo 5、train_window 78,960、fold 0、生产 3s480 基准",
        "读数        独立 MLP peak / 基准 peak（**不是**混合增益）",
        f"两个臂      {' 与 '.join(ARMS)}（λ=0.3，08-19 已预注册）",
        f"门槛        max(两臂) ≥ 基准的 {GATE_FRACTION:.0%}",
        f"条件延长    末档相对前一档仍 ≥ +{EXTENSION_TRIGGER:.0%} ⟹ 追加一档 {EXTENSION_RUNG}",
        "```",
        "",
        f"⭐ **自检**：12 档必须复现 08-19 的 "
        f"{REPRODUCTION_ANCHOR['target_only']:.4%} / {REPRODUCTION_ANCHOR['multitask']:.4%}"
        f"（容差 {REPRODUCTION_TOLERANCE:g}）。对不上则整条阶梯作废。",
        "",
        "⭐ **单轴保护**：四档的 `configuration` 除 `max_iter`/`label` 外必须逐项相同，",
        "汇总时硬校验 —— 多动一项，曲线就无法归因给预算。",
        "",
        f"## 门槛为什么是 {GATE_FRACTION:.0%}",
        "",
        "有仓库数据支撑，不是拍脑袋：旧 screen 里 MLP 达到基准 40.2% 时 oracle 混合增益",
        "才刚进两位数（+10.64%）；达到 4.8~6.2% 时 ≈ 0；08-19 在 17~20% 时是 +0.026%。",
        "",
        "## 三种结局（现在就定好怎么写）",
        "",
        "| 曲线 | 判定 | 动作 |",
        "|---|---|---|",
        f"| 在 <{GATE_FRACTION:.0%} 处饱和或掉头 | `REJECTED` | 写清是**这个配方**不行，连曲线形状一起归档 |",
        f"| ≥{GATE_FRACTION:.0%} 且已饱和 | `RESULT` | v5 列为 8/31 后正式候选，记录所需预算 |",
        f"| ≥{GATE_FRACTION:.0%} 且仍在爬 | `RESULT` | 同上，且容量/预算是绑定约束 —— v5 最值得做的信号 |",
        "",
        "## ⚠️ 结论的适用范围（必须一起写出去）",
        "",
        "本阶梯测的是**一个特定 NN 配方**，不是「NN 这个模型族」。三处对 NN 不利且本轮不动：",
        "",
        "1. 特征是按 `|corr(feature, e)|` 选的 top-200 —— 那是为**线性/树**挑的判据；",
        "2. `asset_id` 给的是 15 维 one-hot，不是 embedding（树那边是原生 categorical）；",
        "3. sklearn `MLPRegressor`：Adam 定学习率，无调度、无 LayerNorm、无 dropout。",
        "",
        "⟹ 即便曲线在 35% 饱和，也只能否掉这个配方。上面三条正是 v5 要处理的东西。",
        "",
        "## 不做",
        "",
        "- 不建 `strategies/v5_*`、不装 torch —— 8/31 之后的事。",
        "- 不动容量 / 学习率 / batch / alpha / 激活 / 特征数（单轴）。",
        "- 不因为看到曲线而改门槛或改档位。",
        "- 不把任何 NN 产物接进生产或私榜提交路径。",
    ]) + "\n"


def render_summary(payload: dict) -> str:
    verdict = payload["verdict"]
    lines = [
        "# NN 独立能力阶梯（`nn_capacity_ladder`）",
        "",
        f"预注册：`{payload['plan_path']}`（sha256 `{payload['plan_sha256'][:16]}…`）"
        " ⟹ 判据先于结果落盘，可核验。",
        "",
        "## 曲线：独立 MLP peak / 基准 peak",
        "",
        "| max_iter | target_only | multitask | 较好者 | 相对上一档 | 耗时 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    previous = None
    for rung in payload["rungs"]:
        step = ("—" if previous is None or previous <= 0
                else f"{rung['best_relative'] / previous - 1:+.1%}")
        lines.append(
            f"| {rung['max_iter']} | {rung['arms']['target_only']:.1%} | "
            f"{rung['arms']['multitask']:.1%} | **{rung['best_relative']:.1%}** | {step} | "
            f"{rung['elapsed_seconds'] / 60:.1f} min |")
        previous = rung["best_relative"]
    anchor = payload["reproduction_check"]
    lines += [
        "",
        f"基准 peak（生产 3s480，fold 0）= {payload['baseline_peak']:.8f}",
        "",
        (f"⭐ **自检通过**：12 档与 08-19 的最大偏差 {anchor['max_abs_drift']:.2e} "
         f"< 容差 {REPRODUCTION_TOLERANCE:g} ⟹ 环境与数据一致，曲线可解读。"
         if anchor.get("checked") else f"⚠️ {anchor.get('note')}"),
        "",
        "## 判定",
        "",
        "```text",
        f"最好一档        max_iter={verdict['best_at_max_iter']}  "
        f"（{verdict['best_arm']}）  {verdict['best_relative']:.1%}",
        f"门槛            {verdict['gate_fraction']:.0%}",
        f"末档相对前一档  "
        + ("—" if verdict["climb_rate_last_step"] is None
           else f"{verdict['climb_rate_last_step']:+.1%}")
        + f"   （≥ +{EXTENSION_TRIGGER:.0%} 视为仍在爬）",
        "```",
        "",
        f"## **{verdict['verdict']}**",
        "",
        verdict["action"],
        "",
    ]
    if verdict["extension_due"]:
        lines += [
            f"⟹ **触发预注册的条件延长**：追加一档 `--max-iter {verdict['extension_rung']}`。",
            "这条是跑之前写死的，不是看到曲线才决定的。",
            "",
        ]
    lines += [
        "## ⚠️ 适用范围",
        "",
        "本阶梯测的是**一个特定 NN 配方**，不是「NN 这个模型族」。三处对 NN 不利且本轮未动：",
        "特征按 `|corr(feature, e)|` 选的 top-200（为线性/树挑的判据）、`asset_id` 是 15 维",
        "one-hot 而非 embedding、sklearn `MLPRegressor` 无学习率调度 / 无 LayerNorm / 无 dropout。",
        "⟹ 上面三条正是 v5（8/31 之后）要处理的东西；本阶梯为它定范围，不替它下结论。",
        "",
    ]
    return "\n".join(lines)


def write(out_dir: Path, label: str, payload: dict | None, text: str, force: bool) -> Path:
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


def emit_plan(args: argparse.Namespace) -> None:
    payload = {
        "experiment": "nn_capacity_ladder_plan",
        "role": "PRE-REGISTRATION —— 必须先于任何一档跑起来之前落盘",
        "question": "MLP 独立 peak 只有基准的 20%，是预算造成的还是能力上限？",
        "ladder": list(LADDER),
        "arms": list(ARMS),
        "gate_fraction": GATE_FRACTION,
        "gate_reads": "max(两臂) 的独立 MLP peak / 基准 peak",
        "gate_rationale": ("旧 screen 里 MLP 到 40.2% 时 oracle 混合增益才 +10.64%，"
                           "到 4.8~6.2% 时 ≈ 0，08-19 在 17~20% 时是 +0.026%"),
        "extension_trigger": EXTENSION_TRIGGER,
        "extension_rung": EXTENSION_RUNG,
        "reproduction_anchor": dict(REPRODUCTION_ANCHOR),
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "config_allowed_to_differ": sorted(CONFIG_ALLOWED_TO_DIFFER),
        "scope_limits": [
            "特征按 |corr(feature, e)| 选的 top-200 —— 为线性/树挑的判据",
            "asset_id 是 15 维 one-hot，不是 embedding",
            "sklearn MLPRegressor：无学习率调度 / 无 LayerNorm / 无 dropout",
        ],
    }
    write(Path(args.output_dir), PLAN_LABEL, payload, render_plan(), args.force)
    print(f"\n阶梯 {list(LADDER)}；门槛 {GATE_FRACTION:.0%}；"
          f"条件延长 +{EXTENSION_TRIGGER:.0%} ⟹ 追加 {EXTENSION_RUNG}", flush=True)


def summarize(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    plan_path = out_dir / f"{PLAN_LABEL}.json"
    if not plan_path.is_file():
        raise SystemExit(f"没有找到预注册文件 {plan_path} —— 先跑 `--emit-plan`。"
                         "判据必须先于结果落盘（CLAUDE.md §5.1）")

    result_dir = Path(args.result_dir)
    rungs = []
    for rung in sorted(args.rungs):
        path = rung_path(result_dir, args.label_prefix, rung)
        if not path.is_file():
            raise SystemExit(f"缺少 {path} —— 先把这一档跑完（--max-iter {rung}）")
        rungs.append(read_rung(path))

    assert_single_axis(rungs)
    reproduction = assert_reproduces_anchor(rungs)
    verdict = judge(rungs)

    payload = {
        "experiment": "nn_capacity_ladder",
        "plan_path": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "baseline_peak": rungs[0]["baseline_peak"],
        "rungs": rungs,
        "reproduction_check": reproduction,
        "verdict": verdict,
    }
    write(out_dir, SUMMARY_LABEL, payload, render_summary(payload), args.force)
    print(f"\n最好一档 max_iter={verdict['best_at_max_iter']}（{verdict['best_arm']}）"
          f" = 基准的 {verdict['best_relative']:.1%}；门槛 {GATE_FRACTION:.0%}", flush=True)
    print(f"判定：{verdict['verdict']}", flush=True)
    if verdict["extension_due"]:
        print(f"⟹ 触发条件延长：追加一档 --max-iter {verdict['extension_rung']}", flush=True)


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args)
    else:
        emit_plan(args)


if __name__ == "__main__":
    main()
