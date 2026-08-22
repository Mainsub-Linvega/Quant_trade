"""8/23 回补包里到底有没有 `responder_*` 列 —— 读 D0.2 已经落盘的审计 JSON 判定。

## 为什么需要这一条

`docs/data_description.md:173` 只说「公榜截止后会发布**标签回补数据**，该部分数据将作为
**扩展训练数据**使用」，`competition_description.md:213` 的 Timeline 也只写「发布扩展训练数据」
—— **主办方原文从未逐字列出回补包的字段清单**。两种读法都成立：

```text
窄读   「标签」= target（+ weight，评分要用）        ⟹ 不含 47 列 responder
宽读   「作为**扩展训练数据**使用」                   ⟹ 训练数据的定义（data_description.md:17）
                                                      是含 responder 的
```

而 2026-08-22 收口的 responder 四项 `REJECTED`（Stage C 补测 / 选列判据 / P9 范围项 ③ /
更早的 A0·A5·重新审计）**共用同一条重开条件**：

> 8/23 回补包若含 responder 列 ⟹ 按原规格复验一次。

⟹ 这个问题的答案决定 8/23–8/31 要不要重开一条已经关掉的线，所以它必须在 D0.2 当天就有
一个**落盘的、可核的**答案，而不是靠当时翻 schema 拍脑袋。

## 判据（跑前钉死）

审计 JSON 已经逐文件存了 `columns`（`scripts/audit_data_release.py:92`），且 `columns` 本身
就在 `file_identity` 的比较键里（:133）⟹ 本脚本不重新扫 parquet，只读那份产物。

```text
train split 无 added/modified            ⟹ 数据没变，responder 轴维持关闭（D0.2 主判据也会拦下重训）
回补文件**全部**带 responder             ⟹ 触发重开条件，按原规格复验（RUNBOOK D3.5）
回补文件**全部不**带 responder           ⟹ responder 轴维持关闭，8/31 前不再碰
回补文件**部分**带 —— 退出码 1           ⟹ schema 在回补包内部就不一致，**先查为什么**，
                                            不要按任一分支走
```

⚠️ 只看 train split。test split 按主办方原文（`:175`）本来就不含 responder，
而回补的是**公榜期那段的标签**，会以 train 分区的形式出现。

用法：
    .venv/bin/python scripts/check_backfill_responders.py \\
        --audit outputs/data_audits/data_release_20260823.json \\
        --output outputs/data_audits/backfill_responders_20260823.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
RESPONDER_PREFIX = "responder_"

VERDICT_UNCHANGED = "train_unchanged"
VERDICT_WITH_RESPONDERS = "backfill_has_responders"
VERDICT_WITHOUT_RESPONDERS = "backfill_has_no_responders"
VERDICT_INCONSISTENT = "backfill_schema_inconsistent"

REOPENS = (
    "responder_stage_c_fill",
    "responder_selection_probe",
    "nn_capacity_ladder_respsel",
    "responder_reaudit_20260814",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", required=True,
                   help="D0.2 产出的审计 JSON（必须是带 comparison 段的那份）")
    p.add_argument("--output", default=None,
                   help="把判定落盘（RUNBOOK 的习惯：不落盘等于没审过）")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def responder_count(item: dict[str, Any]) -> int:
    return sum(1 for name in (item.get("columns") or [])
               if str(name).startswith(RESPONDER_PREFIX))


def evaluate(audit: dict[str, Any]) -> dict[str, Any]:
    comparison = audit.get("comparison") or {}
    train_files = {item["path"]: item for item in audit.get("splits", {}).get("train", [])}
    train_diff = (comparison.get("splits") or {}).get("train") or {}
    touched = sorted(set(train_diff.get("added") or []) | set(train_diff.get("modified") or []))

    rows, missing = [], []
    for path in touched:
        item = train_files.get(path)
        if item is None:                       # 出现在 comparison 里却不在当前 splits ⟹ removed
            missing.append(path)
            continue
        columns = item.get("columns") or []
        rows.append({"path": path, "n_columns": len(columns),
                     "n_responders": responder_count(item),
                     "has_target": "target" in columns, "has_weight": "weight" in columns})

    with_resp = [r for r in rows if r["n_responders"] > 0]
    if not touched:
        verdict, reopens = VERDICT_UNCHANGED, False
    elif not rows:
        verdict, reopens = VERDICT_INCONSISTENT, False
    elif len(with_resp) == len(rows):
        verdict, reopens = VERDICT_WITH_RESPONDERS, True
    elif not with_resp:
        verdict, reopens = VERDICT_WITHOUT_RESPONDERS, False
    else:
        verdict, reopens = VERDICT_INCONSISTENT, False

    return {
        "experiment": "check_backfill_responders",
        "audit": None,
        "train_files_total": len(train_files),
        "train_added_or_modified": touched,
        "paths_in_comparison_but_absent": missing,
        "per_file": rows,
        "n_with_responders": len(with_resp),
        "verdict": verdict,
        "reopens_responder_line": reopens,
        "reopen_targets": list(REOPENS) if reopens else [],
        "source_of_ambiguity": ("docs/data_description.md:173 只说「标签回补 / 扩展训练数据」，"
                                "未逐字列出字段清单 ⟹ 只能实测"),
    }


def render(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# 8/23 回补包 responder 列核查（`check_backfill_responders`）",
        "",
        f"审计来源：`{payload['audit']}`",
        "",
        f"train 分区共 {payload['train_files_total']} 个；"
        f"本次 added/modified **{len(payload['train_added_or_modified'])}** 个",
        "",
    ]
    if payload["per_file"]:
        lines += ["| 文件 | 列数 | responder | target | weight |", "|---|---:|---:|:--:|:--:|"]
        lines += [f"| `{Path(r['path']).name}` | {r['n_columns']} | **{r['n_responders']}** | "
                  f"{'✅' if r['has_target'] else '❌'} | {'✅' if r['has_weight'] else '❌'} |"
                  for r in payload["per_file"]]
        lines.append("")
    if payload["paths_in_comparison_but_absent"]:
        lines += ["⚠️ 下列路径在 comparison 里但不在当前 splits（removed）：",
                  ", ".join(f"`{p}`" for p in payload["paths_in_comparison_but_absent"]), ""]

    reading = {
        VERDICT_UNCHANGED: "train split 未变 ⟹ responder 轴维持关闭"
                           "（D0.2 主判据也会拦下一切重训）。",
        VERDICT_WITH_RESPONDERS:
            "回补的 train 文件**全部带 responder 列** ⟹ **触发 2026-08-22 收口的 responder "
            "四项 `REJECTED` 的统一重开条件**，按**原规格**各复验一次（不得借机改设计）。",
        VERDICT_WITHOUT_RESPONDERS:
            "回补的 train 文件**都不带 responder 列** ⟹ responder 轴维持关闭，"
            "8/31 之前不再碰这条线。",
        VERDICT_INCONSISTENT:
            "⚠️⚠️ 回补包**内部 schema 不一致**（部分文件带 responder、部分不带，或 comparison "
            "指到了不存在的路径）⟹ **先查为什么**，不要按任何一支走。",
    }[verdict]
    lines += [f"## 判定：`{verdict}`", "", reading, ""]
    if payload["reopens_responder_line"]:
        lines += ["需按原规格复验的产物：",
                  ", ".join(f"`{name}`" for name in payload["reopen_targets"]), ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit)
    if not audit_path.is_file():
        raise SystemExit(f"找不到审计 JSON {audit_path} —— 先跑 D0.2 的 audit_data_release.py")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if "comparison" not in audit:
        print("⚠️ 这份审计没有 comparison 段（没传 --baseline？）—— "
              "本脚本只能报「未变」，不能证明回补包的 schema", flush=True)

    payload = evaluate(audit)
    payload["audit"] = str(audit_path)

    for row in payload["per_file"]:
        print(f"  {Path(row['path']).name:<34} {row['n_columns']:>4} 列  "
              f"responder={row['n_responders']:>3}  target={row['has_target']}", flush=True)
    print(f"\n判定：{payload['verdict']}"
          f"（重开 responder 线：{payload['reopens_responder_line']}）", flush=True)

    if args.output:
        out = Path(args.output)
        md = out.with_suffix(".md")
        if not args.force and (out.exists() or md.exists()):
            raise SystemExit(f"{out} 或 {md} 已存在；要覆盖请加 --force")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md.write_text(render(payload), encoding="utf-8")
        print(f"wrote {out}\nwrote {md}", flush=True)

    if payload["verdict"] == VERDICT_INCONSISTENT:
        raise SystemExit("回补包内部 schema 不一致 —— 停下来查原因（退出码 1）")


if __name__ == "__main__":
    main()
