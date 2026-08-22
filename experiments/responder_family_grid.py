"""responder 族群表：47 列其实是一张「维度 × 窗口」的网格。

## 这份东西回答什么

仓库现有的 responder 分组只有一份 —— `responder_analysis.py` 按 `1 − |corr|` 做层次聚类，
得到 24 族。那把刀切出来的其实是**窗口组**（例如 cluster 13 = {25,26,27,44,45,46} 横跨两个
不同量纲的维度，但窗口相同）。本模块提供**正交的另一把刀**：按维度分组。

两个指纹都直接来自 parquet 的 row-group 统计信息，**不加载任何数据**：

```text
缺失数 null_count   ⟹ 窗口指纹。responder 构造自「未来不可见区间」，窗口越长，分区末端
                       越多行算不出来 ⟹ 同 null_count = 同窗口。全 47 列只取 11 个离散值。
取值域 (min, max)    ⟹ 维度指纹。[0,1] / 非正 / 非负 / 双向四类，对应主办方原文说的
                       「收益类、风险类、路径类以及流动性摩擦类」。
```

## 分族规则（先写死，再看结果）

1. 先按 `sign_class` 把连续下标切成极长游程；
2. 再在每个游程内部，于**窗口梯子重启点**切开 —— 即 `null_count` 在**已经上升过之后**
   又下降的位置。（α 族开头的 279 → 9 → 1 → 0 是上升之前的下降，不算重启。）

这条规则跑出来是 8 个族、大小 7/7/7/7/3/7/5/4，合计 **47**。`tests/` 里有断言钉住。

## ⚠️ 这份表是什么、不是什么

- **是**：一组可复现的测量 + 一条写死的分族规则。
- **不是**：主办方公布的语义。「像什么」那一列是解读，不得当事实引用（CLAUDE.md §3）。
- ⚠️ **不得用它按「与 target 相关高」挑臂** —— `responder_targets_stage1.md:14-22` 已经证伪
  过这个论证形式：同期相关最高的 `responder_03`（0.817）当训练目标是全场最差（−15.47%、0/7），
  而相关只有 0.394 的 `responder_04` 排第一（+15.80%）。见 `CLAUDE.md:119`。

用法：
    .venv/bin/python experiments/responder_family_grid.py
输出：outputs/experiments/responder_family_grid.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]

N_RESPONDERS = 47
RESPONDER_COLUMNS = [f"responder_{index:02d}" for index in range(N_RESPONDERS)]

def family_reading(sign_class_name: str, value_range: Sequence[float]) -> str:
    """「像什么」—— 解读，不是主办方语义，不得当事实引用（CLAUDE.md §3）。

    ⚠️ 主办方原文只说 responder「可能反映不同类型的未来状态，例如收益类、风险类、路径类
    以及流动性摩擦类等」（`docs/data_description.md:171`），**没有公布哪一列是哪一类**。
    非正 / 双向 / 饱和到 1 这三种签名足够把三个族对上号；但**四个非负族里哪个是上行、
    哪个是路径、哪个是摩擦，本表判不出来** —— 只报量级，不编标签。
    """
    if sign_class_name == "nonpositive":
        return "非正 ⟹ 下行 / 回撤类"
    if sign_class_name == "unit_interval":
        return "上界饱和到 1 ⟹ 概率 / CDF 类"
    if sign_class_name == "bidirectional":
        return "双向、与 target 同量级（±2.23）⟹ 收益类"
    return f"非负，量级 ~{value_range[1]:.0e}（上行 / 路径 / 摩擦之一，**本表判不出是哪个**）"


# `unit_interval` 的上界容差。**不是随手取的**，两端都实测过：
#   最紧的成员   responder_00  max = 0.9999723   （H=1，单步经验 CDF 到不了 1）
#   最近的非成员 responder_27  max = 0.8709947   （路径类最长窗）
# ⟹ 间隔 0.13，这条线离两边都很远，不是刀刃。
# ⚠️ 反面教训：先用过 `max > 0.5`（太松，把路径类长窗成员误判进 CDF 类 ⟹ 切出 9 族）；
#    又用过 `abs(max − 1.0) <= 1e-6`（太紧，把 responder_00 自己踢出去 ⟹ 还是 9 族）。
UNIT_INTERVAL_TOL = 1e-3


def sign_class(minimum: float, maximum: float, tol: float = 1e-6) -> str:
    """由取值域读出维度类别。判据写死，不看下标。

    `unit_interval` 要求上界**饱和到 1**（见 `UNIT_INTERVAL_TOL` 的实测依据），
    而不是「≤1 且比较大」—— 路径类的上界随窗口单调增长（0.1717 → 0.8710）但从不饱和。
    """
    if maximum <= tol:
        return "nonpositive"
    if minimum >= -tol and abs(maximum - 1.0) <= UNIT_INTERVAL_TOL:
        return "unit_interval"
    if minimum >= -tol:
        return "nonnegative"
    return "bidirectional"


def read_column_stats(parquet_path: Path) -> list[dict[str, Any]]:
    """只读 row-group 统计信息 —— 不加载任何数据。"""
    handle = pq.ParquetFile(parquet_path)
    metadata = handle.metadata
    position = {metadata.schema.column(i).name: i for i in range(metadata.num_columns)}
    missing = [name for name in RESPONDER_COLUMNS if name not in position]
    if missing:
        raise SystemExit(f"{parquet_path} 缺少 responder 列：{missing[:5]}… "
                         "（test 分区没有 responder，这里必须给 train 分区）")
    if metadata.num_row_groups != 1:
        raise SystemExit(f"{parquet_path} 有 {metadata.num_row_groups} 个 row group；"
                         "本模块假设 1 个（data/manifest.json 记的就是 1 个）")

    group = metadata.row_group(0)
    rows = []
    for index, name in enumerate(RESPONDER_COLUMNS):
        stats = group.column(position[name]).statistics
        if stats is None:
            raise SystemExit(f"{name} 没有统计信息 ⟹ 无法用元数据路径分族")
        rows.append({
            "responder": name,
            "index": index,
            "null_count": int(stats.null_count),
            "min": float(stats.min),
            "max": float(stats.max),
            "sign_class": sign_class(float(stats.min), float(stats.max)),
        })
    return rows


def _ladder_restarts(null_counts: Sequence[int]) -> list[int]:
    """窗口梯子重启点：在已经上升过之后又下降的位置（返回下标偏移）。"""
    cuts, risen = [], False
    for offset in range(1, len(null_counts)):
        if null_counts[offset] > null_counts[offset - 1]:
            risen = True
        elif null_counts[offset] < null_counts[offset - 1] and risen:
            cuts.append(offset)
            risen = False
    return cuts


def build_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按分族规则切出维度族。规则见模块 docstring，跑前写死。"""
    runs: list[list[dict[str, Any]]] = []
    for row in rows:
        if runs and runs[-1][-1]["sign_class"] == row["sign_class"]:
            runs[-1].append(row)
        else:
            runs.append([row])

    families = []
    for run in runs:
        bounds = [0, *_ladder_restarts([r["null_count"] for r in run]), len(run)]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            block = run[start:stop]
            value_range = [min(r["min"] for r in block), max(r["max"] for r in block)]
            families.append({
                "family": chr(ord("a") + len(families)),
                "members": [r["responder"] for r in block],
                "sign_class": block[0]["sign_class"],
                "value_range": value_range,
                "null_ladder": [r["null_count"] for r in block],
                "truncation_rungs": truncation_rungs([r["null_count"] for r in block]),
                "reading": family_reading(block[0]["sign_class"], value_range),
            })
    return families


def truncation_rungs(null_ladder: Sequence[int]) -> list[int]:
    """截断梯子 = `null_count` 的最长**严格递增后缀**里的正值。

    ⚠️ 只能取后缀，不能取全部非零值。CDF 族开头是 `279, 9, 1, 0, …` —— 那三个数是
    H=1/2 附近的另一种现象（单步经验 CDF 的退化），不是窗口截断；把它们算进去会让该族
    的梯子看起来独一无二，从而**掩盖掉它的尾巴其实与另外三族逐位相同**这个事实。
    """
    start = len(null_ladder) - 1
    while start > 0 and null_ladder[start - 1] < null_ladder[start]:
        start -= 1
    return [n for n in null_ladder[start:] if n > 0]


def shared_ladders(families: list[dict[str, Any]]) -> dict[str, list[str]]:
    """把截断梯子相同的族归到一起 —— 这是「同缺失数 = 同窗口」的直接证据。"""
    groups: dict[str, list[str]] = {}
    for family in families:
        rungs = truncation_rungs(family["null_ladder"])
        key = ",".join(str(n) for n in rungs) or "(无截断档)"
        groups.setdefault(key, []).append(family["family"])
    return groups


def untested_stage_c_cells(stage_b_json: Path, already_probed: Sequence[str]) -> dict[str, Any]:
    """从 Stage B 的 JSON **派生**未测名单，不硬编码（CLAUDE.md §7）。"""
    if not stage_b_json.is_file():
        raise SystemExit(f"找不到 Stage B 结果 {stage_b_json}")
    summary = json.loads(stage_b_json.read_text(encoding="utf-8"))["summary"]["clusters"]

    passed = [c for c in summary if c["pass"]]
    blocked, other_failures = [], []
    for cluster in summary:
        if cluster["pass"]:
            continue
        failed = sorted(k for k, v in cluster["checks"].items() if not v)
        (blocked if failed == ["multi_member_family"] else other_failures).append(
            {"cluster": cluster["cluster"], "members": cluster["members"], "failed": failed})

    members = sorted(m for c in blocked for m in c["members"])
    unknown = [name for name in already_probed if name not in members]
    if unknown:
        raise SystemExit(f"{unknown} 不在「只错 multi_member_family」的名单里 —— "
                         "已补测名单与 Stage B JSON 对不上，先查口径")
    return {
        "source": str(stage_b_json),
        "clusters_total": len(summary),
        "clusters_passed": len(passed),
        "clusters_blocked_by_heuristic_only": len(blocked),
        "clusters_failed_on_evidence": other_failures,
        "blocked_members": members,
        "already_probed": sorted(already_probed),
        "untested": [name for name in members if name not in already_probed],
    }


def render(payload: dict[str, Any]) -> str:
    families, cells = payload["families"], payload["stage_c_gap"]
    lines = [
        "# responder 族群表（`responder_family_grid`）",
        "",
        "> 由 parquet 的 row-group 统计信息派生，**未加载任何数据**。",
        "> 「像什么」一列是**解读**，不是主办方公布的语义，不得当事实引用（CLAUDE.md §3）。",
        "",
        f"来源：`{payload['source']}`（{payload['n_responders']} 个 responder，"
        f"target 取值域 [{payload['target_range'][0]:+.4f}, {payload['target_range'][1]:+.4f}]）",
        "",
        "## 维度族",
        "",
        "| 族 | 成员 | 维度类别 | 取值域 | 缺失数梯子 | 像什么 |",
        "|---|---|---|---|---|---|",
    ]
    for family in families:
        members = f"`{family['members'][0]}`–`{family['members'][-1]}`" \
            if len(family["members"]) > 1 else f"`{family['members'][0]}`"
        low, high = family["value_range"]
        lines.append(
            f"| {family['family']} | {members}（{len(family['members'])}） | "
            f"`{family['sign_class']}` | `[{low:+.4f}, {high:+.4f}]` | "
            f"{', '.join(str(n) for n in family['null_ladder'])} | {family['reading']} |")

    lines += [
        "",
        f"合计 {' + '.join(str(len(f['members'])) for f in families)} = "
        f"**{sum(len(f['members']) for f in families)}** ✅",
        "",
        "## 共用窗口梯子 —— 「同缺失数 = 同窗口，不同维度」的直接证据",
        "",
        "| 缺失数梯子（截断档） | 共用它的族 |",
        "|---|---|",
    ]
    for ladder, owners in payload["shared_ladders"].items():
        lines.append(f"| `{ladder}` | {', '.join(owners)} |")

    lines += [
        "",
        "responder 构造自「未来不可见区间」（`docs/data_description.md:169`）⟹ 窗口越长，",
        "分区末端越多行算不出来。缺失数因此是**窗口的精确指纹**，且跨维度族逐位对齐。",
        "",
        "⭐ 这与 `responder_analysis.py` 的 24 族聚类是**正交的两把刀**：那把按 `1 − |corr|` 聚，",
        "切出来的是**窗口组**（cluster 13 = {25,26,27,44,45,46} 横跨两个量纲不同的维度但窗口相同）；",
        "本表切的是**维度组**。",
        "",
        "## Stage B 的启发式缺口",
        "",
        f"来源：`{cells['source']}`",
        "",
        "```text",
        f"{cells['clusters_total']} 族   通过 {cells['clusters_passed']}   "
        f"只被 multi_member_family 挡住 {cells['clusters_blocked_by_heuristic_only']}   "
        f"因证据不过 {len(cells['clusters_failed_on_evidence'])}",
        "```",
        "",
        f"被挡住的 {len(cells['blocked_members'])} 个单成员族里，08-18 的 "
        "`horizon_auxiliary_cache_probe` 只补测了 "
        f"{', '.join('`' + m + '`' for m in cells['already_probed'])} ⟹ "
        f"**剩 {len(cells['untested'])} 个从未进过 Stage C**：",
        "",
        "```text",
        ", ".join(cells["untested"]),
        "```",
        "",
        "⚠️ **这个缺口不构成「有收益」的理由。** 两条必须一起读：",
        "",
        "1. 其中最显眼的 `responder_03`/`responder_28`/`responder_29`/`responder_04` "
        "在 A0 阶段 1 **已被逐列量过**（当训练目标），且**同期相关最高的 `responder_03` 是全场最差**"
        "（−15.47%、0/7 阶梯）—— 见 `responder_targets_stage1.md:14-22`、`CLAUDE.md:119`。",
        "2. 「把 responder 的预测值叠进 blend」属于 `responder_reaudit_20260814.md:93-100` "
        "母条件**明令排除**的「线性叠加 / 对预测值做二层校准」机制族 ⟹ 补测它的价值是**结案**，不是找收益。",
        "",
    ]
    return "\n".join(lines)


def build_payload(data_root: Path, stage_b_json: Path,
                  already_probed: Sequence[str]) -> dict[str, Any]:
    source = data_root / "train" / "train_partition_000.parquet"
    rows = read_column_stats(source)
    families = build_families(rows)

    total = sum(len(f["members"]) for f in families)
    if total != N_RESPONDERS:
        raise SystemExit(f"分族覆盖 {total} 列，应为 {N_RESPONDERS} —— 分族规则失效，先查规则")
    covered = [m for f in families for m in f["members"]]
    if sorted(covered) != sorted(RESPONDER_COLUMNS):
        raise SystemExit("分族有重复或遗漏 —— 分族规则失效，先查规则")

    handle = pq.ParquetFile(source)
    metadata = handle.metadata
    position = {metadata.schema.column(i).name: i for i in range(metadata.num_columns)}
    target_stats = metadata.row_group(0).column(position["target"]).statistics

    return {
        "experiment": "responder_family_grid",
        "source": str(source),
        "n_responders": N_RESPONDERS,
        "target_range": [float(target_stats.min), float(target_stats.max)],
        "unit_interval_tol": UNIT_INTERVAL_TOL,
        "columns": rows,
        "families": families,
        "shared_ladders": shared_ladders(families),
        "stage_c_gap": untested_stage_c_cells(stage_b_json, already_probed),
        "scope": ("测量 + 一条写死的分族规则；「像什么」是解读不是主办方语义。"
                  "不得据此按「与 target 相关高」挑臂 —— 该论证形式已被 "
                  "responder_targets_stage1 证伪。"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--stage-b-json", default=str(
        _REPO_ROOT / "outputs" / "experiments"
        / "responder_predictability_reaudit_phasebal_prodwindow.json"))
    p.add_argument("--already-probed", nargs="+", default=["responder_00", "responder_02"],
                   help="08-18 horizon_auxiliary_cache_probe 已补测的列")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="responder_family_grid")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(Path(args.data_root), Path(args.stage_b_json), args.already_probed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} 或 {md_path} 已存在；要覆盖请加 --force")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    md_path.write_text(render(payload), encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)

    sizes = [len(f["members"]) for f in payload["families"]]
    print(f"\n{len(payload['families'])} 个维度族，大小 {sizes}，合计 {sum(sizes)}", flush=True)
    gap = payload["stage_c_gap"]
    print(f"Stage B 被启发式挡住 {gap['clusters_blocked_by_heuristic_only']} 族，"
          f"已补测 {len(gap['already_probed'])} 个 ⟹ 未测 {len(gap['untested'])} 个", flush=True)


if __name__ == "__main__":
    main()
