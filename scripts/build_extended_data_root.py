"""把 8/23 标签回补包拼成两个可训练的数据根 —— 不碰 `data/`，边界写进 `root_identity.json`。

## 为什么需要这个脚本

`RUNBOOK_8_23.md` 的 D0/D1 假设「主办方刷新整个 `data/`」，但 2026-08-24 实际收到的
`public_release_20260823` 是一个**增量包**，形态与预注册假设有三处对不上：

1. ⚠️ **文件名冲突。** 回补包里叫 `train_partition_000/001/002.parquet`，与本地
   `data/train/` 前三个分区**同名但内容全异**（本地 p000 = `time_id 0–99999`，
   回补 p000 = `888480–988479`）。任何 `cp` 进 `data/train/` 都会**静默覆盖训练集前 1/3**，
   而且行数相近（1,499,352 vs 1,499,703）不易察觉。⟹ 在这里重命名成 009/010/011。

2. **审计口径。** `audit_data_release.py` 是整目录对比。直接拿回补包当 `--data-root`
   会报「3 modified + 6 removed + test 全 removed」—— 那是在比两个不同的东西。
   重命名成 009/010/011 之后，同一个比较器给出的是
   `added=[009,010,011] / removed=[] / row_delta=+3,217,458`，才是「增量包」的真实语义。

3. ⚠️⚠️ **密封段边界无处安放。** RUNBOOK D1 写死「决策期重训必须止于
   `time_id 1,045,889`，训进去 D2 之后的一切比较全部作废 —— 而且**不会报错**」，但
   `strategies/{v1_ridge,v3_hybrid}/train.py` **都没有时间截断参数**，
   `src/io.py:20` 按 manifest 顺序整分区读。而密封段起点 `1,045,920`
   **落在回补 p001 内部**（该分区 59.1% = 860,986 行在边界前）⟹ 分区级切分做不到。

⟹ 本脚本用**数据根**承载截断：训练脚本一行不改，"训练段止于哪里" 变成
   `<root>/root_identity.json` 里一个可被门禁读取的数字（`retrain_extended.py --role` 消费它）。

## 产出

| role | train 分区 | time_id 上界 | 用途 |
|---|---|---|---|
| `extended_full` | 本地 000–008 + 回补 000/001/002 → 009/010/011 | 1,105,919 | D0.2 审计、D4.5 最终交付件 |
| `decision`      | 本地 000–008 + 回补 000 → 009 + 回补 001 截断 → 010 | 1,045,889 | D1 决策期重训 |

原始分区一律用**符号链接**（4.1 GB 不复制）；只有截断分区是新写的物理文件。
`data/` 全程只读 —— CLAUDE.md §1.1。

边界常量来自 `outputs/experiments/sealed_period_plan.json` 的
`geometry.decision_train_time_id_max`，**不在本文件写死**（CLAUDE.md §7）。

用法：
    .venv/bin/python scripts/build_extended_data_root.py                    # dry-run，先看计划
    .venv/bin/python scripts/build_extended_data_root.py --execute
    .venv/bin/python scripts/build_extended_data_root.py --execute --force  # 覆盖已存在的根
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 回补包（主办方 2026-08-23 发布）解压后的 data/ 目录 —— 位置因机器而异，
# 从 $QUANT_BACKFILL_ROOT 读，或显式传 --backfill-root。
_ENV_BACKFILL = os.environ.get("QUANT_BACKFILL_ROOT", "").strip()
DEFAULT_BACKFILL = Path(_ENV_BACKFILL) if _ENV_BACKFILL else None
SEALED_PLAN = _REPO_ROOT / "outputs" / "experiments" / "sealed_period_plan.json"

ROLE_FULL = "extended_full"
ROLE_DECISION = "decision"
# ⚠️ 2026-08-24 补：`original` 只含主公开包的 train（time_id 0–888,479），**一行回补数据都不收**。
# 它是「本地公榜」协议的训练根 —— 这样训出来的模型在整个公榜窗口
# （888,480–1,105,919）上都是样本外，打出来的分与「当年真的提交上去」逐位可比
# （21 份历史 CSV 已验证）。做成一个带 root_identity.json 的正式 role，
# 而不是直接传 `--data-root data`，是为了让 retrain_extended 的边界门禁照常生效。
ROLE_ORIGINAL = "original"
ROLES = (ROLE_FULL, ROLE_DECISION, ROLE_ORIGINAL)

# 截断时按批过滤再写，避免把 861k × 375 列 float32（约 1.3 GB）一次性拉进内存。
_BATCH_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"),
                   help="主公开包（只读，不会被写入）")
    p.add_argument("--backfill", default=DEFAULT_BACKFILL,
                   help="8/23 回补包的 data/ 目录（缺省取 $QUANT_BACKFILL_ROOT）")
    p.add_argument("--out-dir", default=str(_REPO_ROOT / "outputs" / "data_roots"))
    p.add_argument("--sealed-plan", default=str(SEALED_PLAN),
                   help="密封期几何的唯一真值源；决策期边界从这里读")
    p.add_argument("--roles", nargs="*", default=list(ROLES), choices=list(ROLES))
    p.add_argument("--execute", action="store_true", help="不加就是 dry-run，只打印计划")
    p.add_argument("--force", action="store_true", help="覆盖已存在的数据根")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decision_cutoff(plan_path: Path) -> int:
    """决策期训练段的 time_id 上界 —— 只从密封期计划读，不在本文件写第二份。"""
    if not plan_path.is_file():
        raise SystemExit(f"找不到密封期计划 {plan_path} —— 边界不能靠本文件猜")
    geometry = json.loads(plan_path.read_text(encoding="utf-8")).get("geometry") or {}
    cutoff = geometry.get("decision_train_time_id_max")
    if not isinstance(cutoff, int):
        raise SystemExit(f"{plan_path} 的 geometry.decision_train_time_id_max 缺失或不是整数")
    seal_min = geometry.get("seal_time_id_min")
    embargo = geometry.get("embargo_real_time_ids")
    # embargo 是**空出来的 time_id 个数**：seal_min-1 往前数 embargo 个都不训，
    # 所以训练段末尾是 seal_min - embargo - 1（30 个空档 = 1,045,890–1,045,919）。
    if isinstance(seal_min, int) and isinstance(embargo, int) \
            and cutoff != seal_min - embargo - 1:
        raise SystemExit(f"密封期计划自相矛盾：cutoff {cutoff} != seal_min {seal_min} - "
                         f"embargo {embargo} - 1；先修计划再建数据根")
    return cutoff


def manifest_train_paths(root: Path) -> list[Path]:
    """与 `src/io.train_files` / `audit_data_release.manifest_paths` 同口径。"""
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        names = (json.loads(manifest_path.read_text(encoding="utf-8"))
                 .get("files", {}).get("train", []))
        if names:
            return [root / str(name) for name in names]
    return sorted((root / "train").glob("*.parquet"))


def probe(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    out: dict[str, Any] = {"rows": parquet.metadata.num_rows, "n_columns": len(schema.names)}
    for column in ("time_id", "row_id"):
        index = schema.get_field_index(column)
        low = high = None
        for group in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(group).column(index).statistics
            if stats is None or not stats.has_min_max:
                low = high = None
                break
            low = int(stats.min) if low is None else min(low, int(stats.min))
            high = int(stats.max) if high is None else max(high, int(stats.max))
        out[f"{column}_min"], out[f"{column}_max"] = low, high
    return out


def plan_members(data_root: Path, backfill: Path, role: str, cutoff: int) -> list[dict[str, Any]]:
    """返回 [{name, source, truncate_at|None}]，name 是它在新根里的文件名。"""
    local = manifest_train_paths(data_root)
    if not local:
        raise SystemExit(f"{data_root} 里没有 train 分区")
    extra = sorted((backfill / "train").glob("*.parquet"))
    if not extra:
        raise SystemExit(f"{backfill}/train 里没有 parquet —— 回补包路径对吗？")

    members = [{"name": path.name, "source": path, "truncate_at": None} for path in local]
    if role == ROLE_ORIGINAL:
        return members                      # 一行回补数据都不收
    index = len(local)
    for path in extra:
        info = probe(path)
        name = f"train_partition_{index:03d}.parquet"
        if role == ROLE_DECISION:
            if info["time_id_min"] > cutoff:            # 整个分区都在密封段里 ⟹ 不收
                continue
            if info["time_id_max"] > cutoff:            # 边界落在分区内部 ⟹ 截断，且它是最后一个
                members.append({"name": name, "source": path, "truncate_at": cutoff})
                index += 1
                break
        members.append({"name": name, "source": path, "truncate_at": None})
        index += 1
    return members


def write_truncated(source: Path, target: Path, cutoff: int) -> None:
    """按 `time_id <= cutoff` 流式过滤重写，schema / 列序 / dtype / 压缩全部沿用源文件。"""
    reader = pq.ParquetFile(source)
    writer = pq.ParquetWriter(target, reader.schema_arrow, compression="zstd")
    kept = 0
    try:
        for batch in reader.iter_batches(batch_size=_BATCH_ROWS):
            time_ids = batch.column(batch.schema.get_field_index("time_id")).to_numpy()
            mask = time_ids <= cutoff
            if not mask.any():
                break                                   # time_id 单调不减 ⟹ 后面不会再有
            table = pa.Table.from_batches([batch]).filter(pa.array(mask))
            writer.write_table(table)
            kept += int(mask.sum())
            if not mask.all():
                break
    finally:
        writer.close()
    print(f"  截断写出 {target.name}: {kept:,} 行（time_id <= {cutoff:,}）", flush=True)


def build(root: Path, data_root: Path, members: list[dict[str, Any]], role: str,
          cutoff: int, execute: bool) -> dict[str, Any]:
    if execute:
        (root / "train").mkdir(parents=True, exist_ok=True)
        for member in members:
            target = root / "train" / member["name"]
            if member["truncate_at"] is None:
                target.symlink_to(member["source"].resolve())
            else:
                write_truncated(member["source"], target, member["truncate_at"])
        # test/ 与 sample_submission 一并链接进来，保持 audit 的 split 比较与
        # `--data-root` 的通用性（官方 runner 也认这个布局）。
        for name in ("test", "sample_submission.csv"):
            source = data_root / name
            if source.exists():
                (root / name).symlink_to(source.resolve())

    entries: list[dict[str, Any]] = []
    for member in members:
        target = root / "train" / member["name"]
        entry = {"name": member["name"], "source": str(member["source"].resolve()),
                 "truncated_at": member["truncate_at"]}
        if execute:
            entry |= probe(target) | {"sha256": sha256_file(target),
                                      "bytes": target.stat().st_size}
        entries.append(entry)

    manifest = {
        "competition": "quantcontest2026",
        "version": f"extended_{role}_20260824",
        "description": (f"派生数据根（role={role}）：主公开包 train + 8/23 标签回补包。"
                        f"由 scripts/build_extended_data_root.py 生成，data/ 只读未改。"),
        "files": {"train": [f"train/{m['name']}" for m in members],
                  "test": [f"test/{p.name}" for p in sorted((data_root / "test").glob("*.parquet"))],
                  "sample_submission": "sample_submission.csv"},
        "rows": {"train": sum(e.get("rows", 0) for e in entries) if execute else None},
    }
    identity = {
        "role": role,
        "generated_by": "scripts/build_extended_data_root.py",
        "data_root": str(data_root.resolve()),
        "sealed_plan_cutoff": cutoff,
        "train_partitions": len(members),
        "train_rows": sum(e.get("rows", 0) for e in entries) if execute else None,
        "train_time_id_min": min((e["time_id_min"] for e in entries
                                  if e.get("time_id_min") is not None), default=None),
        "train_time_id_max": max((e["time_id_max"] for e in entries
                                  if e.get("time_id_max") is not None), default=None),
        "truncated_member": next((e["name"] for e in entries
                                  if e["truncated_at"] is not None), None),
        "members": entries,
    }
    if execute:
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (root / "root_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        expected = cutoff if role == ROLE_DECISION else None
        if expected is not None and identity["train_time_id_max"] != expected:
            raise SystemExit(f"⚠️ {role} 根的 train_time_id_max = "
                             f"{identity['train_time_id_max']} != {expected}；产物不可用")
    return identity


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not args.backfill:
        raise SystemExit(
            "找不到回补包：设置 $QUANT_BACKFILL_ROOT，或传 --backfill <解压后的 data/ 目录>")
    backfill = Path(args.backfill)
    out_dir = Path(args.out_dir)
    cutoff = decision_cutoff(Path(args.sealed_plan))
    print(f"决策期训练段上界（来自 {Path(args.sealed_plan).name}）：time_id <= {cutoff:,}\n",
          flush=True)

    summary: dict[str, Any] = {}
    for role in args.roles:
        root = out_dir / role
        members = plan_members(data_root, backfill, role, cutoff)
        print(f"=== role={role} → {root} ===", flush=True)
        for member in members:
            mark = "  ← 链接" if member["truncate_at"] is None else \
                   f"  ← 截断至 time_id <= {member['truncate_at']:,}"
            print(f"  {member['name']:<32} {member['source']}{mark}", flush=True)

        if args.execute:
            if root.exists():
                if not args.force:
                    raise SystemExit(f"{root} 已存在；要重建请加 --force")
                # 只删本脚本自己生成的派生根，且逐项确认是链接或本目录内的文件。
                for child in sorted(root.rglob("*"), reverse=True):
                    if child.is_symlink() or not child.is_dir():
                        child.unlink()
                    else:
                        child.rmdir()
                root.rmdir()
        identity = build(root, data_root, members, role, cutoff, args.execute)
        summary[role] = identity
        print(f"  分区 {identity['train_partitions']} 个，"
              f"train_rows={identity['train_rows']}，"
              f"time_id ≤ {identity['train_time_id_max']}\n", flush=True)

    if not args.execute:
        print("（dry-run；确认计划无误后加 --execute）", flush=True)
    else:
        print(json.dumps({role: {k: v for k, v in ident.items() if k != "members"}
                          for role, ident in summary.items()},
                         ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
