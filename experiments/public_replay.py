"""公榜分数离线复算器 —— 8/23 回补标签一到就能修尺子。

## 它要回答什么

`NOTES.md` §6「未解问题」第一条：**「为什么多个拟合紧密度参数在本地与公榜量反；
需要真实回补标签按时期分解。」** 这是全项目最高价值的未解项 —— 它修的是**尺子**不是模型。
8/23 之后公榜停更、8 天内交私榜，本地尺子是唯一还有的东西。

主办方 timeline 写明 8/23「发布扩展训练数据供最终模型训练与**本地验证**使用」，
`docs/data_description.md:173` 说公榜截止后发布标签回补数据 ⟹ 那天起可以对**每一份历史
公榜提交**离线算出真实分数，进而按时期拆解「本地 Δ% → 公榜 Δ%」的实测关系。

## 材料（全部现成）

- `experiments/ledger.csv`：31 行**全部**有 `public_score`；
- `outputs/submission_*.csv`：21 份，含逐行预测值；
- `outputs/experiments/public_csv_fingerprints.json`：另 18 份的 sha256 + 公榜分数 + 模型归属。

⚠️ 指纹存档**只有 Gram 矩阵、没有逐行值**（见该脚本 docstring）。其中 14 份的 CSV 已删，
且 `legacy_a2e6` / `legacy_a5e5` 两个模型「已经不在仓库里」⟹ **那几份永久不可复算**，
本脚本如实列出，不假装能补。

## ⭐ 自校验优先

第一个产出不是分析，是**「离线复算的分数 == 主办方公布的分数」**。CSV 是 8 位小数，
预期吻合到 ~1e-8。对不上就先修复算器，**不解释现象** —— 否则后面所有拆解都是在解释 bug。
归属表里 `inferred` 那些也靠这一步验证：复算分数应当落在它被指派的那个 ledger 行上。

用法：
    # 本轮：干跑，不需要标签
    .venv/bin/python experiments/public_replay.py
    # 8/23：有了回补标签
    .venv/bin/python experiments/public_replay.py --labels <回补数据目录或 parquet>
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import csv
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

from src.metric import weighted_zero_mean_r2  # noqa: E402

# 文件名 → (归属置信度, ledger 行号 0-based)
#   sha256      —— public_csv_fingerprints.json 里有该文件的 sha256 与公榜分数，最硬
#   ledger_text —— 文件名在 ledger 的 change 原文里出现
#   inferred    —— 按模型名/参数推断，**待 8/23 用复算分数验证**
ATTRIBUTION: dict[str, tuple[str, int]] = {
    "submission_replace_r80_s116.csv":   ("sha256", 11),
    "submission_replace_s116.csv":       ("sha256", 10),
    "submission_r320_s116.csv":          ("sha256", 12),
    "submission_r480_s116.csv":          ("sha256", 13),
    "submission_mkt_shrunk_slowfast.csv": ("ledger_text", 28),
    "submission_xs_moderate.csv":        ("ledger_text", 27),
    "submission_market_s130.csv":        ("inferred", 14),
    "submission_mix2_r480.csv":          ("inferred", 15),
    "submission_phasebal_r480_s116.csv": ("inferred", 16),
    "submission_hist_r480_s116.csv":     ("inferred", 17),
    "submission_hist_r480_s090.csv":     ("inferred", 18),
    "submission_hist_r160_s116.csv":     ("inferred", 19),
    "submission_hist_c80_s116.csv":      ("inferred", 20),
    "submission_hist_r320_s116.csv":     ("inferred", 21),
    "submission_mktwe_s116.csv":         ("inferred", 22),
    "submission_mkt_shrunk.csv":         ("inferred", 23),
    "submission_mkt_moderate.csv":       ("inferred", 24),
    "submission_r960_pb_hist_mktwe.csv": ("inferred", 25),
    "submission_xs_shrunk.csv":          ("inferred", 26),
    "submission_asset_cross.csv":        ("inferred", 29),
    "submission_slowfast_runner.csv":    ("inferred", 30),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--submissions-dir", default=str(_REPO_ROOT / "outputs"))
    p.add_argument("--ledger", default=str(_REPO_ROOT / "experiments" / "ledger.csv"))
    p.add_argument("--fingerprints", default=str(_REPO_ROOT / "outputs" / "experiments" /
                                                 "public_csv_fingerprints.json"))
    p.add_argument("--labels", default=None,
                   help="8/23 回补标签（parquet 文件或目录）。缺省 = 干跑，只出配对清单")
    p.add_argument("--period-buckets", type=int, default=5,
                   help="按 time_id 均分成几段做时期拆解")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    label = args.label or ("public_replay_scored" if args.labels else "public_replay_inventory")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    import pyarrow.parquet as pq

    ledger = list(csv.DictReader(Path(args.ledger).open(encoding="utf-8")))
    fingerprints = json.loads(Path(args.fingerprints).read_text(encoding="utf-8"))
    fp_by_file = {f["file"]: f for f in fingerprints.get("files", [])}

    # ---- test 索引：row_id → time_id / asset_id（无需标签，现在就能建）
    test_files = sorted((Path(args.data_root) / "test").glob("*.parquet"))
    frames = [pq.ParquetFile(p).read(columns=["row_id", "time_id", "asset_id"]).to_pandas()
              for p in test_files]
    import pandas as pd
    index = pd.concat(frames, ignore_index=True)
    del frames
    row_id = index["row_id"].to_numpy()
    test_time = index["time_id"].to_numpy(np.int64)
    test_asset = index["asset_id"].to_numpy(np.int64)
    order = np.argsort(row_id, kind="stable")
    row_sorted = row_id[order]
    if len(np.unique(row_sorted)) != len(row_sorted):
        raise AssertionError("test row_id 有重复")
    print(f"test 索引：{len(row_sorted):,} 行 / {len(np.unique(test_time)):,} 个 time_id",
          flush=True)

    sample_path = Path(args.data_root) / "sample_submission.csv"
    sample_rows = pd.read_csv(sample_path, usecols=["row_id"])["row_id"].to_numpy()
    if not np.array_equal(np.sort(sample_rows), row_sorted):
        raise AssertionError("sample_submission.csv 的 row_id 与 test 分区对不上")

    # ---- 标签（8/23 才有）
    labels = None
    if args.labels:
        lp = Path(args.labels)
        label_files = sorted(lp.glob("*.parquet")) if lp.is_dir() else [lp]
        want = ["row_id", "target", "weight"]
        got = []
        for p in label_files:
            names = set(pq.ParquetFile(p).schema_arrow.names)
            missing = [c for c in want if c not in names]
            if missing:
                raise SystemExit(
                    f"{p.name} 缺列 {missing}。公榜口径是 Σw(y−ŷ)²/Σw·y²，"
                    "**没有 weight 就不能算**，本脚本拒绝静默退化成无权。")
            got.append(pq.ParquetFile(p).read(columns=want).to_pandas())
        labels = pd.concat(got, ignore_index=True)
        print(f"标签：{len(labels):,} 行", flush=True)

    # ---- 逐份 CSV
    sub_dir = Path(args.submissions_dir)
    entries: list[dict[str, Any]] = []
    for path in sorted(sub_dir.glob("submission_*.csv")):
        rec: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size,
                               "sha256": sha256_file(path)}
        conf, led_idx = ATTRIBUTION.get(path.name, ("unattributed", -1))
        rec["attribution"] = conf
        if led_idx >= 0:
            row = ledger[led_idx]
            rec["ledger_row"] = led_idx
            rec["ledger_date"] = row["date"]
            rec["published_public_score"] = float(row["public_score"])
        fp = fp_by_file.get(path.name)
        if fp:
            rec["fingerprint_public"] = fp.get("public")
            rec["fingerprint_model"] = fp.get("model")
            rec["sha256_matches_fingerprint"] = (fp.get("sha256") == rec["sha256"])
            if rec.get("published_public_score") is not None and fp.get("public") is not None:
                rec["fingerprint_vs_ledger_consistent"] = bool(
                    abs(float(fp["public"]) - rec["published_public_score"]) < 1e-12)

        frame = pd.read_csv(path)
        pred_col = "target" if "target" in frame.columns else frame.columns[-1]
        ids = frame["row_id"].to_numpy()
        pred = frame[pred_col].to_numpy(np.float64)
        rec["rows"] = int(len(ids))
        rec["rows_match_test"] = bool(len(ids) == len(row_sorted))
        pos = np.searchsorted(row_sorted, ids)
        hit = (pos < len(row_sorted)) & (row_sorted[np.minimum(pos, len(row_sorted) - 1)] == ids)
        rec["row_id_join_coverage"] = float(hit.mean())
        rec["non_finite"] = int((~np.isfinite(pred)).sum())
        rec["max_abs_prediction"] = float(np.max(np.abs(pred)))

        if labels is not None:
            merged = frame[["row_id", pred_col]].merge(labels, on="row_id", how="inner")
            coverage = float(len(merged) / max(len(frame), 1))
            rec["label_join_coverage"] = coverage
            # ⚠️ 部分 join 会算出一个**看起来正常但错的**分数。回补数据若换了 row_id 口径，
            # 这里必须响 —— 不响的话 8/23 会拿一个静默错误的尺子去做采纳决策。
            if coverage < 1.0:
                rec["join_warning"] = (
                    f"标签只覆盖 {coverage:.4%} 的预测行 ⟹ 离线分数**不可与公布分数比较**。"
                    "若回补数据的 row_id 与 test 不同口径，改按 (time_id, asset_id) 连接。")
            y = merged["target"].to_numpy(np.float64)
            w = np.maximum(merged["weight"].to_numpy(np.float64), 0.0)
            p = merged[pred_col].to_numpy(np.float64)
            offline = weighted_zero_mean_r2(y, p, w)
            rec["offline_score"] = float(offline)
            pub = rec.get("published_public_score")
            if pub is not None:
                rec["offline_minus_published"] = float(offline - pub)
                # join 不完整时一律判为「未复现」，不给它蒙对的机会
                rec["reproduces_published"] = bool(coverage >= 1.0 and abs(offline - pub) < 1e-7)
            # 时期拆解
            tpos = np.searchsorted(row_sorted, merged["row_id"].to_numpy())
            tid = test_time[order][tpos]
            edges = np.quantile(tid, np.linspace(0, 1, args.period_buckets + 1))
            rec["by_period"] = []
            for b in range(args.period_buckets):
                m = (tid >= edges[b]) & (tid <= edges[b + 1] if b == args.period_buckets - 1
                                         else tid < edges[b + 1])
                if m.sum():
                    rec["by_period"].append({
                        "bucket": b, "rows": int(m.sum()),
                        "time_id_range": [int(tid[m].min()), int(tid[m].max())],
                        "score": float(weighted_zero_mean_r2(y[m], p[m], w[m]))})
        entries.append(rec)
        print(f"  {path.name:<42} {rec['attribution']:<12} "
              f"public={rec.get('published_public_score')} "
              + (f"offline={rec['offline_score']:.10f}" if labels is not None else ""),
              flush=True)

    # ---- 不可复算的那些
    on_disk = {e["file"] for e in entries}
    lost = [{"file": f, "public": info.get("public"), "model": info.get("model"),
             "recoverable": info.get("model") not in ("legacy_a2e6", "legacy_a5e5")}
            for f, info in fp_by_file.items() if f not in on_disk]
    attributed = {e["ledger_row"] for e in entries if "ledger_row" in e}
    ledger_uncovered = [{"row": i, "date": r["date"], "public_score": r["public_score"]}
                        for i, r in enumerate(ledger) if i not in attributed]

    payload = {
        "experiment": "public_replay",
        "mode": "scored" if labels is not None else "inventory",
        "question": "把每一份历史公榜提交在回补标签上离线复算，回答本地↔公榜量反的机制",
        "self_check_note": ("第一个产出必须是「离线复算 == 公布分数」。对不上先修复算器，"
                            "不解释现象 —— 否则后面的拆解是在解释 bug"),
        "counts": {
            "csv_on_disk": len(entries),
            "attributed": sum(1 for e in entries if e["attribution"] != "unattributed"),
            "by_confidence": {c: sum(1 for e in entries if e["attribution"] == c)
                              for c in ("sha256", "ledger_text", "inferred", "unattributed")},
            "ledger_rows_with_public_score": sum(1 for r in ledger if (r["public_score"] or "").strip()),
            "ledger_rows_uncovered": len(ledger_uncovered),
            "fingerprint_only_lost_csv": len(lost),
            "permanently_unrecoverable": sum(1 for x in lost if not x["recoverable"]),
        },
        "entries": entries,
        "fingerprint_only": lost,
        "ledger_uncovered": ledger_uncovered,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    c = payload["counts"]
    lines = [f"# 公榜分数离线复算（{payload['mode']}）", "",
             f"> {payload['self_check_note']}", "",
             "## 配对清单", "",
             f"- 盘上 CSV **{c['csv_on_disk']}** 份，全部已归属"
             f"（sha256 {c['by_confidence']['sha256']} / ledger 原文 "
             f"{c['by_confidence']['ledger_text']} / 推断 {c['by_confidence']['inferred']}"
             f" / 未归属 {c['by_confidence']['unattributed']}）",
             f"- ledger 有公榜分数的行：**{c['ledger_rows_with_public_score']}**，"
             f"其中 {c['ledger_rows_uncovered']} 行没有对应的逐行 CSV",
             f"- 只剩指纹、CSV 已删：**{c['fingerprint_only_lost_csv']}** 份，"
             f"其中 **{c['permanently_unrecoverable']}** 份模型已不在仓库 ⟹ 永久不可复算", "",
             "| CSV | 归属 | ledger 日期 | 公布分数 | 行数 | row_id 覆盖 |"
             + ("离线分数 | 差值 | 复现 |" if labels is not None else ""),
             "|---|---|---|---:|---:|---:|" + ("---:|---:|:--:|" if labels is not None else "")]
    for e in entries:
        base = (f"| `{e['file']}` | {e['attribution']} | {e.get('ledger_date','—')} | "
                f"{e.get('published_public_score','—')} | {e['rows']:,} | "
                f"{e['row_id_join_coverage']:.4%} |")
        if labels is not None:
            base += (f" {e.get('offline_score', float('nan')):.10f} | "
                     f"{e.get('offline_minus_published', float('nan')):+.2e} | "
                     f"{'✅' if e.get('reproduces_published') else '❌'} |")
        lines.append(base)
    if lost:
        lines += ["", "## 只剩指纹（CSV 已删）", "",
                  "| 文件 | 公榜分数 | 模型 | 能否靠重跑补回 |", "|---|---:|---|:--:|"]
        lines += [f"| `{x['file']}` | {x['public']} | {x['model']} | "
                  f"{'重跑可补' if x['recoverable'] else '❌ 模型已不在仓库'} |" for x in lost]
    lines += ["", f"## 8/23 待办", "",
              "1. `--labels <回补数据>` 重跑本脚本；**先看「复现」那一列全绿**，再看任何拆解；",
              "2. `inferred` 归属靠复算分数落在指派的 ledger 行上来验证；",
              "3. 复现通过后按时期/分块/资产拆解，出「本地 Δ% → 公榜 Δ%」实测斜率。", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
