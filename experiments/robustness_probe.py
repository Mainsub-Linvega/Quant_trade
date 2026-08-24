"""稳健性诊断器 —— 把「分数」拆成「分数从哪来、有多容易塌」。

## 为什么需要它

私榜是 9/1–9/30 的**一次** 30 天实现，不是多期平均。而 2026-08-24 的两个测量说明
当前模型对「落在哪一段」极其敏感：

```text
同一模型逐段分数   0.0033680 … 0.0058618   最大/最小 1.74×
密封期 4 块 peak   0.0047286 … 0.0088569   最大/最小 1.87×
```

⟹ 全窗均值是一个**期望**，而我们只抽一次。此前所有判据（peak、折均、块均）量的都是
期望，没有任何一个量过「这一次可能有多差」。

## 它拆哪几件事

1. **逐段**：5 段分数、最差段、段间极差/中位 —— 「抽到坏 regime 有多惨」。
2. **逐资产**：单资产分、留一后的全局分、占分母比重 —— 「优势压在谁身上」。
   ⭐ 2026-08-24 首测（`submission_long512.csv`，训练段 ≤888,479 ⟹ 公榜窗口是真样本外）：
   拿掉 asset 8 全局 **−31.2%**；15 个资产里 **5 个单独看是负分**。
3. **逐资产 × 逐段的符号稳定性** —— 区分「稳定有害」与「噪声」，这是决定能不能动手的关键。
   首测：**8/15 个 5/5 段全正**、asset 0 与 3 **5/5 段全负**、5 个符号乱跳。
4. **集中度**：按贡献排序后前 k 个资产占了多少 Δ 分。

## 口径

- 分数 = 加权零均值 R²，与比赛/公榜逐位同口径（`src.metric.weighted_zero_mean_r2`）。
- 「单资产分」是**只在该资产的行上**算的 R²，分母也只用该资产 ⟹ 可正可负，
  与全局分不可直接相加。
- 「留一分」是**去掉该资产的行后**重算的全局 R²，与全局分同口径、可直接比较。
- 分段沿用 `public_replay` 的 `--period-buckets` 口径（按 time_id 分位数等分）。

⚠️ 本脚本**只打分、不训练、不写任何提交格式 CSV**。

用法：
    .venv/bin/python experiments/robustness_probe.py \\
        --labels /path/to/public_release_20260823/data/train \\
        --predictions outputs/submission_long512.csv \\
        --label robustness_long512
    # 不传 --predictions 就体检 outputs/ 下所有 submission_*.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.metric import weighted_zero_mean_r2  # noqa: E402

PREDICTION_COLUMN = "_prediction"          # 与标签的 target/weight 都不撞（同 public_replay）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="8/23 回补包的 data/train 目录")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"),
                   help="用来取 row_id → time_id/asset_id 的索引（test 分区）")
    p.add_argument("--predictions", nargs="*", default=None,
                   help="预测 CSV；省略则体检 --submissions-dir 下全部 submission_*.csv")
    p.add_argument("--submissions-dir", default=str(_REPO_ROOT / "outputs"))
    p.add_argument("--period-buckets", type=int, default=5)
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="robustness_probe")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_index(data_root: Path) -> pd.DataFrame:
    parts = [pq.read_table(path, columns=["row_id", "time_id", "asset_id"]).to_pandas()
             for path in sorted((data_root / "test").glob("*.parquet"))]
    if not parts:
        raise SystemExit(f"{data_root / 'test'} 下没有 parquet")
    return pd.concat(parts, ignore_index=True)


def load_labels(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    want = ["row_id", "weight", "target"]
    frames = []
    for f in files:
        missing = [c for c in want if c not in set(pq.ParquetFile(f).schema_arrow.names)]
        if missing:
            raise SystemExit(f"{f.name} 缺列 {missing} —— 公榜口径需要 weight，拒绝退化成无权")
        frames.append(pq.ParquetFile(f).read(columns=want).to_pandas())
    return pd.concat(frames, ignore_index=True)


def segment_ids(time_id: np.ndarray, n_buckets: int) -> np.ndarray:
    """按 time_id 分位数等分 —— 与 public_replay 的 by_period 同口径。"""
    edges = np.quantile(time_id, np.linspace(0.0, 1.0, n_buckets + 1))
    seg = np.searchsorted(edges, time_id, side="right") - 1
    return np.clip(seg, 0, n_buckets - 1)


def probe(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray,
          asset_id: np.ndarray, time_id: np.ndarray, n_buckets: int) -> dict[str, Any]:
    seg = segment_ids(time_id, n_buckets)
    num_all = float(np.dot(weight, (target - prediction) ** 2))
    den_all = float(np.dot(weight, target * target))
    overall = weighted_zero_mean_r2(target, prediction, weight)

    segments = []
    for s in range(n_buckets):
        k = seg == s
        segments.append({
            "segment": s, "rows": int(k.sum()),
            "time_id_range": [int(time_id[k].min()), int(time_id[k].max())],
            "score": weighted_zero_mean_r2(target[k], prediction[k], weight[k]),
        })
    seg_scores = np.array([b["score"] for b in segments], dtype=np.float64)

    assets = []
    for aid in sorted(set(asset_id.tolist())):
        k = asset_id == aid
        num_k = float(np.dot(weight[k], (target[k] - prediction[k]) ** 2))
        den_k = float(np.dot(weight[k], target[k] * target[k]))
        per_seg = []
        for s in range(n_buckets):
            m = k & (seg == s)
            per_seg.append(weighted_zero_mean_r2(target[m], prediction[m], weight[m])
                           if m.any() else float("nan"))
        arr = np.array(per_seg, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        assets.append({
            "asset_id": int(aid), "rows": int(k.sum()),
            "denominator_share": den_k / den_all if den_all else float("nan"),
            # 只在该资产行上算，分母也只用该资产 ⟹ 可正可负，不与全局分相加
            "solo_score": 1.0 - num_k / den_k if den_k > 0 else float("nan"),
            # 去掉该资产后重算的全局分 ⟹ 与 overall 同口径、可直接比较
            "leave_one_out_score": (1.0 - (num_all - num_k) / (den_all - den_k)
                                    if den_all - den_k > 0 else float("nan")),
            "per_segment_score": per_seg,
            "positive_segments": int((finite > 0).sum()),
            "n_segments": int(finite.size),
        })

    loo = np.array([a["leave_one_out_score"] for a in assets], dtype=np.float64)
    # 贡献 = 拿掉它之后全局掉了多少（正数 = 该资产在贡献分数）
    contribution = overall - loo
    order = np.argsort(-contribution)
    cum = np.cumsum(contribution[order])
    total_positive = float(contribution[contribution > 0].sum())

    return {
        "overall_score": overall,
        "rows": int(target.size),
        "n_assets": len(assets),
        "segments": segments,
        "segment_summary": {
            "worst": float(seg_scores.min()), "best": float(seg_scores.max()),
            "median": float(np.median(seg_scores)),
            "worst_over_overall": float(seg_scores.min() / overall) if overall else float("nan"),
            "max_over_min": float(seg_scores.max() / seg_scores.min())
            if seg_scores.min() > 0 else float("inf"),
            "spread_over_median": float((seg_scores.max() - seg_scores.min())
                                        / np.median(seg_scores)) if np.median(seg_scores) else float("nan"),
        },
        "assets": assets,
        "asset_summary": {
            "n_solo_negative": int(sum(1 for a in assets if a["solo_score"] < 0)),
            "n_all_segments_positive": int(sum(1 for a in assets
                                               if a["positive_segments"] == a["n_segments"])),
            "n_all_segments_negative": int(sum(1 for a in assets
                                               if a["positive_segments"] == 0)),
            "all_segments_negative_ids": [a["asset_id"] for a in assets
                                          if a["positive_segments"] == 0],
            "worst_leave_one_out": float(loo.min()),
            "worst_loo_relative": float(loo.min() / overall - 1.0) if overall else float("nan"),
            "top1_contribution_share": float(cum[0] / total_positive) if total_positive > 0 else float("nan"),
            "top3_contribution_share": float(cum[2] / total_positive)
            if total_positive > 0 and len(cum) > 2 else float("nan"),
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = ["# 稳健性诊断（`robustness_probe`）", "",
             "分数 = 加权零均值 R²，与公榜逐位同口径。**只打分、不训练。**", "",
             "| 预测 | 全窗 | 最差段 | 最差/全窗 | 段最大/最小 | 单资产负分数 | 5/5段全负 | 去最好资产后 |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for name, r in payload["results"].items():
        s, a = r["segment_summary"], r["asset_summary"]
        lines.append(
            f"| `{name}` | {r['overall_score']:.7f} | {s['worst']:.7f} | {s['worst_over_overall']:.0%} "
            f"| {s['max_over_min']:.2f}× | {a['n_solo_negative']}/{r['n_assets']} "
            f"| {a['all_segments_negative_ids'] or '—'} | {a['worst_loo_relative']:+.1%} |")
    lines += ["", "## 读法", "",
              "- **最差/全窗**：私榜若落在类似最差段的 regime，实际拿到的是这个比例的分。",
              "- **5/5段全负**：该资产在每一段都在扣分 —— 与「符号乱跳」不同，这是可动手的证据。",
              "- **去最好资产后**：优势集中度。数字越负，说明越依赖单个资产。", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    index = load_index(Path(args.data_root))
    labels = load_labels(Path(args.labels))
    base = labels.merge(index, on="row_id", how="inner")
    print(f"标签 {len(labels):,} 行；与 test 索引连上 {len(base):,} 行", flush=True)

    paths = ([Path(p) for p in args.predictions] if args.predictions
             else sorted(Path(args.submissions_dir).glob("submission_*.csv")))
    if not paths:
        raise SystemExit("没有找到任何预测 CSV")

    results: dict[str, Any] = {}
    for path in paths:
        frame = pd.read_csv(path)
        col = "target" if "target" in frame.columns else frame.columns[-1]
        # ⚠️ 预测列常常就叫 `target`，与标签列同名 —— 必须先改名再 merge，
        # 否则 pandas 会给出 target_x/target_y（public_replay 曾因此当场 KeyError）。
        merged = (frame[["row_id", col]].rename(columns={col: PREDICTION_COLUMN})
                  .merge(base, on="row_id", how="inner"))
        coverage = len(merged) / max(len(frame), 1)
        if coverage < 1.0:
            print(f"  ⚠️ {path.name}: 标签只覆盖 {coverage:.4%} ⟹ 跳过", flush=True)
            continue
        r = probe(merged["target"].to_numpy(np.float64),
                  merged[PREDICTION_COLUMN].to_numpy(np.float64),
                  np.maximum(merged["weight"].to_numpy(np.float64), 0.0),
                  merged["asset_id"].to_numpy(np.int64),
                  merged["time_id"].to_numpy(np.int64),
                  args.period_buckets)
        r["label_join_coverage"] = coverage
        results[path.name] = r
        s, a = r["segment_summary"], r["asset_summary"]
        print(f"  {path.name:<42} 全窗={r['overall_score']:.7f} 最差段={s['worst']:.7f}"
              f"（{s['worst_over_overall']:.0%}）5/5全负={a['all_segments_negative_ids'] or '—'}",
              flush=True)

    payload = {"experiment": "robustness_probe",
               "question": "分数从哪来、有多容易塌 —— 私榜是一次抽样，不是期望",
               "labels": str(args.labels), "period_buckets": args.period_buckets,
               "results": results, "elapsed_seconds": time.perf_counter() - started}
    out = Path(args.output_dir) / f"{args.label}.json"
    md = out.with_suffix(".md")
    if not args.force and (out.exists() or md.exists()):
        raise SystemExit(f"{out} 或 {md} 已存在；要覆盖请加 --force")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md.write_text(render(payload), encoding="utf-8")
    print(f"\nwrote {out}\nwrote {md}", flush=True)


if __name__ == "__main__":
    main()
