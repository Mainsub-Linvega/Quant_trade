"""本地公榜 —— 对训练段止于 888,479 的候选，在公榜窗口上无限次精确打分。

## 它为什么成立

8/23 的标签回补给了公榜窗口（`time_id 888,480–1,105,919`，3,217,458 行）的真标签。
`public_replay` 已用 **21 份历史提交**验过：离线复算与主办方公布的分数逐位一致
（最大偏差 **1.9e-09**）。⟹ 只要候选的训练段**止于 888,479**，
在这个窗口上打出来的分就与「当年真的提交上去」等价，且可以打无限次。

## 但无限次不等于免费

在 R²≈0.004 的信号上无限次打同一个固定的历史窗口，必然刷出假赢家 ——
ROADMAP 自己就是这么推断榜首的（约 270 次评分 ⟹ 那个 0.0060 是**上界**）。
所以窗口被切成两半，纪律由本脚本强制：

```text
搜索段  888,480 – 1,045,919   157,440 time_id / 2,361,139 行   随便打
确认段  1,045,920 – 1,105,919   60,000 time_id /   856,319 行   **每个候选只打一次**
```

⭐ **确认段对本协议的候选是干净的**：它只对「训练到那里」的生产模型失效；
训练段 ≤888,479 的候选从未见过它。

⚠️ 确认段的调用次数记在 `outputs/experiments/local_lb_confirm_ledger.json`，
同一个候选第二次调用**直接拒绝**。这不是提醒，是门禁。

用法：
    # 搜索段（可反复）
    .venv/bin/python experiments/local_public_lb.py --labels <回补目录> \\
        --baseline seal_seeds3 --arm seeds10=seal_seeds10 --segment search
    # 确认段（每个候选一次）
    .venv/bin/python experiments/local_public_lb.py ... --segment confirm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.metric import scale_invariant_score, weighted_zero_mean_r2  # noqa: E402
from sealed_period_eval import CACHE_PREFIX, load_backfill_labels, seal_geometry  # noqa: E402

TRAIN_CUT = 888_479                 # 本协议要求候选训练段止于此
N_BLOCKS = 4                        # 每段切 4 块，与密封期同口径
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026
CONFIRM_LEDGER = _REPO_ROOT / "outputs" / "experiments" / "local_lb_confirm_ledger.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--cache-dir", default=str(_REPO_ROOT / "outputs" / "cache"))
    p.add_argument("--baseline", required=True, help="基准臂的 cache label")
    p.add_argument("--arms", nargs="+", required=True, metavar="NAME=LABEL")
    p.add_argument("--segment", choices=("search", "confirm"), required=True)
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_pred(cache_dir: Path, label: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = cache_dir / f"{CACHE_PREFIX}{label}.npz"
    if not path.is_file():
        raise SystemExit(f"找不到 {path} —— 先用 sealed_period_eval --candidate 出预测")
    d = np.load(path, allow_pickle=True)
    return (d["row_id"].astype(np.int64), d["prediction"].astype(np.float64),
            json.loads(str(d["meta_json"])))


def test_index(data_root: Path) -> pd.DataFrame:
    return pd.concat([pq.read_table(p, columns=["row_id", "time_id"]).to_pandas()
                      for p in sorted((data_root / "test").glob("*.parquet"))],
                     ignore_index=True)


def blocks_for(time_id: np.ndarray, lo: int, hi: int) -> list[dict[str, Any]]:
    """把 [lo, hi] 等分成 N_BLOCKS 块 —— 与密封期的分块口径一致。"""
    span = hi - lo + 1
    size = span // N_BLOCKS
    out = []
    for b in range(N_BLOCKS):
        b_lo = lo + b * size
        b_hi = hi if b == N_BLOCKS - 1 else b_lo + size - 1
        out.append({"block": b, "time_id_min": b_lo, "time_id_max": b_hi})
    return out


def arm_metrics(target, pred, weight, time_id, blks) -> dict[str, Any]:
    per_block = []
    for b in blks:
        k = (time_id >= b["time_id_min"]) & (time_id <= b["time_id_max"])
        per_block.append({**b, "rows": int(k.sum()),
                          "peak": scale_invariant_score(target[k], pred[k], weight[k])["peak"]})
    return {"score": weighted_zero_mean_r2(target, pred, weight),
            "pooled": dict(scale_invariant_score(target, pred, weight)),
            "per_block": per_block}


def paired_bootstrap(target, base, arm, weight, time_id, blks, rng) -> dict[str, float]:
    """按块内 chunk 重抽的配对 bootstrap —— 与密封期同思路。"""
    chunks = []
    for b in blks:
        k = (time_id >= b["time_id_min"]) & (time_id <= b["time_id_max"])
        idx = np.flatnonzero(k)
        for part in np.array_split(idx, 25):
            if part.size:
                chunks.append(part)
    rel = []
    for _ in range(BOOTSTRAP_SAMPLES):
        pick = rng.integers(0, len(chunks), len(chunks))
        sel = np.concatenate([chunks[i] for i in pick])
        pb = scale_invariant_score(target[sel], base[sel], weight[sel])["peak"]
        pa = scale_invariant_score(target[sel], arm[sel], weight[sel])["peak"]
        if pb > 0:
            rel.append(pa / pb - 1.0)
    a = np.array(rel)
    return {"samples": int(a.size), "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5)), "median": float(np.median(a))}


def confirm_guard(labels: list[str], write: bool) -> None:
    """确认段每个候选只准打一次 —— 第二次直接拒绝。"""
    led = json.loads(CONFIRM_LEDGER.read_text(encoding="utf-8")) if CONFIRM_LEDGER.is_file() else {}
    used = [c for c in labels if c in led]
    if used:
        raise SystemExit(
            f"⛔ 这些候选已经在确认段上打过分：{used}\n"
            f"   打分时间：{ {c: led[c] for c in used} }\n"
            "   确认段每个候选只准用一次 —— 再打就是在用留出集做搜索，本脚本拒绝。")
    if write:
        led.update({c: time.strftime("%Y-%m-%d %H:%M:%S") for c in labels})
        CONFIRM_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        CONFIRM_LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cache_dir = Path(args.cache_dir)
    arms = dict(s.split("=", 1) for s in args.arms)

    if args.segment == "confirm":
        confirm_guard(list(arms.values()) + [args.baseline], write=False)

    base_row, base_pred, base_meta = load_pred(cache_dir, args.baseline)
    label_row, target_all, weight_all = load_backfill_labels(Path(args.labels))
    index = test_index(Path(args.data_root))

    order = np.argsort(label_row)
    pos = np.searchsorted(label_row[order], base_row)
    if not np.array_equal(label_row[order][pos], base_row):
        raise SystemExit("预测 row_id 与标签对不齐")
    target = target_all[order][pos]
    weight = np.maximum(weight_all[order][pos], 0.0)

    iorder = np.argsort(index["row_id"].to_numpy(np.int64))
    irow = index["row_id"].to_numpy(np.int64)[iorder]
    itime = index["time_id"].to_numpy(np.int64)[iorder]
    time_id = itime[np.searchsorted(irow, base_row)]

    geom = seal_geometry(int(time_id.min()), int(time_id.max()))
    seal_lo = geom["seal_time_id_min"]
    if args.segment == "search":
        lo, hi = int(time_id.min()), seal_lo - 1
    else:
        lo, hi = seal_lo, int(time_id.max())
    k = (time_id >= lo) & (time_id <= hi)
    blks = blocks_for(time_id, lo, hi)
    print(f"{args.segment} 段：time_id {lo:,}–{hi:,}   {int(k.sum()):,} 行   {N_BLOCKS} 块",
          flush=True)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_m = arm_metrics(target[k], base_pred[k], weight[k], time_id[k], blks)
    print(f"  基准 {args.baseline}: score={base_m['score']:.7f} peak={base_m['pooled']['peak']:.7f}",
          flush=True)

    results = []
    for name, lab in arms.items():
        row, pred, meta = load_pred(cache_dir, lab)
        if not np.array_equal(row, base_row):
            raise SystemExit(f"{lab} 的 row_id 与基准不同 —— 不是配对比较")
        m = arm_metrics(target[k], pred[k], weight[k], time_id[k], blks)
        rel_blocks = [a["peak"] / b["peak"] - 1.0
                      for a, b in zip(m["per_block"], base_m["per_block"])]
        bs = paired_bootstrap(target[k], base_pred[k], pred[k], weight[k], time_id[k], blks, rng)
        rec = {"name": name, "cache_label": lab, "identity": meta.get("identity"),
               "score": m["score"], "score_relative": m["score"] / base_m["score"] - 1.0,
               "pooled_peak": m["pooled"]["peak"],
               "peak_relative": m["pooled"]["peak"] / base_m["pooled"]["peak"] - 1.0,
               "block_mean_relative": float(np.mean(rel_blocks)),
               "per_block_relative": rel_blocks,
               "positive_blocks": int(sum(1 for r in rel_blocks if r > 0)),
               "n_blocks": N_BLOCKS,
               "drop_best_block_relative": float(np.mean(sorted(rel_blocks)[:-1])),
               "bootstrap": bs}
        results.append(rec)
        print(f"  {name:<14} score={m['score']:.7f}（{100*rec['score_relative']:+.2f}%）"
              f" 块均 peak {100*rec['block_mean_relative']:+.2f}%"
              f" 正块 {rec['positive_blocks']}/{N_BLOCKS}"
              f" CI [{100*bs['ci_low']:+.2f}%, {100*bs['ci_high']:+.2f}%]", flush=True)

    if args.segment == "confirm":
        confirm_guard(list(arms.values()) + [args.baseline], write=True)
        print("\n⚠️ 已记入确认段账本 —— 这些候选不能再在确认段上打分。", flush=True)

    payload = {"experiment": "local_public_lb", "segment": args.segment,
               "train_cut_required": TRAIN_CUT,
               "window": {"time_id_min": lo, "time_id_max": hi, "rows": int(k.sum())},
               "blocks": blks, "baseline": {"cache_label": args.baseline, **base_m},
               "arms": results, "elapsed_seconds": time.perf_counter() - started}
    label = args.label or f"local_lb_{args.segment}"
    out = Path(args.output_dir) / f"{label}.json"
    if not args.force and out.exists():
        raise SystemExit(f"{out} 已存在；要覆盖请加 --force")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
