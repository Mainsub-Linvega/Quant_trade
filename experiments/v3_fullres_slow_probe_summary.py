"""把多个 fold 的全分辨率探针窗合并成一个估计。

单个 20,000 真实 time_id 的窗口只有主实验约 4% 的数据量，CI 必然跨 0。
本脚本读各 fold 已落盘的预测缓存（`v3_fullres_slow_probe.py` 产生），
每折用**自己的**冻结系数（在它之前的折上解出）打分，再把 delta 与能量across fold 汇总，
最后做一次跨折的 block bootstrap。

这只提高「口径是否翻向」这个问题的功效，**不是**对效应大小的独立确认 ——
主证据仍然是 `v3_slow_variance` 的 5 折 OOF。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \\
        .venv/bin/python experiments/v3_fullres_slow_probe_summary.py --folds 2 3 4
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import time_sample_mask  # noqa: E402
from v3_fullres_slow_probe import (K_REAL, K_SAMPLED, SAMPLE_MODULO, SAMPLING,  # noqa: E402
                                    SELECTED_K_SAMPLED, causal_trailing_mean,
                                    frozen_coefficients, group_moment_rows, score_frozen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--cache-dir", default=str(_REPO_ROOT / "outputs" / "cache"))
    p.add_argument("--slow-variance-json",
                   default=str(_REPO_ROOT / "outputs" / "experiments" / "v3_slow_variance_3s480.json"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_fullres_slow_probe_summary")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    # 每折：读缓存 → 逐 K、逐分辨率算出「逐 time_id 的矩」和该折自己的冻结系数
    per_fold: dict[int, dict[str, Any]] = {}
    for fold in args.folds:
        cache = Path(args.cache_dir) / f"v3_fullres_slow_probe_fold{fold}_predictions.npz"
        if not cache.exists():
            raise SystemExit(f"missing probe cache for fold {fold}: {cache}")
        with np.load(cache, allow_pickle=False) as c:
            y, w = c["target"], c["weight"]
            tid, aid = c["time_id"], c["asset_id"]
            raw = c["prediction_raw"]
        frozen = frozen_coefficients(Path(args.slow_variance_json), fold)
        sampled = time_sample_mask(tid, SAMPLE_MODULO, sampling=SAMPLING)
        entry: dict[str, Any] = {"rows_full": int(len(y)), "rows_sampled": int(sampled.sum()),
                                 "window": [int(tid.min()), int(tid.max())], "by_K": {}}
        for k_sampled, k_real in zip(K_SAMPLED, K_REAL):
            key = str(k_sampled)
            if key not in frozen:
                continue
            c_vec, scale = frozen[key]["coefficients"], frozen[key]["baseline_scale"]
            slow_full = causal_trailing_mean(raw, tid, aid, k_real)
            slow_sampled = causal_trailing_mean(raw[sampled], tid[sampled], aid[sampled], k_real)
            entry["by_K"][key] = {
                "coefficients": c_vec, "baseline_scale": scale,
                "full": group_moment_rows(y, w, slow_full, raw - slow_full, tid),
                "sampled": group_moment_rows(y[sampled], w[sampled], slow_sampled,
                                             raw[sampled] - slow_sampled, tid[sampled]),
            }
        per_fold[fold] = entry
        print(f"fold {fold}: window {entry['window']}, "
              f"{entry['rows_full']:,} full rows / {entry['rows_sampled']:,} sampled", flush=True)

    def pooled(key: str, resolution: str) -> dict[str, float]:
        delta, baseline, energy = 0.0, 0.0, 0.0
        for fold in args.folds:
            block = per_fold[fold]["by_K"][key]
            totals = block[resolution].sum(axis=0)
            result = score_frozen(totals, block["coefficients"], block["baseline_scale"])
            weight = float(totals[0])          # 该折窗口的 target 能量
            delta += result["delta"] * weight
            baseline += result["baseline"] * weight
            energy += weight
        return {"delta": delta / energy, "baseline": baseline / energy,
                "relative": (delta / energy) / abs(baseline / energy) if baseline else float("nan")}

    def pooled_bootstrap(key: str, resolution: str) -> dict[str, float]:
        rng = np.random.default_rng(args.boot_seed)
        prepared = []
        for fold in args.folds:
            block = per_fold[fold]["by_K"][key]
            rows = block[resolution]
            prepared.append((np.vstack([np.zeros(rows.shape[1]), np.cumsum(rows, axis=0)]),
                             len(rows), block["coefficients"], block["baseline_scale"]))
        reference = abs(pooled(key, resolution)["baseline"])
        samples = []
        for _ in range(args.n_boot):
            delta, energy = 0.0, 0.0
            for prefix, n_groups, coefficients, scale in prepared:
                n_blocks = int(np.ceil(n_groups / args.block_size))
                starts = rng.integers(0, max(n_groups - args.block_size, 0) + 1, size=n_blocks)
                stops = np.minimum(starts + args.block_size, n_groups)
                totals = (prefix[stops] - prefix[starts]).sum(axis=0)
                result = score_frozen(totals, coefficients, scale)
                delta += result["delta"] * float(totals[0])
                energy += float(totals[0])
            samples.append(delta / energy)
        array = np.asarray(samples, dtype=float)
        return {"p2.5": float(np.nanpercentile(array, 2.5) / reference),
                "p50": float(np.nanpercentile(array, 50.0) / reference),
                "p97.5": float(np.nanpercentile(array, 97.5) / reference)}

    keys = [str(k) for k in K_SAMPLED if str(k) in per_fold[args.folds[0]]["by_K"]]
    table = {}
    for key in keys:
        table[key] = {}
        for resolution in ("full", "sampled"):
            point = pooled(key, resolution)
            table[key][resolution] = {**point, "bootstrap": pooled_bootstrap(key, resolution)}
        print(f"  K sampled={key:>4}: full {table[key]['full']['relative']*100:+6.2f}% "
              f"[{table[key]['full']['bootstrap']['p2.5']*100:+.1f}%, "
              f"{table[key]['full']['bootstrap']['p97.5']*100:+.1f}%] | "
              f"sampled {table[key]['sampled']['relative']*100:+6.2f}% "
              f"[{table[key]['sampled']['bootstrap']['p2.5']*100:+.1f}%, "
              f"{table[key]['sampled']['bootstrap']['p97.5']*100:+.1f}%]", flush=True)

    selected = str(SELECTED_K_SAMPLED)
    primary = table[selected]["full"]
    verdict = {
        "selected_K_sampled": selected,
        "pooled_full_resolution_relative": primary["relative"],
        "pooled_full_ci": primary["bootstrap"],
        "ci_excludes_zero": bool(primary["bootstrap"]["p2.5"] > 0),
        "full_positive_K_count": f"{sum(1 for k in keys if table[k]['full']['relative'] > 0)}/{len(keys)}",
        "decision": ("换到全分辨率后仍为正且可分辨" if primary["bootstrap"]["p2.5"] > 0
                     else "换到全分辨率后仍为正，但合并后仍不可分辨"
                     if primary["relative"] > 0 else "换到全分辨率后翻向"),
    }

    payload = {
        "experiment": "v3_fullres_slow_probe_summary",
        "question": "把多个探针窗合并后，换全分辨率口径会不会让慢分量降权翻向？",
        "scope_note": "只提高「是否翻向」的功效；效应大小的主证据仍是 v3_slow_variance 的 5 折 OOF。",
        "folds": args.folds,
        "windows": {str(f): per_fold[f]["window"] for f in args.folds},
        "rows": {"full": sum(per_fold[f]["rows_full"] for f in args.folds),
                 "sampled": sum(per_fold[f]["rows_sampled"] for f in args.folds)},
        "by_K": {k: {r: {kk: vv for kk, vv in table[k][r].items()} for r in ("full", "sampled")}
                 for k in keys},
        "verdict": verdict,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# P1 口径核对：多窗口合并", "",
             f"folds {args.folds}，每折验证段末尾 20,000 个连续真实 time_id；"
             f"合计全分辨率 {payload['rows']['full']:,} 行 / 采样格 {payload['rows']['sampled']:,} 行。", "",
             f"> {payload['scope_note']}", "",
             "系数按折冻结（每折用它之前那些折解出的那组），窗口内不做任何拟合。", "",
             "| K（采样步） | K（真实步） | 全分辨率 | 95% CI | 采样格 | 95% CI |",
             "|---:|---:|---:|---|---:|---|"]
    for key in keys:
        f_, s_ = table[key]["full"], table[key]["sampled"]
        lines.append(f"| {key} | {int(key) * SAMPLE_MODULO} | {f_['relative']*100:+.2f}% | "
                     f"[{f_['bootstrap']['p2.5']*100:+.1f}%, {f_['bootstrap']['p97.5']*100:+.1f}%] | "
                     f"{s_['relative']*100:+.2f}% | "
                     f"[{s_['bootstrap']['p2.5']*100:+.1f}%, {s_['bootstrap']['p97.5']*100:+.1f}%] |")
    lines += ["", "## 判定", "",
              f"- 选中的 K（采样步 {selected}）上，合并后的全分辨率相对增益 "
              f"**{verdict['pooled_full_resolution_relative']*100:+.2f}%**，"
              f"95% CI [{primary['bootstrap']['p2.5']*100:+.1f}%, "
              f"{primary['bootstrap']['p97.5']*100:+.1f}%]",
              f"- 全分辨率为正的 K 数：**{verdict['full_positive_K_count']}**",
              "", f"### 结论：{verdict['decision']}", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{json.dumps(verdict, ensure_ascii=False, indent=2)}")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
