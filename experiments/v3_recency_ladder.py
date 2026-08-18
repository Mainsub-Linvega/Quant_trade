"""P4：训练窗阶梯（recency）—— 未来更像最近的训练窗，还是完整历史窗？

## 配对的前提（这条不成立整个实验就作废）

各臂必须落在**完全相同的验证行**上。`rolling_time_folds` 的
`first_valid_idx = train_window + embargo` —— 直接改 `--train-window` 会把验证段一起挪走，
各臂就落在不同数据上。所以 `v3_production_oof.py` 用的是 `--train-truncate`：
**固定 fold 版图，只砍训练段前端**。本脚本开头对 `time_id / asset_id / fold` 逐位断言，
不一致直接停。

## 臂

| 臂 | 训练段 | `min_data_in_leaf` |
|---|---|---|
| 基准 | 78,960（生产值） | 随行数缩放（生产行为） |
| `w60000_scaled` / `w40000_scaled` | 60,000 / 40,000 | 随行数缩放 |
| `w60000_frozen` / `w40000_frozen` | 60,000 / 40,000 | **冻结**在 78,960 档的值 |

冻结臂是用来拆混淆的：窗口变短 ⟹ 行数变少 ⟹ `MIN_DATA_FRAC × 行数` 自动变小 ⟹
**有效容量也被动变了**。只跑缩放臂的话，「数据变少」和「容量变大」分不开。

## 判据（预注册，先写死）

折均 Δpeak > 0；≥4/5 折同号；去最好折 > 0；相对 ≥ +1%；`2ΔA > ΔB`；
配对 block bootstrap 95% CI 下界 > 0。

## ⚠️ 晋级限制

这是第②类「拟合紧密度」轴。本项目在该轴上已三次本地↔公榜量反（alpha、轮数 160/320/960、
history 宽度 c80）⟹ **本地结果不单独晋级**，必须由公榜或回补标签裁决（CLAUDE.md §8.1）。
本地为负也不能直接结案 —— 要对照 bootstrap 宽度报「检出下限」。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_recency_ladder.py
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

from src.metric import scale_invariant_score  # noqa: E402
from market_model import sign_test_p  # noqa: E402

CACHE_DIR = _REPO_ROOT / "outputs" / "cache"
BASELINE = "v3_production_oof_confirm_3s480_phasebal_prodwindow"
ARMS = {"w60000_scaled": "v3_recency_w60000_scaled",
        "w60000_frozen": "v3_recency_w60000_frozen",
        "w40000_scaled": "v3_recency_w40000_scaled",
        "w40000_frozen": "v3_recency_w40000_frozen"}
MIN_RELATIVE_GAIN = 0.01
MIN_POSITIVE_FOLDS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--baseline", default=BASELINE,
                   help="基准 OOF cache 的 label。⚠️ 必须与各臂**同口径**（种子数/轮数/采样/"
                        "fold 版图），否则配对断言会挡下来。")
    p.add_argument("--arms", nargs="*", default=None, metavar="NAME=LABEL",
                   help="覆盖默认臂，形如 expanding=v3_recency_expanding_1s160")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_recency_ladder_3s480")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load(label: str, cache_dir: Path) -> dict[str, np.ndarray]:
    path = cache_dir / f"{label}.npz"
    if not path.exists():
        raise SystemExit(f"missing OOF cache: {path}")
    with np.load(path, allow_pickle=False) as d:
        keep = d["fold"] >= 0
        return {"target": d["target"][keep].astype(np.float64),
                "weight": np.maximum(d["weight"][keep].astype(np.float64), 0.0),
                "time_id": d["time_id"][keep].astype(np.int64),
                "asset_id": d["asset_id"][keep].astype(np.int64),
                "fold": d["fold"][keep].astype(np.int16),
                "raw": d["prediction_raw"][keep].astype(np.float64)}


def moment_rows(data: dict[str, np.ndarray], group_index: np.ndarray, n_groups: int) -> np.ndarray:
    """逐 time_id 的 (D, Σw·y·p, Σw·p²)。配对 bootstrap 就对这些行重采样。"""
    y, w, p = data["target"], data["weight"], data["raw"]
    return np.column_stack([
        np.bincount(group_index, weights=w * y * y, minlength=n_groups),
        np.bincount(group_index, weights=w * y * p, minlength=n_groups),
        np.bincount(group_index, weights=w * p * p, minlength=n_groups)])


def peak_of(totals: np.ndarray) -> dict[str, float]:
    D, ayp, bpp = float(totals[0]), float(totals[1]), float(totals[2])
    A, B = ayp / D, bpp / D
    return {"A": A, "B": B, "peak": (A * A / B) if B > 0 else 0.0}


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    arms = ARMS
    if args.arms:
        arms = {}
        for item in args.arms:
            if "=" not in item:
                raise SystemExit(f"--arms 需要 NAME=LABEL 形式，收到 {item!r}")
            name, label = item.split("=", 1)
            arms[name] = label
    base = load(args.baseline, cache_dir)
    available = {name: label for name, label in arms.items()
                 if (cache_dir / f"{label}.npz").exists()}
    if not available:
        raise SystemExit("no recency arm cache found yet")
    missing = sorted(set(arms) - set(available))
    if missing:
        print(f"⚠️ 尚未产出的臂（本次跳过）：{missing}", flush=True)

    loaded = {name: load(label, cache_dir) for name, label in available.items()}

    # ---- 配对前提：验证行必须逐位相同。不成立就停，绝不静默比错。
    for name, data in loaded.items():
        for key in ("time_id", "asset_id", "fold", "target", "weight"):
            if not np.array_equal(data[key], base[key]):
                raise AssertionError(
                    f"臂 {name} 的 `{key}` 与基准不一致 ⟹ 不是同一批验证行，配对比较无效")
    print(f"配对前提已核实：{len(available)} 个臂与基准的验证行逐位相同 "
          f"（{len(base['target']):,} 行）", flush=True)

    starts = np.r_[0, np.flatnonzero(base["time_id"][1:] != base["time_id"][:-1]) + 1]
    counts = np.diff(np.r_[starts, len(base["time_id"])]).astype(np.int64)
    n_groups = len(starts)
    group_index = np.repeat(np.arange(n_groups), counts)
    group_fold = base["fold"][starts]
    folds = sorted(np.unique(group_fold))

    base_rows = moment_rows(base, group_index, n_groups)
    arm_rows = {name: moment_rows(data, group_index, n_groups) for name, data in loaded.items()}

    reference = scale_invariant_score(base["target"], base["raw"], base["weight"])
    check = peak_of(base_rows.sum(axis=0))
    if abs(check["peak"] - float(reference["peak"])) > 1e-12:
        raise AssertionError("moment-based peak disagrees with src.metric")

    def per_fold(rows: np.ndarray) -> dict[int, dict[str, float]]:
        return {int(f): peak_of(rows[group_fold == f].sum(axis=0)) for f in folds}

    base_fold = per_fold(base_rows)
    base_peaks = np.array([base_fold[int(f)]["peak"] for f in folds])

    rng = np.random.default_rng(args.boot_seed)
    n_blocks = int(np.ceil(n_groups / args.block_size))
    prefix_base = np.vstack([np.zeros(3), np.cumsum(base_rows, axis=0)])
    prefix_arm = {n: np.vstack([np.zeros(3), np.cumsum(r, axis=0)]) for n, r in arm_rows.items()}
    # 配对重采样：同一批区块同时作用于基准与所有臂 ⟹ 消掉共同的期间波动
    block_starts = [rng.integers(0, max(n_groups - args.block_size, 0) + 1, size=n_blocks)
                    for _ in range(args.n_boot)]

    results: dict[str, Any] = {}
    for name, rows in arm_rows.items():
        arm_fold = per_fold(rows)
        arm_peaks = np.array([arm_fold[int(f)]["peak"] for f in folds])
        delta = arm_peaks - base_peaks
        drop_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
        positive = int((delta > 0).sum())
        dA = np.mean([arm_fold[int(f)]["A"] - base_fold[int(f)]["A"] for f in folds])
        dB = np.mean([arm_fold[int(f)]["B"] - base_fold[int(f)]["B"] for f in folds])
        meanA = np.mean([base_fold[int(f)]["A"] for f in folds])
        meanB = np.mean([base_fold[int(f)]["B"] for f in folds])
        rel_A, rel_B = float(dA / meanA), float(dB / meanB)

        samples = []
        for starts_b in block_starts:
            stops = np.minimum(starts_b + args.block_size, n_groups)
            b = peak_of((prefix_base[stops] - prefix_base[starts_b]).sum(axis=0))["peak"]
            a = peak_of((prefix_arm[name][stops] - prefix_arm[name][starts_b]).sum(axis=0))["peak"]
            samples.append(a - b)
        boot = np.asarray(samples, dtype=float)
        ci = {k: float(np.nanpercentile(boot, q)) for k, q in
              (("p2.5", 2.5), ("p50", 50.0), ("p97.5", 97.5))}

        checks = {
            "1_mean_delta_positive": bool(delta.mean() > 0),
            "2_at_least_4_of_5_folds_positive": bool(positive >= MIN_POSITIVE_FOLDS),
            "3_survives_drop_best_fold": bool(drop_best.mean() > 0),
            "4_relative_gain_at_least_1pct": bool(delta.mean() / base_peaks.mean() >= MIN_RELATIVE_GAIN),
            "5_two_delta_A_exceeds_delta_B": bool(2.0 * rel_A > rel_B),
            "6_paired_bootstrap_ci_lower_bound_positive": bool(ci["p2.5"] > 0),
        }
        results[name] = {
            "arm_cache": available[name],
            "per_fold_peak": {str(f): arm_fold[int(f)]["peak"] for f in folds},
            "per_fold_delta": [float(v) for v in delta],
            "mean_delta": float(delta.mean()),
            "mean_delta_drop_best": float(drop_best.mean()),
            "relative": float(delta.mean() / base_peaks.mean()),
            "relative_drop_best": float(drop_best.mean() / base_peaks.mean()),
            "positive_folds": positive, "n_folds": len(folds),
            "sign_test_p": sign_test_p(positive, len(folds)),
            "mechanism": {"relative_delta_A": rel_A, "relative_delta_B": rel_B},
            "paired_bootstrap": ci,
            "checks": checks, "pass": all(checks.values()),
        }
        print(f"  {name:16s} Δ折均 {delta.mean():+.3e}（{delta.mean()/base_peaks.mean()*100:+.2f}%）"
              f" 正折 {positive}/{len(folds)} 去最好折 {drop_best.mean():+.3e}"
              f" CI [{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] "
              f"{'PASS' if all(checks.values()) else 'FAIL'}", flush=True)

    detection_floor = float(np.mean([abs(r["paired_bootstrap"]["p97.5"]
                                         - r["paired_bootstrap"]["p2.5"]) / 2
                                     for r in results.values()]))
    payload = {
        "experiment": "v3_recency_ladder",
        "question": "未来更像最近的训练窗，还是完整历史窗？",
        "paired_guarantee": "各臂与基准的 time_id/asset_id/fold/target/weight 逐位相同（已断言）；"
                            "做法是固定 fold 版图、只截短训练段（--train-truncate），"
                            "**不是**改 --train-window（那会把验证段一起挪走）",
        "baseline": {"cache": args.baseline, "train_window_sampled_time_ids": 78_960,
                     "per_fold_peak": {str(f): base_fold[int(f)]["peak"] for f in folds},
                     "mean_peak": float(base_peaks.mean())},
        "confounder_note": "窗口变短 ⟹ 行数变少 ⟹ min_data_in_leaf 自动变小 ⟹ 有效容量被动改变。"
                           "`frozen` 臂把 min_data_in_leaf 冻结在 78,960 档，用来拆开这两件事。",
        "promotion_limit": "第②类拟合紧密度轴，本项目已三次本地↔公榜量反 ⟹ "
                           "本地结果不单独晋级，须公榜或回补标签裁决（CLAUDE.md §8.1）。",
        "detection_floor_estimate": detection_floor,
        "arms": results,
        "skipped_arms": missing,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# P4：训练窗阶梯（recency）", "",
             f"基准 = 生产窗 78,960，逐折 peak 均值 **{base_peaks.mean():.8f}**；"
             f"{len(base['target']):,} 行验证数据。", "",
             f"> **配对保证**：{payload['paired_guarantee']}。", "",
             f"> **混淆项**：{payload['confounder_note']}", "",
             "| 臂 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 配对 CI | 判定 |",
             "|---|---:|---:|---:|---:|---:|---:|---|:--:|"]
    for name, r in results.items():
        ci = r["paired_bootstrap"]
        lines.append(
            f"| `{name}` | {r['mean_delta']:+.3e} | {r['relative']*100:+.2f}% | "
            f"{r['positive_folds']}/{r['n_folds']} | {r['relative_drop_best']*100:+.2f}% | "
            f"{r['mechanism']['relative_delta_A']*100:+.2f}% | "
            f"{r['mechanism']['relative_delta_B']*100:+.2f}% | "
            f"[{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] | "
            f"{'✅' if r['pass'] else '❌'} |")
    lines += ["", f"检出下限（配对 bootstrap 半宽均值）≈ `{detection_floor:.2e}`，"
              f"相当于基准 peak 的 {detection_floor/base_peaks.mean()*100:.1f}% —— "
              "效应没明显超过它就写「测不出来」，不写「没有效果」。", "",
              "## 逐臂门槛", ""]
    for name, r in results.items():
        lines.append(f"### `{name}`（{'✅ PASS' if r['pass'] else '❌ 不通过'}）")
        lines += [f"- {'✅' if ok else '❌'} {k}" for k, ok in r["checks"].items()]
        lines.append("")
    lines += ["## 晋级限制", "", payload["promotion_limit"], ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
