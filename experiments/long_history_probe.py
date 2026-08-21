"""长历史窗探针：逐 asset 的 64~4096 个观测里，还有没有生产模型没用到的信息？

## 为什么现在问这个

生产 history 块只有 `window=5`（上一次观测 / 差分 / 5 步滚动均值 / 偏离），
而 slow/fast 证明**预测**在 K=2000 真实步的尺度上仍有可用结构（公榜 +2.93%）。
中间那段从没被看过 —— `experiments/temporal_multiscale.py:49` 写着 `MAX_LAG = 20`，
temporal 全族（`t1_lags`/`t2_state`/`t3_full`/`f_lags`/`f_changes`/`f_volatility`/`f_trend`）
的滞后都止步于 20 个真实步。**20 到 2000 是 100 倍的未测跨度。**

## 为什么主臂可以用线性

同日的 `function_class_probe` 实测：在**同一批输入**上线性 r=0.611 / 核 r=0.798 /
树 r=1.000，三者互相 ρ≈0.6~0.7 ⟹ 它们读到的是同一个东西。
⟹ **线性是「信息在不在」的有效探测器**：线性看不见的，换函数类也看不见。
这条是本实验用线性做主臂的全部依据，不是图省事。

## 长窗块怎么算

40 个 history 特征 × 窗口 {64, 512, 4096} × {滚动均值, 当前值−滚动均值} = 240 列。
窗口按**观测数**（逐 asset 行序），与生产 `AssetHistory` 同语义；滚动均值**严格滞后**
（只用 `t−W .. t−1`，不含当前行），无历史时取 0 —— 与 `history.py` 的边界规则一致。

⚠️ 这里用 float64 `cumsum` 逐 asset 向量化。`strategies/v3_hybrid/history.py` 刻意
**不**用 cumsum（为了离线整块与在线逐 time_id 逐位一致，见其 docstring）；
本脚本是研究探针、没有在线路径要对齐，故不受该约束。若本实验 PASS、要进生产，
必须换回定序求和的实现。

判据先于结果落盘在 `outputs/experiments/long_history_probe_plan.json`。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import apply_robust_transform, cross_sectional_deviation  # noqa: E402
from lgbm_xs import load_rows  # noqa: E402
from mt_predictability import group_starts  # noqa: E402
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402
from v3_production_oof import group_mean, row_slice  # noqa: E402
from function_class_probe import (  # noqa: E402
    ALPHA_LADDER, CACHE_PATH, D_RFF, EMBARGO, FEATURE_COUNT, HISTORY_COUNT,
    HISTORY_WINDOW, N_ASSETS, N_FOLDS, SAMPLE_MODULO, SAMPLING, TRAIN_WINDOW,
    fit_arm, median_bandwidth, weighted_ic)

PLAN_PATH = _REPO_ROOT / "outputs" / "experiments" / "long_history_probe_plan.json"
WINDOWS = (64, 512, 4096)          # 预注册，不搜索
SEED = 20260821
BOOTSTRAP_BLOCKS = 200
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 2026
GATE_MIN_GAIN = 0.03
GATE_MIN_POSITIVE_FOLDS = 4


def build_long_window_blocks(data_root: Path, history_names: list[str],
                             history_stats: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """读全量行的 history 列，逐 asset 算严格滞后的滚动均值与偏离，只留采样行。

    必须扫全量行（不能只扫采样行）—— 滚动状态要在每一行上推进，这与
    `strategies/v3_hybrid/train.py:stream_history_blocks` 的理由完全相同。
    """
    import pyarrow.parquet as pq
    from src.io import time_sample_mask, train_files

    lower, upper, center, scale = history_stats
    feats, aids, masks = [], [], []
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=200_000, columns=["time_id", "asset_id", *history_names]):
            frame = batch.to_pandas()
            block = frame[history_names].to_numpy(dtype=np.float32, copy=True)
            apply_robust_transform(block, lower, upper, center, scale)
            feats.append(block)
            aids.append(frame["asset_id"].to_numpy(dtype=np.int64, copy=True))
            masks.append(time_sample_mask(frame["time_id"].to_numpy(copy=False),
                                          SAMPLE_MODULO, sampling=SAMPLING))
    all_features = np.concatenate(feats); del feats
    all_assets = np.concatenate(aids); del aids
    sampled = np.concatenate(masks); del masks
    gc.collect()

    width = len(history_names)
    n_out = int(sampled.sum())
    out = [np.zeros((n_out, width), dtype=np.float32) for _ in range(2 * len(WINDOWS))]
    out_index = np.cumsum(sampled) - 1
    for asset in range(N_ASSETS):
        rows = np.flatnonzero(all_assets == asset)            # 已按时间序
        series = all_features[rows].astype(np.float64)
        cumulative = np.vstack([np.zeros((1, width)), np.cumsum(series, axis=0)])
        index = np.arange(len(series))
        keep = sampled[rows]
        target_rows = out_index[rows[keep]]
        for w_index, window in enumerate(WINDOWS):
            left = np.maximum(index - window, 0)
            count = np.maximum(index - left, 1)                # idx=0 时占位，下面覆盖
            mean = (cumulative[index] - cumulative[left]) / count[:, None]
            mean[0] = 0.0                                      # 无历史 ⟹ 0（同 history.py）
            out[2 * w_index][target_rows] = mean[keep].astype(np.float32)
            out[2 * w_index + 1][target_rows] = (series[keep] - mean[keep]).astype(np.float32)
        del series, cumulative
        gc.collect()
    del all_features, all_assets, sampled
    gc.collect()
    return out


def paired_bootstrap_lower_bound(gain_num: np.ndarray, gain_den: np.ndarray) -> float:
    """逐折配对 block bootstrap 的 95% 下界。gain = Σnum/Σden - 1，按块重抽。"""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(gain_num), size=(BOOTSTRAP_DRAWS, len(gain_num)))
    ratios = gain_num[draw].sum(1) / gain_den[draw].sum(1) - 1.0
    return float(np.percentile(ratios, 2.5))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="long_history_probe")
    p.add_argument("--stage1", action="store_true", help="只跑 fold 0（预注册的降级路径）")
    p.add_argument("--skip-rff", action="store_true", help="内存不足时只跑线性主臂")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="覆盖已有产物（CLAUDE.md §5.10）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    print(f"预注册判据 {PLAN_PATH.name}  sha256={plan_sha}")
    for gate in plan["gates"]:
        print(f"  {gate['id']}. {gate['name']}: {gate['rule']}")
    print(f"窗口={WINDOWS}（观测数）  长窗块 = {HISTORY_COUNT}×{len(WINDOWS)}×2 = "
          f"{HISTORY_COUNT*len(WINDOWS)*2} 列")
    print(f"诚实先验：{plan['honest_prior']}")
    if args.dry_run:
        print("\n--dry-run：未读数据。")
        return

    out_json = Path(args.output_dir) / f"{args.label}.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"产物已存在：{out_json}；要覆盖请加 --force（CLAUDE.md §5.10）")

    started = time.perf_counter()
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")     # 同 function_class_probe 的理由
    if v3_path not in sys.path:
        sys.path.append(v3_path)
    from strategies.v3_hybrid.train import stream_history_blocks
    from src.oof_cache import assert_reproducible_cache
    assert_reproducible_cache(CACHE_PATH)
    cache = np.load(CACHE_PATH)

    rows = load_rows(Path(args.data_root), SAMPLE_MODULO, SAMPLING)
    features, target = rows["features"], rows["target"]
    weight, time_ids, asset_ids = rows["weight"], rows["time_id"], rows["asset_id"]
    del rows
    order = np.argsort(time_ids, kind="stable")
    features, target = features[order], target[order]
    weight, time_ids, asset_ids = weight[order], time_ids[order], asset_ids[order]
    folds = rolling_time_folds(np.unique(time_ids), N_FOLDS, TRAIN_WINDOW, EMBARGO)
    print(f"{len(target):,} 行 / {len(np.unique(time_ids)):,} 采样 time_id / {len(folds)} 折"
          f"（{time.perf_counter()-started:.0f}s）", flush=True)

    cache_key = cache["time_id"] * 16 + cache["asset_id"]
    cache_order = np.argsort(cache_key, kind="stable")
    cache_key_sorted = cache_key[cache_order]

    rng = np.random.default_rng(SEED)
    fold_rows: list[dict] = []
    for index, (train_ids, valid_ids) in enumerate(folds[:1] if args.stage1 else folds):
        t0 = time.perf_counter()
        tr, va = row_slice(time_ids, train_ids), row_slice(time_ids, valid_ids)
        y_tr, y_va, w_tr, w_va = target[tr], target[va], weight[tr], weight[va]
        tid_tr, tid_va = time_ids[tr], time_ids[va]
        aid_tr, aid_va = asset_ids[tr], asset_ids[va]
        tr_starts, va_starts = group_starts(tid_tr), group_starts(tid_va)
        tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)
        va_counts = np.diff(np.r_[va_starts, len(tid_va)]).astype(np.float64)

        transformed_train, stats = robust_transform_fit(features[tr].copy())
        transformed_valid = features[va].copy()
        apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                               stats["center"], stats["scale"])
        e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
        e_va = y_va - group_mean(y_va, va_starts, va_counts)
        unit = np.ones_like(e_tr)
        xs_selected = select_features(transformed_train, e_tr, unit, FEATURE_COUNT)
        xs_tr = cross_sectional_deviation(transformed_train[:, xs_selected].copy(), tid_tr)
        xs_va = cross_sectional_deviation(transformed_valid[:, xs_selected].copy(), tid_va)
        history_positions = np.sort(select_features(xs_tr, e_tr, unit, HISTORY_COUNT).astype(np.int64))
        history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
        history_stats = tuple(stats[key][xs_selected[history_positions]]
                              for key in ("lower", "upper", "center", "scale"))
        del transformed_train, transformed_valid
        gc.collect()

        short = stream_history_blocks(Path(args.data_root), SAMPLE_MODULO, SAMPLING,
                                      history_names, history_stats, HISTORY_WINDOW)
        long_blocks = build_long_window_blocks(Path(args.data_root), history_names, history_stats)
        print(f"fold {index}: history/长窗块就绪（{time.perf_counter()-t0:.0f}s）", flush=True)

        # 列序：[xs_dev200 ‖ history160 ‖ onehot15 ‖ long240]
        # one-hot 放中间 ⟹ base 臂就是前 375 列的**连续切片**，省掉一整份设计矩阵拷贝。
        def stack(rows_slice, xs_block, aid):
            onehot = np.zeros((len(aid), N_ASSETS), dtype=np.float32)
            onehot[np.arange(len(aid)), aid] = 1.0
            return np.ascontiguousarray(np.column_stack(
                [xs_block, *[b[rows_slice] for b in short], onehot,
                 *[b[rows_slice] for b in long_blocks]]), dtype=np.float32)

        design_tr = stack(tr, xs_tr, aid_tr)
        design_va = stack(va, xs_va, aid_va)
        del xs_tr, xs_va, short, long_blocks
        gc.collect()
        base_cols = FEATURE_COUNT + 4 * HISTORY_COUNT + N_ASSETS
        mean = design_tr[:, :].mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = design_tr.std(axis=0, dtype=np.float64).astype(np.float32)
        sd[sd <= 0] = 1.0
        mean[FEATURE_COUNT + 4 * HISTORY_COUNT: base_cols] = 0.0    # one-hot 不去中心
        sd[FEATURE_COUNT + 4 * HISTORY_COUNT: base_cols] = 1.0
        for design in (design_tr, design_va):
            for start in range(0, len(design), 200_000):
                stop = min(start + 200_000, len(design))
                np.subtract(design[start:stop], mean, out=design[start:stop])
                np.divide(design[start:stop], sd, out=design[start:stop])
        print(f"fold {index}: 设计 {design_tr.shape[1]} 列（base {base_cols}），"
              f"train {len(y_tr):,} / valid {len(y_va):,}（{time.perf_counter()-t0:.0f}s）", flush=True)

        probe_key = tid_va * 16 + aid_va
        pos = np.searchsorted(cache_key_sorted, probe_key)
        take = cache_order[pos]
        if not (np.max(np.abs(cache["target"][take] - y_va)) == 0.0
                and np.max(np.abs(cache["weight"][take] - w_va)) == 0.0):
            raise AssertionError(f"fold {index}: join 后 target/weight 不逐位相同")
        e_lgbm = cache["e_lgbm"][take]
        ic_base_tree, *_ = weighted_ic(e_va, e_lgbm, w_va)
        print(f"  join 通过：{len(take):,} 行；生产 e_lgbm IC={ic_base_tree:+.5f}", flush=True)

        row: dict = {"fold": index, "n_valid": int(len(take)),
                     "ic_e_lgbm": ic_base_tree, "arms": {}}
        specs = [("linear_base", None, base_cols), ("linear_long", None, design_tr.shape[1])]
        if not args.skip_rff:
            specs += [("rff_base", "rff", base_cols), ("rff_long", "rff", design_tr.shape[1])]
        for name, kind, ncols in specs:
            t1 = time.perf_counter()
            sub_tr = design_tr[:, :ncols]
            sub_va = design_va[:, :ncols]
            if kind == "rff":
                sigma = median_bandwidth(sub_tr, rng)
                proj = (rng.standard_normal((ncols, D_RFF)) / sigma).astype(np.float32)
            else:
                proj = None
            pred, info = fit_arm(name, sub_tr, sub_va, e_tr, w_tr, tid_tr, proj, rng)
            pred = pred - group_mean(pred, va_starts, va_counts)
            ic, a, b, d = weighted_ic(e_va, pred, w_va)
            row["arms"][name] = {**info, "ic": ic, "A": a, "B": b, "D": d,
                                 "r_vs_tree": ic / ic_base_tree,
                                 "seconds": time.perf_counter() - t1}
            print(f"  {name:12s} IC={ic:+.5f}  r_vs_tree={ic/ic_base_tree:+.4f}  "
                  f"alpha={info['alpha_relative']:.0e} ({time.perf_counter()-t1:.0f}s)", flush=True)
            del sub_tr, sub_va, proj, pred
            gc.collect()
        fold_rows.append(row)
        del design_tr, design_va
        gc.collect()

    report = build_report(fold_rows, plan_sha, args, time.perf_counter() - started)
    write_outputs(Path(args.output_dir), args.label, report)


def build_report(fold_rows: list[dict], plan_sha: str, args, elapsed: float) -> dict:
    summary: dict = {}
    for family in ("linear", "rff"):
        base, long = f"{family}_base", f"{family}_long"
        if base not in fold_rows[0]["arms"]:
            continue
        gains = np.array([r["arms"][long]["ic"] / r["arms"][base]["ic"] - 1.0 for r in fold_rows])
        num = np.array([r["arms"][long]["ic"] for r in fold_rows])
        den = np.array([r["arms"][base]["ic"] for r in fold_rows])
        pooled = float(num.sum() / den.sum() - 1.0)
        d_a = float(np.mean([r["arms"][long]["A"] / r["arms"][base]["A"] - 1.0 for r in fold_rows]))
        d_b = float(np.mean([r["arms"][long]["B"] / r["arms"][base]["B"] - 1.0 for r in fold_rows]))
        drop_best = (float(np.delete(num, gains.argmax()).sum()
                           / np.delete(den, gains.argmax()).sum() - 1.0)
                     if len(gains) > 1 else float("nan"))
        ci_low = paired_bootstrap_lower_bound(num, den) if len(gains) > 1 else float("nan")
        gates = {
            "1_pooled_relative_gain_at_least_3pct": bool(pooled >= GATE_MIN_GAIN),
            "2_at_least_4_of_5_folds_positive": bool((gains > 0).sum() >= min(GATE_MIN_POSITIVE_FOLDS, len(gains))),
            "3_survives_drop_best_fold": bool(drop_best > 0) if len(gains) > 1 else None,
            "4_two_delta_A_exceeds_delta_B": bool(2 * d_a > d_b),
            "5_paired_bootstrap_ci_lower_bound_positive": bool(ci_low > 0) if len(gains) > 1 else None,
        }
        summary[family] = {"pooled_gain": pooled, "fold_gains": gains.tolist(),
                           "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
                           "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
                           "delta_A": d_a, "delta_B": d_b, "gates": gates,
                           "passed": all(v for v in gates.values() if v is not None)}
    verdict = "PASS" if summary.get("linear", {}).get("passed") else "REJECTED"
    return {"experiment": "long_history_probe", "plan_sha256": plan_sha,
            "windows": list(WINDOWS), "stage1_only": bool(args.stage1),
            "baseline_cache": CACHE_PATH.name, "folds": fold_rows,
            "summary": summary, "verdict": verdict, "elapsed_seconds": elapsed}


def write_outputs(output_dir: Path, label: str, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# 长历史窗探针（`{label}`）", "",
             f"> 预注册判据 sha256 `{report['plan_sha256']}`，先于结果落盘。",
             f"> 窗口 {report['windows']}（观测数）。基准臂 = 生产截面块逐列相同的设计。", ""]
    if report["stage1_only"]:
        lines += ["⚠️ **Stage 1 单折**（预注册降级路径），不构成五折裁决。", ""]
    lines += ["| 函数类 | pooled 配对增益 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 | 判定 |",
              "|---|--:|--:|--:|--:|--:|:--:|--:|:--:|"]
    for family, s in report["summary"].items():
        lines.append(
            f"| `{family}` | **{100*s['pooled_gain']:+.2f}%** | {s['positive_folds']}/{s['n_folds']} | "
            f"{100*s['drop_best_gain']:+.2f}% | {100*s['delta_A']:+.2f}% | {100*s['delta_B']:+.2f}% | "
            f"{'✅' if s['gates']['4_two_delta_A_exceeds_delta_B'] else '❌'} | "
            f"{100*s['bootstrap_ci_lower']:+.2f}% | {'✅' if s['passed'] else '❌'} |")
    for family, s in report["summary"].items():
        lines += ["", f"### `{family}` 逐门槛", ""]
        lines += [f"- {'✅' if ok else '❌'} {g}" for g, ok in s["gates"].items()]
    lines += ["", "### 逐折 IC", "",
              "| fold | 生产 e_lgbm | " + " | ".join(sorted(report["folds"][0]["arms"])) + " |",
              "|---|--:|" + "--:|" * len(report["folds"][0]["arms"])]
    for r in report["folds"]:
        lines.append(f"| {r['fold']} | {r['ic_e_lgbm']:+.5f} | " +
                     " | ".join(f"{r['arms'][a]['ic']:+.5f}" for a in sorted(r["arms"])) + " |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    (output_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / (label + '.json')}\nwrote {output_dir / (label + '.md')}")


if __name__ == "__main__":
    main()
