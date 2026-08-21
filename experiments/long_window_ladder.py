"""长窗列数阶梯：把 240 列拆成三个 80 列的单窗口臂，看有没有哪一档由负转正。

## 为什么还要问

`long_history_probe`（同日）一次加了 **240 列**（40 特征 × 窗口 {64,512,4096} ×
{滚动均值, 偏离}），pooled +0.69%、`REJECTED`。但那个设计**把「信号」和「估计代价」
焊死在一起了** —— 240 根列一起进一起出，分不清「长窗里没信号」和「有信号但被
240 列的估计方差淹掉」。

⚠️ 当天曾用 `ΔA/ΔB` 声称「有信号但付不起方差」，**那个读数是错的并已收回**
（实测 `A比/√(B比) ≡ IC比`，ΔA/ΔB 混着两臂解的共同尺度）⟹ **到目前为止这个问题没有答案。**

拆成单窗口各 80 列，列数降到 1/3、估计代价大致同比下降 —— 若某一档由负转正，
就说明信号集中在那个窗口。

## 两处跑前声明的设计变更（见 plan.json 的 amendment_1 / amendment_2）

1. **主评价器改成树**（LightGBM 1s × 160 轮，只跑截面块）。`long_history_probe` 唯一的
   实质缺陷就是线性的内层 alpha 选参 —— fold 4 两臂都选到梯底 `1e-6`（≈不正则），
   造出 −19.49% 那个坏折。换成树，**这一整类脆弱性直接消失**（树不需要 alpha 梯子）。
2. **线性降为次臂，改用本文件独立声明的 `WIDE_ALPHA_LADDER`**。
   ⚠️ **不动** `function_class_probe.ALPHA_LADDER` —— 它被 `long_history_probe.py`
   **import** 走，就地加宽会让两个已结案实验复跑时得出与落盘产物不同的数
   （08-18「缓存出自已不存在的代码版本」那类事故的形状）。
   次臂顺带回答：那个 −19.49% 是不是选参失效造成的。

判据先于结果落盘在 `outputs/experiments/long_window_ladder_plan.json`。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
from v3_production_oof import (XS_SPEC, fit_predict_lgbm, group_mean,  # noqa: E402
                               row_slice)
from function_class_probe import linear_gram, linear_predict, solve_ridge  # noqa: E402
from selection_criterion_probe import peak_of, paired_bootstrap_lower_bound  # noqa: E402

PLAN_PATH = _REPO_ROOT / "outputs" / "experiments" / "long_window_ladder_plan.json"
CACHE_PATH = (_REPO_ROOT / "outputs" / "cache"
              / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz")
# ---- 预注册常量 ----
FEATURE_COUNT = 200
HISTORY_COUNT = 40
HISTORY_WINDOW = 5
TRAIN_WINDOW = 78_960
EMBARGO = 6
N_FOLDS = 5
SAMPLE_MODULO = 5
SAMPLING = "phase_balanced"
WINDOWS = (64, 512, 4096)          # 只用这三个，不搜
NUM_ITERATION = 160
N_SEEDS = 1
SEED = 2026
# ⚠️ 独立常量。绝不改 function_class_probe.ALPHA_LADDER —— 它被已结案实验 import 走。
WIDE_ALPHA_LADDER = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2)
INNER_VALID_FRACTION = 0.2
GATE_MIN_GAIN = 0.03
GATE_MIN_POSITIVE_FOLDS = 4
DETECTION_FLOOR = 0.061
ARMS = ("base", "w64", "w512", "w4096")


# ------------------------------------------------------------------ 长窗块

def trailing_mean_and_deviation(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """一个 asset 的严格滞后滚动均值与偏离。

    `mean[t] = mean(series[max(0,t−W) .. t−1])`，**不含当前行**；`t=0` 取 0。
    边界规则与 `strategies/v3_hybrid/history.py` 一致。
    ⚠️ 这里用 float64 cumsum 向量化 —— 生产那份为了离线/在线逐位一致刻意不用 cumsum，
    本脚本是研究探针、没有在线路径要对齐，故不受该约束。
    """
    values = np.asarray(series, dtype=np.float64)
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    index = np.arange(len(values))
    left = np.maximum(index - window, 0)
    count = np.maximum(index - left, 1)
    mean = (cumulative[index] - cumulative[left]) / count[:, None]
    mean[0] = 0.0
    return mean, values - mean


def build_window_blocks(data_root: Path, history_names: list[str],
                        history_stats: tuple[np.ndarray, ...],
                        windows: tuple[int, ...]) -> dict[int, list[np.ndarray]]:
    """读全量行的 history 列，逐 asset 算每个窗口的 (滚动均值, 偏离)，只留采样行。

    必须扫全量行（不能只扫采样行）—— 滚动状态要在每一行上推进，
    理由与 `strategies/v3_hybrid/train.py:stream_history_blocks` 完全相同。
    三个窗口共用**同一次**读盘。
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
    out = {w: [np.zeros((n_out, width), dtype=np.float32) for _ in range(2)] for w in windows}
    out_index = np.cumsum(sampled) - 1
    for asset in np.unique(all_assets):
        rows = np.flatnonzero(all_assets == asset)          # 已按时间序
        series = all_features[rows]
        keep = sampled[rows]
        target_rows = out_index[rows[keep]]
        for window in windows:
            mean, deviation = trailing_mean_and_deviation(series, window)
            out[window][0][target_rows] = mean[keep].astype(np.float32)
            out[window][1][target_rows] = deviation[keep].astype(np.float32)
            del mean, deviation
        del series
        gc.collect()
    del all_features, all_assets, sampled
    gc.collect()
    return out


# ------------------------------------------------------------------ 线性次臂

def fit_linear_wide(design_tr, design_va, e_tr, w_tr, tid_tr) -> tuple[np.ndarray, dict]:
    """加宽梯子的线性臂。**记录选中 alpha 是否落在梯子两端**，撞端即标 invalid。"""
    uniq = np.unique(tid_tr)
    cut_id = uniq[int(len(uniq) * (1.0 - INNER_VALID_FRACTION))]
    cut = int(np.searchsorted(tid_tr, cut_id, side="left"))
    gram_a, rhs_a, _ = linear_gram(design_tr[:cut], w_tr[:cut], e_tr[:cut])
    gram_b, rhs_b, dss_b = linear_gram(design_tr[cut:], w_tr[cut:], e_tr[cut:])
    best_alpha, best_peak, trace = None, -np.inf, []
    for alpha in WIDE_ALPHA_LADDER:
        beta = solve_ridge(gram_a, rhs_a, alpha)
        a = float(np.dot(beta, rhs_b))
        b = float(beta @ gram_b @ beta)
        inner = (a * a / (b * dss_b)) if b > 0 and dss_b > 0 else -np.inf
        trace.append({"alpha_relative": alpha, "inner_peak": inner})
        if inner > best_peak:
            best_alpha, best_peak = alpha, inner
    at_boundary = best_alpha in (WIDE_ALPHA_LADDER[0], WIDE_ALPHA_LADDER[-1])
    beta = solve_ridge(gram_a + gram_b, rhs_a + rhs_b, best_alpha)
    del gram_a, gram_b
    gc.collect()
    return linear_predict(design_va, beta), {"alpha_relative": float(best_alpha),
                                             "alpha_at_boundary": bool(at_boundary),
                                             "alpha_trace": trace}


# ------------------------------------------------------------------ 主流程

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="long_window_ladder")
    p.add_argument("--stage1", action="store_true", help="只跑 fold 0（预注册降级路径）")
    p.add_argument("--num-threads", type=int, default=32)
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
    print(f"  ⊘ 剔除：{plan['excluded_gate']['name']}")
    print(f"臂：{list(ARMS)}（每个候选臂 +{HISTORY_COUNT*2} 列，对照 240 列那次的 1/3）")
    print(f"主评价器：LightGBM {N_SEEDS}s × {NUM_ITERATION} 轮，只跑截面块（变更 1）")
    print(f"次臂：线性，WIDE_ALPHA_LADDER {WIDE_ALPHA_LADDER[0]:.0e}~{WIDE_ALPHA_LADDER[-1]:.0e}"
          f"（{len(WIDE_ALPHA_LADDER)} 档，变更 2；撞端即标 invalid）")
    print(f"检出下限 {100*DETECTION_FLOOR:.1f}%    诚实先验：{plan['honest_prior'][:70]}…")
    if args.dry_run:
        print("\n--dry-run：未读数据。")
        return

    out_json = Path(args.output_dir) / f"{args.label}.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"产物已存在：{out_json}；要覆盖请加 --force（CLAUDE.md §5.10）")

    started = time.perf_counter()
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")
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
    lgbm_args = SimpleNamespace(num_iteration=NUM_ITERATION, n_seeds=N_SEEDS,
                                seed=SEED, num_threads=args.num_threads)

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
        long_blocks = build_window_blocks(Path(args.data_root), history_names,
                                          history_stats, WINDOWS)
        print(f"fold {index}: history + 三窗块就绪（{time.perf_counter()-t0:.0f}s）", flush=True)

        probe_key = tid_va * 16 + aid_va
        take = cache_order[np.searchsorted(cache_key_sorted, probe_key)]
        if not (np.max(np.abs(cache["target"][take] - y_va)) == 0.0
                and np.max(np.abs(cache["weight"][take] - w_va)) == 0.0):
            raise AssertionError(f"fold {index}: join 后 target/weight 不逐位相同")
        print(f"  join 通过：{len(take):,} 行", flush=True)

        row: dict = {"fold": index, "n_valid": int(len(take)), "arms": {}}
        for name in ARMS:
            t1 = time.perf_counter()
            extra_tr, extra_va = [], []
            if name != "base":
                window = int(name[1:])
                extra_tr = [b[tr] for b in long_blocks[window]]
                extra_va = [b[va] for b in long_blocks[window]]
            design_tr = np.column_stack([xs_tr, *[b[tr] for b in short], *extra_tr,
                                         aid_tr.astype(np.float32)])
            design_va = np.column_stack([xs_va, *[b[va] for b in short], *extra_va,
                                         aid_va.astype(np.float32)])
            pred = fit_predict_lgbm(design_tr, e_tr, w_tr, design_va, lgbm_args,
                                    f"{name}_cross", XS_SPEC,
                                    categorical_columns=[design_tr.shape[1] - 1])
            pred = pred - group_mean(pred, va_starts, va_counts)
            pk, a, b = peak_of(e_va, pred, w_va)
            lin_pred, lin_info = fit_linear_wide(design_tr, design_va, e_tr, w_tr, tid_tr)
            lin_pred = lin_pred - group_mean(lin_pred, va_starts, va_counts)
            lin_pk, _, _ = peak_of(e_va, lin_pred, w_va)
            row["arms"][name] = {"peak": pk, "A": a, "B": b, "n_columns": int(design_tr.shape[1]),
                                 "linear_peak": lin_pk, **{f"linear_{k}": v for k, v in lin_info.items()
                                                           if k != "alpha_trace"},
                                 "seconds": time.perf_counter() - t1}
            flag = " ⚠️撞端" if lin_info["alpha_at_boundary"] else ""
            print(f"  {name:6s} {design_tr.shape[1]:3d} 列  树 peak={pk:.6e}   "
                  f"线性 peak={lin_pk:.6e} (alpha={lin_info['alpha_relative']:.0e}{flag})"
                  f"  ({time.perf_counter()-t1:.0f}s)", flush=True)
            del design_tr, design_va, pred, lin_pred, extra_tr, extra_va
            gc.collect()
        fold_rows.append(row)
        del short, long_blocks, xs_tr, xs_va
        gc.collect()

    report = build_report(fold_rows, plan_sha, args, time.perf_counter() - started)
    write_outputs(Path(args.output_dir), args.label, report)


def build_report(fold_rows: list[dict], plan_sha: str, args, elapsed: float) -> dict:
    summary: dict = {}
    for key, label in (("peak", "tree"), ("linear_peak", "linear")):
        base = np.array([r["arms"]["base"][key] for r in fold_rows])
        for name in ARMS:
            if name == "base":
                continue
            arm = np.array([r["arms"][name][key] for r in fold_rows])
            gains = arm / base - 1.0
            pooled = float(arm.sum() / base.sum() - 1.0)
            drop_best = (float(np.delete(arm, gains.argmax()).sum()
                               / np.delete(base, gains.argmax()).sum() - 1.0)
                         if len(gains) > 1 else float("nan"))
            ci_low = paired_bootstrap_lower_bound(arm, base) if len(gains) > 1 else float("nan")
            gates = {
                "1_mean_delta_peak_positive": bool(gains.mean() > 0),
                "2_at_least_4_of_5_folds_positive": bool((gains > 0).sum() >= min(GATE_MIN_POSITIVE_FOLDS, len(gains))),
                "3_survives_drop_best_fold": bool(drop_best > 0) if len(gains) > 1 else None,
                "4_relative_gain_at_least_3pct": bool(pooled >= GATE_MIN_GAIN),
                "5_paired_bootstrap_ci_lower_bound_positive": bool(ci_low > 0) if len(gains) > 1 else None,
            }
            entry = {"pooled_gain": pooled, "fold_gains": gains.tolist(),
                     "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
                     "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
                     "exceeds_detection_floor": bool(pooled >= DETECTION_FLOOR),
                     "gates": gates, "passed": all(v for v in gates.values() if v is not None)}
            if label == "linear":
                boundary = [r["arms"][n]["linear_alpha_at_boundary"]
                            for r in fold_rows for n in (name, "base")]
                entry["any_alpha_at_boundary"] = bool(any(boundary))
                entry["reading_valid"] = not entry["any_alpha_at_boundary"]
            summary.setdefault(label, {})[name] = entry
    passing = [n for n, s in summary["tree"].items() if s["passed"]]
    return {"experiment": "long_window_ladder", "plan_sha256": plan_sha,
            "baseline_cache": CACHE_PATH.name, "stage1_only": bool(args.stage1),
            "windows": list(WINDOWS), "wide_alpha_ladder": list(WIDE_ALPHA_LADDER),
            "primary_evaluator": f"LightGBM {N_SEEDS}s x {NUM_ITERATION} rounds, cross block only",
            "detection_floor": DETECTION_FLOOR, "folds": fold_rows, "summary": summary,
            "passing_arms": passing, "verdict": "PASS" if passing else "REJECTED",
            "elapsed_seconds": elapsed}


def write_outputs(output_dir: Path, label: str, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    n_cols = {r["fold"]: {n: r["arms"][n]["n_columns"] for n in ARMS} for r in report["folds"]}
    lines = [f"# 长窗列数阶梯（`{label}`）", "",
             f"> 预注册判据 sha256 `{report['plan_sha256']}`，先于结果落盘。",
             f"> **主评价器：{report['primary_evaluator']}**（变更 1：`long_history_probe` 的唯一实质缺陷"
             f"是线性的 alpha 选参，换成树后这类脆弱性消失）。",
             f"> 窗口 {report['windows']}（观测数）；每个候选臂 +80 列 = 240 列那次的 **1/3**"
             f"（{n_cols[0]['base']} → {n_cols[0]['w64']} 列）。",
             f"> 检出下限 **{100*report['detection_floor']:.1f}%**。", ""]
    if report["stage1_only"]:
        lines += ["⚠️ **Stage 1 单折**（预注册降级路径），不构成五折裁决。", ""]
    for label_name, title in (("tree", "树（主判据）"), ("linear", "线性（次臂，仅参考）")):
        lines += [f"## {title}", "",
                  "| 臂 | pooled Δpeak | 正折 | 去最好折 | bootstrap CI 下界 | 超检出下限 | 判定 |",
                  "|---|--:|--:|--:|--:|:--:|:--:|"]
        for name, s in report["summary"][label_name].items():
            drop = "—" if not np.isfinite(s["drop_best_gain"]) else f"{100*s['drop_best_gain']:+.2f}%"
            ci = "—" if not np.isfinite(s["bootstrap_ci_lower"]) else f"{100*s['bootstrap_ci_lower']:+.2f}%"
            mark = "✅" if s["passed"] else "❌"
            if label_name == "linear" and not s.get("reading_valid", True):
                mark = "⚠️ alpha 撞端，读数无效"
            lines.append(f"| `{name}` | **{100*s['pooled_gain']:+.2f}%** | {s['positive_folds']}/{s['n_folds']} | "
                         f"{drop} | {ci} | {'✅' if s['exceeds_detection_floor'] else '❌'} | {mark} |")
        lines.append("")
    lines += ["> ⊘ 判据里没有 `2ΔA>ΔB`：它混着两臂解的共同尺度，且是两分量配比判别式，"
              "不适用于「同一模型换输入」。只用尺度不变的 `peak = A²/(B·D)`。", "",
              "### 逐折树 peak", "",
              "| fold | " + " | ".join(ARMS) + " |", "|---|" + "--:|" * len(ARMS)]
    for r in report["folds"]:
        lines.append(f"| {r['fold']} | " + " | ".join(f"{r['arms'][n]['peak']:.4e}" for n in ARMS) + " |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    (output_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / (label + '.json')}\nwrote {output_dir / (label + '.md')}")


if __name__ == "__main__":
    main()
