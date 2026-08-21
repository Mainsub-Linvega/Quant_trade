"""长窗 w512 的**确认档**（3 种子 × 480 轮）—— 筛选档结果在生产强度下还成立吗？

`long_window_ladder`（同日，筛选档 1s160、只截面块）给出 `w512` pooled **+6.80%**、
**5/5 折**、去最好折 +5.84%、bootstrap CI 下界 +3.87%，五道门槛全过且超 6.1% 检出下限。
本实验是仓库既有阶梯的第二级：**筛选(1s160) → 确认(3s480) → 公榜**。

## 只留两个臂

`w64`（线性 +9.96%/5-of-5 但树只有 +4.51%/3-of-5）与 `w4096`（树 +3.43%/4-of-5 但线性
−0.33%/1-of-5）**两个评价器不一致** ⟹ 窗口只在 512 上被稳健识别。
带它们上确认档就是多重比较捞鱼。

## ⚠️ 确认档的检出下限**更高**

ROADMAP P2-R：1s160 是 **6.1%**，3s480 是 **8.7%**。
⟹ 筛选档那个 +6.80% 若原样迁移过来，将**低于** 8.7%。
预注册已写死这种情形的读法：**「五道全过但低于检出下限」= 方向可信、幅度测不出**，
只够作为花一次公榜额度的理由，**不构成晋级依据**。

## 自带的一致性检查

线性次臂不依赖树的超参 ⟹ 本实验的线性 peak 必须与 `long_window_ladder` 的对应值
**逐位相同**。不同即说明数据路径有差异，实验判无效。
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
from selection_criterion_probe import peak_of, paired_bootstrap_lower_bound  # noqa: E402
from long_window_ladder import (CACHE_PATH, EMBARGO, FEATURE_COUNT,  # noqa: E402
                                HISTORY_COUNT, HISTORY_WINDOW, N_FOLDS,
                                SAMPLE_MODULO, SAMPLING, TRAIN_WINDOW,
                                build_window_blocks, fit_linear_wide)

PLAN_PATH = _REPO_ROOT / "outputs" / "experiments" / "long_window_confirm_plan.json"
LADDER_PATH = _REPO_ROOT / "outputs" / "experiments" / "long_window_ladder.json"
WINDOW = 512
ARMS = ("base", "w512")
NUM_ITERATION = 480          # 确认档
N_SEEDS = 3                  # 确认档
SEED = 2026
GATE_MIN_GAIN = 0.03
GATE_MIN_POSITIVE_FOLDS = 4
DETECTION_FLOOR = 0.087      # ⚠️ 3s480 的下限，比筛选档的 6.1% 更高


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="long_window_confirm")
    p.add_argument("--stage1", action="store_true")
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
    print(f"臂：{list(ARMS)}   评价器：LightGBM {N_SEEDS}s × {NUM_ITERATION} 轮，只跑截面块")
    print(f"⚠️ 检出下限 {100*DETECTION_FLOOR:.1f}%（3s480），比筛选档的 6.1% **更高**")
    print(f"   {plan['detection_floor']['warning'][:120]}…")
    print(f"诚实先验：{plan['honest_prior'][:90]}…")
    if args.dry_run:
        print("\n--dry-run：未读数据。")
        return

    out_json = Path(args.output_dir) / f"{args.label}.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"产物已存在：{out_json}；要覆盖请加 --force（CLAUDE.md §5.10）")
    ladder = json.loads(LADDER_PATH.read_text(encoding="utf-8"))

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
    print(f"{len(target):,} 行 / {len(folds)} 折（{time.perf_counter()-started:.0f}s）", flush=True)

    cache_key = cache["time_id"] * 16 + cache["asset_id"]
    cache_order = np.argsort(cache_key, kind="stable")
    cache_key_sorted = cache_key[cache_order]
    lgbm_args = SimpleNamespace(num_iteration=NUM_ITERATION, n_seeds=N_SEEDS,
                                seed=SEED, num_threads=args.num_threads)

    fold_rows: list[dict] = []
    linear_mismatch: list[str] = []
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
                                          history_stats, (WINDOW,))[WINDOW]
        print(f"fold {index}: 数据就绪（{time.perf_counter()-t0:.0f}s）", flush=True)

        probe_key = tid_va * 16 + aid_va
        take = cache_order[np.searchsorted(cache_key_sorted, probe_key)]
        if not (np.max(np.abs(cache["target"][take] - y_va)) == 0.0
                and np.max(np.abs(cache["weight"][take] - w_va)) == 0.0):
            raise AssertionError(f"fold {index}: join 后 target/weight 不逐位相同")

        row: dict = {"fold": index, "n_valid": int(len(take)), "arms": {}}
        for name in ARMS:
            t1 = time.perf_counter()
            extra_tr = [b[tr] for b in long_blocks] if name != "base" else []
            extra_va = [b[va] for b in long_blocks] if name != "base" else []
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
            # ---- 自带一致性检查：线性不依赖树超参 ⟹ 必须与筛选档逐位相同
            expected = ladder["folds"][index]["arms"][name]["linear_peak"]
            if lin_pk != expected:
                linear_mismatch.append(
                    f"fold {index}/{name}: 线性 peak {lin_pk!r} != 筛选档 {expected!r}")
            row["arms"][name] = {"peak": pk, "A": a, "B": b,
                                 "n_columns": int(design_tr.shape[1]),
                                 "linear_peak": lin_pk,
                                 "linear_matches_ladder": bool(lin_pk == expected),
                                 "linear_alpha_relative": lin_info["alpha_relative"],
                                 "linear_alpha_at_boundary": lin_info["alpha_at_boundary"],
                                 "seconds": time.perf_counter() - t1}
            print(f"  {name:6s} {design_tr.shape[1]:3d} 列  树 peak={pk:.6e}  "
                  f"线性对拍={'✅' if lin_pk == expected else '❌'}"
                  f"  ({time.perf_counter()-t1:.0f}s)", flush=True)
            del design_tr, design_va, pred, lin_pred, extra_tr, extra_va
            gc.collect()
        if len(row["arms"]) == 2:
            g = row["arms"]["w512"]["peak"] / row["arms"]["base"]["peak"] - 1.0
            print(f"  ⟹ fold {index} Δpeak = {100*g:+.2f}%", flush=True)
        fold_rows.append(row)
        del short, long_blocks, xs_tr, xs_va
        gc.collect()

    report = build_report(fold_rows, plan_sha, args, linear_mismatch,
                          time.perf_counter() - started)
    write_outputs(Path(args.output_dir), args.label, report)


def build_report(fold_rows, plan_sha, args, linear_mismatch, elapsed) -> dict:
    base = np.array([r["arms"]["base"]["peak"] for r in fold_rows])
    arm = np.array([r["arms"]["w512"]["peak"] for r in fold_rows])
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
    passed = all(v for v in gates.values() if v is not None)
    above_floor = bool(pooled >= DETECTION_FLOOR)
    if linear_mismatch:
        verdict = "INVALID_LINEAR_CROSSCHECK_FAILED"
    elif passed and above_floor:
        verdict = "PASS"
    elif passed:
        verdict = "PASS_BUT_BELOW_DETECTION_FLOOR"
    else:
        verdict = "REJECTED"
    return {"experiment": "long_window_confirm", "plan_sha256": plan_sha,
            "tier": f"confirmation {N_SEEDS}s x {NUM_ITERATION}", "window": WINDOW,
            "stage1_only": bool(args.stage1), "baseline_cache": CACHE_PATH.name,
            "detection_floor": DETECTION_FLOOR, "folds": fold_rows,
            "linear_crosscheck_mismatches": linear_mismatch,
            "summary": {"pooled_gain": pooled, "fold_gains": gains.tolist(),
                        "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
                        "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
                        "exceeds_detection_floor": above_floor,
                        "floor_multiple": pooled / DETECTION_FLOOR,
                        "gates": gates, "passed": passed},
            "verdict": verdict, "elapsed_seconds": elapsed}


def write_outputs(output_dir: Path, label: str, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    s = report["summary"]
    lines = [f"# 长窗 w512 确认档（`{label}`）", "",
             f"> 预注册判据 sha256 `{report['plan_sha256']}`，先于结果落盘。",
             f"> 档位：**{report['tier']}**，只跑截面块。窗口 {report['window']}（观测数）。",
             f"> ⚠️ 检出下限 **{100*report['detection_floor']:.1f}%**（3s480），"
             f"比筛选档的 6.1% **更高**。", ""]
    if report["linear_crosscheck_mismatches"]:
        lines += ["## ❌ 线性对拍失败 —— 实验判无效", ""]
        lines += [f"- {m}" for m in report["linear_crosscheck_mismatches"]]
        lines += ["", "线性次臂不依赖树超参，本应与 `long_window_ladder` 逐位相同。"
                  "对不上 ⟹ 数据路径有差异，树的读数也不可信。", ""]
    else:
        lines += ["> ✅ 线性对拍：10 次拟合全部与 `long_window_ladder` **逐位相同** "
                  "⟹ 两次运行的数据路径一致。", ""]
    lines += ["| 指标 | 值 |", "|---|--:|",
              f"| pooled Δpeak | **{100*s['pooled_gain']:+.2f}%** |",
              f"| 正折 | {s['positive_folds']}/{s['n_folds']} |",
              f"| 去最好折 | {100*s['drop_best_gain']:+.2f}% |",
              f"| bootstrap CI 下界 | {100*s['bootstrap_ci_lower']:+.2f}% |",
              f"| 检出下限倍数 | {s['floor_multiple']:.2f}× |", "", "### 逐门槛", ""]
    lines += [f"- {'✅' if ok else '❌'} {g}" for g, ok in s["gates"].items()]
    lines += ["", "### 逐折", "", "| fold | base | w512 | Δpeak |", "|---|--:|--:|--:|"]
    for r, g in zip(report["folds"], s["fold_gains"]):
        lines.append(f"| {r['fold']} | {r['arms']['base']['peak']:.4e} | "
                     f"{r['arms']['w512']['peak']:.4e} | {100*g:+.2f}% |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    if report["verdict"] == "PASS_BUT_BELOW_DETECTION_FLOOR":
        lines += ["> 五道门槛全过，但幅度低于 3s480 的 8.7% 检出下限 ⟹ **方向可信、幅度测不出**。",
                  "> 按预注册：只够作为花一次公榜额度的理由，**不构成晋级依据**。", ""]
    (output_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / (label + '.json')}\nwrote {output_dir / (label + '.md')}")


if __name__ == "__main__":
    main()
