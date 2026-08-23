"""截面块市场态交互探针：喂 market_pred_t 给 XS 树，能不能学出 asset×market 交互？

判据先于结果落盘在 `outputs/experiments/xs_market_state_interaction_plan.json`。

## 与 8/14 `asset × observable regime`（REJECTED）的区别

那次是**训练完成后**在残差上打一个 2-bin 手工线性 adapter，regime 用**模型自己预测的
截面 RMS** 定义（自指）。本探针把市场态做成**训练时的输入列**，容量交给树自己决定
（沿用生产 SPEC/160 轮），regime 换成**不自指**的 `market_pred_t`——由一个只在训练折内
拟合的行级 LGBM 打 `y`、取其逐 time_id 无权截面均值得到，与 XS 树的输出无关。

## market_pred_t 怎么算（工程简化，已在计划里预注册）

计划里写的是"市场块自身预测值"。为了把探针控制在一次筛选档的成本内，这里只用
**行级 LGBM（1 种子 / 160 轮）**打 `y` 来近似市场块，不混岭回归的 0.5 权重
（生产的完整市场块是 Ridge+LGBM 各半）。这是一个更简的代理，不是生产口径本身——
若本探针 PASS、要进候选，必须换成 `train.py` 的完整市场块（含冻结岭回归）。

## 因果性

市场森林只在**训练折**上拟合（label=y_tr，无权，设计矩阵与生产市场块同构：
`raw ‖ dev ‖ history ‖ asset_id`），预测覆盖训练折与验证折两段——这与生产模型
上线后对未来行连续预测完全同构，不是「验证段信息泄漏进训练」。

## 为什么评价器必须是树

假设的机制是"树对 asset_id × market_pred_t 列的非线性分裂"，线性探测器结构性
看不到这种交互，所以本探针跳过 `long_history_probe`/`function_class_probe` 那套
线性主臂，直接用 LightGBM 当评价器——这是本探针预注册里写明的理由，不是图省事。
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
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402
from v3_production_oof import group_mean, row_slice  # noqa: E402
from function_class_probe import (  # noqa: E402
    CACHE_PATH, EMBARGO, FEATURE_COUNT, HISTORY_COUNT, HISTORY_WINDOW,
    N_FOLDS, SAMPLE_MODULO, SAMPLING, TRAIN_WINDOW, weighted_ic)
from strategies.v3_hybrid.train import (  # noqa: E402
    MIN_DATA_FRAC, SPEC, stream_history_blocks)

PLAN_PATH = _REPO_ROOT / "outputs" / "experiments" / "xs_market_state_interaction_plan.json"
SEED = 20260823
NUM_ITERATION = 160                # 与生产 XS/market 块的 num_iteration 相同
BOOTSTRAP_BLOCKS = 200
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 2026
GATE_MIN_GAIN = 0.03
GATE_MIN_POSITIVE_FOLDS = 4
DETECTION_FLOOR_1S160 = 0.061       # v3_recency_expanding_ladder_1s160.md 实测


def fit_lgbm(design_tr: np.ndarray, label_tr: np.ndarray, weight_tr,
             design_va: np.ndarray, cat_index: int, seed: int) -> np.ndarray:
    """一个种子 / NUM_ITERATION 轮，与生产 SPEC 同超参，asset_id 恒为最后一列。"""
    import lightgbm as lgb

    min_data = max(20, int(round(MIN_DATA_FRAC * len(design_tr))))
    params = {
        **SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
        "num_threads": 8, "min_data_in_leaf": min_data,
        "bagging_fraction": 0.7, "bagging_freq": 1,
        "deterministic": True, "force_row_wise": True, "feature_pre_filter": False,
        "seed": seed, "bagging_seed": seed + 1000, "feature_fraction_seed": seed + 2000,
    }
    ds = lgb.Dataset(design_tr, label=label_tr, weight=weight_tr, params=params,
                     categorical_feature=[cat_index], free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=NUM_ITERATION)
    pred = booster.predict(design_va, num_iteration=NUM_ITERATION)
    del booster, ds
    return pred


def paired_bootstrap_lower_bound(gain_num: np.ndarray, gain_den: np.ndarray) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(gain_num), size=(BOOTSTRAP_DRAWS, len(gain_num)))
    ratios = gain_num[draw].sum(1) / gain_den[draw].sum(1) - 1.0
    return float(np.percentile(ratios, 2.5))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="xs_market_state_probe")
    p.add_argument("--stage1", action="store_true", help="只跑 fold 0（预注册的降级路径）")
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
    print(f"诚实先验：{plan['honest_prior']}")
    if args.dry_run:
        print("\n--dry-run：未读数据。")
        return

    out_json = Path(args.output_dir) / f"{args.label}.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"产物已存在：{out_json}；要覆盖请加 --force（CLAUDE.md §5.10）")

    started = time.perf_counter()
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")   # 同 long_history_probe 的理由：
    if v3_path not in sys.path:                              # history.py/lgbm_numpy.py 是
        sys.path.append(v3_path)                             # train.py 内部的裸 import，只能事后补
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

    fold_rows: list[dict] = []
    for index, (train_ids, valid_ids) in enumerate(folds[:1] if args.stage1 else folds):
        t0 = time.perf_counter()
        tr, va = row_slice(time_ids, train_ids), row_slice(time_ids, valid_ids)
        y_tr, y_va, w_tr, w_va = target[tr], target[va], weight[tr], weight[va]
        tid_tr, tid_va = time_ids[tr], time_ids[va]
        aid_tr, aid_va = asset_ids[tr], asset_ids[va]
        tr_starts = np.r_[0, np.flatnonzero(tid_tr[1:] != tid_tr[:-1]) + 1]
        va_starts = np.r_[0, np.flatnonzero(tid_va[1:] != tid_va[:-1]) + 1]
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
        raw_tr = transformed_train[:, xs_selected].copy()
        raw_va = transformed_valid[:, xs_selected].copy()
        dev_tr = cross_sectional_deviation(raw_tr.copy(), tid_tr)
        dev_va = cross_sectional_deviation(raw_va.copy(), tid_va)
        history_positions = np.sort(select_features(dev_tr, e_tr, unit, HISTORY_COUNT).astype(np.int64))
        history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
        history_stats = tuple(stats[key][xs_selected[history_positions]]
                              for key in ("lower", "upper", "center", "scale"))
        del transformed_train, transformed_valid
        gc.collect()

        # 全量扫一遍、按采样掩码留行 —— 与 long_history_probe 同构：history 状态从数据集
        # 开头连续推进（含更早的折），返回顺序与全局采样行序一致，用 tr/va 两个 slice 切出本折。
        short = stream_history_blocks(Path(args.data_root), SAMPLE_MODULO, SAMPLING,
                                      history_names, history_stats, HISTORY_WINDOW)
        history_tr = [b[tr] for b in short]
        history_va = [b[va] for b in short]
        del short
        print(f"fold {index}: history 块就绪（{time.perf_counter()-t0:.0f}s）", flush=True)

        base_tr = np.ascontiguousarray(
            np.column_stack([dev_tr, *history_tr, aid_tr.astype(np.float32)]))
        base_va = np.ascontiguousarray(
            np.column_stack([dev_va, *history_va, aid_va.astype(np.float32)]))
        base_cat = base_tr.shape[1] - 1

        # ---- market_pred_t：只在训练折上拟合的行级 LGBM 打 y，覆盖训练折与验证折两段
        market_tr = np.ascontiguousarray(
            np.column_stack([raw_tr, dev_tr, *history_tr, aid_tr.astype(np.float32)]))
        market_va = np.ascontiguousarray(
            np.column_stack([raw_va, dev_va, *history_va, aid_va.astype(np.float32)]))
        market_cat = market_tr.shape[1] - 1
        market_pred_tr = fit_lgbm(market_tr, y_tr, None, market_tr, market_cat, SEED)
        market_pred_va = fit_lgbm(market_tr, y_tr, None, market_va, market_cat, SEED)
        market_state_tr = group_mean(market_pred_tr, tr_starts, tr_counts).astype(np.float32)
        market_state_va = group_mean(market_pred_va, va_starts, va_counts).astype(np.float32)
        del market_tr, market_va, market_pred_tr, market_pred_va, raw_tr, raw_va
        gc.collect()

        arm_tr = np.ascontiguousarray(np.column_stack(
            [base_tr[:, :base_cat], market_state_tr, base_tr[:, base_cat]]))
        arm_va = np.ascontiguousarray(np.column_stack(
            [base_va[:, :base_cat], market_state_va, base_va[:, base_cat]]))
        arm_cat = arm_tr.shape[1] - 1
        del history_tr, history_va
        gc.collect()

        probe_key = tid_va * 16 + aid_va
        pos = np.searchsorted(cache_key_sorted, probe_key)
        take = cache_order[pos]
        if not (np.max(np.abs(cache["target"][take] - y_va)) == 0.0
                and np.max(np.abs(cache["weight"][take] - w_va)) == 0.0):
            raise AssertionError(f"fold {index}: join 后 target/weight 不逐位相同")
        e_lgbm = cache["e_lgbm"][take]
        ic_prod, *_ = weighted_ic(e_va, e_lgbm, w_va)
        print(f"  join 通过：{len(take):,} 行；生产 e_lgbm IC={ic_prod:+.5f}", flush=True)

        row: dict = {"fold": index, "n_valid": int(len(take)), "ic_e_lgbm": ic_prod, "arms": {}}
        for name, design_tr, design_va, cat in (
                ("tree_base", base_tr, base_va, base_cat),
                ("tree_market_pred", arm_tr, arm_va, arm_cat)):
            t1 = time.perf_counter()
            pred = fit_lgbm(design_tr, e_tr, w_tr, design_va, cat, SEED)
            pred = pred - group_mean(pred, va_starts, va_counts)
            ic, a, b, d = weighted_ic(e_va, pred, w_va)
            row["arms"][name] = {"ic": ic, "A": a, "B": b, "D": d,
                                 "r_vs_prod": ic / ic_prod,
                                 "seconds": time.perf_counter() - t1}
            print(f"  {name:20s} IC={ic:+.5f}  r_vs_prod={ic/ic_prod:+.4f} "
                  f"({time.perf_counter()-t1:.0f}s)", flush=True)
            del pred
            gc.collect()
        fold_rows.append(row)
        del base_tr, base_va, arm_tr, arm_va, market_state_tr, market_state_va
        gc.collect()

    report = build_report(fold_rows, plan_sha, args, time.perf_counter() - started)
    write_outputs(Path(args.output_dir), args.label, report)


def build_report(fold_rows: list[dict], plan_sha: str, args, elapsed: float) -> dict:
    base, arm = "tree_base", "tree_market_pred"
    gains = np.array([r["arms"][arm]["ic"] / r["arms"][base]["ic"] - 1.0 for r in fold_rows])
    num = np.array([r["arms"][arm]["ic"] for r in fold_rows])
    den = np.array([r["arms"][base]["ic"] for r in fold_rows])
    pooled = float(num.sum() / den.sum() - 1.0)
    d_a = float(np.mean([r["arms"][arm]["A"] / r["arms"][base]["A"] - 1.0 for r in fold_rows]))
    d_b = float(np.mean([r["arms"][arm]["B"] / r["arms"][base]["B"] - 1.0 for r in fold_rows]))
    drop_best = (float(np.delete(num, gains.argmax()).sum()
                       / np.delete(den, gains.argmax()).sum() - 1.0)
                 if len(gains) > 1 else float("nan"))
    ci_low = paired_bootstrap_lower_bound(num, den) if len(gains) > 1 else float("nan")
    gates = {
        "1_pooled_relative_gain_at_least_3pct": bool(pooled >= GATE_MIN_GAIN),
        "2_at_least_4_of_5_folds_positive": bool((gains > 0).sum() >= min(GATE_MIN_POSITIVE_FOLDS, len(gains))),
        "3_survives_drop_best_fold": (bool(drop_best > 0) if len(gains) > 1 else None),
        "4_two_delta_A_exceeds_delta_B": bool(2 * d_a > d_b),
        "5_paired_bootstrap_ci_lower_bound_positive": (bool(ci_low > 0) if len(gains) > 1 else None),
        "6_exceeds_detection_floor": bool(pooled >= DETECTION_FLOOR_1S160),
    }
    passed = all(v for v in gates.values() if v is not None)
    summary = {"pooled_gain": pooled, "fold_gains": gains.tolist(),
              "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
              "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
              "delta_A": d_a, "delta_B": d_b, "gates": gates, "passed": passed,
              "detection_floor": DETECTION_FLOOR_1S160}
    return {"experiment": "xs_market_state_probe", "plan_sha256": plan_sha,
            "stage1_only": bool(args.stage1), "baseline_cache": CACHE_PATH.name,
            "folds": fold_rows, "summary": summary,
            "verdict": "PASS" if passed else "REJECTED", "elapsed_seconds": elapsed}


def write_outputs(output_dir: Path, label: str, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    s = report["summary"]
    lines = [f"# 截面块市场态交互探针（`{label}`）", "",
             f"> 预注册判据 sha256 `{report['plan_sha256']}`，先于结果落盘。",
             f"> 基准臂 `tree_base` = 生产截面块逐列相同的设计；"
             f"主臂 `tree_market_pred` = base + market_pred_t（训练折内拟合的行级 LGBM 打 y 的截面均值）。",
             ""]
    if report["stage1_only"]:
        lines += ["⚠️ **Stage 1 单折**（预注册降级路径），不构成五折裁决。", ""]
    lines += ["| pooled 配对增量 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 | 检出下限 | 判定 |",
              "|--:|--:|--:|--:|--:|:--:|--:|--:|:--:|",
              f"| **{100*s['pooled_gain']:+.2f}%** | {s['positive_folds']}/{s['n_folds']} | "
              f"{100*s['drop_best_gain']:+.2f}% | {100*s['delta_A']:+.2f}% | {100*s['delta_B']:+.2f}% | "
              f"{'✅' if s['gates']['4_two_delta_A_exceeds_delta_B'] else '❌'} | "
              f"{100*s['bootstrap_ci_lower']:+.2f}% | {100*s['detection_floor']:.1f}% | "
              f"{'✅' if s['passed'] else '❌'} |", "",
              "### 逐门槛", ""]
    lines += [f"- {'✅' if ok else ('—' if ok is None else '❌')} {g}" for g, ok in s["gates"].items()]
    lines += ["", "### 逐折 IC", "",
              "| fold | 生产 e_lgbm | tree_base | tree_market_pred | 增量 |",
              "|---|--:|--:|--:|--:|"]
    for r, g in zip(report["folds"], s["fold_gains"]):
        lines.append(f"| {r['fold']} | {r['ic_e_lgbm']:+.5f} | {r['arms']['tree_base']['ic']:+.5f} | "
                     f"{r['arms']['tree_market_pred']['ic']:+.5f} | {100*g:+.2f}% |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    (output_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / (label + '.json')}\nwrote {output_dir / (label + '.md')}")


if __name__ == "__main__":
    main()
