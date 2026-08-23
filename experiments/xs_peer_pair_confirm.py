"""截面块窄 peer 对确认档（3 种子×480 轮）——把筛选档的低于检出下限的结果测清楚。

## 为什么要跑这一档

`xs_peer_pair_probe.py`（1s160 筛选档）结果：pooled +2.39%、4/5 折、去最好折 +1.31%、
`2ΔA(+3.13%) > ΔB(+1.18%)`——方向和机制都对，但没过 3% 门槛、bootstrap 下界跨 0
（−0.49%）、且只有 1s160 检出下限 6.1% 的 0.39×。按仓库惯例（`long_history_ladder` 走的
就是这条路径：1s160 筛选 PASS → 3s480 确认档才拿到能看清的幅度），这个结果现在只是
"够格进确认档"，不是"证明有效"也不是"证明无效"。

## 与筛选档的差别

只有规模变了：`NUM_ITERATION` 160→480，`N_SEEDS` 1→3（预测取 3 个种子平均，
与生产 XS/market 块同构），检出下限换成 3s480 的实测值 8.7%
（`v3_recency_expanding_ladder_1s160.md`/`v3_slow_variance_3s480.md` 一系列实验统一引用的口径）。
特征、门禁、seed 基数、peer 对 `(0,6)(2,14)(1,13)` 与折版图全部沿用筛选档，未重新挑选、
未重新搜索——这条轴此前已经预注册"只跑一次确认档，不因为差一点点就再调"。
"""

from __future__ import annotations

import gc
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
from xs_market_state_probe import GATE_MIN_GAIN, GATE_MIN_POSITIVE_FOLDS, paired_bootstrap_lower_bound  # noqa: E402
from xs_peer_pair_probe import PAIRS, build_peer_feature  # noqa: E402
from strategies.v3_hybrid.train import MIN_DATA_FRAC, SPEC  # noqa: E402

SEED = 20260823
N_SEEDS = 3
NUM_ITERATION = 480                 # 与生产 XS/market 块的 num_iteration 相同
DETECTION_FLOOR_3S480 = 0.087       # 3s480 规模的实测检出下限
OUTPUT_DIR = _REPO_ROOT / "outputs" / "experiments"
LABEL = "xs_peer_pair_confirm_3s480"


def fit_lgbm_multiseed(design_tr: np.ndarray, label_tr: np.ndarray, weight_tr,
                       design_va: np.ndarray, cat_index: int) -> np.ndarray:
    """N_SEEDS 个种子 × NUM_ITERATION 轮，预测取平均——与生产 XS/market 块同构。"""
    import lightgbm as lgb

    min_data = max(20, int(round(MIN_DATA_FRAC * len(design_tr))))
    total = np.zeros(len(design_va), dtype=np.float64)
    for s in range(N_SEEDS):
        seed = SEED + s
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
        total += booster.predict(design_va, num_iteration=NUM_ITERATION)
        del booster, ds
    return total / N_SEEDS


def main() -> None:
    started = time.perf_counter()
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")
    if v3_path not in sys.path:
        sys.path.append(v3_path)
    from strategies.v3_hybrid.train import stream_history_blocks
    from src.oof_cache import assert_reproducible_cache
    assert_reproducible_cache(CACHE_PATH)
    cache = np.load(CACHE_PATH)
    keep = cache["fold"] >= 0
    cache = {k: cache[k][keep] for k in ("time_id", "asset_id", "target", "weight", "e_lgbm")}

    rows = load_rows(Path("data"), SAMPLE_MODULO, SAMPLING)
    features, target = rows["features"], rows["target"]
    weight, time_ids, asset_ids = rows["weight"], rows["time_id"], rows["asset_id"]
    del rows
    order = np.argsort(time_ids, kind="stable")
    features, target = features[order], target[order]
    weight, time_ids, asset_ids = weight[order], time_ids[order], asset_ids[order]
    peer_feature = build_peer_feature(time_ids, asset_ids, target)

    folds = rolling_time_folds(np.unique(time_ids), N_FOLDS, TRAIN_WINDOW, EMBARGO)
    print(f"{len(target):,} 行 / {len(np.unique(time_ids)):,} 采样 time_id / {len(folds)} 折 / "
          f"{N_SEEDS} 种子 × {NUM_ITERATION} 轮（{time.perf_counter()-started:.0f}s）", flush=True)

    cache_key = cache["time_id"] * 16 + cache["asset_id"]
    cache_order = np.argsort(cache_key, kind="stable")
    cache_key_sorted = cache_key[cache_order]

    fold_rows: list[dict] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
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
        dev_tr = cross_sectional_deviation(transformed_train[:, xs_selected].copy(), tid_tr)
        dev_va = cross_sectional_deviation(transformed_valid[:, xs_selected].copy(), tid_va)
        history_positions = np.sort(select_features(dev_tr, e_tr, unit, HISTORY_COUNT).astype(np.int64))
        history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
        history_stats = tuple(stats[key][xs_selected[history_positions]]
                              for key in ("lower", "upper", "center", "scale"))
        del transformed_train, transformed_valid
        gc.collect()

        short = stream_history_blocks(Path("data"), SAMPLE_MODULO, SAMPLING,
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

        arm_tr = np.ascontiguousarray(np.column_stack(
            [base_tr[:, :base_cat], peer_feature[tr], base_tr[:, base_cat]]))
        arm_va = np.ascontiguousarray(np.column_stack(
            [base_va[:, :base_cat], peer_feature[va], base_va[:, base_cat]]))
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
                ("tree_peer_pair", arm_tr, arm_va, arm_cat)):
            t1 = time.perf_counter()
            pred = fit_lgbm_multiseed(design_tr, e_tr, w_tr, design_va, cat)
            pred = pred - group_mean(pred, va_starts, va_counts)
            ic, a, b, d = weighted_ic(e_va, pred, w_va)
            row["arms"][name] = {"ic": ic, "A": a, "B": b, "D": d,
                                 "r_vs_prod": ic / ic_prod, "seconds": time.perf_counter() - t1}
            print(f"  {name:20s} IC={ic:+.5f}  r_vs_prod={ic/ic_prod:+.4f} "
                  f"({time.perf_counter()-t1:.0f}s)", flush=True)
            del pred
            gc.collect()
        fold_rows.append(row)
        del base_tr, base_va, arm_tr, arm_va
        gc.collect()
        print(f"fold {index} 总耗时 {time.perf_counter()-t0:.0f}s，累计 {time.perf_counter()-started:.0f}s",
              flush=True)

    report = build_report(fold_rows, time.perf_counter() - started)
    write_outputs(report)


def build_report(fold_rows: list[dict], elapsed: float) -> dict:
    base, arm = "tree_base", "tree_peer_pair"
    gains = np.array([r["arms"][arm]["ic"] / r["arms"][base]["ic"] - 1.0 for r in fold_rows])
    num = np.array([r["arms"][arm]["ic"] for r in fold_rows])
    den = np.array([r["arms"][base]["ic"] for r in fold_rows])
    pooled = float(num.sum() / den.sum() - 1.0)
    d_a = float(np.mean([r["arms"][arm]["A"] / r["arms"][base]["A"] - 1.0 for r in fold_rows]))
    d_b = float(np.mean([r["arms"][arm]["B"] / r["arms"][base]["B"] - 1.0 for r in fold_rows]))
    drop_best = float(np.delete(num, gains.argmax()).sum() / np.delete(den, gains.argmax()).sum() - 1.0)
    ci_low = paired_bootstrap_lower_bound(num, den)
    gates = {
        "1_pooled_relative_gain_at_least_3pct": bool(pooled >= GATE_MIN_GAIN),
        "2_at_least_4_of_5_folds_positive": bool((gains > 0).sum() >= GATE_MIN_POSITIVE_FOLDS),
        "3_survives_drop_best_fold": bool(drop_best > 0),
        "4_two_delta_A_exceeds_delta_B": bool(2 * d_a > d_b),
        "5_paired_bootstrap_ci_lower_bound_positive": bool(ci_low > 0),
        "6_exceeds_detection_floor": bool(pooled >= DETECTION_FLOOR_3S480),
    }
    passed = all(gates.values())
    summary = {"pooled_gain": pooled, "fold_gains": gains.tolist(),
              "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
              "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
              "delta_A": d_a, "delta_B": d_b, "gates": gates, "passed": passed,
              "detection_floor": DETECTION_FLOOR_3S480}
    return {"experiment": LABEL, "pairs": PAIRS, "n_seeds": N_SEEDS, "num_iteration": NUM_ITERATION,
            "folds": fold_rows, "summary": summary,
            "verdict": "PASS" if passed else "REJECTED", "elapsed_seconds": elapsed}


def write_outputs(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{LABEL}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    s = report["summary"]
    lines = [f"# 截面块窄 peer 对确认档（`{LABEL}`）", "",
             f"> 筛选档 `xs_peer_pair_probe.md` 的确认档：{report['n_seeds']} 种子 × "
             f"{report['num_iteration']} 轮，检出下限换成 3s480 的 8.7%。"
             "特征、门禁、peer 对、折版图与筛选档相同，未重新挑选。", "",
             "| pooled 配对增量 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 | 检出下限 | 判定 |",
             "|--:|--:|--:|--:|--:|:--:|--:|--:|:--:|",
             f"| **{100*s['pooled_gain']:+.2f}%** | {s['positive_folds']}/{s['n_folds']} | "
             f"{100*s['drop_best_gain']:+.2f}% | {100*s['delta_A']:+.2f}% | {100*s['delta_B']:+.2f}% | "
             f"{'✅' if s['gates']['4_two_delta_A_exceeds_delta_B'] else '❌'} | "
             f"{100*s['bootstrap_ci_lower']:+.2f}% | {100*s['detection_floor']:.1f}% | "
             f"{'✅' if s['passed'] else '❌'} |", "",
             "### 逐门槛", ""]
    lines += [f"- {'✅' if ok else '❌'} {g}" for g, ok in s["gates"].items()]
    lines += ["", "### 逐折 IC", "",
              "| fold | 生产 e_lgbm | tree_base | tree_peer_pair | 增量 |",
              "|---|--:|--:|--:|--:|"]
    for r, g in zip(report["folds"], s["fold_gains"]):
        lines.append(f"| {r['fold']} | {r['ic_e_lgbm']:+.5f} | {r['arms']['tree_base']['ic']:+.5f} | "
                     f"{r['arms']['tree_peer_pair']['ic']:+.5f} | {100*g:+.2f}% |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    (OUTPUT_DIR / f"{LABEL}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT_DIR / (LABEL + '.json')}\nwrote {OUTPUT_DIR / (LABEL + '.md')}")


if __name__ == "__main__":
    main()
