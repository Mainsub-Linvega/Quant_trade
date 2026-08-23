"""截面块窄 peer 对探针：喂 3 对资产的滞后共动信息，能不能补上树看不到的部分？

## 为什么是这 3 对，不是全量 peer 矩阵

`asset_grouping_diagnostic.py`（08-23）用生产 OOF cache 算了 15 资产两两的 e 相关矩阵。
`e` 逐 time_id 和为 0，机械上把两两相关的均值压到 −1/14≈−0.071；真正有信息量的是**偏离
这条基线的对子**。三对明显偏离：`(0,6)` +0.18（+0.25，约 3.6σ）、`(2,14)` +0.13、
`(1,13)` +0.12——层次聚类在 k=3~5 上稳定把它们分到一起，不是噪声。且"模型解释前的 e"
和"模型解释后的残差 e−e_lgbm"两个相关矩阵**几乎逐位相同**——生产模型目前完全没碰这部分
结构，因为 `asset_id` categorical 分裂只能看"这一行是哪个资产"，看不到"另一个资产这一刻
在干什么"。

`peer_leadlag`（08-14）已经测过**全量**15 资产×5~10 特征广播（75~150 列），`REJECTED`
（−2.31%、1/5 折）——那次是把 2~3 对真信号摊在 13 对纯噪声里，且用的是旧架构（peak
0.00227，远弱于当前生产）。本探针只测上面 3 对，只用 1 列（滞后一期的 e，而非原始特征），
是它报告结尾点名但没做的"窄选 peer 对"。

## 口径（简化版，用户要求"先简单试试"，不写正式预注册 JSON）

- 特征：对资产 `i`（若在 3 对里有搭档 `j`），`peer_e_lag1[t] = e_j[前一个采样 time_id]`；
  没有搭档的 9 个资产该列恒为 0。只用**严格滞后**的 e，无当期信息，无拟合参数——
  是纯算术量，折间不需要重新拟合，全局算一次即可按 tr/va 切片。
- 评价器仍用树（同 `xs_market_state_probe` 的理由：假设的机制是 `asset_id × peer` 的
  非对称交互，线性探测器结构性看不到）。
- 门禁、种子、轮数、5 折设置全部复用 `xs_market_state_probe.py`，便于两次探针互相对照。
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from xs_market_state_probe import (  # noqa: E402
    DETECTION_FLOOR_1S160, GATE_MIN_GAIN, GATE_MIN_POSITIVE_FOLDS, SEED,
    fit_lgbm, paired_bootstrap_lower_bound)

PAIRS = {0: 6, 6: 0, 2: 14, 14: 2, 1: 13, 13: 1}   # 由 asset_grouping_diagnostic 挑出的 3 对
OUTPUT_DIR = _REPO_ROOT / "outputs" / "experiments"
LABEL = "xs_peer_pair_probe"


def build_peer_feature(time_ids: np.ndarray, asset_ids: np.ndarray, target: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)]).astype(np.float64)
    e_all = target - np.repeat(np.add.reduceat(target, starts) / counts, counts.astype(int))

    df = pd.DataFrame({"time_id": time_ids, "asset_id": asset_ids, "e": e_all})
    pivot = df.pivot_table(index="time_id", columns="asset_id", values="e").sort_index()
    prev = pivot.shift(1)   # 上一个**采样**time_id（与 history.py"previous observation"同语义）

    out = np.zeros(len(time_ids), dtype=np.float32)
    for asset, partner in PAIRS.items():
        mask = asset_ids == asset
        vals = prev[partner].reindex(time_ids[mask]).to_numpy()
        out[mask] = np.nan_to_num(vals, nan=0.0).astype(np.float32)
    return out


def main() -> None:
    started = time.perf_counter()
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")
    if v3_path not in sys.path:
        sys.path.append(v3_path)
    from strategies.v3_hybrid.train import stream_history_blocks
    from src.oof_cache import assert_reproducible_cache
    assert_reproducible_cache(CACHE_PATH)
    cache = np.load(CACHE_PATH)
    keep = cache["fold"] >= 0     # 见 asset_grouping_diagnostic 的注释：fold=-1 行 e_lgbm 全 NaN
    cache = {k: cache[k][keep] for k in ("time_id", "asset_id", "target", "weight", "e_lgbm")}

    rows = load_rows(Path("data"), SAMPLE_MODULO, SAMPLING)
    features, target = rows["features"], rows["target"]
    weight, time_ids, asset_ids = rows["weight"], rows["time_id"], rows["asset_id"]
    del rows
    order = np.argsort(time_ids, kind="stable")
    features, target = features[order], target[order]
    weight, time_ids, asset_ids = weight[order], time_ids[order], asset_ids[order]
    peer_feature = build_peer_feature(time_ids, asset_ids, target)
    print(f"peer 列非零占比 {np.count_nonzero(peer_feature) / len(peer_feature):.3f}"
          f"（预期 ≈ 6/15 资产参与 3 对 = 0.4，减去每资产第一次出现时的 0）")

    folds = rolling_time_folds(np.unique(time_ids), N_FOLDS, TRAIN_WINDOW, EMBARGO)
    print(f"{len(target):,} 行 / {len(np.unique(time_ids)):,} 采样 time_id / {len(folds)} 折"
          f"（{time.perf_counter()-started:.0f}s）", flush=True)

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
            pred = fit_lgbm(design_tr, e_tr, w_tr, design_va, cat, SEED)
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
        "6_exceeds_detection_floor": bool(pooled >= DETECTION_FLOOR_1S160),
    }
    passed = all(gates.values())
    summary = {"pooled_gain": pooled, "fold_gains": gains.tolist(),
              "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
              "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
              "delta_A": d_a, "delta_B": d_b, "gates": gates, "passed": passed,
              "detection_floor": DETECTION_FLOOR_1S160}
    return {"experiment": LABEL, "pairs": PAIRS, "folds": fold_rows, "summary": summary,
            "verdict": "PASS" if passed else "REJECTED", "elapsed_seconds": elapsed}


def write_outputs(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{LABEL}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    s = report["summary"]
    lines = [f"# 截面块窄 peer 对探针（`{LABEL}`）", "",
             "> 非正式预注册（用户要求'先简单试试'）；门禁与种子复用 `xs_market_state_probe.py`，"
             "以便两次探针互相对照。特征来源见 `asset_grouping_diagnostic.py` 挑出的 3 对："
             "`(0,6) (2,14) (1,13)`。", "",
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
