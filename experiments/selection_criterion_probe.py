"""选列准则探针：history 那 40 列「选的标准」与「用的方式」对不上，换掉会怎样？

## 问的问题

`history_positions` 是第二道、独立的筛子（`v3_production_oof.py:412`）：在已选出的
200 列 `xs_dev` 里，再按 **`|corr(xs_dev[t], e[t])|`（当期）** 取 top-40。
但这 40 列进模型的形态是 `previous`(lag1) / `difference` / `rolling_mean(5)` /
`rolling_deviation` —— **全是滞后量**。选的标准与用的方式不是一回事。

2026-08-21 现测（`train_partition_008`，同生产口径）：

    top-40 重合   当期 vs lag1 = 24/40   当期 vs rollmean5 = 25/40
                  lag1 vs rollmean5 = 35/40   ← 两个滞后准则彼此高度一致

两个滞后准则彼此重合 35/40、却都只与当期重合 24~25/40 ⟹ **真实区分，不是抽样噪声**。
而 `hist_c80`（top-80 超集，公榜 −0.0032%）只证明「更多列没用」，
**没证明「换一批 40 个没用」**。

## 为什么评价器必须是树

① 生产是树；② 对 `lasso200` 臂，用线性模型评价线性选择器是循环论证。
⟹ 用仓库自己的筛选档：LightGBM 1 seed × 160 rounds、**只跑截面块**。

## 判据里为什么没有 `2ΔA > ΔB`

2026-08-21 实测 `A比/√(B比) ≡ IC比` ⟹ ΔA/ΔB 混着两臂解的**共同尺度**
（`A→cA`、`B→c²B` 时 peak 不变而两个 Δ 都非零）；且它本是**两分量配比**的判别式，
不适用于「同一模型换选列」的比较。ROADMAP 2026-08-19 记着 P8 栽过同一个坑，
同日 `long_history_probe` 又栽一次。⟹ **只用尺度不变量 `peak = A²/(B·D)`**；
ΔA/ΔB 仍打印但**明确标注为非判据**。

判据先于结果落盘在 `outputs/experiments/selection_criterion_probe_plan.json`。
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

PLAN_PATH = _REPO_ROOT / "outputs" / "experiments" / "selection_criterion_probe_plan.json"
CACHE_PATH = (_REPO_ROOT / "outputs" / "cache"
              / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz")
# ---- 预注册常量（与 plan.json 一一对应；改这里必须同步改那里）----
FEATURE_COUNT = 200
HISTORY_COUNT = 40
HISTORY_WINDOW = 5
TRAIN_WINDOW = 78_960
EMBARGO = 6
N_FOLDS = 5
SAMPLE_MODULO = 5
SAMPLING = "phase_balanced"
NUM_ITERATION = 160
N_SEEDS = 1
SEED = 2026
ROLL_WINDOW = 5                    # rollmean5 准则用的窗口，与 HISTORY_WINDOW 同值
ARMS = ("base", "hist_lag1", "hist_roll5", "lasso200")
BOOTSTRAP_BLOCKS = 200
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 2026
GATE_MIN_GAIN = 0.03
GATE_MIN_POSITIVE_FOLDS = 4
DETECTION_FLOOR = 0.061            # ROADMAP P2-R：1s160/5 折的实测检出下限


# ------------------------------------------------------------------ 工具

def peak_of(label: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    """返回 (peak, A, B)。`peak = A²/(B·D)` 对预测的整体缩放**严格不变**。"""
    a = float(np.dot(weight * label, pred))
    b = float(np.dot(weight * pred, pred))
    d = float(np.dot(weight * label, label))
    return (a * a / (b * d) if b > 0 and d > 0 else 0.0), a, b


def lagged_and_rolling(values: np.ndarray, asset_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐 asset 的严格滞后 lag1 与 rollmean(ROLL_WINDOW)。

    与 `strategies/v3_hybrid/history.py` 的边界规则一致：**无历史时取 0**，
    滚动均值只用 `t−W .. t−1`、不含当前行。这里只用来**排序选列**，
    不进设计矩阵，所以可以用 cumsum 向量化（生产那份为了在线/离线逐位一致不能用）。
    """
    lag1 = np.zeros_like(values)
    roll = np.zeros_like(values)
    for asset in np.unique(asset_ids):
        rows = np.flatnonzero(asset_ids == asset)          # 已按 time_id 序
        series = values[rows].astype(np.float64)
        lag1[rows[1:]] = series[:-1].astype(values.dtype)
        cumulative = np.vstack([np.zeros((1, series.shape[1])), np.cumsum(series, axis=0)])
        index = np.arange(len(series))
        left = np.maximum(index - ROLL_WINDOW, 0)
        count = np.maximum(index - left, 1)
        mean = (cumulative[index] - cumulative[left]) / count[:, None]
        mean[0] = 0.0
        roll[rows] = mean.astype(values.dtype)
    return lag1, roll


def lasso_select(features: np.ndarray, label: np.ndarray, count: int) -> np.ndarray:
    """LASSO 路径上**恰好** `count` 个非零的那一点的列集合。

    走 `lars_path_gram(Xy, Gram, n_samples)` —— 只要 `XᵀX`(323²) 与 `Xᵀe`，
    不必 materialize 1.18m × 323 的 float64 设计矩阵。取「恰好 count 个非零」之后
    **没有任何自由超参**（ElasticNet 还要定 l1_ratio 与 λ₂）⟹ 自由度比计划文稿更少。
    """
    from sklearn.linear_model import lars_path_gram

    n_rows, n_cols = features.shape
    gram = np.zeros((n_cols, n_cols), dtype=np.float64)
    rhs = np.zeros(n_cols, dtype=np.float64)
    for start in range(0, n_rows, 100_000):                 # 分块，避免 3 GB 的 float64 拷贝
        block = features[start:min(start + 100_000, n_rows)].astype(np.float64)
        gram += block.T @ block
        rhs += block.T @ label[start:start + len(block)]
        del block
    _, _, coefs = lars_path_gram(Xy=rhs, Gram=gram, n_samples=n_rows,
                                 method="lasso", max_iter=8 * count)
    nonzero = (coefs != 0).sum(axis=0)
    step = int(np.argmax(nonzero >= count))
    if nonzero[step] < count:                               # 路径没走到，回退到最后一步
        step = len(nonzero) - 1
    selected = np.flatnonzero(coefs[:, step] != 0)
    if len(selected) > count:                               # 恰好那步可能一次进多列
        selected = selected[np.argsort(np.abs(coefs[selected, step]))[-count:]]
    return np.sort(selected.astype(np.int64))


def paired_bootstrap_lower_bound(num: np.ndarray, den: np.ndarray) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(num), size=(BOOTSTRAP_DRAWS, len(num)))
    return float(np.percentile(num[draw].sum(1) / den[draw].sum(1) - 1.0, 2.5))


# ------------------------------------------------------------------ 主流程

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="selection_criterion_probe")
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
    print(f"  ⊘ 剔除：{plan['excluded_gate']['name']} —— {plan['excluded_gate']['why_excluded'][:60]}…")
    print(f"臂：{list(ARMS)}   评价器：LightGBM {N_SEEDS}s × {NUM_ITERATION} 轮，只跑截面块")
    print(f"检出下限 {100*DETECTION_FLOOR:.1f}%（{plan['detection_floor']['source']}）")
    print(f"诚实先验：{plan['honest_prior'][:80]}…")
    if args.dry_run:
        print("\n--dry-run：未读数据。")
        return

    out_json = Path(args.output_dir) / f"{args.label}.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"产物已存在：{out_json}；要覆盖请加 --force（CLAUDE.md §5.10）")

    started = time.perf_counter()
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")   # 同 function_class_probe 的理由
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

        # ---- 两套 200 列：现状（当期单变量）与 LASSO
        xs_sets = {"uni": select_features(transformed_train, e_tr, unit, FEATURE_COUNT)}
        xs_sets["lasso"] = lasso_select(transformed_train, e_tr, FEATURE_COUNT)
        overlap = len(set(xs_sets["uni"].tolist()) & set(xs_sets["lasso"].tolist()))
        print(f"fold {index}: 200 列 uni∩lasso = {overlap}/{FEATURE_COUNT}", flush=True)

        xs_dev = {}
        for key, sel in xs_sets.items():
            xs_dev[key] = (cross_sectional_deviation(transformed_train[:, sel].copy(), tid_tr),
                           cross_sectional_deviation(transformed_valid[:, sel].copy(), tid_va))
        del transformed_train, transformed_valid
        gc.collect()

        # ---- 四个臂各自的 history 40 列（只有准则不同）
        lag1, roll5 = lagged_and_rolling(xs_dev["uni"][0], aid_tr)
        criteria = {"base": xs_dev["uni"][0], "hist_lag1": lag1, "hist_roll5": roll5}
        arm_spec: dict[str, dict] = {}
        for name, matrix in criteria.items():
            pos = np.sort(select_features(matrix, e_tr, unit, HISTORY_COUNT).astype(np.int64))
            arm_spec[name] = {"xs_key": "uni", "positions": pos,
                              "global": xs_sets["uni"][pos]}
        del lag1, roll5
        pos_l = np.sort(select_features(xs_dev["lasso"][0], e_tr, unit, HISTORY_COUNT).astype(np.int64))
        arm_spec["lasso200"] = {"xs_key": "lasso", "positions": pos_l,
                                "global": xs_sets["lasso"][pos_l]}
        gc.collect()

        # ---- 并集只跑一次 history 流（四个块逐列独立 ⟹ 算并集再切片与单算等价）
        union = np.sort(np.unique(np.concatenate([s["global"] for s in arm_spec.values()])))
        union_names = [f"feature_{int(i):03d}" for i in union]
        union_stats = tuple(stats[key][union] for key in ("lower", "upper", "center", "scale"))
        print(f"fold {index}: history 并集 {len(union)} 列（4 臂各 {HISTORY_COUNT}），"
              f"流式扫一次（{time.perf_counter()-t0:.0f}s）", flush=True)
        union_blocks = stream_history_blocks(Path(args.data_root), SAMPLE_MODULO, SAMPLING,
                                             union_names, union_stats, HISTORY_WINDOW)

        probe_key = tid_va * 16 + aid_va
        take = cache_order[np.searchsorted(cache_key_sorted, probe_key)]
        if not (np.max(np.abs(cache["target"][take] - y_va)) == 0.0
                and np.max(np.abs(cache["weight"][take] - w_va)) == 0.0):
            raise AssertionError(f"fold {index}: join 后 target/weight 不逐位相同")
        print(f"  join 通过：{len(take):,} 行", flush=True)

        row: dict = {"fold": index, "n_valid": int(len(take)),
                     "xs_overlap_uni_lasso": overlap, "arms": {}}
        base_global = set(arm_spec["base"]["global"].tolist())
        for name in ARMS:
            t1 = time.perf_counter()
            spec = arm_spec[name]
            cols = np.searchsorted(union, spec["global"])       # 该臂的 40 列在并集里的位置
            xs_tr, xs_va = xs_dev[spec["xs_key"]]
            design_tr = np.column_stack(
                [xs_tr, *[b[tr][:, cols] for b in union_blocks], aid_tr.astype(np.float32)])
            design_va = np.column_stack(
                [xs_va, *[b[va][:, cols] for b in union_blocks], aid_va.astype(np.float32)])
            pred = fit_predict_lgbm(design_tr, e_tr, w_tr, design_va, lgbm_args,
                                    f"{name}_cross", XS_SPEC,
                                    categorical_columns=[design_tr.shape[1] - 1])
            pred = pred - group_mean(pred, va_starts, va_counts)
            pk, a, b = peak_of(e_va, pred, w_va)
            row["arms"][name] = {"peak": pk, "A": a, "B": b,
                                 "history_overlap_with_base":
                                     len(base_global & set(spec["global"].tolist())),
                                 "seconds": time.perf_counter() - t1}
            print(f"  {name:11s} peak={pk:.6e}  history∩base={row['arms'][name]['history_overlap_with_base']:2d}/40"
                  f"  ({time.perf_counter()-t1:.0f}s)", flush=True)
            del design_tr, design_va, pred
            gc.collect()
        fold_rows.append(row)
        del union_blocks, xs_dev, arm_spec, xs_sets
        gc.collect()

    report = build_report(fold_rows, plan_sha, args, time.perf_counter() - started)
    write_outputs(Path(args.output_dir), args.label, report)


def build_report(fold_rows: list[dict], plan_sha: str, args, elapsed: float) -> dict:
    base = np.array([r["arms"]["base"]["peak"] for r in fold_rows])
    summary: dict = {}
    for name in ARMS:
        if name == "base":
            continue
        arm = np.array([r["arms"][name]["peak"] for r in fold_rows])
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
        # ΔA/ΔB 只打印不判：见 plan.json 的 excluded_gate
        d_a = float(np.mean([r["arms"][name]["A"] / r["arms"]["base"]["A"] - 1.0 for r in fold_rows]))
        d_b = float(np.mean([r["arms"][name]["B"] / r["arms"]["base"]["B"] - 1.0 for r in fold_rows]))
        summary[name] = {"pooled_gain": pooled, "fold_gains": gains.tolist(),
                         "positive_folds": int((gains > 0).sum()), "n_folds": len(gains),
                         "drop_best_gain": drop_best, "bootstrap_ci_lower": ci_low,
                         "delta_A_not_a_gate": d_a, "delta_B_not_a_gate": d_b,
                         "exceeds_detection_floor": bool(pooled >= DETECTION_FLOOR),
                         "gates": gates,
                         "passed": all(v for v in gates.values() if v is not None)}
    passing = [n for n, s in summary.items() if s["passed"]]
    verdict = "PASS" if passing else "REJECTED"
    return {"experiment": "selection_criterion_probe", "plan_sha256": plan_sha,
            "baseline_cache": CACHE_PATH.name, "stage1_only": bool(args.stage1),
            "evaluator": f"LightGBM {N_SEEDS}s x {NUM_ITERATION} rounds, cross block only",
            "detection_floor": DETECTION_FLOOR, "folds": fold_rows, "summary": summary,
            "passing_arms": passing, "verdict": verdict, "elapsed_seconds": elapsed}


def write_outputs(output_dir: Path, label: str, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# 选列准则探针（`{label}`）", "",
             f"> 预注册判据 sha256 `{report['plan_sha256']}`，先于结果落盘。",
             f"> 评价器：{report['evaluator']}。基准臂 `base` = 现状（当期单变量准则）。",
             f"> 检出下限 **{100*report['detection_floor']:.1f}%**：落在 3%~6.1% 之间读作"
             f"「过门槛但测不出来」，只能作为花公榜额度的理由，不能直接晋级。", ""]
    if report["stage1_only"]:
        lines += ["⚠️ **Stage 1 单折**（预注册降级路径），不构成五折裁决。", ""]
    lines += ["| 臂 | pooled Δpeak | 正折 | 去最好折 | bootstrap CI 下界 | 超检出下限 | 判定 |",
              "|---|--:|--:|--:|--:|:--:|:--:|"]
    for name, s in report["summary"].items():
        drop = "—" if not np.isfinite(s["drop_best_gain"]) else f"{100*s['drop_best_gain']:+.2f}%"
        ci = "—" if not np.isfinite(s["bootstrap_ci_lower"]) else f"{100*s['bootstrap_ci_lower']:+.2f}%"
        lines.append(f"| `{name}` | **{100*s['pooled_gain']:+.2f}%** | {s['positive_folds']}/{s['n_folds']} | "
                     f"{drop} | {ci} | {'✅' if s['exceeds_detection_floor'] else '❌'} | "
                     f"{'✅' if s['passed'] else '❌'} |")
    lines += ["", "> ⊘ **判据里没有 `2ΔA>ΔB`**：它混着两臂解的共同尺度，且是**两分量配比**的判别式，"
              "不适用于「同一模型换选列」。下表的 ΔA/ΔB **仅供参考，不是判据**。", "",
              "| 臂 | ΔA（非判据） | ΔB（非判据） |", "|---|--:|--:|"]
    for name, s in report["summary"].items():
        lines.append(f"| `{name}` | {100*s['delta_A_not_a_gate']:+.2f}% | {100*s['delta_B_not_a_gate']:+.2f}% |")
    for name, s in report["summary"].items():
        lines += ["", f"### `{name}` 逐门槛", ""]
        lines += [f"- {'✅' if ok else '❌'} {g}" for g, ok in s["gates"].items()]
    lines += ["", "### 逐折", "",
              "| fold | " + " | ".join(f"peak({n})" for n in ARMS) +
              " | " + " | ".join(f"hist∩base({n})" for n in ARMS if n != "base") + " | 200列 uni∩lasso |",
              "|---|" + "--:|" * (len(ARMS) + len(ARMS))]
    for r in report["folds"]:
        lines.append("| %d | " % r["fold"] +
                     " | ".join(f"{r['arms'][n]['peak']:.4e}" for n in ARMS) + " | " +
                     " | ".join(str(r["arms"][n]["history_overlap_with_base"]) for n in ARMS if n != "base") +
                     f" | {r['xs_overlap_uni_lasso']} |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    (output_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / (label + '.json')}\nwrote {output_dir / (label + '.md')}")


if __name__ == "__main__":
    main()
