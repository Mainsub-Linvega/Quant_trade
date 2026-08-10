"""岭回归到底缺不缺数据？—— 训练窗阶梯的逐折配对测量。

## 要验证的假设

生产岭回归（`strategies/v1_ridge/train.py` 默认 `--train-partitions 4 × --sample-modulo 5`
⟹ `train_rows = 1,146,659`）**只用了 13.2M 行的 8.7%**。加数据能显著抬高 `m̂` 的质量。

`replace` 之后岭回归**唯一的作用就是产 `m̂`**（400 列里 200 列贡献恒为 0），
所以这里的主指标就是 `m̂` 的质量：尺度无关的 `peak_m = A²/B`。

## 为什么这次值得测（不是拍脑袋）

`market_model_w60k` 那次探针留下一个没被追问的数：把每折训练窗从 394,800 拉到
600,000 个原始 time_id（**+52%**）之后，基准臂 `implied_u` 的 `peak_m`
从 **0.002474 涨到 0.002655（+7.3%）**；同一次里「先取截面均值再回归」的那条臂
**纹丝不动**（0.002524 → 0.002523）。⟹ 增益来自**行级**信息，正是加数据能买到的。

但那两个数来自**两次独立的运行、折的划分不同**，不是配对比较。本脚本把它做成配对的。

## 怎么保证是「配对」的

`rolling_time_folds` 的验证段位置由 `train_window` 决定 —— 直接换窗口跑两遍，
两边评的根本不是同一批行。所以这里：**折按阶梯里最大的那个窗口划一次**，
小窗口取同一段训练区间的**后缀**（`train_ids[-W:]`）。
于是所有档共用**逐位相同的验证段**，Δ 才是配对的。

## alpha 必须跟着行数走

不然「加数据」会混进第②类的「松紧」变化，测出来的就不是数据量的效应。
沿用 `market_model` 的 `fold_alpha = ridge_alpha × train_window / PROD_SAMPLED_WINDOW`
（Ridge 的损失是行的求和 ⟹ alpha 正比于行数 = 每行的正则强度不变）。

## 分类与判据

「加数据」是**第③类（结构性）**改动 ⟹ 本地滚动配对 Δ 约 1:1 迁移，本地尺子能信。
判据写死在 `verdict()` 里，由代码判不由报告文字判（伤疤清单 #2）。

⚠️ **本脚本只量数据量这一根轴**，验证段仍是 `sample_modulo` 抽出来的那个相位。
两个臂看同一批验证行，所以配对 Δ 没问题；但**相位错配是另一根轴**，
要用 `experiments/ridge_phase_sampling.py` 那条流式全相位路径量，别在这里读结论。

用法：
    .venv/bin/python experiments/ridge_data_ladder.py
输出：outputs/experiments/ridge_data_ladder.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import fit_model, predict_array
# 复用已有实现，不另写一份（口径唯一性）
from mt_predictability import group_starts, weighted_group_mean, decompose_score
from market_model import ab_peak, unweighted_group_mean, sign_test_p
from walk_forward_rolling import PROD_SAMPLED_WINDOW, load_all_sampled

# 阶梯（**采样后**的 time_id 数，sample_modulo=10 ⟹ ×10 就是原始 time_id）。
# 39,480 那一档 ≈ 生产的 4 个分区 / 400,000 个原始 time_id —— 基准就是它。
# 30,000 那一档是**故意放的负向档**：真有数据效应的话它必须明显更差，
# 一条单调的阶梯比单个配对 Δ 有说服力得多。
LADDER = (30_000, 39_480, 50_000, 60_000, 70_000)
BASELINE_WINDOW = 39_480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Does the production ridge want more rows?")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="ridge_data_ladder")
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=1.13)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    parser.add_argument("--ladder", type=int, nargs="+", default=list(LADDER))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def row_level_peak(target: np.ndarray, fitted: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """整条预测（不只市场块）的 A/B/峰值，口径与公榜的 `Score(a)=2aA−a²B` 完全一致。

    `predict_array` 必须以 `scale=1 / clip 关掉` 调用，否则 A、B 不再是那个二次式的系数。
    """
    denominator = float(np.dot(weight, target * target))
    a = float(np.dot(weight, target * fitted)) / denominator
    b = float(np.dot(weight, fitted * fitted)) / denominator
    return {"A": a, "B": b, "peak": (a * a / b) if b > 0 else 0.0,
            "optimal_scale": (a / b) if b > 0 else float("nan")}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.label}.md"
    if report_path.exists() and not args.force:
        raise SystemExit(f"{report_path} 已存在；要覆盖请加 --force")

    ladder = sorted(set(int(w) for w in args.ladder))
    if BASELINE_WINDOW not in ladder:
        raise SystemExit(f"阶梯里必须含基准档 {BASELINE_WINDOW}（生产等效），否则没有配对基准")

    print("loading all partitions...", flush=True)
    data = load_all_sampled(train_files(Path(args.data_root)), args.sample_modulo)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)

    # ⭐ 折按**最大**窗口划一次，小窗口取同一训练区间的后缀 → 验证段逐位相同 → Δ 是配对的
    widest = ladder[-1]
    folds = rolling_time_folds(unique_time_ids, args.n_folds, widest, args.embargo)
    print(f"{len(all_time_ids):,} rows, {len(unique_time_ids):,} time_ids "
          f"(sample_modulo {args.sample_modulo} ⟹ 原始 {len(unique_time_ids)*args.sample_modulo:,}), "
          f"{len(folds)} folds，折按最大档 {widest:,} 划定", flush=True)
    for window in ladder:
        print(f"  档 {window:6,d}（原始 {window*args.sample_modulo:9,d}）"
              f"  fold_alpha = {args.ridge_alpha * window / PROD_SAMPLED_WINDOW:.3e}"
              f"{'   ← 基准（生产等效）' if window == BASELINE_WINDOW else ''}")

    fold_results: list[dict[str, object]] = []
    for index, (train_ids_widest, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        valid_set = np.isin(all_time_ids, valid_ids)
        v_time = data["time_id"][valid_set]
        v_features = data["features"][valid_set]
        v_target = data["target"][valid_set].astype(np.float64)
        v_weight = np.maximum(data["weight"][valid_set].astype(np.float64), 0.0)
        v_starts = group_starts(v_time)
        v_counts = np.diff(np.r_[v_starts, len(v_time)]).astype(np.float64)
        group_w = np.add.reduceat(v_weight, v_starts)
        m_valid = weighted_group_mean(v_target, v_weight, v_starts)   # 目标：加权截面均值

        rungs: dict[str, dict[str, object]] = {}
        for window in ladder:
            train_ids = train_ids_widest[-window:]        # 后缀 = 同一段历史的近端
            train_set = np.isin(all_time_ids, train_ids)
            fold_alpha = args.ridge_alpha * window / PROD_SAMPLED_WINDOW

            artifact, selected = fit_model(
                data["features"][train_set], data["target"][train_set],
                data["weight"][train_set], data["time_id"][train_set],
                args.feature_count, fold_alpha,
            )
            # scale=1 / clip 关掉：峰值是尺度无关的，但限幅会破坏线性 → A/B 不再是二次式系数
            raw = predict_array(artifact, v_features, v_time, selected,
                                prediction_scale=1.0, prediction_clip=1e9).astype(np.float64)
            market_u = unweighted_group_mean(raw, v_starts, v_counts)   # 可交付口径（推理拿不到 weight）

            production = predict_array(artifact, v_features, v_time, selected,
                                       args.prediction_scale, args.prediction_clip)
            parts = decompose_score(v_target, production, v_weight, v_starts)
            rungs[str(window)] = {
                "n_train_time_ids": int(len(train_ids)),
                "n_train_rows": int(train_set.sum()),
                "fold_alpha": float(fold_alpha),
                "market_u": ab_peak(m_valid, market_u, group_w),
                "market_w": ab_peak(m_valid, weighted_group_mean(raw, v_weight, v_starts), group_w),
                "full_prediction": row_level_peak(v_target, raw, v_weight),
                "production_score": float(weighted_zero_mean_r2(v_target, production, v_weight)),
                "share_market": float(parts["share_market"]),
            }
            del artifact, selected, raw, market_u, production, train_set
            gc.collect()

        base = rungs[str(BASELINE_WINDOW)]["market_u"]["peak"]
        fold_results.append({
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "n_valid_time_ids": int(len(m_valid)),
            "rungs": rungs,
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        summary = "  ".join(
            f"{w//1000}k={rungs[str(w)]['market_u']['peak']:+.6f}"
            f"({rungs[str(w)]['market_u']['peak']/base-1:+.1%})" for w in ladder)
        print(f"fold {index:2d}: {summary}  ({fold_results[-1]['elapsed_seconds']:.0f}s)", flush=True)
        del valid_set, v_features, v_time, v_target, v_weight
        gc.collect()

    # ------------------------------------------------------------------ 汇总与判据
    def series(window: int, block: str, key: str = "peak") -> np.ndarray:
        return np.array([f["rungs"][str(window)][block][key] for f in fold_results])

    baseline = series(BASELINE_WINDOW, "market_u")
    share_m = float(np.mean([f["rungs"][str(BASELINE_WINDOW)]["share_market"] for f in fold_results]))
    production_score = float(np.mean(
        [f["rungs"][str(BASELINE_WINDOW)]["production_score"] for f in fold_results]))

    comparisons: dict[str, dict[str, object]] = {}
    for window in ladder:
        delta = series(window, "market_u") - baseline
        positive = int((delta > 0).sum())
        without_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
        comparisons[str(window)] = {
            "n_train_time_ids_raw": window * args.sample_modulo,
            "mean_peak_market_u": float(series(window, "market_u").mean()),
            "mean_peak_market_w": float(series(window, "market_w").mean()),
            "mean_peak_full": float(series(window, "full_prediction").mean()),
            "mean_delta": float(delta.mean()),
            "relative_gain": float(delta.mean() / baseline.mean()),
            "positive_folds": positive,
            "sign_test_p": sign_test_p(positive, len(delta)),
            "mean_delta_drop_best": float(without_best.mean()),
            "score_gain_share": float(share_m * delta.mean() / production_score),
        }

    top = comparisons[str(ladder[-1])]
    n_folds = len(fold_results)
    # 5/6 折的符号检验只有 p≈0.109（10 折下的 8/10 是 0.055）—— 功效本来就低，
    # 所以这里额外要求整条阶梯单调，而不是只看最高档赢没赢。
    need_positive = int(math.ceil(n_folds * 0.8))
    rising = [w for w in ladder if w > BASELINE_WINDOW]
    falling = [w for w in ladder if w < BASELINE_WINDOW]
    checks = {
        f"1_paired_delta_positive_{need_positive}of{n_folds}":
            top["mean_delta"] > 0 and top["positive_folds"] >= need_positive,
        "2_survives_drop_best_fold": top["mean_delta_drop_best"] > 0,
        "3_ladder_monotone":
            all(comparisons[str(w)]["mean_delta"] > 0 for w in rising)
            and all(comparisons[str(w)]["mean_delta"] < 0 for w in falling),
        "4_score_gain_at_least_5pct": top["score_gain_share"] >= 0.05,
    }
    verdict = {"checks": checks, "pass": all(checks.values()),
               "evaluated_at_window": ladder[-1]}

    payload = {
        "question": "生产岭回归是不是缺数据？把训练窗从生产等效的 394,800 个原始 time_id "
                    "拉到 700,000，m̂ 的质量（尺度无关的 peak_m）会不会显著变好？",
        "why": "replace 之后岭回归唯一的作用就是产 m̂；而它只用了 13.2M 行的 8.7%。"
               "market_model_w60k 的探针显示训练窗 +52% 时 peak_m +7.3%，但那两个数"
               "来自折划分不同的两次运行，不是配对比较 —— 本脚本把它做成配对的",
        "metric": "peak_m = A²/B（A=ΣW·m·m̂/ΣW·m²，B=ΣW·m̂²/ΣW·m²），"
                  "基准臂与 market_model 的 implied_u 完全一致",
        "pairing": "折按阶梯最大档划定一次，小档取同一训练区间的后缀 → 各档共用逐位相同的验证段",
        "alpha_rule": "fold_alpha = ridge_alpha × train_window / PROD_SAMPLED_WINDOW，"
                      "即每行的正则强度跨档不变（否则混进第②类的松紧变化）",
        "classification": "第③类（结构性）→ 本地滚动配对 Δ 约 1:1 迁移，本地尺子能信",
        "configuration": {
            "ladder": ladder, "baseline_window": BASELINE_WINDOW, "n_folds": args.n_folds,
            "embargo": args.embargo, "sample_modulo": args.sample_modulo,
            "feature_count": args.feature_count, "ridge_alpha": args.ridge_alpha,
            "prod_sampled_window": PROD_SAMPLED_WINDOW,
        },
        "summary": {"baseline_peak_mean": float(baseline.mean()),
                    "share_market_mean": share_m, "production_score_mean": production_score},
        "comparisons": comparisons,
        "verdict": verdict,
        "folds": fold_results,
    }
    (output_dir / f"{args.label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 岭回归缺不缺数据 —— 训练窗阶梯（`{args.label}`）",
        "",
        "主指标 `peak_m = A²/B`（`m̂` 在自己最优 scale 下的 R²_m，**尺度无关**），",
        "基准臂与 `market_model` 的 `implied_u` 逐位同口径 —— 那是**可交付**的那个",
        "（推理端拿不到 `weight`，只能用无权截面均值）。",
        "",
        f"折数 {args.n_folds}，embargo {args.embargo}，sample_modulo {args.sample_modulo}，"
        f"基准档 {BASELINE_WINDOW:,}（≈生产的 4 个分区）。",
        "**折按阶梯最大档划定一次，小档取同一训练区间的后缀 → 各档共用逐位相同的验证段。**",
        f"`fold_alpha = {args.ridge_alpha:.0e} × 窗口 / {PROD_SAMPLED_WINDOW:,}` —— 每行的正则强度跨档不变。",
        "",
        "## 阶梯",
        "",
        "| 训练窗（采样后） | 原始 time_id | 训练行数 | peak_m（无权） | Δ vs 基准 | 相对 | 正折 | 符号 p | 去掉最好一折 | 换算总分 | 整条预测的峰值 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in ladder:
        row = comparisons[str(window)]
        rows_mean = int(np.mean([f["rungs"][str(window)]["n_train_rows"] for f in fold_results]))
        mark = " ⭐" if window == BASELINE_WINDOW else ""
        lines.append(
            f"| {window:,}{mark} | {row['n_train_time_ids_raw']:,} | {rows_mean:,} | "
            f"{row['mean_peak_market_u']:+.6f} | {row['mean_delta']:+.6f} | "
            f"{row['relative_gain']:+.1%} | {row['positive_folds']}/{n_folds} | "
            f"{row['sign_test_p']:.3f} | {row['mean_delta_drop_best']:+.6f} | "
            f"{row['score_gain_share']:+.1%} | {row['mean_peak_full']:+.6f} |")
    lines += [
        "",
        f"生产口径分数均值 {production_score:.8f}，share_m 均值 {share_m:.3f}。",
        "",
        "## 判据（由 `verdict()` 判，不是报告里的评语）",
        "",
        f"**最高档 {ladder[-1]:,}：{'✅ 通过' if verdict['pass'] else '❌ 未通过'}**",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {'✅' if ok else '❌'} {name}")
    lines += [
        "",
        f"⚠️ **功效说明**：{n_folds} 折下 {need_positive}/{n_folds} 的符号检验 "
        f"p ≈ {sign_test_p(need_positive, n_folds):.3f}"
        f"（10 折的 8/10 是 {sign_test_p(8, 10):.3f}）。折少是为了给最大档留出训练窗 —— "
        "所以第 3 条判据要求**整条阶梯单调**（负向档 30,000 必须明显更差），",
        "而不是只看最高档赢没赢。判据没过时要分清「测不出来」和「没效果」（`mt_lagged` 的教训）。",
        "",
        "## 逐折",
        "",
        "| Fold | 验证段 | " + " | ".join(f"{w:,}" for w in ladder) + " |",
        "|---:|---|" + "---:|" * len(ladder),
    ]
    for f in fold_results:
        cells = " | ".join(f"{f['rungs'][str(w)]['market_u']['peak']:+.6f}" for w in ladder)
        lines.append(f"| {f['fold']} | {f['valid_time_range'][0]}–{f['valid_time_range'][1]} | {cells} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n判据 {'通过' if verdict['pass'] else '未通过'} → {report_path}")


if __name__ == "__main__":
    main()
