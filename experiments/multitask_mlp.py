"""多任务辅助监督：共享 trunk 的 MLP，responder 只在**训练时**提供梯度。

## 与 2026-08-12 已否决的那条有什么不同

`NOTES.md` 08-12 否掉的是：把 8 个可预测 responder 族的**预测值**当**输入特征**加进 target
Ridge ⟹ 五折全负、折均 −20.64%。那条路的误差是两阶段累积的（先预测 responder，
再用这个带噪的预测），而且推理端必须真的算出 responder —— 但官方 runner 会剥掉
`responder_*` 列，所以它在线上根本不成立。

本实验是**另一件事**：

| | 已测（被否） | 本实验 |
|---|---|---|
| responder 的角色 | **输入特征** | **辅助损失**（共享 trunk，多头输出） |
| 误差路径 | 两阶段累积 | 单阶段：只在训练时提供梯度 |
| 推理端 | 需要真的算出 responder | **完全不需要**，只留 target 头（第 0 列） |
| runner 剥列 | 致命 | **无影响** |

08-18 `horizon_auxiliary_cache_probe` 写下的重开条件是「出现**不是**『换目标 / 线性叠加 /
对预测值做二层校准』的新机制」。共享 trunk 的辅助损失三条都不是 ⟹ 条件字面上满足。

## 为什么现在才值得跑

`outputs/experiments/target_mlp_oracle_blend.md`（08-19）从 `target_mlp_screen` 的逐折 A/B
反解出交叉项，算出 **oracle 最优配比折均 +6.97%、5/5 折** ⟹ 当年那个
「等权集成 −54.49%」否掉的是**等权掺弱模型**，不是「MLP 没有独立信息」。
⚠️ 但那是 oracle 上界，按仓库量过的冻结系数让步（−2.54%~−3.84%）折算只剩 **+3.1%~+4.4%**，
恰好卡在③类 +3% 门槛上 ⟹ **值得一次预注册筛选，不值得反复调参**。

## 辅助目标：08-18 量出来的那把窗口梯子

`responder_window_atlas` 实测（等权 MA 窗拟合 RMSE 0.024~0.054、与 target 的峰值 shift 单调）：

```text
responder_00 H=1   responder_02 H=2   responder_03 H=4
  target     H=5   responder_04 H=7   responder_05 H=10
```

一组**同源、嵌套、密度递变**的目标，正是稠密辅助监督最想要的配置。

## λ 怎么实现（不需要自写训练循环）

sklearn 的 `MLPRegressor` 原生支持多输出，损失是各输出列平方误差之和。把第 j 个辅助目标列
乘以 `√λ`，网络的最优输出随之变成 `√λ·f_j`，该列的损失就变成 `λ·Σ(y_j − f_j)²`
⟹ **等价于给这个头损失权重 λ**（输出层是自由的线性层，重参数化是精确的）。

⚠️ 严格等价只在 `alpha=0`（无 L2 罚）时成立：`alpha>0` 时输出层权重被缩放 √λ，
其 L2 罚项也随之变化，引入二阶差异。两个臂用**同一个** alpha ⟹ 比较仍然干净。
这条等价性有单测（`tests/test_multitask_mlp.py`）。

## 预注册（跑之前钉死，不得搜索）

```text
λ                0.3        —— 唯一超参，模块常量，不搜索
辅助目标集       梯子 5 个（responder_00/02/03/04/05），不换、不增减
架构/迭代/种子   与 target-only 对照臂**完全相同**
对照臂           target_only —— 没有它，正结果无法归因给辅助损失
Stage 1 门槛     Δpeak > 0 且 Δpeak ≥ 3% 且 multitask > target_only；任一不满足 ⟹ 停
Stage 2 门槛     折均 ≥ +3%、≥3/5 正折、去最好折 > 0（冻结系数）
```

⚠️ **2026-08-19 的口径订正（先说清楚，因为它改的是预注册里的一条）**：
Stage 1 原本写的第二道是仓库惯用的 `2ΔA > ΔB`。那条在这里**不成立** —— 它的前提是两个
预测在同一 scale 约定下比较，而 oracle/frozen 配比会**重解系数**，`A→cA`、`B→c²B`，
peak 不变但 ΔA/ΔB 只反映整体缩放。正确的机制分解是把增益写成「m 中 b 解释不掉的那部分」：

    Δpeak = (A_m − A_b·C/B_b)² / (B_m − C²/B_b)      C = ⟨b, m⟩_w / D

分子是**残差信号**、分母是**残差能量**，两者随 m 的缩放同步变化 ⟹ 商是尺度不变的。
本脚本报告这两个量并自检其商等于实测 Δpeak。⟹ 机制门槛与 `Δpeak > 0` **等价**，
不是独立的一道；原来的三道里有一道是冗余且口径错的。
**订正方向是收紧**（补上本就该有的 3% 幅度门槛），不是放松，且不改变已得结论。

**符号为负直接停** —— 不调 λ、不调 hidden、不换激活、不换辅助目标集、不换第三个种子。
NN 的超参空间足以在 5 折上刷出任何想要的数字；一旦开始搜，这套 OOF 尺子当场失效。

用法：
    # Stage 1：fold 0 符号筛（oracle 上界口径，只看符号）
    .venv/bin/python experiments/multitask_mlp.py --folds 0 --blend-mode oracle \\
        --label multitask_mlp_stage1
    # Stage 2：五折冻结系数终审（只有过了 Stage 1 才跑）
    .venv/bin/python experiments/multitask_mlp.py --folds 0 1 2 3 4 --blend-mode frozen \\
        --label multitask_mlp_stage2
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "experiments"), str(ROOT / "strategies" / "v3_hybrid"),
              str(ROOT / "strategies" / "v1_ridge"), str(ROOT / "strategies" / "v4_mlp")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.metric import scale_invariant_score
from src.validation import rolling_time_folds
from features import cross_sectional_deviation
from history_peak import build_lag_cache, history_blocks, transform_with
from mlp_numpy import NumpyMLP
from mt_predictability import group_starts
from responder_targets import RESPONDER_COLUMNS, load_rows_with_responders
from target_mlp_oracle_blend import oracle_two_component_peak
from train import robust_transform_fit, select_features

# ---- 预注册常量：跑之前钉死，**不得搜索** ----
AUX_LAMBDA = 0.3
# 08-18 responder_window_atlas 实测的窗口梯子：H=1/2/4/7/10，夹着 target 的 H=5
AUX_LADDER = ("responder_00", "responder_02", "responder_03", "responder_04", "responder_05")
ARMS = ("target_only", "multitask")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="multitask_mlp_stage1")
    # 默认对齐生产 OOF 口径（3s480 那份缓存就是这个切分）
    p.add_argument("--baseline-cache", default=str(
        ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"))
    p.add_argument("--candidate-meta", default=str(
        ROOT / "strategies" / "v3_hybrid" / "model" / "hybrid_meta.json"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--folds", type=int, nargs="+", default=[0], help="要评估的折号")
    p.add_argument("--train-window", type=int, default=78_960)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", default="phase_balanced",
                   choices=["periodic", "phase_balanced"])
    p.add_argument("--current-feature-count", type=int, default=200)
    p.add_argument("--blend-mode", default="oracle", choices=["oracle", "frozen"],
                   help="oracle=评估折上重解系数（上界，只用于 Stage 1 看符号）；"
                        "frozen=只用更早的折拟合系数（Stage 2 终审）")
    p.add_argument("--market-hidden", type=int, nargs="+", default=[32])
    p.add_argument("--cross-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--max-iter", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts, axis=0) / counts[:, None]


def standardize(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float, float]:
    """加权标准化。与 `target_mlp.standardize_target` 同一口径。"""
    weight = np.maximum(weight.astype(np.float64), 0.0)
    total = float(weight.sum())
    mean = float(np.dot(weight, values) / total)
    variance = float(np.dot(weight, (values - mean) ** 2) / total)
    std = max(float(np.sqrt(max(variance, 0.0))), 1e-8)
    return ((values - mean) / std).astype(np.float64), mean, std


def impute_missing(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """把 responder 的缺失值补成该列的**训练段加权均值**（标准化后即 0）。

    ⚠️ 为什么不能丢行：`target_only` 对照臂没有 responder，不会丢任何行。两个臂必须看到
    **逐行相同**的训练集，否则「multitask 比 target_only 好」就不再能归因给辅助损失。
    ⚠️ 为什么必须在截面去均值**之前**补：`cross_sectional_deviation` 要减掉每个 `time_id`
    的组均值，组里有一个 NaN 就会把**整组**变成 NaN —— 实测 responder_05 的缺失率只有
    0.06%，但扩散后会污染远大于此的比例。
    缺失率实测（partition_000）：r00 0.019% / r02 0.0001% / r03 0% / r04 0.028% / r05 0.062%；
    `target` 本身**没有**缺失。补成均值 = 「这一格不提供梯度信号」，是辅助头上最保守的处理。
    """
    values = np.array(values, dtype=np.float64, copy=True)
    weight = np.maximum(weight.astype(np.float64), 0.0)
    counts: list[int] = []
    for index in range(values.shape[1]):
        column = values[:, index]
        missing = ~np.isfinite(column)
        counts.append(int(missing.sum()))
        if missing.all():
            raise SystemExit(f"辅助目标第 {index} 列在训练段上全为缺失")
        if missing.any():
            observed = ~missing
            column[missing] = float(np.dot(weight[observed], column[observed])
                                    / max(weight[observed].sum(), 1e-12))
    return values, counts


def build_multitask_targets(e_train: np.ndarray, responder_dev: np.ndarray,
                            weight: np.ndarray, aux_lambda: float
                            ) -> tuple[np.ndarray, float, float]:
    """`[ e_std , √λ·r_std … ]`。**第 0 列必须是 target** —— 推理侧靠这个取列。

    辅助目标用 responder 的**截面去均值**而不是原值：cross 头本来就只建模截面分量
    （market 分量由另一个头负责），保持同一分解才是「共享同一套截面表示」。
    """
    standardized, mean, std = standardize(e_train, weight)
    columns = [standardized]
    root_lambda = float(np.sqrt(aux_lambda))
    for index in range(responder_dev.shape[1]):
        aux, _, _ = standardize(responder_dev[:, index].astype(np.float64), weight)
        columns.append(root_lambda * aux)
    targets = np.column_stack(columns)
    if not np.all(np.isfinite(targets)):
        raise SystemExit("辅助目标矩阵含非有限值 —— 补缺失应当在截面去均值之前完成")
    return targets, mean, std


def fit_mlp(design: np.ndarray, targets: np.ndarray, weight: np.ndarray,
            hidden: tuple[int, ...], args: argparse.Namespace, seed: int):
    """训练并与 NumpyMLP 对拍（多输出也支持：`predict` 返回矩阵）。"""
    estimator = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=args.alpha,
        batch_size=args.batch_size, learning_rate_init=args.learning_rate,
        max_iter=args.max_iter, shuffle=True, random_state=seed, tol=0.0,
        early_stopping=False, n_iter_no_change=args.max_iter + 1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(design, targets, sample_weight=np.maximum(weight, 0.0))
    numpy_model = NumpyMLP.from_sklearn(estimator)
    probe = design[:min(4096, len(design))]
    parity = float(np.max(np.abs(np.atleast_2d(estimator.predict(probe))
                                 - np.atleast_2d(numpy_model.predict(probe)))))
    if parity > 1e-5:
        raise RuntimeError(f"sklearn/NumPy MLP mismatch: {parity}")
    return estimator, numpy_model, parity


def load_baseline(cache_path: Path) -> dict[str, np.ndarray]:
    """生产 OOF 缓存。按 `(time_id<<8)|asset_id` 连接（`horizon_auxiliary_cache_probe` 的做法）。"""
    with np.load(cache_path, allow_pickle=False) as cache:
        return {"key": (cache["time_id"].astype(np.int64) << 8) | cache["asset_id"].astype(np.int64),
                "prediction": cache["prediction"].astype(np.float64),
                "fold": cache["fold"].astype(np.int64)}


def blend_coefficients(y: np.ndarray, base: np.ndarray, extra: np.ndarray,
                       weight: np.ndarray) -> tuple[float, float]:
    """两分量最优 `(c1, c2)` = S⁻¹V（加权口径，与 `src.metric` 同一分母约定）。"""
    w = np.maximum(weight, 0.0)
    gram = np.array([[float(np.dot(w * base, base)), float(np.dot(w * base, extra))],
                     [float(np.dot(w * base, extra)), float(np.dot(w * extra, extra))]])
    rhs = np.array([float(np.dot(w * y, base)), float(np.dot(w * y, extra))])
    return tuple(np.linalg.solve(gram, rhs))


def summarise(rows: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = np.array([row["baseline"]["peak"] for row in rows], dtype=np.float64)
    for arm in arms:
        blended = np.array([row["arms"][arm]["blend"]["peak"] for row in rows], dtype=np.float64)
        delta = blended - base
        drop = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
        signal = np.array([row["arms"][arm]["residual_signal"] for row in rows], dtype=np.float64)
        energy = np.array([row["arms"][arm]["residual_energy"] for row in rows], dtype=np.float64)
        out[arm] = {
            "mean_delta": float(delta.mean()),
            "relative_gain": float(delta.mean() / base.mean()),
            "positive_folds": int((delta > 0).sum()),
            "drop_best_mean_delta": float(drop.mean()),
            # 尺度不变的机制分解：Δpeak = 残差信号² / 残差能量
            "mean_residual_signal": float(signal.mean()),
            "mean_residual_energy": float(energy.mean()),
            "mean_blend_weight_share": float(np.mean(
                [row["arms"][arm]["blend_weight_share"] for row in rows])),
        }
    out["baseline_peak_mean"] = float(base.mean())
    if {"multitask", "target_only"} <= set(arms):
        multitask, control = out["multitask"], out["target_only"]
        out["stage1_gates"] = {
            "delta_positive": multitask["mean_delta"] > 0.0,
            "relative_gain_at_least_3pct": multitask["relative_gain"] >= 0.03,
            "beats_target_only": multitask["mean_delta"] > control["mean_delta"],
        }
        out["stage1_pass"] = all(out["stage1_gates"].values())
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / f"{args.label}.json", out / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} 或 {md_path} 已存在；要覆盖请加 --force")
    if args.blend_mode == "frozen" and min(args.folds) == 0 and len(args.folds) == 1:
        raise SystemExit("frozen 模式下 fold 0 没有更早的折可拟合系数 —— Stage 1 请用 --blend-mode oracle")

    started = time.perf_counter()
    aux_index = [RESPONDER_COLUMNS.index(name) for name in AUX_LADDER]
    print(f"辅助目标（预注册，不得更改）：{list(AUX_LADDER)}，λ={AUX_LAMBDA}", flush=True)

    data = load_rows_with_responders(Path(args.data_root), args.sample_modulo, args.sampling)
    tid, aid = data["time_id"], data["asset_id"]
    folds = rolling_time_folds(np.unique(tid), args.n_folds, args.train_window, args.embargo)
    baseline = load_baseline(Path(args.baseline_cache))
    base_lookup = {int(key): index for index, key in enumerate(baseline["key"])}

    meta = json.loads(Path(args.candidate_meta).read_text(encoding="utf-8"))
    lgbm_global = np.array([int(name.split("_")[-1]) for name in meta["lgbm_features"]],
                           dtype=np.int64)
    history_global = lgbm_global[np.asarray(meta["history_positions"], dtype=np.int64)]
    from src.io import train_files
    lag_cache = build_lag_cache([Path(path) for path in train_files(Path(args.data_root))],
                                history_global, args.sample_modulo, int(meta["history_window"]),
                                sampling=args.sampling)
    if not (np.array_equal(lag_cache["time_id"], tid) and np.array_equal(lag_cache["asset_id"], aid)):
        raise SystemExit("history cache is not row-aligned")

    fold_rows: list[dict[str, Any]] = []
    history: dict[str, list[tuple[float, float]]] = {arm: [] for arm in ARMS}
    for fold_index in sorted(args.folds):
        fold_started = time.perf_counter()
        train_ids, valid_ids = folds[fold_index]
        tr, va = np.isin(tid, train_ids), np.isin(tid, valid_ids)

        keys_valid = (tid[va] << 8) | aid[va]
        matched = np.array([base_lookup.get(int(key), -1) for key in keys_valid], dtype=np.int64)
        keep = matched >= 0
        if keep.sum() < 0.9 * len(keep):
            raise SystemExit(f"fold {fold_index}: 只有 {keep.sum():,}/{len(keep):,} 行能与基准缓存连上")
        base_prediction = baseline["prediction"][matched[keep]]

        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        t_valid = transform_with(data["features"][va], stats)
        y_train = data["target"][tr].astype(np.float64)
        y_valid = data["target"][va].astype(np.float64)[keep]
        w_train = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
        w_valid = np.maximum(data["weight"][va].astype(np.float64), 0.0)[keep]
        tid_train, tid_valid = tid[tr], tid[va]
        aid_train, aid_valid = aid[tr], aid[va]

        starts_train = group_starts(tid_train)
        counts_train = np.diff(np.r_[starts_train, len(tid_train)])
        starts_valid = group_starts(tid_valid)
        counts_valid = np.diff(np.r_[starts_valid, len(tid_valid)])
        market_y_train = np.add.reduceat(y_train, starts_train) / counts_train
        market_weight_train = np.add.reduceat(w_train, starts_train)
        e_train = y_train - np.repeat(market_y_train, counts_train)

        market_selected = select_features(t_train, y_train, w_train, args.current_feature_count)
        cross_selected = select_features(t_train, e_train, np.ones_like(e_train),
                                         args.current_feature_count)
        market_train = group_mean(t_train[:, market_selected], starts_train, counts_train)
        market_valid = group_mean(t_valid[:, market_selected], starts_valid, counts_valid)
        cross_train = cross_sectional_deviation(t_train[:, cross_selected].copy(), tid_train)
        cross_valid = cross_sectional_deviation(t_valid[:, cross_selected].copy(), tid_valid)
        lo, hi, ce, sc = (stats[key][history_global] for key in ("lower", "upper", "center", "scale"))
        hist_train = history_blocks(lag_cache["lags"][tr], lag_cache["count"][tr],
                                    t_train[:, history_global], lo, hi, ce, sc)
        hist_valid = history_blocks(lag_cache["lags"][va], lag_cache["count"][va],
                                    t_valid[:, history_global], lo, hi, ce, sc)
        asset_train = np.eye(15, dtype=np.float32)[aid_train]
        asset_valid = np.eye(15, dtype=np.float32)[aid_valid]
        design_train = np.ascontiguousarray(np.column_stack([cross_train, *hist_train, asset_train]))
        design_valid = np.ascontiguousarray(np.column_stack([cross_valid, *hist_valid, asset_valid]))

        # responder 先补缺失、再做截面去均值（与 e 同一分解口径）。
        # 顺序不能反：组里一个 NaN 会让整个 time_id 组的去均值结果变成 NaN。
        responder_raw, missing_counts = impute_missing(
            data["responders"][tr][:, aux_index], np.ones(int(tr.sum())))
        responder_dev = cross_sectional_deviation(
            responder_raw.astype(np.float32), tid_train)
        if not np.all(np.isfinite(responder_dev)):
            raise SystemExit("responder 截面去均值后仍有非有限值")
        print(f"  fold {fold_index} 辅助目标缺失补齐："
              + "，".join(f"{name}={count:,}"
                          for name, count in zip(AUX_LADDER, missing_counts)), flush=True)

        # ---- market 头两个臂共用（同种子、同数据）⟹ 两臂的唯一差别就是 cross 头的损失 ----
        market_std_target, market_mean, market_scale = standardize(market_y_train,
                                                                  market_weight_train)
        market_est, _, market_parity = fit_mlp(market_train, market_std_target,
                                               market_weight_train, tuple(args.market_hidden),
                                               args, args.seed + fold_index)
        market_prediction = market_est.predict(market_valid) * market_scale + market_mean

        unit = np.ones_like(e_train)
        arms_payload: dict[str, Any] = {}
        base_score = scale_invariant_score(y_valid, base_prediction, w_valid)
        for arm in ARMS:
            if arm == "target_only":
                targets, e_mean, e_scale = standardize(e_train, unit)
            else:
                targets, e_mean, e_scale = build_multitask_targets(e_train, responder_dev,
                                                                   unit, AUX_LAMBDA)
            cross_est, _, cross_parity = fit_mlp(design_train, targets, unit,
                                                 tuple(args.cross_hidden), args,
                                                 args.seed + 100 + fold_index)
            raw = cross_est.predict(design_valid)
            e_prediction = (raw[:, 0] if raw.ndim == 2 else raw) * e_scale + e_mean
            e_prediction -= np.repeat(
                np.add.reduceat(e_prediction, starts_valid) / counts_valid, counts_valid)
            mlp_prediction = (np.repeat(market_prediction, counts_valid) + e_prediction)[keep]

            if args.blend_mode == "oracle":
                c1, c2 = blend_coefficients(y_valid, base_prediction, mlp_prediction, w_valid)
                coefficient_source = f"oracle（在 fold {fold_index} 自身上重解，**上界**）"
            else:
                if not history[arm]:
                    raise SystemExit(f"fold {fold_index}: frozen 模式需要更早的折先跑过")
                past = np.array(history[arm], dtype=np.float64)
                c1, c2 = float(past[:, 0].mean()), float(past[:, 1].mean())
                coefficient_source = f"frozen（fold 0..{fold_index - 1} 的均值）"
            blend = c1 * base_prediction + c2 * mlp_prediction
            blend_score = scale_invariant_score(y_valid, blend, w_valid)
            mlp_score = scale_invariant_score(y_valid, mlp_prediction, w_valid)
            # 交叉项与 A/B 同一分母（Σw·y²）⟹ 可直接喂闭式解，口径与
            # target_mlp_oracle_blend 一致
            denominator = float(np.dot(np.maximum(w_valid, 0.0) * y_valid, y_valid))
            cross_term = float(np.dot(np.maximum(w_valid, 0.0) * base_prediction,
                                      mlp_prediction) / denominator)
            oracle_peak = oracle_two_component_peak(base_score["A"], base_score["B"],
                                                    mlp_score["A"], mlp_score["B"], cross_term)
            # 自检：残差分解必须精确复现 oracle 增益（这是上面那条口径订正的依据）
            residual_signal = mlp_score["A"] - base_score["A"] * cross_term / base_score["B"]
            residual_energy = mlp_score["B"] - cross_term ** 2 / base_score["B"]
            decomposed = oracle_peak - base_score["peak"]
            if abs(residual_signal ** 2 / residual_energy - decomposed) > 1e-9 * max(
                    abs(decomposed), 1e-12) + 1e-18:
                raise SystemExit("残差分解与 oracle 增益不符 —— 机制口径有误")
            arms_payload[arm] = {
                "coefficients": [float(c1), float(c2)],
                "coefficient_source": coefficient_source,
                "mlp": mlp_score,
                "blend": blend_score,
                "oracle_upper_bound_peak": float(oracle_peak),
                # 尺度不变的机制分解：Δpeak = 残差信号²/残差能量
                # （分子 = m 中 b 解释不掉的信号，分母 = m 中 b 解释不掉的能量）
                "residual_signal": float(mlp_score["A"]
                                         - base_score["A"] * cross_term / base_score["B"]),
                "residual_energy": float(mlp_score["B"] - cross_term ** 2 / base_score["B"]),
                # MLP 在混合里占多大幅度：|c2|·√B_m 相对 |c1|·√B_b
                "blend_weight_share": float(
                    abs(c2) * np.sqrt(mlp_score["B"])
                    / max(abs(c1) * np.sqrt(base_score["B"]) + abs(c2) * np.sqrt(mlp_score["B"]),
                          1e-300)),
                "mlp_relative_to_baseline": float(mlp_score["peak"] / base_score["peak"]),
                "correlation_with_baseline": float(np.corrcoef(base_prediction, mlp_prediction)[0, 1]),
                "numpy_parity": cross_parity,
            }
            # oracle 系数存起来供后续折的 frozen 模式使用
            oc1, oc2 = blend_coefficients(y_valid, base_prediction, mlp_prediction, w_valid)
            history[arm].append((float(oc1), float(oc2)))
            del cross_est
            gc.collect()

        fold_rows.append({
            "fold": fold_index,
            "train_rows": int(tr.sum()), "valid_rows": int(keep.sum()),
            "dropped_unmatched_rows": int((~keep).sum()),
            "design_columns": int(design_train.shape[1]),
            "market_numpy_parity": market_parity,
            "auxiliary_missing_imputed": dict(zip(AUX_LADDER, missing_counts)),
            "baseline": base_score,
            "arms": arms_payload,
            "elapsed_seconds": time.perf_counter() - fold_started,
        })
        print(f"fold {fold_index}: base={base_score['peak']:.8f}  "
              + "  ".join(f"{arm}={arms_payload[arm]['blend']['peak']:.8f}"
                          f"({arms_payload[arm]['blend']['peak'] / base_score['peak'] - 1:+.2%})"
                          for arm in ARMS)
              + f"  ({fold_rows[-1]['elapsed_seconds']:.0f}s)", flush=True)
        del t_train, t_valid, design_train, design_valid, hist_train, hist_valid, responder_dev
        del market_est
        gc.collect()

    summary = summarise(fold_rows, ARMS)
    payload = {
        "experiment": "multitask_mlp",
        "question": "共享 trunk 的 responder 辅助损失，能否让 MLP 分量补上生产 v3 的 target 残差？",
        "pre_registration": {
            "aux_lambda": AUX_LAMBDA, "aux_ladder": list(AUX_LADDER),
            "lambda_note": "√λ 缩放辅助目标列 ⟺ 该头损失权重 λ（alpha=0 时严格）",
            "control_arm": "target_only（同架构/同种子/同迭代，仅去掉辅助头）",
            "stage1_gates": ["delta_positive", "relative_gain_at_least_3pct", "beats_target_only"],
            "gate_correction_20260819": (
                "原第二道写的是 2ΔA>ΔB；配比被重解时 A/B 只反映整体缩放，该判据不成立。"
                "已换成尺度不变的残差分解 + 本就该有的 3% 幅度门槛（收紧，非放松）。"),
            "stop_rule": "任一门槛不过即停；不调 λ / hidden / 激活 / 辅助目标集",
        },
        "configuration": vars(args),
        "blend_mode_caveat": (
            "oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 "
            "−2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）"),
        "folds": fold_rows, "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")

    lines = [f"# 多任务辅助监督 MLP（`{args.label}`）", "",
             f"λ={AUX_LAMBDA}（预注册，唯一超参）；辅助目标 {list(AUX_LADDER)}；"
             f"配比口径 **{args.blend_mode}**", "",
             "| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |",
             "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in fold_rows:
        for arm in ARMS:
            block = row["arms"][arm]
            lines.append(
                f"| {row['fold']} | {row['baseline']['peak']:.8f} | `{arm}` | "
                f"{block['mlp_relative_to_baseline']:.1%} | "
                f"{block['blend']['peak']:.8f} | "
                f"{block['blend']['peak'] / row['baseline']['peak'] - 1:+.4%} | "
                f"{block['blend_weight_share']:.1%} | "
                f"{block['correlation_with_baseline']:.3f} |")
    lines += ["", "## 汇总", "",
              "| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        block = summary[arm]
        lines.append(f"| `{arm}` | {block['mean_delta']:+.3e} | {block['relative_gain']:+.4%} | "
                     f"{block['positive_folds']}/{len(fold_rows)} | "
                     f"{block['drop_best_mean_delta']:+.3e} | "
                     f"{block['mean_residual_signal']:+.3e} | "
                     f"{block['mean_residual_energy']:.3e} |")
    if "stage1_pass" in summary:
        lines += ["", "## Stage 1 门禁（跑前写死）", ""]
        for name, ok in summary["stage1_gates"].items():
            lines.append(f"- {'✅' if ok else '❌'} `{name}`")
        lines += ["", f"**判定：{'过 —— 可进 Stage 2 冻结系数五折终审' if summary['stage1_pass'] else '不过 —— 停，不调参'}**"]
    lines += ["", "⚠️ " + payload["blend_mode_caveat"], ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {json_path}\nwrote {md_path}")
    if "stage1_pass" in summary:
        print(json.dumps(summary["stage1_gates"], ensure_ascii=False, indent=2))
        print("Stage 1:", "PASS" if summary["stage1_pass"] else "FAIL")


if __name__ == "__main__":
    main()
