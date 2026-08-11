"""A0：`responder_*` 换训练目标 —— 47 列全扫 + 两层折判决 + multi 臂。

## 要验证的假设

存在某个（或某几个）`responder_*`，**用它当训练目标**训出的模型，
去预测 **`target`** 时的样本外 `peak` 高于直接用 `target` 训练的模型。

## 为什么合法

测试端拿不到 responder ⟹ **不能当特征**。但 `模型(feature) → responder` 在推理端完全合法，
所以可以换训练目标。代码层面已确认：`timeseries_api/runner.py:79` 把 `responder_*`
从可见列里滤掉；`docs/data_description.md:175` 说私榜也不提供。

## 为什么可能更好

`target` 信噪比极低（本地 R² 约 0.0017）。若某个 responder 是**同一底层信号的低噪声版本**
（`responder_01~04` 是 [0,1] 概率/指示类），拟合时就不会被厚尾离群点拖着走。
这是低信噪比金融 ML 的标准手法，也符合「到 0.0045 需要一个**新的信号来源**」这个判断。

⭐ **选哪个不看它与 target 的相关性** —— 8/10 粗筛里 `responder_03` 相关 0.82 却 −9.6%，
`responder_04` 相关 0.41 却是唯一为正。要看「可预测性 × 对齐度」，只能实测。

## ⚠️ 8/10 晚那次 3 折粗筛不作数

ROADMAP A0 专条列了三个缺陷。其中**第 1 条（alpha 不公平）经核实并不成立**，见下一节的更正；
真正站得住的只有「只测了 6 个」「只有 3 折」。
而且**那次的脚本与产物在仓库里没有任何留存**，
`grep` 只在 ROADMAP/NOTES 命中 ⟹ 数字不可独立复现。本脚本从头重做。

## ⚠️⚠️ 更正：ROADMAP 的 `alpha_eff` 规则是错的，本脚本**不采用**

ROADMAP A0 专条与 8/11 计划书都写着：「alpha 按 target 方差标定，而 `responder_04` 的
std 只有 0.29（target 1.09），岭回归里目标缩放 c 倍、等效正则要按 c² 缩放才不变
⟹ 给它的正则**强了约 14 倍**」，并据此把 8/10 那次粗筛判为「测得不对」。

**这条推理不成立。** 正确的缩放律是

    R(y/s, α) = R(y, α) / s          （**不是** R(y, α·s²)/s）

推导：`J̃(b) = Σw(y/s − Xb)² + α‖b‖²`，令 `b = u/s` 得
`J̃ = (1/s²)·[Σw(y − Xu)² + α‖u‖²]` —— 数据项与惩罚项**同步**缩放 `s²`，
提取公因子后 α 原封不动。即**岭回归对目标缩放是严格等变的**：
目标缩放 s 倍，系数就缩放 1/s 倍，方向完全不变。

而 `peak = A²/B` 对预测的缩放**不变** ⟹ **目标的量纲对 peak 没有任何影响**，
一个固定的 α 对 std 0.29 和 std 1.09 的目标是同样紧的。

数值验证（`--check-solver` 里跑，也在 60 列合成数据上验过）：
同一 α 下「缩放目标」与「缩放系数」相对差 **1.4e-15**（机器精度）；
而按 `alpha_eff` 写法调整 α，系数相对差高达 **65%** —— 那条规则不是在修公平性，
**是在制造不公平**（给每个候选不同的有效正则）。

⟹ 本脚本的做法：所有候选走**同一条 alpha 阶梯，α 不随目标方差调整**。
标准化只为数值整洁（并顺带保证中心化），对 peak 是恒等变换。

⟹ 顺带结论：8/10 那次粗筛的**第 1 条缺陷（alpha 不公平）不存在**。
它真正的问题只剩「只测了 6 个」「只有 3 折」，以及那次产物在仓库里没有留存。

⚠️ **中心化不是可选项**：`peak` 对预测的缩放不变，但**对平移不变不了**
（`A = Σwyf/Σwy²`、`B = Σwf²/Σwy²` 都会变）。[0,1] 型 responder 的常数偏移
（均值 ≈ 0.5）若进了预测，`B` 直接爆炸、A/B/peak 全错 —— 这是 8/10 已经踩过的坑 1。
本脚本对 **X 和 y 都做加权中心化**，于是加权最小二乘的截距**恒为 0**，从根上避免。

## 判据（预注册，由 `verdict()` 机器判 —— 伤疤清单 #2）

因为有 **47 选 1** 的选择偏置，比常规 A 段判据严，四条全过才 PASS：

1. 逐折配对 Δ(peak) 均值为正
2. 去掉最好一折仍为正
3. **整条 alpha 阶梯为正**（只在一个 alpha 上赢不算 —— `market_model` 的教训）
4. 点估计换算总分 **≥ +5%**

## 防选择偏置的两层折

- 阶段 1 用**最早一折**单折扫全部 47 列，**该折之后不参与判决**
- 阶段 2 外层用 `rolling_time_folds`（排除阶段 1 那折）；
  **候选只在内层选**（训练段内部再切一刀），外层只评被内层选中的那一个

## 求解器

全程用**加权中心化 + Cholesky** 解正规方程（`(G+αI)b = rhs`），而不是 sklearn 的 lsqr：
G 只算一次、48 个右端项几乎免费，阶段 1 才跑得动。
`--check-solver` 会在一折上与 `sklearn.Ridge(solver="lsqr", tol=1e-8)` 对拍。
门槛取 **peak 相对差 < 1e-4**：lsqr 是迭代近似（严格档实测系数漂移约 4.5e-07，
NOTES 记过），实测两者 peak 相对差 5.6e-06；而本项目的噪声底
（共享折划分、各自拟合）是 peak 的约 2.9e-02 相对量 —— 5.6e-06 比它低约 5000 倍，
完全不影响任何结论。1e-8 那个门槛是对迭代求解器的误用。

⚠️ `np.nan_to_num` 默认把 inf 映射成 1.8e308 —— 全程用
`np.where(np.isfinite(x), x, ...)`，不用默认参数（8/10 踩过的坑 2）。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/responder_targets.py --stage 1
    # 把阶段 1 的前 5 名写进 SHORTLIST 常量，然后
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/responder_targets.py --stage 2
输出：outputs/experiments/responder_targets_stage1.{json,md}、responder_targets.{json,md}
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
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.validation import rolling_time_folds
# 复用已有实现，不另写一份（口径唯一性）
from features import apply_robust_transform, cross_sectional_deviation
from train import robust_transform_fit, select_features
from market_model import sign_test_p
from ridge_data_ladder import row_level_peak
from walk_forward_rolling import PROD_SAMPLED_WINDOW

# ---- 预注册常量（不搜） -------------------------------------------------------
N_RESPONDERS = 47
RESPONDER_COLUMNS = [f"responder_{i:02d}" for i in range(N_RESPONDERS)]
FEATURE_COUNT = 200
TRAIN_WINDOW = 39_480          # = ridge_data_ladder.BASELINE_WINDOW，生产等效
ALPHAS = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
SHORTLIST_SIZE = 5

# ⚠️ 预注册：阶段 1（2026-08-11）单折排名的前 5 名，写死在这里，**不许事后改**。
# 来源 outputs/experiments/responder_targets_stage1.json，其 sha256 记在下面；
# 阶段 2 的判决只对这 5 个 + target 基准进行。
# 单折 peak 相对 target：+15.80% / +11.34% / +7.68% / +5.57% / −3.24%
# （responder_06 是负的也照样入选 —— 名额按预注册的「前 5」取，不按结果挑）
SHORTLIST: tuple[str, ...] | None = (
    "responder_04", "responder_28", "responder_05", "responder_29", "responder_06",
)
SHORTLIST_SOURCE_SHA256: str | None = (
    "0fcff71f2f0466360ed3ee02ca70a863822e67af6159e4a66a9552e391b37e7e"
)

BASELINE_ARM = "target"
PROJECTION_ARM = "projection"   # target 投影到 47 个 responder 空间上的「去噪 target」


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A0: responder_* as training targets")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="responder_targets")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--train-window", type=int, default=TRAIN_WINDOW)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=10)
    p.add_argument("--sampling", default="periodic", choices=["periodic", "phase_balanced"])
    p.add_argument("--feature-count", type=int, default=FEATURE_COUNT)
    p.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    p.add_argument("--inner-fraction", type=float, default=0.25,
                   help="训练段末尾留作内层选择的比例（外层验证段绝不参与）")
    p.add_argument("--with-projection", action="store_true",
                   help="加一个 projection 臂：把 target 投影到 47 个 responder 张成的空间上"
                        "当训练标签（保留 target 作锚点，只削掉 responder 解释不了的部分）。"
                        "这是 A0 唯一没测过的形态 —— 先验低，做它是为了结案。")
    p.add_argument("--check-solver", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ---- 数据 ---------------------------------------------------------------------

def load_rows_with_responders(data_root: Path, sample_modulo: int, sampling: str) -> dict[str, np.ndarray]:
    """与 `lgbm_xs.load_rows` 同构，额外带 47 个 responder 列。

    `walk_forward_rolling.load_all_sampled` 不接受 columns 参数、`src.io.READ_COLUMNS`
    也不含 responder，所以这里自带一份。逐 batch 掩码、从不整体 materialize。
    """
    columns = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target", *RESPONDER_COLUMNS]
    keys = ("features", "responders", "target", "weight", "time_id", "asset_id")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for path in train_files(data_root):
        kept, started = 0, time.perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            mask = time_sample_mask(frame["time_id"].to_numpy(copy=False),
                                    sample_modulo, sampling=sampling)
            if not mask.any():
                continue
            parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            parts["responders"].append(frame.loc[mask, RESPONDER_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float64, copy=True))
            parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float64, copy=True))
            parts["time_id"].append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
            parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int64, copy=True))
            kept += int(mask.sum())
        print(f"  {path.name}: {kept:,} 行 ({time.perf_counter()-started:.0f}s)", flush=True)
    if not parts["features"]:
        raise SystemExit("采样为空")
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---- 加权统计 + 标准化 ---------------------------------------------------------

def weighted_moments(values: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """只用有限值算加权均值与标准差（responder 有极少量 NaN，≤0.03%）。"""
    finite = np.isfinite(values)
    w = np.where(finite, weight, 0.0)
    total = float(w.sum())
    if total <= 0:
        return 0.0, 0.0
    v = np.where(finite, values, 0.0)
    mean = float(np.dot(w, v) / total)
    var = float(np.dot(w, (v - mean) ** 2) / total)
    return mean, float(np.sqrt(max(var, 0.0)))


def standardize_target(values: np.ndarray, weight: np.ndarray,
                       mean: float, std: float) -> np.ndarray:
    """(y − 加权均值)/加权标准差；非有限值填成均值（中心化后即 0，贡献为 0）。

    ⚠️ 不用 `np.nan_to_num` 的默认参数（它把 inf 映射成 1.8e308）。
    """
    v = np.where(np.isfinite(values), values, mean).astype(np.float64)
    return (v - mean) / max(std, 1e-30)


# ---- 加权中心化 + Cholesky 岭回归 ---------------------------------------------

def weighted_center(design: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(weight.sum())
    mean = (design.astype(np.float64).T @ weight) / total
    return mean, total


def gram_and_rhs(design: np.ndarray, weight: np.ndarray, mean: np.ndarray,
                 targets: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """分块累加 `G = X̃ᵀWX̃`（p×p）与 `rhs = X̃ᵀWỸ`（p×k）。

    X 和 y 都已按加权均值中心化 ⟹ 加权最小二乘的截距恒为 0，
    [0,1] 型 responder 的常数偏移进不了预测（坑 1）。
    """
    p = design.shape[1]
    k = targets.shape[1]
    gram = np.zeros((p, p), dtype=np.float64)
    rhs = np.zeros((p, k), dtype=np.float64)
    for start in range(0, len(design), chunk):
        stop = min(start + chunk, len(design))
        xc = design[start:stop].astype(np.float64) - mean
        wc = weight[start:stop]
        xw = xc * wc[:, None]
        gram += xc.T @ xw
        rhs += xw.T @ targets[start:stop]
        del xc, wc, xw
    return gram, rhs


def solve_ridge(gram: np.ndarray, rhs: np.ndarray, alpha: float) -> np.ndarray:
    p = gram.shape[0]
    return np.linalg.solve(gram + alpha * np.eye(p), rhs)


def build_design(t_features: np.ndarray, time_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """`raw_dev` 基底，与 `train.make_design` 逐位同构。"""
    raw = t_features[:, selected].copy()
    deviation = cross_sectional_deviation(raw, time_ids)
    return np.column_stack([raw, deviation]).astype(np.float32, copy=False)


def ladder_for(fold_alpha: float) -> dict[str, float]:
    """整条阶梯 + 一档「生产等效」。

    ⚠️ α **不随目标方差调整**。岭回归对目标缩放严格等变（模块 docstring 的更正一节），
    `fold_alpha` 对 std 0.29 和 std 1.09 的目标一样紧，标准化不改变任何一档的含义。
    """
    return {"production_equivalent": float(fold_alpha),
            **{f"{a:.0e}": float(a) for a in ALPHAS}}


# ---- 阶段 1 -------------------------------------------------------------------

def run_stage1(data, folds, all_time_ids, args) -> dict[str, object]:
    """最早一折，冻结「按 target 选的 200 列」设计矩阵，48 个臂共用 Gram。"""
    train_ids, valid_ids = folds[0]
    tr = np.isin(all_time_ids, train_ids)
    va = np.isin(all_time_ids, valid_ids)
    fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW

    t_train, stats = robust_transform_fit(data["features"][tr].copy())
    w_tr = np.maximum(data["weight"][tr], 0.0)
    w_va = np.maximum(data["weight"][va], 0.0)
    y_tr_target, y_va_target = data["target"][tr], data["target"][va]
    selected = select_features(t_train, y_tr_target, w_tr, args.feature_count)

    t_valid = data["features"][va].copy()
    apply_robust_transform(t_valid, stats["lower"], stats["upper"], stats["center"], stats["scale"])
    d_tr = build_design(t_train, data["time_id"][tr], selected)
    d_va = build_design(t_valid, data["time_id"][va], selected)
    del t_train, t_valid
    gc.collect()

    names = [BASELINE_ARM, *RESPONDER_COLUMNS]
    resp_tr = data["responders"][tr]          # 一次性切出来；逐列 fancy-index 会复制 47 遍
    raw_targets = [y_tr_target] + [resp_tr[:, i] for i in range(N_RESPONDERS)]
    moments, cols, kept_names, skipped = [], [], [], {}
    for name, values in zip(names, raw_targets):
        mean, std = weighted_moments(values, w_tr)
        if std <= 0:
            skipped[name] = "训练段内加权标准差为 0"
            continue
        moments.append({"name": name, "weighted_mean": mean, "weighted_std": std})
        cols.append(standardize_target(values, w_tr, mean, std))
        kept_names.append(name)
    targets = np.column_stack(cols)
    del cols, raw_targets, resp_tr
    gc.collect()

    print(f"阶段 1：{len(kept_names)} 个臂，设计 {d_tr.shape}，验证 {d_va.shape}", flush=True)
    started = time.perf_counter()
    mean_x, _ = weighted_center(d_tr, w_tr)
    gram, rhs = gram_and_rhs(d_tr, w_tr, mean_x, targets)
    print(f"  Gram {gram.shape} 建好 ({time.perf_counter()-started:.0f}s)", flush=True)

    xv = d_va.astype(np.float64) - mean_x
    ladder = ladder_for(fold_alpha)
    results: dict[str, dict[str, dict[str, float]]] = {n: {} for n in kept_names}
    for key, alpha in ladder.items():
        beta = solve_ridge(gram, rhs, alpha)
        preds = xv @ beta
        for j, name in enumerate(kept_names):
            results[name][key] = row_level_peak(y_va_target, preds[:, j], w_va)
        del beta, preds
    print(f"  阶梯扫完 ({time.perf_counter()-started:.0f}s)", flush=True)

    base = results[BASELINE_ARM]["production_equivalent"]["peak"]
    ranking = sorted(
        ({"name": n,
          "peak": results[n]["production_equivalent"]["peak"],
          "relative_to_target": results[n]["production_equivalent"]["peak"] / base - 1.0,
          "peak_best_alpha": max(v["peak"] for v in results[n].values()),
          "positive_rungs": int(sum(results[n][k]["peak"] > results[BASELINE_ARM][k]["peak"]
                                    for k in ladder)),
          } for n in kept_names if n != BASELINE_ARM),
        key=lambda r: -r["peak"])

    return {
        "fold": {"train_time_ids": int(len(train_ids)), "valid_time_ids": int(len(valid_ids)),
                 "n_train_rows": int(tr.sum()), "n_valid_rows": int(va.sum()),
                 "fold_alpha": float(fold_alpha)},
        "design": {"columns": int(d_tr.shape[1]),
                   "note": "冻结「按 target 选的 200 列」—— 48 臂共用 Gram，仅用于剪枝排序"},
        "alpha_ladder": {k: float(v) for k, v in ladder.items()},
        "target_moments": moments,
        "skipped": skipped,
        "baseline_peak": float(base),
        "per_arm": results,
        "ranking": ranking,
        "shortlist_proposal": [r["name"] for r in ranking[:SHORTLIST_SIZE]],
    }


# ---- 阶段 2 + 3 ---------------------------------------------------------------

def fit_betas(t_features_tr, tid_tr, w_tr, y_std_tr, feature_count,
              alphas: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """端到端换目标：**特征选择也按候选目标做**（这才是真会上线的形态）。

    只出系数，不出预测 —— 这样同一个拟合好的模型可以被用到多个验证块上，
    multi 臂才能用「与标定 γ 时同一批模型」去预测外层（见 run_stage2 的说明）。
    """
    selected = select_features(t_features_tr, y_std_tr, w_tr, feature_count)
    d_tr = build_design(t_features_tr, tid_tr, selected)
    mean_x, _ = weighted_center(d_tr, w_tr)
    gram, rhs = gram_and_rhs(d_tr, w_tr, mean_x, y_std_tr[:, None])
    betas = {key: solve_ridge(gram, rhs, alpha)[:, 0] for key, alpha in alphas.items()}
    del d_tr, gram, rhs
    gc.collect()
    return selected, mean_x, betas


def apply_betas(t_features, tid, selected, mean_x, betas) -> dict[str, np.ndarray]:
    xv = build_design(t_features, tid, selected).astype(np.float64) - mean_x
    out = {key: xv @ beta for key, beta in betas.items()}
    del xv
    gc.collect()
    return out


def fit_and_predict(t_features_tr, tid_tr, w_tr, y_std_tr, t_features_va, tid_va,
                    feature_count, alphas: dict[str, float]) -> dict[str, np.ndarray]:
    selected, mean_x, betas = fit_betas(t_features_tr, tid_tr, w_tr, y_std_tr,
                                        feature_count, alphas)
    return apply_betas(t_features_va, tid_va, selected, mean_x, betas)


def run_stage2(data, folds, all_time_ids, args, shortlist) -> dict[str, object]:
    arms = [BASELINE_ARM, *shortlist]
    if args.with_projection:
        arms.append(PROJECTION_ARM)
    fold_rows: list[dict[str, object]] = []

    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        tr = np.isin(all_time_ids, train_ids)
        va = np.isin(all_time_ids, valid_ids)
        fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW

        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        t_valid = data["features"][va].copy()
        apply_robust_transform(t_valid, stats["lower"], stats["upper"],
                               stats["center"], stats["scale"])
        tid_tr, tid_va = data["time_id"][tr], data["time_id"][va]
        w_tr, w_va = np.maximum(data["weight"][tr], 0.0), np.maximum(data["weight"][va], 0.0)
        y_tr_t, y_va_t = data["target"][tr], data["target"][va]
        ladder = ladder_for(fold_alpha)
        resp_tr = data["responders"][tr]

        # 内层：训练段末尾切一刀（留 embargo），候选**只在这里**选
        inner_cut = int(len(train_ids) * (1.0 - args.inner_fraction))
        inner_tr_ids = train_ids[:max(1, inner_cut - args.embargo)]
        inner_va_ids = train_ids[inner_cut:]
        i_tr = np.isin(tid_tr, inner_tr_ids)
        i_va = np.isin(tid_tr, inner_va_ids)

        def projected_target(mask):
            """target 在 47 个 responder 张成空间上的加权最小二乘投影（「去噪 target」）。

            β 只用**这一段训练行**的 responder 与 target 拟合；产出的标签只是训练标签的
            一个变换，**推理端完全不需要 responder** ⟹ 合法。
            R 与 y 都加权中心化 ⟹ 截距恒 0，返回值已是零加权均值。
            ⚠️ NaN 用 `np.where(np.isfinite(...))`，不用 `np.nan_to_num` 默认参数。
            """
            R = np.where(np.isfinite(resp_tr[mask]), resp_tr[mask], 0.0).astype(np.float64)
            y, wt = y_tr_t[mask], w_tr[mask]
            total = float(wt.sum())
            Rc = R - (R.T @ wt) / total
            yc = y - float(wt @ y) / total
            G = (Rc * wt[:, None]).T @ Rc
            rhs = (Rc * wt[:, None]).T @ yc
            beta = np.linalg.solve(G + 1e-10 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), rhs)
            return Rc @ beta

        def raw_target(name, mask):
            if name == BASELINE_ARM:
                return y_tr_t[mask]
            if name == PROJECTION_ARM:
                return projected_target(mask)
            return resp_tr[mask, RESPONDER_COLUMNS.index(name)]

        # --- 内层：为每个候选出 OOF 预测（既用于选，也用于 multi 的系数）
        #     整条阶梯都算 —— multi 的组合系数必须与它将要作用的那一档同档标定
        #
        # ⚠️ multi 臂用**同一批内层模型**去预测外层验证段，而不是用全训练段重训的模型。
        # 理由：γ 是在内层模型的预测尺度上标定的；重训后训练行数变了、收缩程度随之改变，
        # 预测尺度也就变了，γ 直接搬过去会失配（第一版就是这么写的，
        # 结果 multi 在 5 折上从 −41% 摆到 +41%，那是实现缺陷不是结论）。
        # 而按外层验证段的统计量去归一化会用到验证信息 = 泄漏，所以不能那样修。
        inner_pred: dict[str, dict[str, np.ndarray]] = {}
        inner_outer_pred: dict[str, dict[str, np.ndarray]] = {}
        inner_peak: dict[str, float] = {}
        for name in arms:
            values = raw_target(name, i_tr)
            mean, std = weighted_moments(values, w_tr[i_tr])
            if std <= 0:
                continue
            y_std = standardize_target(values, w_tr[i_tr], mean, std)
            selected, mean_x, betas = fit_betas(t_train[i_tr], tid_tr[i_tr], w_tr[i_tr],
                                                y_std, args.feature_count, ladder)
            inner_pred[name] = apply_betas(t_train[i_va], tid_tr[i_va], selected, mean_x, betas)
            inner_outer_pred[name] = apply_betas(t_valid, tid_va, selected, mean_x, betas)
            inner_peak[name] = row_level_peak(
                y_tr_t[i_va], inner_pred[name]["production_equivalent"], w_tr[i_va])["peak"]
        picked = max(inner_peak, key=inner_peak.get)

        # --- multi 臂：组合系数在**训练段内**用 OOF 解，绝不碰外层验证段
        stack_names = [n for n in arms if n in inner_pred]
        wi = w_tr[i_va]
        gammas: dict[str, np.ndarray] = {}
        for key in ladder:
            stack = np.column_stack([inner_pred[n][key] for n in stack_names])
            g = (stack * wi[:, None]).T @ stack
            b = (stack * wi[:, None]).T @ y_tr_t[i_va]
            # 6 个高度相关的预测做无正则 LS 会病态，给一点点对角加载稳住条件数
            gammas[key] = np.linalg.solve(
                g + 1e-8 * np.trace(g) / len(g) * np.eye(len(g)), b)

        # --- 外层：在整个训练段上重训，只评被内层选中的那个 + 基准 + multi
        outer: dict[str, dict[str, dict[str, float]]] = {}
        outer_pred_by_alpha: dict[str, dict[str, np.ndarray]] = {}
        for name in sorted(set([BASELINE_ARM, picked, *stack_names])):
            values = raw_target(name, slice(None))
            mean, std = weighted_moments(values, w_tr)
            y_std = standardize_target(values, w_tr, mean, std)
            preds = fit_and_predict(t_train, tid_tr, w_tr, y_std, t_valid, tid_va,
                                    args.feature_count, ladder)
            outer_pred_by_alpha[name] = preds
            outer[name] = {k: row_level_peak(y_va_t, v, w_va) for k, v in preds.items()}

        # γ 与 inner_outer_pred 出自同一批模型 ⟹ 尺度一致，系数可以原样搬过去
        multi = {}
        for key in ladder:
            combined = sum(gammas[key][j] * inner_outer_pred[n][key]
                           for j, n in enumerate(stack_names))
            multi[key] = row_level_peak(y_va_t, combined, w_va)

        row = {
            "fold": index, "n_train_rows": int(tr.sum()), "n_valid_rows": int(va.sum()),
            "fold_alpha": float(fold_alpha),
            "inner_peaks": inner_peak, "picked_by_inner": picked,
            "multi_weights": {n: float(gammas["production_equivalent"][j])
                              for j, n in enumerate(stack_names)},
            "outer": outer, "multi": multi,
        }
        b_peak = outer[BASELINE_ARM]["production_equivalent"]["peak"]
        p_peak = outer[picked]["production_equivalent"]["peak"]
        m_peak = multi["production_equivalent"]["peak"]
        print(f"  fold {index}: 内层选中 {picked} | baseline {b_peak:.8f} → "
              f"picked {p_peak:.8f} ({(p_peak/b_peak-1)*100:+.2f}%) | "
              f"multi {m_peak:.8f} ({(m_peak/b_peak-1)*100:+.2f}%) "
              f"[{time.perf_counter()-started:.0f}s]", flush=True)
        fold_rows.append(row)
        del t_train, t_valid, resp_tr
        gc.collect()
    return {"arms": arms, "folds": fold_rows}


# ---- 判据 ---------------------------------------------------------------------

def paired_stats(deltas: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    positive = int((deltas > 0).sum())
    without_best = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
    base = float(baseline.mean())
    ratio = (lambda d: float(d / base)) if base > 0 else (lambda d: float("nan"))
    return {
        "mean_delta": float(deltas.mean()), "relative_gain": ratio(deltas.mean()),
        "positive_folds": positive, "n_folds": int(len(deltas)),
        "sign_test_p": sign_test_p(positive, len(deltas)),
        "mean_delta_drop_best": float(without_best.mean()),
        "relative_gain_drop_best": ratio(without_best.mean()),
        "per_fold": [float(v) for v in deltas],
    }


def verdict(stats: dict[str, object], ladder_all_positive: bool) -> dict[str, object]:
    """四条判据由代码判 —— 报告里的文字不得与这里不一致（伤疤清单 #2）。

    比常规 A 段判据严，因为有 47 选 1 的选择偏置。
    """
    checks = {
        "1_paired_delta_positive": stats["mean_delta"] > 0,
        "2_survives_drop_best_fold": stats["mean_delta_drop_best"] > 0,
        "3_positive_across_alpha_ladder": bool(ladder_all_positive),
        "4_relative_gain_at_least_5pct": stats["relative_gain"] >= 0.05,
    }
    return {"checks": checks, "pass": all(checks.values())}


def check_solver(data, folds, all_time_ids, args) -> dict[str, float]:
    """两条必须成立的等式，跑真数据验一次。

    (a) 中心化 + Cholesky ≡ `sklearn.Ridge(solver="lsqr", tol=1e-8, fit_intercept=True)`
        —— Cholesky 是正规方程的精确解，lsqr 是它的迭代近似。
    (b) 「标准化目标 + α」≡「只中心化的原始目标 + **同一个** α」 —— 由等变律
        `R(y/s, α) = R(y, α)/s` 与 peak 的尺度不变性推出，两者 peak 必须相等。
        同时把 ROADMAP 的 `alpha_eff` 写法（α·s²）一并算出来作对照，
        用数字说明那条规则是**引入**差异而不是消除差异。
    """
    from sklearn.linear_model import Ridge

    train_ids, valid_ids = folds[0]
    tr, va = np.isin(all_time_ids, train_ids), np.isin(all_time_ids, valid_ids)
    fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW
    t_train, stats = robust_transform_fit(data["features"][tr].copy())
    t_valid = data["features"][va].copy()
    apply_robust_transform(t_valid, stats["lower"], stats["upper"], stats["center"], stats["scale"])
    w_tr, w_va = np.maximum(data["weight"][tr], 0.0), np.maximum(data["weight"][va], 0.0)
    y_tr, y_va = data["target"][tr], data["target"][va]
    selected = select_features(t_train, y_tr, w_tr, args.feature_count)
    d_tr = build_design(t_train, data["time_id"][tr], selected)
    d_va = build_design(t_valid, data["time_id"][va], selected)

    mean, std = weighted_moments(y_tr, w_tr)
    y_std = standardize_target(y_tr, w_tr, mean, std)
    mean_x, _ = weighted_center(d_tr, w_tr)
    gram, rhs = gram_and_rhs(d_tr, w_tr, mean_x, y_std[:, None])
    xv = d_va.astype(np.float64) - mean_x
    peak_chol = row_level_peak(y_va, (xv @ solve_ridge(gram, rhs, fold_alpha))[:, 0], w_va)["peak"]

    est = Ridge(alpha=fold_alpha, solver="lsqr", tol=1e-8, max_iter=2000, fit_intercept=True)
    est.fit(d_tr, y_std, sample_weight=w_tr)
    peak_lsqr = row_level_peak(y_va, est.intercept_ + d_va @ est.coef_, w_va)["peak"]

    # (b) 只中心化、不缩放的原始目标，配**同一个** α —— 等变律说 peak 必须相同
    y_centered = np.where(np.isfinite(y_tr), y_tr, mean) - mean
    _, rhs_raw = gram_and_rhs(d_tr, w_tr, mean_x, y_centered[:, None])
    peak_same_alpha = row_level_peak(
        y_va, (xv @ solve_ridge(gram, rhs_raw, fold_alpha))[:, 0], w_va)["peak"]
    # 对照：ROADMAP 的 alpha_eff 写法（α·s²）—— 它**不该**相等，差多少就是那条规则的伤害
    peak_alpha_eff_rule = row_level_peak(
        y_va, (xv @ solve_ridge(gram, rhs_raw, fold_alpha * std * std))[:, 0], w_va)["peak"]

    return {
        "peak_cholesky": peak_chol, "peak_sklearn_lsqr": peak_lsqr,
        "relative_difference_solver": abs(peak_chol - peak_lsqr) / peak_chol,
        "peak_standardized": peak_chol, "peak_raw_same_alpha": peak_same_alpha,
        "relative_difference_equivariance": abs(peak_chol - peak_same_alpha) / peak_chol,
        "peak_roadmap_alpha_eff_rule": peak_alpha_eff_rule,
        "relative_difference_alpha_eff_rule": abs(peak_chol - peak_alpha_eff_rule) / peak_chol,
        "note": "等变律要求 relative_difference_equivariance ≈ 0；"
                "alpha_eff 那条规则的偏离量就是它引入的失真",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.label}_stage1" if args.stage == 1 else args.label
    report_path = output_dir / f"{label}.md"
    if report_path.exists() and not args.force:
        raise SystemExit(f"{report_path} 已存在；要覆盖请加 --force")

    print("loading partitions (features + 47 responders)...", flush=True)
    data = load_rows_with_responders(Path(args.data_root), args.sample_modulo, args.sampling)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)
    print(f"{len(all_time_ids):,} 行，{len(unique_time_ids):,} 个采样 time_id，{len(folds)} 折",
          flush=True)

    solver_checks = None
    if args.check_solver:
        print("checking solver + alpha rule ...", flush=True)
        solver_checks = check_solver(data, folds, all_time_ids, args)
        print(json.dumps(solver_checks, indent=1), flush=True)
        # lsqr 是迭代近似，门槛按它的实际精度取（见模块 docstring「求解器」一节）
        if solver_checks["relative_difference_solver"] > 1e-4:
            raise SystemExit("❌ Cholesky 与 sklearn lsqr 对不上 —— 停下来查")
        if solver_checks["relative_difference_equivariance"] > 1e-8:
            raise SystemExit("❌ 目标缩放等变律不成立 —— 中心化/标准化实现有错，停下来查")
        print("✅ 求解器与等变律都对上了（同一次装载，继续跑阶段）", flush=True)
        gc.collect()

    if args.stage == 1:
        payload = {
            "question": "哪些 responder 值得进短名单？（单折剪枝，不作判决）",
            "design_note": "冻结特征选择 + 共用 Gram，只用于排序；阶段 2 才端到端换目标",
            "selection_bias_guard": "本折之后不参与阶段 2 的判决",
            "configuration": vars(args) | {"alpha_ladder": [float(a) for a in ALPHAS]},
            "solver_checks": solver_checks,
            "stage1": run_stage1(data, folds, all_time_ids, args),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        (output_dir / f"{label}.json").write_text(text, encoding="utf-8")
        report_path.write_text(render_stage1(payload), encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f"\nstage1 JSON sha256 = {digest}")
        print(f"短名单候选（前 {SHORTLIST_SIZE}）：{payload['stage1']['shortlist_proposal']}")
        print(f"报告：{report_path}")
        return

    if SHORTLIST is None:
        raise SystemExit("阶段 2 需要先把阶段 1 的前 5 名写进 SHORTLIST 常量（预注册）")
    outer = [f for i, f in enumerate(folds) if i != 0]      # 排除阶段 1 那折
    print(f"阶段 2：短名单 {list(SHORTLIST)}，外层 {len(outer)} 折", flush=True)
    results = run_stage2(data, outer, all_time_ids, args, list(SHORTLIST))
    payload = build_stage2_payload(args, results)
    (output_dir / f"{label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_stage2(payload), encoding="utf-8")
    print(json.dumps(payload["verdicts"], ensure_ascii=False, indent=2), flush=True)
    print(f"报告：{report_path}")


def build_stage2_payload(args, results) -> dict[str, object]:
    rows = results["folds"]
    keys = list(rows[0]["multi"].keys())

    def series(getter) -> np.ndarray:
        return np.asarray([getter(r) for r in rows], dtype=np.float64)

    base = series(lambda r: r["outer"][BASELINE_ARM]["production_equivalent"]["peak"])
    picked = series(lambda r: r["outer"][r["picked_by_inner"]]["production_equivalent"]["peak"])
    multi = series(lambda r: r["multi"]["production_equivalent"]["peak"])

    comparisons, verdicts = {}, {}
    for name, arr in (("inner_selected", picked), ("multi", multi)):
        ladder_ok = True
        ladder_detail = {}
        for key in keys:
            b = series(lambda r, k=key: r["outer"][BASELINE_ARM][k]["peak"])
            v = (series(lambda r, k=key: r["outer"][r["picked_by_inner"]][k]["peak"])
                 if name == "inner_selected" else series(lambda r, k=key: r["multi"][k]["peak"]))
            d = float((v - b).mean())
            ladder_detail[key] = {"mean_delta": d, "relative_gain": d / b.mean()}
            ladder_ok = ladder_ok and d > 0
        stats = paired_stats(arr - base, base)
        comparisons[name] = {"peak": stats, "alpha_ladder": ladder_detail,
                             "alpha_ladder_all_positive": ladder_ok,
                             "baseline_peak_mean": float(base.mean()),
                             "arm_peak_mean": float(arr.mean())}
        verdicts[name] = verdict(stats, ladder_ok)

    return {
        "question": "用 responder 当训练目标，预测 target 的样本外 peak 能不能超过用 target 训练？",
        "metric": "peak = A²/B（尺度无关），在 target 上评",
        "fairness": "每个候选目标在训练段内标准化成零加权均值/单位加权方差 ⟹ "
                    "ROADMAP 的 alpha_eff 规则的等价实现；X、y 均加权中心化 ⟹ 截距恒 0（坑 1）",
        "selection_bias_guard": "阶段 1 那折已排除；候选只在内层选，外层只评被选中的",
        "shortlist": list(SHORTLIST or ()),
        "shortlist_source_sha256": SHORTLIST_SOURCE_SHA256,
        "configuration": vars(args) | {"alpha_ladder": [float(a) for a in ALPHAS]},
        "results": results,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }


def render_stage1(payload) -> str:
    s = payload["stage1"]
    lines = ["# A0 阶段 1：47 个 responder 单折剪枝", "",
             f"**{payload['question']}**", "",
             f"- {payload['design_note']}", f"- {payload['selection_bias_guard']}",
             f"- 折：训练 {s['fold']['n_train_rows']:,} 行 / 验证 {s['fold']['n_valid_rows']:,} 行，"
             f"fold_alpha {s['fold']['fold_alpha']:.4e}",
             f"- 设计矩阵 {s['design']['columns']} 列", "",
             f"**基准（`target` 训练）peak = {s['baseline_peak']:.8f}**", "",
             "| 排名 | 训练目标 | peak | 相对 target | 阶梯上胜出档数 |",
             "|---:|---|---:|---:|---:|"]
    for i, r in enumerate(s["ranking"], 1):
        lines.append(f"| {i} | `{r['name']}` | {r['peak']:.8f} | {r['relative_to_target']*100:+.2f}% | "
                     f"{r['positive_rungs']}/{len(s['alpha_ladder'])} |")
    if s["skipped"]:
        lines += ["", "跳过：" + "；".join(f"`{k}`（{v}）" for k, v in s["skipped"].items())]
    lines += ["", f"**短名单候选（前 {SHORTLIST_SIZE}）**：" +
              "、".join(f"`{n}`" for n in s["shortlist_proposal"]),
              "", "⚠️ 阶段 1 只剪枝，**不作判决** —— 判决在阶段 2 的外层折上，判据见 `verdict()`。", ""]
    return "\n".join(lines) + "\n"


def render_stage2(payload) -> str:
    lines = ["# A0：`responder_*` 换训练目标 —— 外层判决", "",
             f"**{payload['question']}**", "",
             f"- 指标：{payload['metric']}", f"- 公平性：{payload['fairness']}",
             f"- 防选择偏置：{payload['selection_bias_guard']}",
             f"- 短名单（预注册）：" + "、".join(f"`{n}`" for n in payload["shortlist"]),
             f"- 来源 stage1 JSON sha256：`{payload['shortlist_source_sha256']}`", "",
             "## 逐折", "", "| 折 | 内层选中 | baseline peak | 选中臂 peak | multi peak |",
             "|---:|---|---:|---:|---:|"]
    for r in payload["results"]["folds"]:
        b = r["outer"][BASELINE_ARM]["production_equivalent"]["peak"]
        p = r["outer"][r["picked_by_inner"]]["production_equivalent"]["peak"]
        m = r["multi"]["production_equivalent"]["peak"]
        lines.append(f"| {r['fold']} | `{r['picked_by_inner']}` | {b:.8f} | "
                     f"{p:.8f} ({(p/b-1)*100:+.2f}%) | {m:.8f} ({(m/b-1)*100:+.2f}%) |")
    lines.append("")
    for name, c in payload["comparisons"].items():
        st = c["peak"]
        lines += [f"## {name} 臂", "",
                  f"- baseline 折均 **{c['baseline_peak_mean']:.8f}** → 本臂 **{c['arm_peak_mean']:.8f}**",
                  f"- 配对 Δ 均值 {st['mean_delta']:+.3e}（**{st['relative_gain']*100:+.2f}%**），"
                  f"{st['positive_folds']}/{st['n_folds']} 折为正，符号检验 p={st['sign_test_p']:.3f}",
                  f"- 去掉最好一折：{st['relative_gain_drop_best']*100:+.2f}%",
                  f"- 整条 alpha 阶梯为正：{'✅' if c['alpha_ladder_all_positive'] else '❌'}", ""]
    lines += ["## 判据（由 `verdict()` 判，不是报告里的评语）", ""]
    for name, v in payload["verdicts"].items():
        lines.append(f"**{name} —— {'✅ PASS' if v['pass'] else '❌ 不过'}**")
        lines += [f"- {'✅' if ok else '❌'} {k}" for k, ok in v["checks"].items()]
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
