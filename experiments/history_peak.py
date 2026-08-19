"""A1：每资产历史特征，在**尺度无关的 peak 口径**下复现 07-23 那个 +2.5%。

## 要验证的假设

把 `history_features.AssetHistory` 那 4 组按资产滚动特征（`previous` / `difference` /
`rolling_mean` / `rolling_deviation`）加进设计矩阵后，**样本外 `peak = A²/B` 提高**。

## ⚠️ 为什么必须重测：原实验的判据带着一个已经咬过一次的缺陷

`experiments/walk_forward_history.py:37,124` 是在**固定 `prediction_scale=0.5`** 下算
`weighted_zero_mean_r2` 的**绝对分**（baseline 0.00080097 → history 0.00082130，+2.5%）。

这正是 08-10 把「市场模型重建还有 13% 余量」判成幻觉的同一个机制 ——
「基准被 `prediction_scale=0.5` 压扁，换成尺度无关的 peak 后消失」（NOTES / ROADMAP 已结案）。

而且这里更要命。在 `Score(a) = 2aA − a²B` 上，固定 `a=0.5` 时分数对 B 的相对惩罚是

    ∂lnScore/∂lnB = −0.25B / (A − 0.25B) ≈ −0.22        （用生产的 A、B 代入）

而 peak 的是 `∂ln(A²/B)/∂lnB = −1`，折算到同一口径约 **−0.5**。
⟹ **固定低 scale 对「方差膨胀」的惩罚不到 peak 的一半**。
而 history 臂的全部动作就是把设计矩阵从 **400 列加到 560 列** —— 典型的 B 膨胀。
所以那 +2.5% 很可能在 peak 口径下缩水甚至翻负。本脚本就是来钉死这一点的。

报告里会**同时给出两种口径的数字**（peak 与固定 scale 0.5 的旧口径），
用来量化「换尺子」到底改变了多少 —— 这是本次实验最有价值的副产物。

## 关键实现决定：一次流式扫描 + 原始滞后缓存

`history_features.build_history_design` 不能直接用，三个原因：

1. `:135` 的 `mask = time_ids % sample_modulo == 0` 写死 periodic
2. 它要求 `artifact` 已拟合，而 artifact 逐折不同 ⟹ **每折都要重扫 9 个分区**
3. 历史状态必须在**每一行**上推进（`:134` 在 `:135` 的掩码之前），无法从已采样矩阵重建

改成：**流式扫一遍全量，只缓存被采样行的「原始滞后值」**（前 `window` 期，未变换），
所有折与两个臂共用。逐折再套该折的 `robust_transform_fit` 统计量。
缓存的是原始值、不含任何拟合量 ⟹ **无泄漏**。

等价性（`apply_robust_transform` 是逐列逐元素的 ⟹ 与「取滞后」可交换）：

    previous          = T(L1)                        ← 精确
    difference        = T(x) − T(L1)                 ← 精确
    rolling_mean      = mean_{k≤count} T(Lk)         ← 精确（先变换再平均，不能反过来）
    rolling_deviation = T(x) − rolling_mean
    count == 0 的行，previous / rolling_mean 置 0（与 AssetHistory 一致）

`--verify` 会拿这条重建路径与 `AssetHistory.transform` 直接对拍，**对不上就退出**。

## 预注册

- history 列集：在**第 0 折的训练窗**上按加权 |corr| 选一次的**固定 40 列**，全折通用。
  与原实验的「逐折在选中的 200 列里重选」不同 —— 固定下来更保守，也省掉一层选择偏置。
- 窗长 5（`AssetHistory.window_size` 的历史值，原实验也是 5）。
- 折：`rolling_time_folds`，`train_window = 39_480`（= 生产等效，`ridge_data_ladder`
  的 `BASELINE_WINDOW`），`embargo = 6`。
- `fold_alpha = ridge_alpha × train_window / PROD_SAMPLED_WINDOW`（行数归一，同 market_model）。

## 判据（预注册，由 `verdict()` 机器判，不由报告文字判 —— 伤疤清单 #2）

沿用 ROADMAP A 段那三条，**不擅自加严**：

1. 逐折配对 Δ(peak) 均值为正
2. 去掉最好一折仍为正
3. 点估计换算总分 ≥ +1%

不要求本地显著（③类规矩：9 个分区的检出下限就在 +8%，等显著只会漏掉真的）。
⚠️ 但「去掉最好一折」不豁免 —— 点估计不显著和点估计由单折制造是两回事。

⚠️ **必报诊断（不作硬门槛）**：整条 alpha 阶梯的符号，以及 `2ΔA > ΔB` 是否成立。
理由：加 160 列**同时是**③类结构改动和②类拟合紧密度改动，和 A0「换训练目标」
是同一种纠缠。只在一个 alpha 上赢，不足以当③类看待。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/history_peak.py --verify-only
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/history_peak.py --arms ridge
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/history_peak.py --arms ridge lgbm
输出：outputs/experiments/history_peak.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
# 复用已有实现，不另写一份（口径唯一性）
from features import apply_robust_transform, cross_sectional_deviation
from train import robust_transform_fit, select_features
from history_features import AssetHistory, iter_complete_time_batches
from mt_predictability import group_starts
from market_model import sign_test_p
from ridge_data_ladder import row_level_peak
from walk_forward_rolling import PROD_SAMPLED_WINDOW
from lgbm_xs import load_rows   # 支持 phase_balanced；load_all_sampled 只会 periodic

# ---- 预注册常量（不搜） -------------------------------------------------------
HISTORY_WINDOW = 5              # AssetHistory.window_size 的历史值，原实验同
HISTORY_FEATURE_COUNT = 40      # 原实验的 history_feature_count
FEATURE_COUNT = 200             # 生产口径
TRAIN_WINDOW = 39_480           # = ridge_data_ladder.BASELINE_WINDOW，生产等效
ALPHAS = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)   # 与 market_model / mt_predictability 同一条阶梯
LEGACY_SCALE = 0.5              # 原实验的固定 scale —— 只用于「旧口径对照」这一栏
LEGACY_CLIP = 0.5
# LightGBM 实验档：生产是 3 种子 × 480 轮，这里降规模只为判方向
LGBM_SPEC = {"num_leaves": 63, "learning_rate": 0.03, "feature_fraction": 0.7, "lambda_l2": 1.0}
LGBM_MIN_DATA_FRAC = 12000 / 3_500_000
LGBM_ROUNDS = 160


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A1: per-asset history features under the peak metric")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="history_peak")
    p.add_argument("--arms", nargs="+", default=["ridge"], choices=["ridge", "lgbm"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=TRAIN_WINDOW)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--sample-modulo", type=int, default=10)
    p.add_argument("--sampling", default="periodic", choices=["periodic", "phase_balanced"],
                   help="生产是 phase_balanced + modulo 5；历史实验一直是 periodic + modulo 10。"
                        "查『本地为什么低估』时要能切到生产口径。")
    p.add_argument("--lgbm-seeds", type=int, default=1,
                   help="LGBM 臂的种子数（生产是 3）。多种子平均降方差 ⟹ 可能抬 peak，"
                        "这是本地与生产之间一个与 history 无关的口径差。")
    p.add_argument("--feature-count", type=int, default=FEATURE_COUNT)
    p.add_argument("--history-count", type=int, default=HISTORY_FEATURE_COUNT)
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--history-basis", default="raw", choices=["raw", "deviation"],
                   help="A3′：history 块喂什么。raw = 滞后【变换后的原始特征】（A1′ 已上线的口径）；"
                        "deviation = 滞后【截面偏差】—— LGBM 截面块的目标 e 本身就是截面残差，"
                        "对齐更好。**换不是加**，列数/窗长/设计矩阵形状全不变。")
    p.add_argument("--xs-transform", default="deviation", choices=["deviation", "rank"],
                   help="A2：LGBM 设计里主块的截面变换。deviation = 减截面均值（现口径）；"
                        "rank = 逐 time_id 截面排序映射到 [−1,1]，对厚尾免疫、逐截面自适应。"
                        "**换不是加**，列数不变。")
    p.add_argument("--history-scope", default="lgbm_selected", choices=["global", "lgbm_selected"],
                   help="history 列的候选池。lgbm_selected = 先按无权截面残差 e 选出 200 列，"
                        "再在这 200 列内选 history 列 —— 这样 history 列必然是 LGBM 选中列的"
                        "子集，`hybrid_meta.json` 里已有它们的 lower/upper/center/scale，"
                        "推理端不用扩契约。global = 在全部 323 列里按 target 选（会漏到 200 列外）。")
    p.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    p.add_argument("--lgbm-select", default="residual", choices=["residual", "target"],
                   help="LGBM 臂的选列口径。residual = 按无权截面残差 e 选（与生产 "
                        "train.py:184 一致）；target = 按加权 target 选。两臂共用同一套选中列，"
                        "所以 Δ 在两种口径下都是配对的，但 residual 才能读回生产。")
    p.add_argument("--lgbm-rounds", type=int, default=LGBM_ROUNDS)
    p.add_argument("--lgbm-seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--verify-rows", type=int, default=60_000)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ---- 滞后缓存 -----------------------------------------------------------------

def retained_lag_rows(
    time_ids: np.ndarray,
    sample_modulo: int,
    sampling: str,
    *,
    minimum_time_id: int | None = None,
    maximum_time_id: int | None = None,
) -> np.ndarray:
    """Combine the existing sampling mask with an optional inclusive time range."""
    ids = np.asarray(time_ids, dtype=np.int64)
    mask = time_sample_mask(ids, sample_modulo, sampling=sampling)
    if minimum_time_id is not None:
        mask &= ids >= int(minimum_time_id)
    if maximum_time_id is not None:
        mask &= ids <= int(maximum_time_id)
    if (
        minimum_time_id is not None
        and maximum_time_id is not None
        and minimum_time_id > maximum_time_id
    ):
        raise ValueError("minimum_time_id must not exceed maximum_time_id")
    return mask


def build_lag_cache(
    files: list[Path], history_columns: np.ndarray, sample_modulo: int, window: int,
    sampling: str = "periodic", batch_size: int = 120_000, verbose: bool = True,
    minimum_time_id: int | None = None, maximum_time_id: int | None = None,
) -> dict[str, np.ndarray]:
    """流式扫全量，缓存被采样行的**原始**滞后值。

    历史状态在每一行上推进，采样掩码只决定**留不留**这一行的缓存 ——
    与 `build_history_design:134-135` 的顺序一致。
    """
    names = [FEATURE_COLUMNS[i] for i in history_columns]
    columns = ["time_id", "asset_id", *names]
    k = len(history_columns)
    buffers: dict[int, np.ndarray] = {}          # asset -> (m<=window, k)，从旧到新
    lag_parts, cnt_parts, tid_parts, aid_parts = [], [], [], []

    for path in files:
        kept, started = 0, time.perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            cur = frame.loc[:, names].to_numpy(dtype=np.float32, copy=True)
            n = len(tid)
            lags = np.zeros((n, window, k), dtype=np.float32)
            counts = np.zeros(n, dtype=np.int16)
            for asset in np.unique(aid):
                idx = np.flatnonzero(aid == asset)
                buf = buffers.get(int(asset))
                if buf is None:
                    buf = np.empty((0, k), dtype=np.float32)
                combined = np.vstack([buf, cur[idx]])
                pos = len(buf) + np.arange(len(idx))       # 当前行在 combined 里的下标
                for j in range(window):                    # j=0 → 滞后 1 期
                    src = pos - (j + 1)
                    ok = src >= 0
                    if ok.any():
                        lags[idx[ok], j, :] = combined[src[ok]]
                counts[idx] = np.minimum(pos, window)
                buffers[int(asset)] = combined[-window:].astype(np.float32, copy=True)
            mask = retained_lag_rows(
                tid,
                sample_modulo,
                sampling,
                minimum_time_id=minimum_time_id,
                maximum_time_id=maximum_time_id,
            )
            if mask.any():
                lag_parts.append(lags[mask])
                cnt_parts.append(counts[mask])
                tid_parts.append(tid[mask])
                aid_parts.append(aid[mask])
                kept += int(mask.sum())
            del lags, counts, frame, cur
        if verbose:
            print(f"  lag cache {path.name}: {kept:,} 行 ({time.perf_counter()-started:.0f}s)", flush=True)

    return {
        "lags": np.concatenate(lag_parts),
        "count": np.concatenate(cnt_parts),
        "time_id": np.concatenate(tid_parts),
        "asset_id": np.concatenate(aid_parts),
    }


def cross_sectional_rank(features: np.ndarray, time_ids: np.ndarray,
                         chunk_groups: int = 4000) -> np.ndarray:
    """A2：逐 time_id 的截面 rank，映射到对称区间 [−1, 1]。

    `(2·rank − (n−1)) / (n−1)`，rank ∈ 0..n−1。n=1 的截面给 0。

    与 `cross_sectional_deviation` 的区别：后者减去截面均值、离群点仍有杠杆；
    rank 对厚尾**完全免疫**，且逐截面自适应（不像裁剪/IQR 是训练期一次拟合、时间上固定的）。
    ⚠️ `ab_xsstd` 否决的是「除以截面标准差」，不是 rank。
    ⚠️ 面板不平衡（7.06% 的 time_id 少于 15 个资产），所以 n 逐组不同，归一化要按各组的 n。
    ⚠️ 只有 ≤15 个资产 ⟹ rank 只有 ≤15 个取值，比连续值粗 —— 这是它可能输的地方。

    填充值用 `+inf` 排到最后，于是真实元素拿到的 rank 就是 0..n−1。
    """
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    width = int(counts.max())
    out = np.empty_like(features, dtype=np.float32)

    for begin in range(0, len(starts), chunk_groups):
        end = min(begin + chunk_groups, len(starts))
        g_starts, g_counts = starts[begin:end], counts[begin:end]
        row0, row1 = int(g_starts[0]), int(g_starts[-1] + g_counts[-1])
        padded = np.full((len(g_starts), width, features.shape[1]), np.inf, dtype=np.float32)
        valid = np.arange(width)[None, :] < g_counts[:, None]
        padded[valid] = features[row0:row1]
        order = np.argsort(padded, axis=1, kind="stable")
        rank = np.argsort(order, axis=1).astype(np.float32)
        denominator = np.maximum(g_counts.astype(np.float32) - 1.0, 1.0)[:, None, None]
        scaled = (2.0 * rank - (g_counts.astype(np.float32) - 1.0)[:, None, None]) / denominator
        out[row0:row1] = scaled[valid]
        del padded, order, rank, scaled
    return out


def stream_history_deviation(
    files: list[Path], history_columns: np.ndarray, sample_modulo: int, sampling: str,
    window: int, stats: dict[str, np.ndarray], verbose: bool = False,
) -> tuple[np.ndarray, ...]:
    """A3′：history 喂**滞后的截面偏差**而不是滞后的原始特征。

    ⚠️ 为什么不能复用滞后缓存：缓存里只有「该资产自己」的原始滞后值，而
    `cross_sectional_deviation` 需要那个 time_id 的**完整截面**；且 `dev = xs_dev(T(x))`
    里的 `T` 含 clip（非线性）、又逐折不同，事后重建不出来。
    ⚠️ 更关键：面板**不平衡**（7.06% 的 time_id 少于 15 个资产），所以一个资产的
    「上一次观测」未必在 `t−1` —— 想靠「同 time_id 的 15 行凑出 t−k 的截面」也不成立。

    ⟹ 逐折流式：用该折已拟合的 `T`，扫全量、逐完整 time_id 算 dev、喂 AssetHistory，
    只留采样行。实测一遍约 10 秒，可接受。

    必须用 `iter_complete_time_batches` 把跨 parquet batch 的 time_id 拼完整，
    否则截面均值会按半个截面算错。
    """
    names = [FEATURE_COLUMNS[i] for i in history_columns]
    columns = ["time_id", "asset_id", *names]
    lo, hi, ce, sc = (stats[k][history_columns] for k in ("lower", "upper", "center", "scale"))
    history = AssetHistory(feature_count=len(history_columns), window_size=window)
    parts: list[list[np.ndarray]] = [[], [], [], []]
    kept = 0
    for path in files:
        for frame in iter_complete_time_batches(path, columns):
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            current = frame.loc[:, names].to_numpy(dtype=np.float32, copy=True)
            apply_robust_transform(current, lo, hi, ce, sc)
            dev = cross_sectional_deviation(current, tid)          # ← 唯一的区别
            blocks = history.transform(dev, aid)                   # 每一行都推进状态
            mask = time_sample_mask(tid, sample_modulo, sampling=sampling)
            if mask.any():
                for slot, block in zip(parts, blocks):
                    slot.append(block[mask])
                kept += int(mask.sum())
        if verbose:
            print(f"  dev-history {path.name}: 累计 {kept:,} 行", flush=True)
    return tuple(np.concatenate(slot) for slot in parts)


def history_blocks(
    lags: np.ndarray, counts: np.ndarray, current_t: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, center: np.ndarray, scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由原始滞后缓存 + 该折的变换统计量重建 4 个历史块。

    ⚠️ 无效槽位存的是原始 0.0，`apply_robust_transform` 会把它映成 `(0−center)/scale ≠ 0`，
    所以必须**变换之后**再按 count 掩码置 0 —— 与 AssetHistory 的「无历史即 0」一致。
    """
    n, window, _ = lags.shape
    previous = None
    # ⚠️ 必须在 float64 里累加、最后一次性 round 回 float32 —— AssetHistory 走的是
    # `np.cumsum(..., dtype=np.float64)` 再 `.astype(np.float32)`。若在 float32 里累加，
    # 两条路径会差一个 ulp（实测固定 1.907e-06 = 2⁻¹⁹，与 batch 大小无关）。
    rolling_sum = np.zeros((n, lags.shape[2]), dtype=np.float64)
    for j in range(window):
        block = lags[:, j, :].copy()
        apply_robust_transform(block, lower, upper, center, scale)
        block[counts <= j] = 0.0
        if j == 0:
            previous = block.copy()
        rolling_sum += block
    cnt = counts.astype(np.float64)[:, None]
    rolling_mean = np.divide(rolling_sum, cnt, out=np.zeros_like(rolling_sum),
                             where=cnt > 0).astype(np.float32)
    return previous, current_t - previous, rolling_mean, current_t - rolling_mean


def verify_equivalence(path: Path, history_columns: np.ndarray, window: int,
                       n_rows: int, batch_size: int = 12_000) -> dict[str, float]:
    """把重建路径与 `AssetHistory.transform` 对拍。这是整个 A1 复现的地基。"""
    names = [FEATURE_COLUMNS[i] for i in history_columns]
    frame = next(pq.ParquetFile(path).iter_batches(
        batch_size=n_rows, columns=["time_id", "asset_id", *names])).to_pandas()
    aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
    raw = frame.loc[:, names].to_numpy(dtype=np.float32, copy=True)

    transformed, stats = robust_transform_fit(raw.copy())
    lo, hi, ce, sc = stats["lower"], stats["upper"], stats["center"], stats["scale"]

    # 参照路径：AssetHistory 直接吃**已变换**的值，分批推进
    ah = AssetHistory(feature_count=len(history_columns), window_size=window)
    ref = [[], [], [], []]
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        out = ah.transform(transformed[start:stop], aid[start:stop])
        for slot, arr in zip(ref, out):
            slot.append(arr)
    ref_prev, ref_diff, ref_rm, ref_rd = (np.concatenate(s) for s in ref)

    # 重建路径：缓存原始滞后 → 逐折套统计量
    buffers: dict[int, np.ndarray] = {}
    lags = np.zeros((len(raw), window, len(history_columns)), dtype=np.float32)
    counts = np.zeros(len(raw), dtype=np.int16)
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        sl = slice(start, stop)
        for asset in np.unique(aid[sl]):
            idx = np.flatnonzero(aid[sl] == asset) + start
            buf = buffers.get(int(asset), np.empty((0, len(history_columns)), dtype=np.float32))
            combined = np.vstack([buf, raw[idx]])
            pos = len(buf) + np.arange(len(idx))
            for j in range(window):
                src = pos - (j + 1)
                ok = src >= 0
                if ok.any():
                    lags[idx[ok], j, :] = combined[src[ok]]
            counts[idx] = np.minimum(pos, window)
            buffers[int(asset)] = combined[-window:].astype(np.float32, copy=True)
    got_prev, got_diff, got_rm, got_rd = history_blocks(lags, counts, transformed, lo, hi, ce, sc)

    return {
        "rows": int(len(raw)),
        "max_abs_previous": float(np.abs(got_prev - ref_prev).max()),
        "max_abs_difference": float(np.abs(got_diff - ref_diff).max()),
        "max_abs_rolling_mean": float(np.abs(got_rm - ref_rm).max()),
        "max_abs_rolling_deviation": float(np.abs(got_rd - ref_rd).max()),
    }


# ---- 设计矩阵 -----------------------------------------------------------------

def transform_with(features: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    out = features.copy()
    return apply_robust_transform(out, stats["lower"], stats["upper"], stats["center"], stats["scale"])


def ridge_designs(t_features: np.ndarray, time_ids: np.ndarray, selected: np.ndarray,
                  hist: tuple[np.ndarray, ...] | None) -> np.ndarray:
    """`raw_dev` 基底（与 `train.make_design` 逐位同构），history 块追加在后面。"""
    raw = t_features[:, selected].copy()
    deviation = cross_sectional_deviation(raw, time_ids)
    blocks = [raw, deviation]
    if hist is not None:
        blocks.extend(hist)
    return np.column_stack(blocks).astype(np.float32, copy=False)


def fit_ridge(design: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> Ridge:
    """与 `train.fit_model` 的严格档同参数（tol 1e-8 / max_iter 2000）。"""
    est = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=2000,
                fit_intercept=True, copy_X=False)
    est.fit(design, y, sample_weight=np.maximum(w, 0.0))
    return est


def legacy_score(target: np.ndarray, raw_pred: np.ndarray, weight: np.ndarray) -> float:
    """07-23 的旧口径：固定 scale 0.5 + clip 0.5 的绝对分。只作对照，不作判据。"""
    return float(weighted_zero_mean_r2(
        target, np.clip(raw_pred * LEGACY_SCALE, -LEGACY_CLIP, LEGACY_CLIP), weight))


# ---- 判据 ---------------------------------------------------------------------

def paired_stats(deltas: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    positive = int((deltas > 0).sum())
    without_best = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
    # ⚠️ 基准均值 ≤ 0 时「相对增益」会翻号（Δ 更差却报成正的）。peak 恒 ≥ 0 不会碰到，
    # 但固定 scale 0.5 的旧口径分数是可以为负的 —— 那时相对量无意义，只能看绝对 Δ。
    base = float(baseline.mean())
    ratio = (lambda d: float(d / base)) if base > 0 else (lambda d: float("nan"))
    return {
        "mean_delta": float(deltas.mean()),
        "relative_gain": ratio(deltas.mean()),
        "baseline_mean": base,
        "baseline_positive": base > 0,
        "positive_folds": positive,
        "n_folds": int(len(deltas)),
        "sign_test_p": sign_test_p(positive, len(deltas)),
        "mean_delta_drop_best": float(without_best.mean()),
        "relative_gain_drop_best": ratio(without_best.mean()),
        "per_fold": [float(v) for v in deltas],
    }


def verdict(stats: dict[str, object]) -> dict[str, object]:
    """三条判据由代码判 —— 报告里的文字不得与这里不一致（伤疤清单 #2）。

    ROADMAP A 段预注册：配对 Δ 均值为正 + 去掉最好一折仍为正 + 点估计换算总分 ≥ +1%。
    不要求显著（③类规矩）。
    """
    checks = {
        "1_paired_delta_positive": stats["mean_delta"] > 0,
        "2_survives_drop_best_fold": stats["mean_delta_drop_best"] > 0,
        "3_relative_gain_at_least_1pct": stats["relative_gain"] >= 0.01,
    }
    return {"checks": checks, "pass": all(checks.values())}


# ---- 主流程 -------------------------------------------------------------------

def run_ridge_arm(data, cache, folds, all_time_ids, hist_cols, args) -> dict[str, object]:
    fold_rows: list[dict[str, object]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        tr = np.isin(all_time_ids, train_ids)
        va = np.isin(all_time_ids, valid_ids)
        fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW

        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        y_tr = data["target"][tr].astype(np.float64)
        w_tr = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
        tid_tr = data["time_id"][tr]
        selected = select_features(t_train, y_tr, w_tr, args.feature_count)

        t_valid = transform_with(data["features"][va], stats)
        y_va = data["target"][va].astype(np.float64)
        w_va = np.maximum(data["weight"][va].astype(np.float64), 0.0)
        tid_va = data["time_id"][va]

        if args.history_basis == "deviation":
            # A3′：逐折流式重算（滞后缓存重建不出 dev，见 stream_history_deviation 的说明）
            blocks = stream_history_deviation(train_files(Path(args.data_root)), hist_cols,
                                              args.sample_modulo, args.sampling,
                                              args.history_window, stats)
            hist_tr = tuple(b[tr] for b in blocks)
            hist_va = tuple(b[va] for b in blocks)
            del blocks
        else:
            lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
            hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
            hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

        row: dict[str, object] = {
            "fold": index, "n_train_rows": int(tr.sum()), "n_valid_rows": int(va.sum()),
            "n_train_time_ids": int(len(train_ids)), "fold_alpha": float(fold_alpha),
            "arms": {},
        }
        for arm, hist in (("baseline", None), ("history", hist_tr)):
            d_tr = ridge_designs(t_train, tid_tr, selected, hist)
            d_va = ridge_designs(t_valid, tid_va, selected, None if hist is None else hist_va)
            ladder: dict[str, dict[str, float]] = {}
            for alpha in (fold_alpha, *ALPHAS):
                key = "fold_alpha" if alpha == fold_alpha else f"{alpha:.0e}"
                est = fit_ridge(d_tr, y_tr, w_tr, alpha)
                pred = (est.intercept_ + d_va @ est.coef_).astype(np.float64)
                entry = row_level_peak(y_va, pred, w_va)
                entry["legacy_score_at_scale_0.5"] = legacy_score(y_va, pred, w_va)
                entry["n_clipped_at_legacy"] = int((np.abs(pred * LEGACY_SCALE) > LEGACY_CLIP).sum())
                ladder[key] = entry
                del est, pred
            row["arms"][arm] = {"design_columns": int(d_tr.shape[1]), "ladder": ladder}
            del d_tr, d_va
            gc.collect()

        b = row["arms"]["baseline"]["ladder"]["fold_alpha"]
        h = row["arms"]["history"]["ladder"]["fold_alpha"]
        print(f"  ridge fold {index}: peak {b['peak']:.8f} → {h['peak']:.8f} "
              f"({(h['peak']/b['peak']-1)*100:+.2f}%) | 旧口径 {b['legacy_score_at_scale_0.5']:.8f} → "
              f"{h['legacy_score_at_scale_0.5']:.8f} ({(h['legacy_score_at_scale_0.5']/b['legacy_score_at_scale_0.5']-1)*100:+.2f}%) "
              f"[{time.perf_counter()-started:.0f}s]", flush=True)
        fold_rows.append(row)
        del t_train, t_valid, hist_tr, hist_va
        gc.collect()
    return {"folds": fold_rows}


def run_lgbm_arm(data, cache, folds, all_time_ids, hist_cols, args) -> dict[str, object]:
    """LGBM 截面块。岭回归部分两臂**逐位相同** ⟹ 唯一变量是 history 块。

    评的是**整条预测** `m̂ + ê` 的 peak，所以 Δ 与 ridge 臂直接可比，也就是会上线的形态。
    """
    import lightgbm as lgb

    fold_rows: list[dict[str, object]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        tr = np.isin(all_time_ids, train_ids)
        va = np.isin(all_time_ids, valid_ids)
        fold_alpha = args.ridge_alpha * len(train_ids) / PROD_SAMPLED_WINDOW

        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        y_tr = data["target"][tr].astype(np.float64)
        w_tr = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
        tid_tr, aid_tr = data["time_id"][tr], data["asset_id"][tr]
        t_valid = transform_with(data["features"][va], stats)
        y_va = data["target"][va].astype(np.float64)
        w_va = np.maximum(data["weight"][va].astype(np.float64), 0.0)
        tid_va, aid_va = data["time_id"][va], data["asset_id"][va]

        # --- 岭回归（两臂共用，只用来出 m̂）。选列按 target 加权，与生产岭回归一致
        ridge_selected = select_features(t_train, y_tr, w_tr, args.feature_count)
        est = fit_ridge(ridge_designs(t_train, tid_tr, ridge_selected, None), y_tr, w_tr, fold_alpha)
        d_va = ridge_designs(t_valid, tid_va, ridge_selected, None)
        ridge_raw = (est.intercept_ + d_va @ est.coef_).astype(np.float64)
        va_starts = group_starts(tid_va)
        va_counts = np.diff(np.r_[va_starts, len(tid_va)]).astype(np.float64)
        m_hat = np.repeat(np.add.reduceat(ridge_raw, va_starts) / va_counts, va_counts.astype(int))
        del est, d_va, ridge_raw

        # --- 目标：无权截面残差（与生产 train.py:177 同口径）
        tr_starts = group_starts(tid_tr)
        tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)
        e_tr = y_tr - np.repeat(np.add.reduceat(y_tr, tr_starts) / tr_counts, tr_counts.astype(int))

        # 选列口径：residual = 按 e 无权选（生产 train.py:184 就是这么做的）
        lgbm_selected = (select_features(t_train, e_tr, np.ones_like(e_tr), args.feature_count)
                         if args.lgbm_select == "residual" else ridge_selected)
        xs = cross_sectional_rank if args.xs_transform == "rank" else cross_sectional_deviation
        sel_dev_tr = xs(t_train[:, lgbm_selected].copy(), tid_tr)
        sel_dev_va = xs(t_valid[:, lgbm_selected].copy(), tid_va)
        if args.history_basis == "deviation":
            # A3′：逐折流式重算（滞后缓存重建不出 dev，见 stream_history_deviation 的说明）
            blocks = stream_history_deviation(train_files(Path(args.data_root)), hist_cols,
                                              args.sample_modulo, args.sampling,
                                              args.history_window, stats)
            hist_tr = tuple(b[tr] for b in blocks)
            hist_va = tuple(b[va] for b in blocks)
            del blocks
        else:
            lo, hi, ce, sc = (stats[k][hist_cols] for k in ("lower", "upper", "center", "scale"))
            hist_tr = history_blocks(cache["lags"][tr], cache["count"][tr], t_train[:, hist_cols], lo, hi, ce, sc)
            hist_va = history_blocks(cache["lags"][va], cache["count"][va], t_valid[:, hist_cols], lo, hi, ce, sc)

        row: dict[str, object] = {"fold": index, "n_train_rows": int(tr.sum()),
                                  "n_valid_rows": int(va.sum()), "arms": {}}
        min_data = max(20, int(round(LGBM_MIN_DATA_FRAC * int(tr.sum()))))
        for arm, hist in (("baseline", None), ("history", True)):
            # ⚠️ asset_id 必须是最后一列（train.py:196 / main.py:186 / lgbm_numpy 都这么假设）
            parts_tr = [sel_dev_tr] + (list(hist_tr) if hist else []) + [aid_tr.astype(np.float32)[:, None]]
            parts_va = [sel_dev_va] + (list(hist_va) if hist else []) + [aid_va.astype(np.float32)[:, None]]
            d_tr = np.ascontiguousarray(np.column_stack(parts_tr))
            d_va = np.ascontiguousarray(np.column_stack(parts_va))
            cat = d_tr.shape[1] - 1
            # 多种子：与生产一致，先对各种子的预测取平均，再投影成无权零均值
            e_hat = np.zeros(len(d_va), dtype=np.float64)
            for s in range(args.lgbm_seeds):
                seed = args.lgbm_seed + s
                params = {**LGBM_SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
                          "num_threads": args.num_threads, "min_data_in_leaf": min_data,
                          "bagging_fraction": 0.7, "bagging_freq": 1, "deterministic": True,
                          "force_row_wise": True, "feature_pre_filter": False,
                          "seed": seed, "bagging_seed": seed + 1000,
                          "feature_fraction_seed": seed + 2000}
                ds = lgb.Dataset(d_tr, label=e_tr, params=params, categorical_feature=[cat],
                                 free_raw_data=False)
                booster = lgb.train(params, ds, num_boost_round=args.lgbm_rounds)
                e_hat += booster.predict(d_va, num_iteration=args.lgbm_rounds).astype(np.float64)
                del ds, booster
            e_hat /= args.lgbm_seeds
            e_hat -= np.repeat(np.add.reduceat(e_hat, va_starts) / va_counts, va_counts.astype(int))
            full = m_hat + e_hat
            entry = row_level_peak(y_va, full, w_va)
            entry["legacy_score_at_scale_0.5"] = legacy_score(y_va, full, w_va)
            row["arms"][arm] = {"design_columns": int(d_tr.shape[1]), "full": entry}
            del d_tr, d_va, e_hat, full        # ds/booster 已在种子循环里逐个释放
            gc.collect()

        b = row["arms"]["baseline"]["full"]
        h = row["arms"]["history"]["full"]
        print(f"  lgbm fold {index}: peak {b['peak']:.8f} → {h['peak']:.8f} "
              f"({(h['peak']/b['peak']-1)*100:+.2f}%) [{time.perf_counter()-started:.0f}s]", flush=True)
        fold_rows.append(row)
        del t_train, t_valid, hist_tr, hist_va, sel_dev_tr, sel_dev_va
        gc.collect()
    return {"folds": fold_rows}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.label}.md"
    if report_path.exists() and not args.force and not args.verify_only:
        raise SystemExit(f"{report_path} 已存在；要覆盖请加 --force")

    files = train_files(Path(args.data_root))

    # --verify-only 只验机制，与选哪些列无关 —— 走快路径，不必先做几分钟的全量装载
    if args.verify_only:
        probe = np.arange(args.history_count)
        print(f"verifying lag-cache ≡ AssetHistory（前 {args.history_count} 列，"
              f"{args.verify_rows:,} 行）...", flush=True)
        report = verify_equivalence(files[0], probe, args.history_window, args.verify_rows)
        print(json.dumps(report, indent=1), flush=True)
        exact = max(report["max_abs_previous"], report["max_abs_difference"])
        approx = max(report["max_abs_rolling_mean"], report["max_abs_rolling_deviation"])
        if exact != 0.0 or approx > 1e-6:
            raise SystemExit("❌ 等价性不成立 —— 停下来查")
        print("✅ 等价性通过", flush=True)
        return

    print("loading sampled partitions...", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)
    print(f"{len(all_time_ids):,} 行，{len(unique_time_ids):,} 个采样 time_id，{len(folds)} 折，"
          f"train_window {args.train_window:,}", flush=True)

    # ---- 预注册 history 列：第 0 折训练窗上按加权 |corr| 选一次，全折通用
    tr0 = np.isin(all_time_ids, folds[0][0])
    scratch, _ = robust_transform_fit(data["features"][tr0].copy())
    y0 = data["target"][tr0].astype(np.float64)
    if args.history_scope == "lgbm_selected":
        # 与生产同口径：先按无权截面残差 e 选 200 列，history 列只能从这 200 列里出
        tid0 = data["time_id"][tr0]
        s0 = group_starts(tid0)
        c0 = np.diff(np.r_[s0, len(tid0)]).astype(np.float64)
        e0 = y0 - np.repeat(np.add.reduceat(y0, s0) / c0, c0.astype(int))
        pool = select_features(scratch, e0, np.ones_like(e0), args.feature_count)
        inner = select_features(scratch[:, pool], e0, np.ones_like(e0), args.history_count)
        hist_cols = np.sort(pool[inner])
    else:
        hist_cols = select_features(scratch, y0,
                                    np.maximum(data["weight"][tr0].astype(np.float64), 0.0),
                                    args.history_count)
    del scratch
    gc.collect()
    print(f"history 列（第 0 折训练窗预注册，scope={args.history_scope}，{len(hist_cols)} 列）："
          f"{[int(c) for c in hist_cols[:10]]}{' ...' if len(hist_cols) > 10 else ''}", flush=True)

    # ---- 等价性硬验：对不上就停
    verification = None
    if not args.skip_verify:
        print("verifying lag-cache ≡ AssetHistory ...", flush=True)
        verification = verify_equivalence(files[0], hist_cols, args.history_window, args.verify_rows)
        print(json.dumps(verification, indent=1), flush=True)
        exact = max(verification["max_abs_previous"], verification["max_abs_difference"])
        approx = max(verification["max_abs_rolling_mean"], verification["max_abs_rolling_deviation"])
        if exact != 0.0 or approx > 1e-6:
            raise SystemExit("❌ 等价性不成立（previous/difference 必须逐位相同，rolling 两块 <1e-6）——"
                             " 停下来查，不许带着差异往下跑")
        print("✅ 等价性通过", flush=True)
        if args.verify_only:
            return

    print("building lag cache (streams every row) ...", flush=True)
    cache = build_lag_cache(files, hist_cols, args.sample_modulo, args.history_window,
                            sampling=args.sampling)
    if not (np.array_equal(cache["time_id"], all_time_ids)
            and np.array_equal(cache["asset_id"], data["asset_id"].astype(np.int64))):
        raise SystemExit("❌ 滞后缓存与采样矩阵的行不对齐 —— 两条读取路径的掩码/顺序不一致")
    print(f"lag cache {cache['lags'].shape} = {cache['lags'].nbytes/1e9:.2f} GB，行对齐已验", flush=True)

    results: dict[str, object] = {}
    if "ridge" in args.arms:
        print("== ridge 臂 ==", flush=True)
        results["ridge"] = run_ridge_arm(data, cache, folds, all_time_ids, hist_cols, args)
    if "lgbm" in args.arms:
        print("== lgbm 臂 ==", flush=True)
        results["lgbm"] = run_lgbm_arm(data, cache, folds, all_time_ids, hist_cols, args)

    payload = build_payload(args, folds, hist_cols, verification, results)
    (output_dir / f"{args.label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["verdicts"], ensure_ascii=False, indent=2), flush=True)
    print(f"报告：{report_path}", flush=True)


def _series(folds_rows: list[dict], arm: str, path: tuple[str, ...]) -> np.ndarray:
    out = []
    for row in folds_rows:
        node = row["arms"][arm]
        for key in path:
            node = node[key]
        out.append(node)
    return np.asarray(out, dtype=np.float64)


def build_payload(args, folds, hist_cols, verification, results) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    verdicts: dict[str, object] = {}
    for arm_name, res in results.items():
        rows = res["folds"]
        node = ("ladder", "fold_alpha") if arm_name == "ridge" else ("full",)
        base_peak = _series(rows, "baseline", node + ("peak",))
        hist_peak = _series(rows, "history", node + ("peak",))
        base_legacy = _series(rows, "baseline", node + ("legacy_score_at_scale_0.5",))
        hist_legacy = _series(rows, "history", node + ("legacy_score_at_scale_0.5",))
        base_a = _series(rows, "baseline", node + ("A",))
        hist_a = _series(rows, "history", node + ("A",))
        base_b = _series(rows, "baseline", node + ("B",))
        hist_b = _series(rows, "history", node + ("B",))

        stats = paired_stats(hist_peak - base_peak, base_peak)
        legacy_stats = paired_stats(hist_legacy - base_legacy, base_legacy)
        da = float((hist_a / base_a - 1.0).mean())
        db = float((hist_b / base_b - 1.0).mean())
        entry: dict[str, object] = {
            "peak": stats,
            "legacy_scale_0.5": legacy_stats,
            "delta_A_relative": da,
            "delta_B_relative": db,
            "mechanism_2dA_gt_dB": bool(2.0 * da > db),
            "baseline_peak_mean": float(base_peak.mean()),
            "history_peak_mean": float(hist_peak.mean()),
        }
        if arm_name == "ridge":
            ladder: dict[str, object] = {}
            for key in ("fold_alpha", *(f"{a:.0e}" for a in ALPHAS)):
                bp = _series(rows, "baseline", ("ladder", key, "peak"))
                hp = _series(rows, "history", ("ladder", key, "peak"))
                ladder[key] = {"mean_delta": float((hp - bp).mean()),
                               "relative_gain": float((hp - bp).mean() / bp.mean()),
                               "positive_folds": int(((hp - bp) > 0).sum())}
            entry["alpha_ladder"] = ladder
            entry["alpha_ladder_all_positive"] = bool(
                all(v["mean_delta"] > 0 for v in ladder.values()))
        comparisons[arm_name] = entry
        verdicts[arm_name] = verdict(stats)

    return {
        "question": "每资产历史特征在尺度无关的 peak 口径下还提高样本外表现吗？",
        "why": "原实验 walk_forward_history 是在固定 prediction_scale=0.5 下算绝对分的，"
               "而固定低 scale 对 B 膨胀的惩罚不到 peak 的一半；history 臂恰恰把设计矩阵"
               "从 400 列加到 560 列。这与 08-10 把「市场模型重建 13% 余量」判成幻觉的机制相同。",
        "metric": "peak = A²/B（尺度无关）；同时报固定 scale 0.5 的旧口径作对照",
        "pairing": "两臂共用同一折划分、同一变换统计量、同一选中的 200 列 —— 唯一变量是 history 块",
        "alpha_rule": "fold_alpha = ridge_alpha × train_window / PROD_SAMPLED_WINDOW（行数归一）",
        "classification": "③类结构改动，但加 160 列同时是②类拟合紧密度改动 —— 故必报整条 alpha 阶梯",
        "prior": {"source": "outputs/experiments/walk_forward_history.json",
                  "baseline_mean": 0.0008009714156413198, "history_mean": 0.0008213024421204196,
                  "relative_gain": 0.02538, "metric": "weighted_zero_mean_r2 @ prediction_scale 0.5",
                  "note": "3 折、按分区划分、legacy 求解器 tol 1e-4/max_iter 100"},
        "configuration": {
            "n_folds": len(folds), "train_window": args.train_window, "embargo": args.embargo,
            "sample_modulo": args.sample_modulo, "sampling": args.sampling,
            "lgbm_seeds": args.lgbm_seeds, "feature_count": args.feature_count,
            "history_count": args.history_count, "history_window": args.history_window,
            "history_columns": [int(c) for c in hist_cols],
            "history_scope": args.history_scope,
            "history_basis": args.history_basis,
            "xs_transform": args.xs_transform,
            "lgbm_select": args.lgbm_select,
            "history_columns_note": "第 0 折训练窗上选一次，全折通用（预注册）；"
                                    "scope=lgbm_selected 时限定在按 e 选出的 200 列内，"
                                    "以保证 history 列 ⊂ LGBM 选中列（推理端已有其统计量）",
            "ridge_alpha": args.ridge_alpha, "alpha_ladder": [float(a) for a in ALPHAS],
            "ridge_solver": "lsqr tol=1e-8 max_iter=2000（严格档，与 train.fit_model 一致）",
            "lgbm": {"rounds": args.lgbm_rounds, "seeds": 1, "spec": LGBM_SPEC,
                     "note": "实验档降规模（生产是 3 种子 × 480 轮），只用于判方向"},
        },
        "equivalence_verification": verification,
        "results": results,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }


def render_report(payload: dict[str, object]) -> str:
    cfg = payload["configuration"]
    lines = [
        "# A1：每资产历史特征 —— peak 口径复现", "",
        f"**问题**：{payload['question']}", "",
        f"**为什么重测**：{payload['why']}", "",
        "## 原实验（07-23）的结论与它的口径", "",
        f"- baseline {payload['prior']['baseline_mean']:.8f} → history {payload['prior']['history_mean']:.8f}"
        f"（**{payload['prior']['relative_gain']*100:+.2f}%**，3/3 折为正，报告写 `Accepted: True`）",
        f"- 口径：`{payload['prior']['metric']}` —— **不是尺度无关的 peak**",
        f"- {payload['prior']['note']}", "",
        "## 本次配置", "",
        f"- 折 {cfg['n_folds']} × train_window {cfg['train_window']:,}，embargo {cfg['embargo']}，"
        f"sample_modulo {cfg['sample_modulo']}",
        f"- 特征 {cfg['feature_count']} 列 + history {cfg['history_count']} 列 × 4 块，窗长 {cfg['history_window']}",
        f"- 求解器 {cfg['ridge_solver']}",
        f"- history 列：{cfg['history_columns_note']}", "",
    ]
    ver = payload.get("equivalence_verification")
    if ver:
        lines += ["## 等价性对拍（滞后缓存 ≡ AssetHistory）", "",
                  f"- 行数 {ver['rows']:,}",
                  f"- `previous` max|Δ| = {ver['max_abs_previous']:.3e}（要求逐位相同）",
                  f"- `difference` max|Δ| = {ver['max_abs_difference']:.3e}（要求逐位相同）",
                  f"- `rolling_mean` max|Δ| = {ver['max_abs_rolling_mean']:.3e}（要求 <1e-6）",
                  f"- `rolling_deviation` max|Δ| = {ver['max_abs_rolling_deviation']:.3e}（要求 <1e-6）", ""]

    for arm, entry in payload["comparisons"].items():
        peak, legacy = entry["peak"], entry["legacy_scale_0.5"]
        lines += [f"## {arm} 臂", "",
                  f"- baseline peak 折均 **{entry['baseline_peak_mean']:.8f}** → "
                  f"history **{entry['history_peak_mean']:.8f}**",
                  "", "| 口径 | 配对 Δ 均值 | 相对 | 正折 | 去掉最好一折 | 符号检验 p |",
                  "|---|---:|---:|---:|---:|---:|",
                  f"| **peak（判据口径）** | {peak['mean_delta']:+.3e} | **{peak['relative_gain']*100:+.2f}%** | "
                  f"{peak['positive_folds']}/{peak['n_folds']} | {peak['relative_gain_drop_best']*100:+.2f}% | "
                  f"{peak['sign_test_p']:.3f} |",
                  f"| 固定 scale 0.5（旧口径） | {legacy['mean_delta']:+.3e} | {legacy['relative_gain']*100:+.2f}% | "
                  f"{legacy['positive_folds']}/{legacy['n_folds']} | {legacy['relative_gain_drop_best']*100:+.2f}% | "
                  f"{legacy['sign_test_p']:.3f} |", "",
                  f"- 逐折 Δpeak：{['%+.3e' % v for v in peak['per_fold']]}",
                  f"- ΔA {entry['delta_A_relative']*100:+.2f}%，ΔB {entry['delta_B_relative']*100:+.2f}%，"
                  f"`2ΔA > ΔB` {'成立' if entry['mechanism_2dA_gt_dB'] else '**不成立**'}", ""]
        if "alpha_ladder" in entry:
            lines += ["### alpha 阶梯（必报诊断，非硬门槛）", "",
                      "| alpha | Δpeak 均值 | 相对 | 正折 |", "|---|---:|---:|---:|"]
            for key, v in entry["alpha_ladder"].items():
                lines.append(f"| {key} | {v['mean_delta']:+.3e} | {v['relative_gain']*100:+.2f}% | "
                             f"{v['positive_folds']} |")
            lines += ["", f"整条阶梯为正：{'✅' if entry['alpha_ladder_all_positive'] else '❌'}", ""]

    lines += ["## 判据（由 `verdict()` 判，不是报告里的评语）", ""]
    for arm, v in payload["verdicts"].items():
        lines.append(f"**{arm} 臂 —— {'✅ PASS' if v['pass'] else '❌ 不过'}**")
        lines += [f"- {'✅' if ok else '❌'} {name}" for name, ok in v["checks"].items()]
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
