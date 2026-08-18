"""v1_ridge 的特征预处理与线性推理 —— train.py 与 main.py 共用的唯一实现。

本文件必须自包含（只依赖 numpy）：提交包只含本目录，main.py 不允许 import
仓库其他位置的代码（提交 zip 里没有 src/）。任何预处理 / 推理口径的改动都
只能改这里，训练与推理两侧自动同步。

注意存在两条截面去均值路径（见各自 docstring），它们数学等价但浮点求和顺序
不同，结果可能差最后一个 ulp；scripts/check_consistency.py 负责断言两条路径
在 1e-6 内一致。为了保持已提交公榜版本（0.00119088）的逐位可复现性，
不要合并这两条路径。
"""

from __future__ import annotations

import numpy as np


_ROBUST_STAT_NAMES = ("lower", "upper", "center", "scale")


def resolve_feature_contract(meta: dict[str, object]) -> dict[str, list[object]]:
    """Resolve separate XS/market feature metadata with an old-model fallback."""
    xs_features = list(meta["lgbm_features"])
    contract: dict[str, list[object]] = {"xs_features": xs_features}
    for name in _ROBUST_STAT_NAMES:
        values = list(meta[name])
        if len(values) != len(xs_features):
            raise ValueError(f"{name} length does not match lgbm_features")
        contract[f"xs_{name}"] = values

    optional_keys = ["market_features", *[f"market_{name}" for name in _ROBUST_STAT_NAMES]]
    present = [key in meta for key in optional_keys]
    if any(present) and not all(present):
        missing = [key for key, exists in zip(optional_keys, present) if not exists]
        raise ValueError(f"incomplete market feature metadata: {missing}")

    market_features = (list(meta["market_features"])
                       if all(present) else list(xs_features))
    contract["market_features"] = market_features
    for name in _ROBUST_STAT_NAMES:
        values = (list(meta[f"market_{name}"])
                  if all(present) else list(contract[f"xs_{name}"]))
        if len(values) != len(market_features):
            raise ValueError(f"market_{name} length does not match market_features")
        contract[f"market_{name}"] = values
    return contract


def map_history_positions(
    xs_indices: list[int] | np.ndarray,
    history_indices: list[int] | np.ndarray,
) -> list[int]:
    """Map manifest-global history indices into the selected XS feature order."""
    xs = [int(index) for index in xs_indices]
    history = [int(index) for index in history_indices]
    if len(set(xs)) != len(xs) or len(set(history)) != len(history):
        raise ValueError("XS and history feature indices must be unique")
    positions = {index: position for position, index in enumerate(xs)}
    missing = [index for index in history if index not in positions]
    if missing:
        raise ValueError(f"history features must be selected by XS: {missing}")
    return [positions[index] for index in history]


def selection_sets_from_manifest(
    manifest: dict[str, object], n_features: int
) -> dict[str, list[int]]:
    """Validate and resolve the four global feature sets from a selection manifest."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")

    def indices_for(task: str, *, allow_empty: bool = False) -> list[int]:
        section = manifest.get(task)
        if not isinstance(section, dict) or section.get("selected_indices") is None:
            raise ValueError(f"selection manifest is missing {task}.selected_indices")
        indices = [int(index) for index in section["selected_indices"]]
        if not allow_empty and not indices:
            raise ValueError(f"selection manifest has an empty {task} feature set")
        if len(set(indices)) != len(indices):
            raise ValueError(f"selection manifest has duplicate {task} indices")
        if any(index < 0 or index >= n_features for index in indices):
            raise ValueError(f"selection manifest has out-of-range {task} indices")
        return indices

    ridge = indices_for("ridge")
    xs = indices_for("xs")
    market = indices_for("market")
    history = indices_for("history", allow_empty=True)
    return {
        "ridge": ridge,
        "xs": xs,
        "market": market,
        "history": history,
        "history_positions": map_history_positions(xs, history),
    }


def apply_robust_transform(
    features: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """套用训练期拟合好的稳健标准化统计量（就地修改并返回）。

    步骤：NaN/inf→0 → 按训练期分位数裁剪 → 减中位数 → 除 IQR → 裁到 ±10。
    统计量一律来自训练期（robust_transform_fit），本函数绝不重新估计。
    """
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(features, lower, upper, out=features)
    features -= center
    features /= scale
    np.clip(features, -10.0, 10.0, out=features)
    return features


# 截面标准差归一化的护栏。sd 下限防 0/0；结果裁到 ±10 与 apply_robust_transform 的
# 最后一步同口径 —— 没有上界的话，某个 time_id 上分散度塌缩会让输入冲到极大值，
# 再强的正则也压不住（这个教训来自 experiments/mt_lagged.py 的第一版）。
CROSS_SECTIONAL_SD_FLOOR = np.float32(1e-2)
CROSS_SECTIONAL_CLIP = np.float32(10.0)


def cross_sectional_deviation(
    features: np.ndarray, time_ids: np.ndarray, scaling: str = "none"
) -> np.ndarray:
    """多 time_id 数组的截面去均值（离线训练 / 验证路径）。

    依赖行已按 time_id 排序（分区文件天然满足）。

    scaling="std" 时再除以每个 time_id 内的截面标准差（总体标准差，与均值同口径），
    把时变的截面离散度归一化掉。scaling="none" 是历史行为，逐位不变。
    """
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    means = np.add.reduceat(features, starts, axis=0) / counts[:, None]
    deviation = features.copy()
    # Chunk the expansion so the temporary repeated-mean matrix stays small.
    for group_start in range(0, len(starts), 20_000):
        group_stop = min(group_start + 20_000, len(starts))
        row_start = int(starts[group_start])
        row_stop = int(starts[group_stop]) if group_stop < len(starts) else len(features)
        deviation[row_start:row_stop] -= np.repeat(
            means[group_start:group_stop], counts[group_start:group_stop], axis=0
        )
    if scaling == "none":
        return deviation
    if scaling != "std":
        raise ValueError(f"unknown cross-sectional scaling: {scaling}")

    variance = np.add.reduceat(deviation * deviation, starts, axis=0) / counts[:, None]
    sd = np.maximum(np.sqrt(variance), CROSS_SECTIONAL_SD_FLOOR)
    for group_start in range(0, len(starts), 20_000):
        group_stop = min(group_start + 20_000, len(starts))
        row_start = int(starts[group_start])
        row_stop = int(starts[group_stop]) if group_stop < len(starts) else len(features)
        deviation[row_start:row_stop] /= np.repeat(
            sd[group_start:group_stop], counts[group_start:group_stop], axis=0
        )
    np.clip(deviation, -CROSS_SECTIONAL_CLIP, CROSS_SECTIONAL_CLIP, out=deviation)
    return deviation


def single_time_deviation(features: np.ndarray, scaling: str = "none") -> np.ndarray:
    """单个 time_id 批次的截面去均值（Time-Series API 在线推理路径）。

    官方 runner 每次 predict 恰好喂一个 time_id 的全部行，直接对整批求均值。
    与 cross_sectional_deviation 数学等价；浮点上不保证逐位相同（求和顺序不同），
    公榜 0.00119088 的提交走的是 scaling="none" 这条路径，勿改。

    scaling="std" 必须与 cross_sectional_deviation 的同名分支保持一致 ——
    两侧口径由 scripts/check_consistency.py 断言。
    """
    deviation = features - features.mean(axis=0, keepdims=True)
    if scaling == "none":
        return deviation
    if scaling != "std":
        raise ValueError(f"unknown cross-sectional scaling: {scaling}")
    sd = np.maximum(np.sqrt((deviation * deviation).mean(axis=0, keepdims=True)),
                    CROSS_SECTIONAL_SD_FLOOR)
    return np.clip(deviation / sd, -CROSS_SECTIONAL_CLIP, CROSS_SECTIONAL_CLIP)


def linear_predict(
    raw: np.ndarray,
    deviation: np.ndarray,
    intercept,
    coef: np.ndarray,
    prediction_scale,
    prediction_clip,
) -> np.ndarray:
    """设计矩阵 [raw ‖ deviation] 的线性预测 + 保守缩放 + 限幅。

    dtype 跟随调用方：main.py 传 float32 标量（在线路径全程 float32）；
    train.py 传 python float 截距（离线路径提升到 float64）。两侧历史数值
    均由此保持不变。
    """
    design = np.column_stack([raw, deviation])
    prediction = (intercept + design @ coef) * prediction_scale
    return np.clip(prediction, -prediction_clip, prediction_clip)


def asset_scaled_zero_mean(
    values: np.ndarray,
    asset_ids: np.ndarray,
    scales: np.ndarray,
    time_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Apply per-asset scales and project back to an unweighted zero-mean cross section.

    ``time_ids=None`` is the online one-time-id path. Passing ordered ``time_ids`` handles an
    offline block containing multiple complete cross sections. The returned array is float64
    because LightGBM ensemble predictions are accumulated in float64.
    """
    values = np.asarray(values, dtype=np.float64)
    asset_ids = np.asarray(asset_ids, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float64)
    if values.ndim != 1 or asset_ids.ndim != 1 or values.shape != asset_ids.shape:
        raise ValueError("values and asset_ids must be matching 1D arrays")
    if scales.ndim != 1 or len(scales) == 0 or not np.all(np.isfinite(scales)):
        raise ValueError("scales must be a finite non-empty 1D array")
    if len(asset_ids) == 0:
        raise ValueError("empty cross section")
    if asset_ids.min() < 0 or asset_ids.max() >= len(scales):
        raise ValueError("asset_id is outside scales")

    adjusted = values * scales[asset_ids]
    if time_ids is None:
        adjusted -= adjusted.mean()
        return adjusted

    time_ids = np.asarray(time_ids, dtype=np.int64)
    if time_ids.shape != values.shape or np.any(np.diff(time_ids) < 0):
        raise ValueError("time_ids must be a sorted 1D array matching values")
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    adjusted -= np.repeat(np.add.reduceat(adjusted, starts) / counts, counts)
    return adjusted
