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


def cross_sectional_deviation(features: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    """多 time_id 数组的截面去均值（离线训练 / 验证路径）。

    依赖行已按 time_id 排序（分区文件天然满足）。
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
    return deviation


def single_time_deviation(features: np.ndarray) -> np.ndarray:
    """单个 time_id 批次的截面去均值（Time-Series API 在线推理路径）。

    官方 runner 每次 predict 恰好喂一个 time_id 的全部行，直接对整批求均值。
    与 cross_sectional_deviation 数学等价；浮点上不保证逐位相同（求和顺序不同），
    公榜 0.00119088 的提交走的是本路径，勿改。
    """
    return features - features.mean(axis=0, keepdims=True)


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
