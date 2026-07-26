"""比赛评分指标 —— 全项目唯一实现。

Score = 1 − Σwᵢ(yᵢ−ŷᵢ)² / Σwᵢyᵢ²（加权零均值 R²）。全零预测得 0 分。
任何脚本要算分一律 import 这里，不得自行再写一份。
"""

from __future__ import annotations

import numpy as np


def weighted_zero_mean_r2(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    target64 = target.astype(np.float64)
    prediction64 = prediction.astype(np.float64)
    weight64 = np.maximum(weight.astype(np.float64), 0.0)
    denominator = float(np.dot(weight64, target64 * target64))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 - np.dot(weight64, (target64 - prediction64) ** 2) / denominator)
