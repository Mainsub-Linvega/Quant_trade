"""比赛评分指标 —— 全项目唯一实现。

Score = 1 − Σwᵢ(yᵢ−ŷᵢ)² / Σwᵢyᵢ²（加权零均值 R²）。全零预测得 0 分。
任何脚本要算分一律 import 这里，不得自行再写一份。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class ScaleInvariantScore(TypedDict):
    """未限幅预测沿全局尺度变化时的二次评分系数。"""

    A: float
    B: float
    peak: float
    optimal_scale: float
    score_at_unit_scale: float
    denominator: float


def _metric_arrays(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    weight64 = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    if target64.shape != prediction64.shape or target64.shape != weight64.shape:
        raise ValueError("target, prediction, and weight must have identical shapes")
    if target64.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if not (np.all(np.isfinite(target64)) and np.all(np.isfinite(prediction64))):
        raise ValueError("target and prediction must be finite")
    return target64, prediction64, weight64


def weighted_zero_mean_r2_from_sums(weighted_sse: float, weighted_target_energy: float) -> float:
    """由可流式累积的分子/分母计算比赛分数。"""
    if weighted_target_energy <= 0.0:
        return 0.0
    return float(1.0 - weighted_sse / weighted_target_energy)


def weighted_zero_mean_r2(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    target64, prediction64, weight64 = _metric_arrays(target, prediction, weight)
    denominator = float(np.dot(weight64, target64 * target64))
    weighted_sse = float(np.dot(weight64, (target64 - prediction64) ** 2))
    return weighted_zero_mean_r2_from_sums(weighted_sse, denominator)


def scale_invariant_score(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> ScaleInvariantScore:
    """返回 ``Score(a)=2aA-a²B`` 的系数和闭式最优值。

    ``prediction`` 必须是尚未乘发布尺度、也未限幅的预测。若 target 能量或预测能量为零，
    peak 定义为 0，optimal_scale 返回 NaN。权重口径与比赛指标一致：负权重按 0 处理。
    """
    target64, prediction64, weight64 = _metric_arrays(target, prediction, weight)
    denominator = float(np.dot(weight64, target64 * target64))
    if denominator <= 0.0:
        return {
            "A": 0.0,
            "B": 0.0,
            "peak": 0.0,
            "optimal_scale": float("nan"),
            "score_at_unit_scale": 0.0,
            "denominator": denominator,
        }

    a = float(np.dot(weight64, target64 * prediction64)) / denominator
    b = float(np.dot(weight64, prediction64 * prediction64)) / denominator
    peak = a * a / b if b > 0.0 else 0.0
    optimal_scale = a / b if b > 0.0 else float("nan")
    return {
        "A": a,
        "B": b,
        "peak": peak,
        "optimal_scale": optimal_scale,
        "score_at_unit_scale": 2.0 * a - b,
        "denominator": denominator,
    }
