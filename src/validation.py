"""walk-forward 切分。

partition_folds：按分区索引切（粗粒度，3 fold）。
rolling_time_folds：按 time_id 滚动细切（细粒度，8–12 fold + embargo + 可平移边界）。
"""

from __future__ import annotations

import numpy as np


def partition_folds(validation_partitions: list[int], window: int) -> list[tuple[list[int], int]]:
    """按分区索引生成 (train_indices, valid_index) 折序列 —— 当前 walk_forward 的切法。

    每折用 valid_index 前 window 个连续分区做训练，valid_index 分区做验证。
    """
    if window <= 0:
        raise ValueError("window must be positive")
    return [
        (list(range(valid_index - window, valid_index)), int(valid_index))
        for valid_index in sorted(validation_partitions)
    ]


def rolling_fold_chunk_size(
    n_time_ids: int,
    n_folds: int,
    train_window: int,
    embargo: int = 6,
    offset: int = 0,
) -> int:
    """每个验证段的长度（time_id 个数）—— rolling_time_folds 的切分算术，单独暴露。

    调用方用它算 P0-3 的「平移半个 fold」偏移量，不必自己重算一遍公式。
    """
    return (n_time_ids - train_window - embargo - offset) // n_folds


def rolling_time_folds(
    unique_time_ids: np.ndarray,
    n_folds: int,
    train_window: int,
    embargo: int = 6,
    offset: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 time_id 滚动细切多 fold，训练段与验证段之间空出 embargo 个 time_id。

    验证区域被均分为 n_folds 个不重叠的段。每个验证段的训练段是它前面
    train_window 个 time_id（滑动窗口），中间跳过 embargo 个 time_id。

    offset 把所有 fold 边界整体右移 offset 个 time_id（P0-3 噪声地板用：
    同一配置换一套边界再跑一遍，结果的漂移量就是检出下限）。offset=0 与
    不带该参数时逐位等价。

    返回 list[(train_time_ids, valid_time_ids)]。
    """
    ids = np.sort(unique_time_ids)
    n = len(ids)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    first_valid_idx = train_window + embargo + offset
    if first_valid_idx >= n:
        raise ValueError(
            f"train_window({train_window}) + embargo({embargo}) + offset({offset}) "
            f">= total time_ids({n})"
        )
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    chunk_size = rolling_fold_chunk_size(n, n_folds, train_window, embargo, offset)
    if chunk_size == 0:
        raise ValueError(f"not enough validation time_ids for {n_folds} folds")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        v_start = first_valid_idx + i * chunk_size
        v_end = first_valid_idx + (i + 1) * chunk_size if i < n_folds - 1 else n
        t_end = v_start - embargo
        t_start = max(0, t_end - train_window)
        folds.append((ids[t_start:t_end].copy(), ids[v_start:v_end].copy()))

    return folds
