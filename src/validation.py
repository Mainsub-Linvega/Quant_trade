"""walk-forward 切分。

partition_folds：按分区索引切（粗粒度，3 fold）。
rolling_time_folds：按 time_id 滚动细切（细粒度，8–12 fold + embargo）。
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


def rolling_time_folds(
    unique_time_ids: np.ndarray,
    n_folds: int,
    train_window: int,
    embargo: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 time_id 滚动细切多 fold，训练段与验证段之间空出 embargo 个 time_id。

    验证区域被均分为 n_folds 个不重叠的段。每个验证段的训练段是它前面
    train_window 个 time_id（滑动窗口），中间跳过 embargo 个 time_id。

    返回 list[(train_time_ids, valid_time_ids)]。
    """
    ids = np.sort(unique_time_ids)
    n = len(ids)
    first_valid_idx = train_window + embargo
    if first_valid_idx >= n:
        raise ValueError(
            f"train_window({train_window}) + embargo({embargo}) >= total time_ids({n})"
        )
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    valid_total = n - first_valid_idx
    chunk_size = valid_total // n_folds
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
