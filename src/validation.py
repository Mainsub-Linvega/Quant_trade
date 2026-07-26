"""walk-forward 切分。

当前只有按分区切的 partition_folds（与既有 walk_forward 实验行为完全一致）。
P0（HANDOFF §5：按 time_id 细切 8–12 个 fold + embargo + 配对 delta + 噪声地板）
将在这里实现 —— 那是统计判断密集的工作，实现前先与人确认方案。
"""

from __future__ import annotations


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


def rolling_time_folds(unique_time_ids, n_folds: int, train_window: int, embargo: int = 0):
    """P0 占位：按 time_id 滚动细切多 fold，训练段与验证段之间空出 embargo 个 time_id。

    embargo 大小取决于 target 窗口重叠长度 H（HANDOFF §4① 的自相关衰减诊断），
    先跑诊断定 H，再实现本函数。
    """
    raise NotImplementedError("P0：见 HANDOFF_20260724 §5 与 ROADMAP.md")
