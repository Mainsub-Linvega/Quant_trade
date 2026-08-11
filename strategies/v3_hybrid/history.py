"""每资产滚动历史特征 —— train.py 与 main.py 共用的唯一实现。

本文件必须自包含（只依赖 numpy）：提交包只含本目录，`main.py` 不允许 import
仓库其他位置的代码（zip 里没有 src/ 也没有 experiments/）。
`experiments/history_features.py` 那份**不能直接用** —— 它模块级 import
pandas/pyarrow，还往 sys.path 注 v1_ridge。

## 产出的 4 个块

对每个资产、按行序（= time_id 序）看它自己的历史：

    previous          = 上一次观测                       （无历史时 0）
    difference        = 当前 − previous
    rolling_mean      = 前 min(count, window) 次观测的均值 （无历史时 0）
    rolling_deviation = 当前 − rolling_mean

**严格滞后**：`rolling_mean` 只用 `t−window .. t−1`，绝不含当前行 —— 否则就是泄漏。

## ⚠️ 为什么不照抄 experiments 里那份的 cumsum 写法

那份用「整段 float64 cumsum 再取差」算 rolling_mean。离线整块调用时 cumsum 会累到
很大的量级，与在线逐 time_id 调用（每次只有 ≤6 行）的小 cumsum **不是逐位相同**的。
而这 4 个块要喂给 LightGBM —— 工程坑第 7 条记着：`cross_sectional_deviation` 的
1-ulp 差异曾被树的阶跃放大到 `max|Δpred| = 2.85e-03`。

本实现改成**对 ≤window 个滞后量做定序直接求和**（k = 1..count，float64 累加，
最后一次性 round 回 float32）。同一行的滞后值与求和顺序在两条路径下完全相同
⟹ **离线整块与在线逐 time_id 逐位一致**，`scripts/check_consistency.py` 才过得去。
"""

from __future__ import annotations

import numpy as np


class AssetHistory:
    """按 `asset_id` 维护最近 ``window_size`` 次观测。**有状态**，调用即推进。

    `transform` 一次可以吃任意多行（离线整块）或恰好一个 time_id 的那几行（在线）；
    两种调用方式给出逐位相同的结果。
    """

    def __init__(self, feature_count: int, window_size: int = 5):
        if feature_count <= 0 or window_size <= 0:
            raise ValueError("feature_count 与 window_size 必须为正")
        self.feature_count = int(feature_count)
        self.window_size = int(window_size)
        self.values: dict[int, np.ndarray] = {}

    def transform(self, current: np.ndarray, asset_ids: np.ndarray):
        """返回 (previous, difference, rolling_mean, rolling_deviation) 并推进状态。

        `current` 必须是**已经过 apply_robust_transform** 的那 `feature_count` 列。
        """
        current = np.asarray(current, dtype=np.float32)
        if current.ndim != 2 or current.shape[1] != self.feature_count:
            raise ValueError(f"current 形状应为 (n, {self.feature_count})，收到 {current.shape}")
        n = len(current)
        window = self.window_size

        # 逐行的滞后值：lags[i, j] = 第 i 行所属资产往前第 (j+1) 次观测
        lags = np.zeros((n, window, self.feature_count), dtype=np.float32)
        counts = np.zeros(n, dtype=np.int64)

        for asset in np.unique(asset_ids):
            index = np.flatnonzero(asset_ids == asset)
            buffer = self.values.get(int(asset))
            if buffer is None:
                buffer = np.empty((0, self.feature_count), dtype=np.float32)
            combined = np.vstack([buffer, current[index]])
            position = len(buffer) + np.arange(len(index))      # 当前行在 combined 里的下标
            for j in range(window):                             # j=0 → 滞后 1 期
                source = position - (j + 1)
                usable = source >= 0
                if usable.any():
                    lags[index[usable], j, :] = combined[source[usable]]
            counts[index] = np.minimum(position, window)
            self.values[int(asset)] = combined[-window:].astype(np.float32, copy=True)

        return self._blocks(current, lags, counts)

    def _blocks(self, current: np.ndarray, lags: np.ndarray, counts: np.ndarray):
        # ⚠️ float64 累加、最后一次性 round —— 在 float32 里累加会与整块路径差一个 ulp
        rolling_sum = np.zeros((len(current), self.feature_count), dtype=np.float64)
        previous = None
        for j in range(self.window_size):
            block = lags[:, j, :].copy()
            block[counts <= j] = 0.0        # 无该期历史 ⟹ 该期不参与，与「无历史即 0」一致
            if j == 0:
                previous = block
            rolling_sum += block
        count = counts.astype(np.float64)[:, None]
        rolling_mean = np.divide(rolling_sum, count,
                                 out=np.zeros_like(rolling_sum),
                                 where=count > 0).astype(np.float32)
        return previous, current - previous, rolling_mean, current - rolling_mean

    # -- 存档/复原（当前未启用；冷启动一律按「无历史即 0」，与训练端一致）
    def as_payload(self) -> dict[str, list[list[float]]]:
        return {str(asset): values.astype(float).tolist()
                for asset, values in sorted(self.values.items())}

    @classmethod
    def from_payload(cls, payload, feature_count: int, window_size: int) -> "AssetHistory":
        history = cls(feature_count=feature_count, window_size=window_size)
        history.values = {int(asset): np.asarray(values, dtype=np.float32)
                          for asset, values in payload.items()}
        return history


def history_design_blocks(transformed: np.ndarray, asset_ids: np.ndarray,
                          history_positions: np.ndarray, window_size: int,
                          history: AssetHistory | None = None):
    """便捷封装：从**已变换**的 LGBM 特征里取出 history 列，产出 4 个块。

    `history_positions` 是**在 `lgbm_features` 里的下标**（0..len-1），不是 323 列里的下标 ——
    这样推理端直接复用 `hybrid_meta.json` 里那 200 列的 lower/upper/center/scale，
    不必为 history 列另存一套统计量。
    """
    positions = np.asarray(history_positions, dtype=np.int64)
    history = history or AssetHistory(feature_count=len(positions), window_size=window_size)
    blocks = history.transform(transformed[:, positions], asset_ids)
    return blocks, history
