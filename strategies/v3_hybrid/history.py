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
        # 在线快路径的状态（见 transform_online）。与 `values` **互斥**，混用会被拒。
        self._ring: np.ndarray | None = None
        self._seen: np.ndarray | None = None

    # ------------------------------------------------------------------ 在线快路径

    def _ensure_slots(self, n_slots: int) -> None:
        """按 asset_id 开定槽位缓冲；出现更大的 asset_id 就补零扩容。"""
        if self._ring is not None and len(self._ring) >= n_slots:
            return
        ring = np.zeros((n_slots, self.window_size, self.feature_count), dtype=np.float32)
        seen = np.zeros(n_slots, dtype=np.int64)
        if self._ring is not None:
            ring[: len(self._ring)] = self._ring
            seen[: len(self._seen)] = self._seen
        self._ring, self._seen = ring, seen

    def transform_online(self, current: np.ndarray, asset_ids: np.ndarray):
        """与 `transform` **逐位相同**的在线版本：一次恰好一个 time_id。

        为什么要另开一条：`transform` 每次都 `np.unique(asset_ids)` 再逐资产
        `flatnonzero` / `vstack`，一次调用约上百个小 numpy 操作。离线整块只调几次无所谓，
        在线要调 21.4 万次 —— 实测 **0.338 ms/次，占整次 `predict` 的 16%**，
        比 1440 棵树的一半还贵。定槽位数组版实测 **0.019 ms**。

        逐位相同的依据（不是「应该一样」，是构造上一样）：

        1. 两条路径**共用 `_blocks`** —— float64 累加顺序、`counts<=j` 置零、
           最后一次性 round 回 float32，全部是同一段代码；
        2. 差别只在 `lags` 怎么取。在线每个 asset 至多一行 ⟹
           `transform` 里 `position = len(buffer)`、`lags[j] = buffer[position-1-j]`，
           恰好就是「最近一次在前」的环形缓冲第 j 格；`counts = min(已见次数, window)` 同理；
        3. 存进去的都是**原样的 float32 值**，没有任何算术。

        前提（官方 runner 满足）：本批内 asset_id 不重复。不满足直接抛，绝不静默算错。
        """
        current = np.asarray(current, dtype=np.float32)
        if current.ndim != 2 or current.shape[1] != self.feature_count:
            raise ValueError(f"current 形状应为 (n, {self.feature_count})，收到 {current.shape}")
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        if len(asset_ids) != len(current):
            raise ValueError("asset_ids 与 current 行数不一致")
        if self.values:
            raise RuntimeError("同一个 AssetHistory 不能混用 transform() 与 transform_online()")
        if not len(asset_ids):
            raise ValueError("空批次")
        if asset_ids.min() < 0:
            raise ValueError("asset_id 不能为负")

        self._ensure_slots(int(asset_ids.max()) + 1)
        if len(asset_ids) > 1 and np.bincount(asset_ids, minlength=len(self._ring)).max() > 1:
            raise ValueError("同一批内出现重复 asset_id —— 在线快路径要求一次一个 time_id")

        lags = self._ring[asset_ids]                       # 高级索引 ⟹ 已是拷贝，随后写 ring 不会串
        counts = np.minimum(self._seen[asset_ids], self.window_size)
        if self.window_size > 1:                           # 整体右移一格（RHS 是拷贝，无别名问题）
            self._ring[asset_ids, 1:] = self._ring[asset_ids, :-1]
        self._ring[asset_ids, 0] = current
        self._seen[asset_ids] += 1
        return self._blocks(current, lags, counts)

    def transform(self, current: np.ndarray, asset_ids: np.ndarray):
        """返回 (previous, difference, rolling_mean, rolling_deviation) 并推进状态。

        `current` 必须是**已经过 apply_robust_transform** 的那 `feature_count` 列。

        ⚠️ 这是**离线整块**路径（一次吃任意多行）。在线逐 time_id 请用
        `transform_online` —— 逐位相同但快 18 倍。
        """
        current = np.asarray(current, dtype=np.float32)
        if self._ring is not None:
            raise RuntimeError("同一个 AssetHistory 不能混用 transform() 与 transform_online()")
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


class AssetLongWindow:
    """逐 asset 的**长窗**滚动均值与偏离。O(1)/行，离线整块与在线逐 time_id **逐位相同**。

    ## 为什么不能直接把 AssetHistory 的 window_size 调大

    `AssetHistory` 是 O(window)：每行要对 window 个滞后量**定序求和**，在线还要把
    `(window, feature_count)` 的环形缓冲整体右移一格。window=512 时每个 time_id 要搬
    512 × 40 × 15 ≈ 307K 个元素、共 21.4 万次 —— 跑不动。

    ## 改用「持久累积和相减」

        mean[t] = (cum[n] − cum[max(n−W, 0)]) / min(n, W)

    与 `main.PredictionTrail` 同一套路。逐位一致的依据是**构造上一样**，不是「应该一样」：

    1. 离线用 `np.cumsum(..., dtype=np.float64)`，它本身就是**定序累加**；
    2. 在线维护**持久** running total，逐行 `running = running + current`；
    3. 实测两者逐位相同（`np.array_equal` 为 True、max|Δ| = 0）。

    ⚠️ **分块重起**的 cumsum 与整段 cumsum **不**逐位相同 —— 那正是本文件开头
    警告的写法。持久 running total 才是整段 cumsum 的正确在线对偶；离线批处理时
    也必须把上一批的 `running` 作为 cumsum 的**第一行**接上，不能算完再加偏移
    （`r + (s0+s1)` 与 `(r+s0)+s1` 的舍入不同）。

    边界与 `AssetHistory` 一致：无历史（n=0）⟹ mean = 0 ⟹ deviation = current。

    ⚠️ **冷启动**：推理端从空状态起步，测试期前 W 个观测的窗口比训练期短。
    W=512 时这影响每个 asset 的前 512 次观测（公榜 217,440 个 time_id 里的 0.24%）。
    与训练端「无历史即 0」的语义一致，不额外补状态。
    """

    __slots__ = ("window", "feature_count", "_ring", "_running", "_seen")

    def __init__(self, feature_count: int, window: int):
        if feature_count <= 0 or window <= 0:
            raise ValueError("feature_count 与 window 必须为正")
        self.feature_count = int(feature_count)
        self.window = int(window)
        self._ring: np.ndarray | None = None      # (slots, window+1, F)  ring[k % (W+1)] = cum[k]
        self._running: np.ndarray | None = None   # (slots, F)            = cum[n]
        self._seen: np.ndarray | None = None      # (slots,)              = n

    def _ensure_slots(self, n_slots: int) -> None:
        if self._ring is not None and len(self._ring) >= n_slots:
            return
        ring = np.zeros((n_slots, self.window + 1, self.feature_count), dtype=np.float64)
        running = np.zeros((n_slots, self.feature_count), dtype=np.float64)
        seen = np.zeros(n_slots, dtype=np.int64)
        if self._ring is not None:
            ring[: len(self._ring)] = self._ring
            running[: len(self._running)] = self._running
            seen[: len(self._seen)] = self._seen
        self._ring, self._running, self._seen = ring, running, seen

    def _emit(self, current: np.ndarray, cum_n: np.ndarray, cum_left: np.ndarray,
              denom: np.ndarray):
        """(rolling_mean, current − rolling_mean)。float64 里算完一次性 round 回 float32。"""
        mean = np.divide(cum_n - cum_left, denom,
                         out=np.zeros_like(cum_n), where=denom > 0).astype(np.float32)
        return mean, current - mean

    def transform_online(self, current: np.ndarray, asset_ids: np.ndarray):
        """一次恰好一个 time_id。与 `transform` 逐位相同，但全程向量化、无 Python 循环。

        前提（官方 runner 满足）：本批内 asset_id 不重复。不满足直接抛，绝不静默算错。
        """
        current = np.asarray(current, dtype=np.float32)
        if current.ndim != 2 or current.shape[1] != self.feature_count:
            raise ValueError(f"current 形状应为 (n, {self.feature_count})，收到 {current.shape}")
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        if len(asset_ids) != len(current):
            raise ValueError("asset_ids 与 current 行数不一致")
        if not len(asset_ids):
            raise ValueError("空批次")
        if asset_ids.min() < 0:
            raise ValueError("asset_id 不能为负")
        self._ensure_slots(int(asset_ids.max()) + 1)
        if len(asset_ids) > 1 and np.bincount(asset_ids, minlength=len(self._ring)).max() > 1:
            raise ValueError("同一批内出现重复 asset_id —— 在线快路径要求一次一个 time_id")

        seen = self._seen[asset_ids]
        left = np.maximum(seen - self.window, 0)
        cum_n = self._running[asset_ids]                          # 高级索引 ⟹ 已是拷贝
        cum_left = self._ring[asset_ids, left % (self.window + 1)]
        blocks = self._emit(current, cum_n, cum_left,
                            (seen - left).astype(np.float64)[:, None])
        self._ring[asset_ids, seen % (self.window + 1)] = cum_n   # 先存 cum[n]
        self._running[asset_ids] = cum_n + current                # 再推进到 cum[n+1]
        self._seen[asset_ids] += 1
        return blocks

    def transform(self, current: np.ndarray, asset_ids: np.ndarray):
        """离线整块（一次吃任意多行，行内按 time_id 升序）。与 `transform_online` 逐位相同。"""
        current = np.asarray(current, dtype=np.float32)
        if current.ndim != 2 or current.shape[1] != self.feature_count:
            raise ValueError(f"current 形状应为 (n, {self.feature_count})，收到 {current.shape}")
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        if len(asset_ids) != len(current):
            raise ValueError("asset_ids 与 current 行数不一致")
        if not len(asset_ids):
            raise ValueError("空批次")
        self._ensure_slots(int(asset_ids.max()) + 1)
        width, span = self.feature_count, self.window + 1
        mean = np.zeros((len(current), width), dtype=np.float32)

        for asset in np.unique(asset_ids):
            index = np.flatnonzero(asset_ids == asset)
            start = int(self._seen[asset])
            # ⚠️ 把上一批的 running 作为 cumsum 的**第一行**接上 —— 不能算完再加偏移。
            forward = np.cumsum(
                np.vstack([self._running[asset][None, :], current[index]]),
                axis=0, dtype=np.float64)                      # forward[i] = cum[start + i]
            base = max(start - self.window, 0)
            if start > base:                                   # 本批之前的 cum，取自环
                past = self._ring[asset][np.arange(base, start) % span]
                lookup = np.vstack([past, forward])
            else:
                lookup = forward
            positions = start + np.arange(len(index))
            cum_n = lookup[positions - base]
            left = np.maximum(positions - self.window, 0)
            cum_left = lookup[left - base]
            block_mean, _ = self._emit(current[index], cum_n, cum_left,
                                       (positions - left).astype(np.float64)[:, None])
            mean[index] = block_mean
            # 落环：本批产生的 cum[start .. start+T]，只需保留最后 span 个
            keep = np.arange(max(start, positions[-1] + 1 - self.window), positions[-1] + 1)
            self._ring[asset][keep % span] = lookup[keep - base]
            self._running[asset] = forward[-1]
            self._seen[asset] = positions[-1] + 1
        return mean, current - mean


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
