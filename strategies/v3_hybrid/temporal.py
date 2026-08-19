"""可部署的多尺度每资产时间状态（训练/实验/推理共用）。

所有统计都严格使用当前行之前的观测；``transform`` 返回后才推进状态。实现只依赖 NumPy，
可直接随提交包入包。整块调用与按 ``time_id`` 在线调用必须逐位一致。
"""

from __future__ import annotations

import numpy as np


BASELINE_KEYS = ("lag1", "difference", "mean5", "deviation5")
T1_EXTRA_KEYS = ("lag2", "lag5")
T2_EXTRA_KEYS = ("ema3", "ema10", "std5", "std20", "slope5", "slope20")
T3_EXTRA_KEYS = (*T1_EXTRA_KEYS, *T2_EXTRA_KEYS, "observation_gap")
# Family-isolated arms: unlike t2/t3, each asks one mechanism question.
F_LAG_KEYS = ("lag3", "lag10")
F_CHANGE_KEYS = ("delta3", "delta5", "delta10", "acceleration1")
F_VOLATILITY_KEYS = ("std5", "std20")
F_TREND_KEYS = ("slope5", "slope20")
# T5：`deviation5` 的**波动归一化**版本。baseline 已有未归一化的 deviation5，
# t2_state 已有裸 std5 —— 但两者的比值从没被测过，而 residual atlas 显示
# market_vol_quartile 桶间 model-vs-market delta 相差 4 倍。
T5_EXTRA_KEYS = ("zscore5",)
REGIME_KEYS = ("regime_current", "regime_lag1", "regime_difference")
ARMS = ("baseline", "t1_lags", "t2_state", "t3_full", "t4_regime", "t5_zscore",
        "f_lags", "f_changes", "f_volatility", "f_trend")
ALL_ASSET_KEYS = tuple(dict.fromkeys((*BASELINE_KEYS, *T1_EXTRA_KEYS, *T2_EXTRA_KEYS,
                                     *T5_EXTRA_KEYS, *F_LAG_KEYS, *F_CHANGE_KEYS)))

# std5 的下限。特征已过 robust transform（尺度约为 1），1e-3 只挡住真正恒定的历史，
# 避免除以接近 0 的标准差把 z-score 炸开。
ZSCORE_STD_FLOOR = np.float32(1e-3)


def zscore_from(deviation: np.ndarray, std: np.ndarray) -> np.ndarray:
    """`deviation5 / max(std5, floor)` —— 在线与离线两条路径**共用这一份实现**。"""
    return (deviation / np.maximum(std, ZSCORE_STD_FLOOR)).astype(np.float32)


class MultiScaleAssetHistory:
    """维护每个资产最多 20 次观测及严格滞后的 EMA 状态。"""

    def __init__(self, feature_count: int, max_window: int = 20):
        if feature_count <= 0 or max_window < 20:
            raise ValueError("feature_count 必须为正，max_window 至少为 20")
        self.feature_count = int(feature_count)
        self.max_window = int(max_window)
        self.values: dict[int, np.ndarray] = {}
        self.last_time: dict[int, int] = {}

    @staticmethod
    def _mean_std_slope(history: np.ndarray, window: int, width: int):
        used = history[-window:].astype(np.float64, copy=False)
        if not len(used):
            zero = np.zeros(width, dtype=np.float32)
            return zero, zero.copy(), zero.copy()
        mean = used.mean(axis=0, dtype=np.float64)
        centred = used - mean
        std = np.sqrt(np.maximum((centred * centred).mean(axis=0), 0.0))
        if len(used) < 2:
            slope = np.zeros(width, dtype=np.float64)
        else:
            x = np.arange(len(used), dtype=np.float64)
            x -= x.mean()
            slope = (x[:, None] * centred).sum(axis=0) / float(np.dot(x, x))
        return mean.astype(np.float32), std.astype(np.float32), slope.astype(np.float32)

    @staticmethod
    def _ewm(history: np.ndarray, span: int, width: int) -> np.ndarray:
        used = history[-20:].astype(np.float64, copy=False)
        if not len(used):
            return np.zeros(width, dtype=np.float32)
        alpha = 2.0 / (span + 1.0)
        # oldest -> newest；越新的滞后权重越大。固定最多 20 期，便于从原始 lag cache 精确重建。
        weights = (1.0 - alpha) ** np.arange(len(used) - 1, -1, -1, dtype=np.float64)
        weights /= weights.sum()
        return (weights @ used).astype(np.float32)

    def transform(self, current: np.ndarray, asset_ids: np.ndarray, time_ids: np.ndarray):
        current = np.asarray(current, dtype=np.float32)
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        time_ids = np.asarray(time_ids, dtype=np.int64)
        if current.ndim != 2 or current.shape[1] != self.feature_count:
            raise ValueError(f"current 形状应为 (n, {self.feature_count})，收到 {current.shape}")
        if len(current) != len(asset_ids) or len(current) != len(time_ids):
            raise ValueError("current/asset_ids/time_ids 行数必须一致")
        n, width = current.shape
        out = {key: np.zeros((n, width), dtype=np.float32) for key in ALL_ASSET_KEYS}
        gap = np.zeros((n, 1), dtype=np.float32)

        # 必须逐行推进：同一批可含多个 time_id，同一资产后面的行要看见前面的行。
        for row in range(n):
            asset = int(asset_ids[row])
            time_id = int(time_ids[row])
            value = current[row]
            history = self.values.get(asset)
            if history is None:
                history = np.empty((0, width), dtype=np.float32)

            count = len(history)
            if count:
                out["lag1"][row] = history[-1]
            if count >= 2:
                out["lag2"][row] = history[-2]
            if count >= 3:
                out["lag3"][row] = history[-3]
            if count >= 5:
                out["lag5"][row] = history[-5]
            if count >= 10:
                out["lag10"][row] = history[-10]
            out["difference"][row] = value - out["lag1"][row]
            out["delta3"][row] = value - out["lag3"][row]
            out["delta5"][row] = value - out["lag5"][row]
            out["delta10"][row] = value - out["lag10"][row]
            out["acceleration1"][row] = value - 2.0 * out["lag1"][row] + out["lag2"][row]

            mean5, std5, slope5 = self._mean_std_slope(history, 5, width)
            _, std20, slope20 = self._mean_std_slope(history, 20, width)
            out["mean5"][row] = mean5
            out["deviation5"][row] = value - mean5
            out["zscore5"][row] = zscore_from(out["deviation5"][row], std5)
            out["std5"][row] = std5
            out["std20"][row] = std20
            out["slope5"][row] = slope5
            out["slope20"][row] = slope20
            out["ema3"][row] = self._ewm(history, 3, width)
            out["ema10"][row] = self._ewm(history, 10, width)
            if asset in self.last_time:
                delta = time_id - self.last_time[asset]
                if delta <= 0:
                    raise ValueError("同一资产的 time_id 必须严格递增")
                gap[row, 0] = float(delta)

            # 当前行不进入上面的任何输出；到这里才推进。
            self.values[asset] = np.vstack([history, value[None, :]])[-self.max_window:].astype(
                np.float32, copy=True)
            self.last_time[asset] = time_id

        out["observation_gap"] = gap
        return out


class MarketRegimeHistory:
    """把完整截面压成固定 10 组 × (均值, 标准差) 的 20 维市场状态。"""

    def __init__(self, feature_count: int, group_count: int = 10):
        if feature_count <= 0 or group_count <= 0 or feature_count % group_count != 0:
            raise ValueError("feature_count 必须能被 group_count 整除")
        self.feature_count = int(feature_count)
        self.group_count = int(group_count)
        self.previous: np.ndarray | None = None

    def transform(self, current: np.ndarray, time_ids: np.ndarray) -> dict[str, np.ndarray]:
        current = np.asarray(current, dtype=np.float32)
        time_ids = np.asarray(time_ids, dtype=np.int64)
        if current.ndim != 2 or current.shape[1] != self.feature_count or len(current) != len(time_ids):
            raise ValueError("current/time_ids 形状不匹配")
        width = self.group_count * 2
        output = {key: np.zeros((len(current), width), dtype=np.float32) for key in REGIME_KEYS}
        starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
        stops = np.r_[starts[1:], len(time_ids)]
        group_width = self.feature_count // self.group_count
        for start, stop in zip(starts, stops):
            block = current[start:stop].astype(np.float64)
            mean = block.mean(axis=0).reshape(self.group_count, group_width).mean(axis=1)
            std = block.std(axis=0).reshape(self.group_count, group_width).mean(axis=1)
            state = np.r_[mean, std].astype(np.float32)
            previous = np.zeros_like(state) if self.previous is None else self.previous
            output["regime_current"][start:stop] = state
            output["regime_lag1"][start:stop] = previous
            output["regime_difference"][start:stop] = state - previous
            self.previous = state.copy()
        return output


def temporal_atoms_from_lags(current: np.ndarray, lags: np.ndarray, counts: np.ndarray,
                             observation_gap: np.ndarray) -> dict[str, np.ndarray]:
    """从**已变换**的 lag cache 重建与 ``MultiScaleAssetHistory`` 完全相同的原子块。

    ``lags[:, j]`` 是往前第 ``j+1`` 次观测；无效槽位值任意，必须由 ``counts`` 掩掉。
    """
    current = np.asarray(current, dtype=np.float32)
    lags = np.asarray(lags, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    n, window, width = lags.shape
    if window < 20 or current.shape != (n, width):
        raise ValueError("lags 至少需要 20 期且 current/lag 形状必须一致")
    atoms = {key: np.zeros((n, width), dtype=np.float32) for key in ALL_ASSET_KEYS}
    for key, index in (("lag1", 0), ("lag2", 1), ("lag3", 2), ("lag5", 4), ("lag10", 9)):
        valid = counts > index
        atoms[key][valid] = lags[valid, index]
    atoms["difference"] = current - atoms["lag1"]
    atoms["delta3"] = current - atoms["lag3"]
    atoms["delta5"] = current - atoms["lag5"]
    atoms["delta10"] = current - atoms["lag10"]
    atoms["acceleration1"] = current - 2.0 * atoms["lag1"] + atoms["lag2"]

    def statistics(max_count: int):
        means = np.zeros((n, width), dtype=np.float32)
        stds = np.zeros((n, width), dtype=np.float32)
        slopes = np.zeros((n, width), dtype=np.float32)
        for count in range(1, max_count + 1):
            rows = np.flatnonzero(np.minimum(counts, max_count) == count)
            if not len(rows):
                continue
            # lags 是 newest -> oldest；翻转成 oldest -> newest 与在线实现一致。
            values = lags[rows, :count, :][:, ::-1, :].astype(np.float64)
            mean = values.mean(axis=1)
            centred = values - mean[:, None, :]
            means[rows] = mean.astype(np.float32)
            stds[rows] = np.sqrt(np.maximum((centred * centred).mean(axis=1), 0.0)).astype(np.float32)
            if count >= 2:
                x = np.arange(count, dtype=np.float64); x -= x.mean()
                slopes[rows] = ((centred * x[None, :, None]).sum(axis=1)
                                / float(np.dot(x, x))).astype(np.float32)
        return means, stds, slopes

    mean5, std5, slope5 = statistics(5)
    _, std20, slope20 = statistics(20)
    atoms["mean5"] = mean5
    atoms["deviation5"] = current - mean5
    atoms["zscore5"] = zscore_from(atoms["deviation5"], std5)
    atoms["std5"], atoms["std20"] = std5, std20
    atoms["slope5"], atoms["slope20"] = slope5, slope20

    for key, span in (("ema3", 3), ("ema10", 10)):
        for count in range(1, 21):
            rows = np.flatnonzero(np.minimum(counts, 20) == count)
            if not len(rows):
                continue
            values = lags[rows, :count, :][:, ::-1, :].astype(np.float64)
            alpha = 2.0 / (span + 1.0)
            weight = (1.0 - alpha) ** np.arange(count - 1, -1, -1, dtype=np.float64)
            weight /= weight.sum()
            atoms[key][rows] = np.einsum("t,ntf->nf", weight, values).astype(np.float32)
    gap = np.asarray(observation_gap, dtype=np.float32).reshape(n, 1)
    atoms["observation_gap"] = gap
    return atoms


def temporal_arm_blocks(atoms: dict[str, np.ndarray], arm: str) -> tuple[np.ndarray, ...]:
    """固定列序组装预注册臂；所有候选都保留已验证的 v3 baseline history。"""
    if arm not in ARMS:
        raise ValueError(f"未知 temporal arm: {arm}")
    keys = list(BASELINE_KEYS)
    if arm == "t1_lags":
        keys.extend(T1_EXTRA_KEYS)
    elif arm == "t2_state":
        keys.extend(T2_EXTRA_KEYS)
    elif arm == "t3_full":
        keys.extend(T3_EXTRA_KEYS)
    elif arm == "t4_regime":
        keys.extend(REGIME_KEYS)
    elif arm == "t5_zscore":
        keys.extend(T5_EXTRA_KEYS)
    elif arm == "f_lags":
        keys.extend(F_LAG_KEYS)
    elif arm == "f_changes":
        keys.extend(F_CHANGE_KEYS)
    elif arm == "f_volatility":
        keys.extend(F_VOLATILITY_KEYS)
    elif arm == "f_trend":
        keys.extend(F_TREND_KEYS)
    return tuple(atoms[key] for key in keys)


def temporal_arm_width(feature_count: int, arm: str) -> int:
    return int(sum(block.shape[1] for block in temporal_arm_blocks(
        {key: np.empty((0, (1 if key == "observation_gap" else
                                  20 if key in REGIME_KEYS else feature_count)), dtype=np.float32)
         for key in (*ALL_ASSET_KEYS, "observation_gap", *REGIME_KEYS)}, arm)))
