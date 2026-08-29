"""v3_hybrid 推理入口：生产岭回归 + LightGBM 截面分量五五混合。

## 这个模型在干什么

任何一组预测都能按 time_id 拆成两块（15 个资产，加起来精确等于原值）：

    ŷᵢ = m̂ + (ŷᵢ − m̂)         m̂ = 该 time_id 内 15 个预测的**无权**均值
         ↑市场分量  ↑截面分量

实测（`lgbm_mt` / `lgbm_mt_v2` 两轮）：**市场分量那块树打不过线性**，
所以 `m̂` 原样保留生产岭回归的；而截面分量那块（`lgbm_xs` / `lgbm_blend_unweighted`）
LightGBM 是岭回归的 2.5~2.7 倍，所以把它**五五混进来**。

五五而不是全替换：`replace` 的原始分更高（+42.2% vs +28.9%），
但 ΔB 涨 17.9%（靠增加预测方差），按 A/B 折扣规则折完只剩 +9.8%；
`blend50` 的 ΔB 只涨 2.8%，折后 **+11.1%**，是四个臂里最高的。
0.5/0.5 是先验、不拟合权重（ROADMAP §5）。

## 为什么截面均值一律用**无权**

`data/test/*.parquet` 没有 `weight` 列，且 `timeseries_api/runner.py` 的
`forbidden = {"weight", "target", ...}` 会在交给 `predict` 之前剥掉它 ——
**推理端拿不到权重**，也重建不出来（每 asset 约 10 万个不同值、跨分区大幅漂移）。
所以拆解与投影全部走无权均值。这一点在 `lgbm_blend_unweighted` 里重测过，PASS 依然成立。

## ê 为什么必须投影

LightGBM 的输出 `ê_raw` 在 time_id 内的均值不保证为 0，不为 0 的那部分会污染 `m̂` ——
而 `m̂` 那块岭回归已经比树好。实测 `ê_raw` 里约 20% 的幅度是这种虚假市场分量。
投影 `ê = ê_raw − mean(ê_raw)` 只赚不亏（`Σ(e−ê)² = Σ(e−ê_dev)² + ê_mean²·n`）。

官方 runner 每次 `predict` 恰好喂一个 time_id 的全部行，所以这些截面运算都是因果的。

## 树推理走哪条路

主办方确认评测环境有 lightgbm，所以默认走 `lightgbm`（更快，21.4 万次约 0.8 分钟）。
但模型文本是 **`version=v4`**（LightGBM 4.x 格式），评测端若装的是 3.3.x 会读错 ——
那是「import 成功但结果不对」，**`try: import` 兜不住**。
所以 `__init__` 里无条件建一份纯 numpy 的森林（`lgbm_numpy.py`，0.09 s / 2.9 MB），
拿固定种子生成的一批合成输入**对拍一次**：

- import 失败 / Booster 构造失败 / 对拍不一致 → 自动退到 numpy（慢约 0.4 ms/次，仍够用）
- 显式传 `backend="lightgbm"` 则不许降级，对拍不过直接抛 —— 离线校验用

两条路径逐位等价（实测 `max|Δ|/std ≈ 1e-14`，差异只来自 480 个 double 的求和顺序）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 提交包内的同目录模块。官方 runner 加载 main.py 时会把本目录压入 sys.path。
# 绝不 import 仓库的 src/ —— 提交包里没有它。
from features import (apply_robust_transform, cross_sectional_deviation,
                      single_time_deviation)
from lgbm_numpy import NumpyForest
from history import AssetHistory, AssetLongWindow

# 开机对拍的门限。两条路径只该差求和顺序（~1e-18）；真翻了一个分裂，输出会跳一个叶子值
# （~1e-3）。1e-10 落在两者中间好几个数量级，怎么定都不会误判。
_BACKEND_SELFCHECK_ATOL = 1e-10
_BACKEND_SELFCHECK_SEED = 20260809

# ⚠️ 2026-08-29：树推理的线程数**必须由本文件显式给出**，不能靠环境变量。
# 官方评测环境是 4 核（`docs/competition_description.md:158`），而 lightgbm 的默认
# `num_threads=-1` 是「用全部**可见**核」——评测容器按配额限核时可见核数远大于配额，
# 于是起 128 个线程去挤 4 核。云端实测（`experiments/thread_default_probe.py`，
# 128 可见核）：默认 **99.249 ms/次** vs 显式 4 线程 **1.355 ms/次**，慢 **73.27×**。
# 折算到全量 214,538 次调用 = 6.04 小时，是官方总预算 2.98 小时的 **2.03 倍**
# ⟹ 约 49% 处撞总闸、其后全部 time_id 填 0。
# 此前所有交付读数的「4 线程」都来自外部 `OMP_NUM_THREADS=4`，**它不随提交包走**。
_PREDICT_NUM_THREADS = 4


def _asset_scaled_zero_mean(values: np.ndarray, asset_ids: np.ndarray, scales: np.ndarray,
                            time_ids: np.ndarray | None = None) -> np.ndarray:
    """Scale the cross block by asset and re-project to zero mean."""
    values = np.asarray(values, dtype=np.float64)
    asset_ids = np.asarray(asset_ids, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.ndim != 1 or len(scales) == 0 or not np.all(np.isfinite(scales)):
        raise ValueError("asset_cross_scales must be a finite non-empty 1D array")
    if asset_ids.min() < 0 or asset_ids.max() >= len(scales):
        raise ValueError("asset_id is outside asset_cross_scales")
    adjusted = values * scales[asset_ids]
    if time_ids is None:
        return adjusted - adjusted.mean()
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    return adjusted - np.repeat(np.add.reduceat(adjusted, starts) / counts, counts)


class PredictionTrail:
    """逐 asset 维护**自身预测**的因果滚动均值（slow 分量）。**有状态**，调用即推进。

    窗口按**真实 time_id 步长**定义（不是「多少个观测」）：一行的 slow 是该资产在
    ``[t − window, t)`` 内**此前**所有预测的均值。与离线端
    ``experiments/slow_fast_csv.py:causal_trailing_mean`` 逐位一致，包括两个边界规则：

    1. 该资产**首次**出现 ⟹ slow = 当期值本身（于是 fast = 0，不制造假信号）；
    2. 见过但窗口内**没有**任何历史（跳号超过 window）⟹ slow = 0.0。
       这看着别扭，但离线实现就是这个语义（``count = max(index − left, 1)``、分子为 0）。
       两端必须一致，所以照搬而不是「修正」。

    ⚠️ **为什么维护累积和而不是直接求窗口和**：离线算的是
    ``(cumsum[index] − cumsum[left]) / count``。直接对窗口求和虽然数学等价，
    但求和顺序不同 —— 实测两者差 1.29e-15（99.3% 的行不逐位相同）。这里改成同样的
    「增量累积和相减」，两边就是**同一个算式**、逐位相同。
    （幅度上 1e-15 本来无害：slow 是**线性**进入最终预测的，不像 history 那样喂给树 ——
    树差一个 ulp 会翻叶子，实测能放大到 2.85e-03。但能免费做到逐位一致就不该留口子。）

    官方 runner 每次 predict 恰好喂一个 time_id 的全部行、且 time_id 严格递增 ⟹
    窗口左端只需单向右移，均摊 O(1)。
    """

    __slots__ = ("window", "_times", "_cum", "_count", "_head")

    _INITIAL = 1024

    def __init__(self, window: int):
        if window <= 0:
            raise ValueError("slow_fast_window must be positive")
        self.window = int(window)
        self._times: dict[int, np.ndarray] = {}
        self._cum: dict[int, np.ndarray] = {}     # cum[k] = 前 k 个值之和，cum[0] = 0
        self._count: dict[int, int] = {}
        self._head: dict[int, int] = {}

    def _append(self, asset: int, time_id: int, value: float) -> None:
        n = self._count[asset]
        times, cum = self._times[asset], self._cum[asset]
        if n + 1 >= len(times):                    # 几何扩容，避免逐次 realloc
            times = np.resize(times, len(times) * 2)
            cum = np.resize(cum, len(cum) * 2)
            self._times[asset], self._cum[asset] = times, cum
        times[n] = time_id
        cum[n + 1] = cum[n] + value                # 与 np.cumsum 的顺序累加逐位相同
        self._count[asset] = n + 1

    def transform_online(self, values: np.ndarray, asset_ids: np.ndarray,
                         time_id: int) -> np.ndarray:
        """返回这批行的 slow 分量，然后把当期值推进状态。"""
        values = np.asarray(values, dtype=np.float64)
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        if values.ndim != 1 or len(values) != len(asset_ids):
            raise ValueError("values 与 asset_ids 行数必须一致")
        cutoff = int(time_id) - self.window
        slow = np.empty(len(values), dtype=np.float64)
        for row in range(len(values)):
            asset = int(asset_ids[row])
            if asset not in self._count:           # 规则 1：首次出现
                self._times[asset] = np.zeros(self._INITIAL, dtype=np.int64)
                self._cum[asset] = np.zeros(self._INITIAL, dtype=np.float64)
                self._count[asset] = 0
                self._head[asset] = 0
                slow[row] = values[row]
                self._append(asset, int(time_id), float(values[row]))
                continue
            n = self._count[asset]
            times, cum = self._times[asset], self._cum[asset]
            head = self._head[asset]
            while head < n and times[head] < cutoff:
                head += 1
            self._head[asset] = head
            count = n - head
            slow[row] = ((cum[n] - cum[head]) / count) if count > 0 else 0.0
            self._append(asset, int(time_id), float(values[row]))
        return slow


class Model:
    """Sequential inference: production ridge + LightGBM cross-sectional blend."""

    def __init__(self, model_path: str | Path | None = None, backend: str | None = None):
        model_dir = (Path(model_path) if model_path is not None
                     else Path(__file__).resolve().parent / "model")
        if model_dir.is_file():                  # 兼容传 json 路径的调用方
            model_dir = model_dir.parent

        ridge = json.loads((model_dir / "baseline_model.json").read_text(encoding="utf-8"))
        meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))

        # ---- 岭回归部分（生产模型原样冻结，sha 与 v1_ridge 一致）
        self.ridge_features = list(ridge["selected_features"])
        self.r_lower = np.asarray(ridge["lower"], dtype=np.float32)
        self.r_upper = np.asarray(ridge["upper"], dtype=np.float32)
        self.r_center = np.asarray(ridge["center"], dtype=np.float32)
        self.r_scale = np.asarray(ridge["scale"], dtype=np.float32)
        self.r_intercept = np.float32(ridge["intercept"])
        self.r_coef = np.asarray(ridge["coef"], dtype=np.float32)
        self.cross_sectional_scaling = str(ridge.get("cross_sectional_scaling", "none"))

        # ---- LightGBM 部分（自己的一套选列与预处理统计量）
        self.lgbm_features = list(meta["lgbm_features"])
        self.l_lower = np.asarray(meta["lower"], dtype=np.float32)
        self.l_upper = np.asarray(meta["upper"], dtype=np.float32)
        self.l_center = np.asarray(meta["center"], dtype=np.float32)
        self.l_scale = np.asarray(meta["scale"], dtype=np.float32)
        self.num_iteration = int(meta["num_iteration"])
        self.market_num_iteration = int(meta.get("market_num_iteration", self.num_iteration))
        if self.num_iteration <= 0 or self.market_num_iteration <= 0:
            raise ValueError("num_iteration and market_num_iteration must be positive")
        # ---- 每资产滚动历史（**唯一的跨 predict 调用状态**，除 last_time_id 外）
        # 下标是 lgbm_features 内的位置 ⟹ 统计量复用上面那套 l_*，不必另存。
        self.history_positions = list(meta.get("history_positions") or [])
        self.history_window = int(meta.get("history_window", 0) or 0)
        self.history = (AssetHistory(feature_count=len(self.history_positions),
                                     window_size=self.history_window)
                        if self.history_positions else None)
        # ---- 长窗滚动均值（可选，第二个跨 predict 状态）。复用 history 的**同一批列**，
        # 所以不必扩输入契约、meta 也只多一个键。⚠️ **只进截面设计**：确认档
        # （long_window_confirm，3s480 +7.77%/5-of-5）只测了截面块，市场块未测。
        # 缺键 ⟹ None ⟹ 下面的组装退化成原来的 blocks ⟹ 旧模型逐位不变。
        long_window = meta.get("long_window")
        self.long_window = (AssetLongWindow(feature_count=len(self.history_positions),
                                            window=int(long_window))
                            if long_window and self.history_positions else None)
        self.blend_weight = float(meta["blend_weight"])       # ê 里 LGBM 占的比重
        self.prediction_scale = np.float32(meta["prediction_scale"])
        self.prediction_clip = np.float32(meta["prediction_clip"])
        # Optional OOF-fitted per-asset calibration for the cross-sectional block.
        # Missing key keeps every historical model byte-for-byte equivalent at inference.
        scales = meta.get("asset_cross_scales")
        self.asset_cross_scales = (np.asarray(scales, dtype=np.float64) if scales is not None else None)
        if self.asset_cross_scales is not None:
            if self.asset_cross_scales.ndim != 1 or len(self.asset_cross_scales) == 0:
                raise ValueError("asset_cross_scales must be a non-empty 1D array")
            if not np.all(np.isfinite(self.asset_cross_scales)):
                raise ValueError("asset_cross_scales must be finite")

        # ---- slow/fast 分离（①类后处理）。逐 asset 对**自身预测**做因果滚动均值，
        # 慢/快两块各给一个 scale。公榜实测 0.0039977510 → 0.0041150085（+2.93%）。
        # ⚠️ 两个 relative 是**相对 prediction_scale** 的乘数，不是绝对 scale ——
        # OOF 的全局最优 scale 是 0.7296、公榜标定是 1.16（差 59%，本项目已知的尺子分歧），
        # 只搬 OOF 的**相对模式**、保留公榜标定的绝对水平。缺键 ⟹ 关闭 ⟹ 旧模型逐位不变。
        window = meta.get("slow_fast_window")
        self.slow_fast = PredictionTrail(int(window)) if window else None
        if self.slow_fast is not None:
            self.slow_relative = np.float64(meta["slow_fast_slow_relative"])
            self.fast_relative = np.float64(meta["slow_fast_fast_relative"])
            if not (np.isfinite(self.slow_relative) and np.isfinite(self.fast_relative)):
                raise ValueError("slow_fast_*_relative must be finite")

        # ---- 第二个市场分量：行级 LGBM 打 y，取逐 time_id 无权截面均值（`combo_market_weight`）
        # 设计矩阵 = [raw ‖ xs_dev ‖ history ‖ asset_id]，只比截面块多前面那 200 列 raw。
        # λ 是先验、不拟合；缺这两个键的旧模型 λ=0 ⟹ 行为与以前逐位相同。
        self.market_files = [model_dir / name for name in (meta.get("market_model_files") or [])]
        self.market_lambda = float(meta.get("market_lambda", 0.0)) if self.market_files else 0.0

        # ---- 树推理后端：numpy 森林无条件建好（它同时是兜底路径和自检基准）
        self.model_files = [model_dir / name for name in meta["lgbm_model_files"]]
        self.n_models = len(self.model_files)
        self.forest = NumpyForest.from_files(self.model_files, self.num_iteration)
        self.market_forest = (NumpyForest.from_files(self.market_files, self.market_num_iteration)
                              if self.market_files else None)
        self.boosters: list | None = None
        self.market_boosters: list | None = None
        # `validate_features` 是 lightgbm 4.x 才有的参数，评测端版本未知 ⟹ 探测一次再用，
        # 绝不无条件传（传给 3.3.x 会在第一次 predict 就 TypeError）。
        self.predict_kwargs: dict = {}
        self.backend = self._select_backend(backend)

        # make_submission 的烟测按 feature_columns 造表，所以这里给两套的并集
        self.feature_columns = sorted(set(self.ridge_features) | set(self.lgbm_features))
        self.last_time_id: int | None = None

        # ---- 取列的快路径缓存（见 _feature_blocks）。None = 还没解析过；False = 已放弃
        self._columns: object = None
        self._ridge_positions: np.ndarray | None = None
        self._lgbm_positions: np.ndarray | None = None

    def _select_backend(self, requested: str | None) -> str:
        """选树推理后端。`None` = 自动（lightgbm 优先、对拍不过就退 numpy）。"""
        if requested == "numpy":
            return "numpy"
        if requested not in (None, "lightgbm"):
            raise ValueError(f"未知 backend {requested!r}，只认 'lightgbm' / 'numpy' / None")
        strict = requested == "lightgbm"

        try:
            import lightgbm as lgb                # 延迟 import：只有推理时才需要
            boosters = [lgb.Booster(model_file=str(path)) for path in self.model_files]
            market_boosters = [lgb.Booster(model_file=str(path)) for path in self.market_files]
            # 先探测可选参数再对拍：这样 __init__ 里那几次自检也走钉死的线程数。
            self.predict_kwargs = self._probe_predict_kwargs(boosters[0])
            difference = max(self._selfcheck_difference(boosters, self.forest, self.num_iteration),
                             (self._selfcheck_difference(market_boosters, self.market_forest,
                                                        self.market_num_iteration)
                              if self.market_forest is not None else 0.0))
        except Exception as error:                # noqa: BLE001 —— 任何失败都该退到兜底
            if strict:
                raise
            print(f"[v3_hybrid] lightgbm 不可用，改用纯 numpy 树遍历：{error!r}", flush=True)
            return "numpy"

        if not np.isfinite(difference) or difference > _BACKEND_SELFCHECK_ATOL:
            message = (f"lightgbm 与纯 numpy 的对拍不一致（max|Δ| = {difference:.3e} > "
                       f"{_BACKEND_SELFCHECK_ATOL:g}）—— 多半是模型文本 version=v4 "
                       f"被旧版 lightgbm 读错了")
            if strict:
                raise RuntimeError(message)
            print(f"[v3_hybrid] {message}；改用纯 numpy 树遍历", flush=True)
            return "numpy"

        self.boosters = boosters
        self.market_boosters = market_boosters or None
        return "lightgbm"

    def _probe_predict_kwargs(self, booster) -> dict:
        """逐个探测 `predict` 的可选参数，只留下这份 lightgbm 真的认的。

        必须**逐个**探测、且每次都带上已经确认可用的那些：评测端版本未知，
        `validate_features` 是 lightgbm 4.x 才有的具名参数（传给 3.3.x 会在第一次
        predict 就 TypeError），而 `num_threads` 是老早就有的训练/预测参数。
        一次性全传会让不被认的那个把被认的那个一起拖没 —— 而这两者的代价完全不对等：
        少了 `validate_features` 只是慢一点，少了 `num_threads` 会直接撞穿总超时。
        所以 `num_threads` 排在前面先探。
        """
        kwargs: dict = {}
        probe = np.zeros((1, self.forest.n_features), dtype=np.float32)
        for key, value in (("num_threads", _PREDICT_NUM_THREADS),
                           ("validate_features", False)):
            candidate = {**kwargs, key: value}
            try:
                booster.predict(probe, num_iteration=self.num_iteration, **candidate)
            except (TypeError, ValueError):
                print(f"[v3_hybrid] 本机 lightgbm 不接受 predict(..., {key}=…)，跳过该参数",
                      flush=True)
                continue
            kwargs = candidate
        if "num_threads" not in kwargs:
            # 兜底也要有声音：这不是「慢一点」，是可能超总时限的那一档。
            print("[v3_hybrid] ⚠️ 无法设置 num_threads，树推理将用 lightgbm 默认线程数"
                  "（在多核容器上可能极慢）", flush=True)
        return kwargs

    def _selfcheck_difference(self, boosters: list, forest: NumpyForest, num_iteration: int) -> float:
        """固定种子造一批合成设计矩阵，两条路径各跑一次，返回最大绝对差。

        合成而不是拿真数据：提交包里没有数据，这个自检必须在评测机上也能跑。
        15 行 × 480 棵树 = 7200 条路径，足够把「读错模型」这类系统性错误暴露出来。
        两片森林（截面块 361 列 / 市场块 561 列）各校一次，取最大值。
        """
        rng = np.random.default_rng(_BACKEND_SELFCHECK_SEED)
        assets = np.arange(forest.n_assets, dtype=np.int64)
        probe = np.empty((len(assets), forest.n_features), dtype=np.float32)
        probe[:, :-1] = rng.normal(0.0, 1.0, size=(len(assets), forest.n_features - 1))
        probe[:, -1] = assets                     # 最后一列是 asset_id（分类特征）
        reference = np.zeros(len(assets), dtype=np.float64)
        for booster in boosters:
            reference += booster.predict(probe, num_iteration=num_iteration)
        return float(np.max(np.abs(reference - forest.predict_sum(probe, assets))))

    def _feature_blocks(self, test):
        """取出岭回归与 LGBM 各自那 200 列（float32，可就地改）。

        `test.loc[:, 200 个列名]` 每次都要按标签重建索引器 —— 实测**两次合计
        0.575 ms/次**，占整次 `predict` 的 **27%**，比 1440 棵树还贵。
        整个 frame 一次 `to_numpy(float32)` 再按**位置**切两刀只要 **0.006 ms**。

        逐位相同：源列本来就是 float32，`to_numpy(float32)` 与
        `.loc[...].to_numpy(float32)` 都是纯拷贝，不含任何算术。

        列集合与首次见到的不一致、或帧里有转不成 float32 的列 → 永久退回 `.loc` 老路
        （评测端的帧长什么样只有主办方知道，这里不赌）。
        """
        if self._columns is False:
            return (test.loc[:, self.ridge_features].to_numpy(dtype=np.float32, copy=True),
                    test.loc[:, self.lgbm_features].to_numpy(dtype=np.float32, copy=True))
        columns = test.columns
        if self._columns is None or not columns.equals(self._columns):
            try:
                positions = {name: index for index, name in enumerate(columns)}
                self._ridge_positions = np.array([positions[n] for n in self.ridge_features])
                self._lgbm_positions = np.array([positions[n] for n in self.lgbm_features])
                test.to_numpy(dtype=np.float32)          # 先试一次，转不动就别走这条路
            except Exception as error:                   # noqa: BLE001 —— 任何失败都退回老路
                print(f"[v3_hybrid] 取列快路径不可用，退回 .loc：{error!r}", flush=True)
                self._columns = False
                return self._feature_blocks(test)
            self._columns = columns
        block = test.to_numpy(dtype=np.float32)
        return block[:, self._ridge_positions], block[:, self._lgbm_positions]

    def predict(self, test):
        time_ids = test["time_id"].to_numpy(dtype=np.int64)
        time_id = int(time_ids[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        # ---- 岭回归的**原始**预测（不乘 scale、不 clip —— 限幅只在最后做一次）
        raw, lraw = self._feature_blocks(test)
        raw = apply_robust_transform(raw, self.r_lower, self.r_upper, self.r_center, self.r_scale)
        deviation = single_time_deviation(raw, self.cross_sectional_scaling)
        ridge_raw = self.r_intercept + np.column_stack([raw, deviation]) @ self.r_coef

        # ---- 按无权截面均值拆成 市场分量 ‖ 截面分量
        market = ridge_raw.mean()
        e_ridge = ridge_raw - market

        # ---- LightGBM 的截面分量，投影成无权零均值
        lraw = apply_robust_transform(lraw, self.l_lower, self.l_upper, self.l_center, self.l_scale)
        # ⚠️ 这里必须用 cross_sectional_deviation 而不是 single_time_deviation。
        # 两者数学等价，但求和顺序不同（reduceat 顺序累加 vs mean 的成对求和），
        # float32 下差一个 ulp（实测 4.77e-07、60% 元素不逐位相同）。
        # **树是阶跃函数** —— 输入在分裂阈值附近差一个 ulp 就翻到另一枝，
        # 输出直接跳一个叶子值：实测 max|Δpred| 被放大到 2.85e-03（0.52% 的行）。
        # 训练端走的是 cross_sectional_deviation，所以推理端必须**逐位一致**地走同一个。
        # （岭回归那半是线性的，同样的输入差只产生 9.5e-07 的输出差，
        #   所以它继续用 single_time_deviation，保持与 v1_ridge 生产路径逐位相同。）
        ldev = cross_sectional_deviation(lraw, time_ids)
        asset_ids = test["asset_id"].to_numpy(dtype=np.int64)
        blocks = [ldev]
        if self.history is not None:
            # ⚠️ 有状态：每次 predict 都会推进。首个 time_id 无历史 ⟹ previous /
            # rolling_mean 为 0（与训练端最前面那些行同口径，模型见过这种输入），不出 NaN。
            # transform_online 与离线的 transform 逐位相同，见 history.py 的推导。
            blocks.extend(self.history.transform_online(lraw[:, self.history_positions], asset_ids))
        # ⚠️ 有状态：与 history 同样每次 predict 都推进。离线整块与在线逐 time_id
        # 逐位相同（AssetLongWindow 用持久累积和相减，不是分块重起的 cumsum）。
        long_blocks = ([] if self.long_window is None
                       else list(self.long_window.transform_online(
                           lraw[:, self.history_positions], asset_ids)))
        blocks.append(asset_ids.astype(np.float32))   # ⚠️ asset_id 必须留在最后一列
        # ⚠️ 长窗块只插在**截面**设计里、且在 asset_id 之前，与 train.py 逐列对应。
        # 市场设计下面仍用未加长窗的 `blocks` ⟹ 市场块一列不动。
        # long_blocks 为空时 `blocks[:-1] + [] + blocks[-1:]` 就是 `blocks` ⟹ 旧模型逐位不变。
        design = np.column_stack(blocks[:-1] + long_blocks + blocks[-1:])
        e_lgbm = self._forest_mean(design, asset_ids, self.boosters, self.forest,
                                   self.num_iteration)
        e_lgbm -= e_lgbm.mean()
        if self.asset_cross_scales is not None:
            # Scaling reintroduces a group mean; project it out so the adapter cannot leak
            # into the independently modelled market component.
            e_lgbm = _asset_scaled_zero_mean(e_lgbm, asset_ids, self.asset_cross_scales)

        # ---- 第二个市场分量。设计矩阵只比上面多前面那 200 列 raw（与训练端逐列对应）。
        # ⚠️ 这里用的是**未加长窗**的 `blocks` —— 与 train.py 的 market_design 一致。
        # `m̂_lgbm` 取无权截面均值 ⟹ 它是纯市场量，不碰截面块。λ 缺省 0 ⟹ 旧模型行为不变。
        if self.market_lambda:
            market_design = np.column_stack([lraw, *blocks])
            m_lgbm = self._forest_mean(market_design, asset_ids,
                                       self.market_boosters, self.market_forest,
                                       self.market_num_iteration).mean()
            market = (1.0 - self.market_lambda) * market + self.market_lambda * m_lgbm

        blended = market + (1.0 - self.blend_weight) * e_ridge + self.blend_weight * e_lgbm
        if self.slow_fast is None:
            return np.clip(blended * self.prediction_scale,
                           -self.prediction_clip, self.prediction_clip)
        # ⚠️ 有状态：喂给 trail 的必须是**未乘 scale**的 blended，与离线端
        # `raw = pred / prediction_scale` 反解出来的量同口径。
        slow = self.slow_fast.transform_online(blended, asset_ids, time_id)
        fast = blended - slow
        emitted = (self.prediction_scale * self.slow_relative) * slow \
            + (self.prediction_scale * self.fast_relative) * fast
        return np.clip(emitted, -self.prediction_clip, self.prediction_clip)

    def _forest_mean(self, design, asset_ids, boosters, forest, num_iteration: int) -> np.ndarray:
        """一片森林在这批行上的**平均**预测（lightgbm 主路径 / numpy 兜底两条）。"""
        if boosters is not None:
            total = np.zeros(len(design), dtype=np.float64)
            for booster in boosters:
                total += booster.predict(design, num_iteration=num_iteration,
                                         **self.predict_kwargs)
            return total / len(boosters)
        return forest.predict_sum(design, asset_ids) / forest.n_models
