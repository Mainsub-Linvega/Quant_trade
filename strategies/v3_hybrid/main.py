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

# 开机对拍的门限。两条路径只该差求和顺序（~1e-18）；真翻了一个分裂，输出会跳一个叶子值
# （~1e-3）。1e-10 落在两者中间好几个数量级，怎么定都不会误判。
_BACKEND_SELFCHECK_ATOL = 1e-10
_BACKEND_SELFCHECK_SEED = 20260809


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
        self.blend_weight = float(meta["blend_weight"])       # ê 里 LGBM 占的比重
        self.prediction_scale = np.float32(meta["prediction_scale"])
        self.prediction_clip = np.float32(meta["prediction_clip"])

        # ---- 树推理后端：numpy 森林无条件建好（它同时是兜底路径和自检基准）
        self.model_files = [model_dir / name for name in meta["lgbm_model_files"]]
        self.n_models = len(self.model_files)
        self.forest = NumpyForest.from_files(self.model_files, self.num_iteration)
        self.boosters: list | None = None
        self.backend = self._select_backend(backend)

        # make_submission 的烟测按 feature_columns 造表，所以这里给两套的并集
        self.feature_columns = sorted(set(self.ridge_features) | set(self.lgbm_features))
        self.last_time_id: int | None = None

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
            difference = self._selfcheck_difference(boosters)
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
        return "lightgbm"

    def _selfcheck_difference(self, boosters: list) -> float:
        """固定种子造一批合成设计矩阵，两条路径各跑一次，返回最大绝对差。

        合成而不是拿真数据：提交包里没有数据，这个自检必须在评测机上也能跑。
        15 行 × 480 棵树 = 7200 条路径，足够把「读错模型」这类系统性错误暴露出来。
        """
        rng = np.random.default_rng(_BACKEND_SELFCHECK_SEED)
        assets = np.arange(self.forest.n_assets, dtype=np.int64)
        probe = np.empty((len(assets), self.forest.n_features), dtype=np.float32)
        probe[:, :-1] = rng.normal(0.0, 1.0, size=(len(assets), self.forest.n_features - 1))
        probe[:, -1] = assets                     # 最后一列是 asset_id（分类特征）
        reference = np.zeros(len(assets), dtype=np.float64)
        for booster in boosters:
            reference += booster.predict(probe, num_iteration=self.num_iteration)
        return float(np.max(np.abs(reference - self.forest.predict_sum(probe, assets))))

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        # ---- 岭回归的**原始**预测（不乘 scale、不 clip —— 限幅只在最后做一次）
        raw = test.loc[:, self.ridge_features].to_numpy(dtype=np.float32, copy=True)
        raw = apply_robust_transform(raw, self.r_lower, self.r_upper, self.r_center, self.r_scale)
        deviation = single_time_deviation(raw, self.cross_sectional_scaling)
        ridge_raw = self.r_intercept + np.column_stack([raw, deviation]) @ self.r_coef

        # ---- 按无权截面均值拆成 市场分量 ‖ 截面分量
        market = ridge_raw.mean()
        e_ridge = ridge_raw - market

        # ---- LightGBM 的截面分量，投影成无权零均值
        lraw = test.loc[:, self.lgbm_features].to_numpy(dtype=np.float32, copy=True)
        lraw = apply_robust_transform(lraw, self.l_lower, self.l_upper, self.l_center, self.l_scale)
        # ⚠️ 这里必须用 cross_sectional_deviation 而不是 single_time_deviation。
        # 两者数学等价，但求和顺序不同（reduceat 顺序累加 vs mean 的成对求和），
        # float32 下差一个 ulp（实测 4.77e-07、60% 元素不逐位相同）。
        # **树是阶跃函数** —— 输入在分裂阈值附近差一个 ulp 就翻到另一枝，
        # 输出直接跳一个叶子值：实测 max|Δpred| 被放大到 2.85e-03（0.52% 的行）。
        # 训练端走的是 cross_sectional_deviation，所以推理端必须**逐位一致**地走同一个。
        # （岭回归那半是线性的，同样的输入差只产生 9.5e-07 的输出差，
        #   所以它继续用 single_time_deviation，保持与 v1_ridge 生产路径逐位相同。）
        ldev = cross_sectional_deviation(lraw, test["time_id"].to_numpy(dtype=np.int64))
        asset_ids = test["asset_id"].to_numpy(dtype=np.int64)
        design = np.column_stack([ldev, asset_ids.astype(np.float32)])
        if self.boosters is not None:
            e_lgbm = np.zeros(len(design), dtype=np.float64)
            for booster in self.boosters:
                e_lgbm += booster.predict(design, num_iteration=self.num_iteration)
        else:
            e_lgbm = self.forest.predict_sum(design, asset_ids)
        e_lgbm /= self.n_models
        e_lgbm -= e_lgbm.mean()

        blended = market + (1.0 - self.blend_weight) * e_ridge + self.blend_weight * e_lgbm
        return np.clip(blended * self.prediction_scale,
                       -self.prediction_clip, self.prediction_clip)
