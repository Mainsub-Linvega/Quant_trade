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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 提交包内的同目录模块。官方 runner 加载 main.py 时会把本目录压入 sys.path。
# 绝不 import 仓库的 src/ —— 提交包里没有它。
from features import (apply_robust_transform, cross_sectional_deviation,
                      single_time_deviation)


class Model:
    """Sequential inference: production ridge + LightGBM cross-sectional blend."""

    def __init__(self, model_path: str | Path | None = None):
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

        import lightgbm as lgb                    # 延迟 import：只有推理时才需要
        self.boosters = [lgb.Booster(model_file=str(model_dir / name))
                         for name in meta["lgbm_model_files"]]

        # make_submission 的烟测按 feature_columns 造表，所以这里给两套的并集
        self.feature_columns = sorted(set(self.ridge_features) | set(self.lgbm_features))
        self.last_time_id: int | None = None

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
        design = np.column_stack([ldev, test["asset_id"].to_numpy(dtype=np.float32)])
        e_lgbm = np.zeros(len(design), dtype=np.float64)
        for booster in self.boosters:
            e_lgbm += booster.predict(design, num_iteration=self.num_iteration)
        e_lgbm /= len(self.boosters)
        e_lgbm -= e_lgbm.mean()

        blended = market + (1.0 - self.blend_weight) * e_ridge + self.blend_weight * e_lgbm
        return np.clip(blended * self.prediction_scale,
                       -self.prediction_clip, self.prediction_clip)
