# 选列筛子：323 vs 200（截面块 / 市场块分开测）

> 把 323→200 的线性单变量筛子拆掉，树会不会变好？

基准折均 peak **0.00281206**；236,079 行验证数据。

> **单变量保证**：history 列名与基准逐折相同（已断言）—— top-200 的 top-40 == 全 323 的 top-40，所以换宽度不动 history 块

> ⚠️ **基准来源**：基准由**当前代码**现跑。outputs/cache 里的 v3_production_oof_phasebal_prodwindow_exact.npz 时间戳 08-14 11:12，早于该脚本首次提交 08-15 11:18 ⟹ 出自已不存在的代码版本，与当前输出差 max|Δ(market_ridge)|=3.37e-05，不可用作配对基准

| 臂 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 检出下限 | 配对 CI | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|:--:|
| `extended` | +1.189e-04 | +4.23% | 2/4 | +4.109e-05 | +4.09% | +3.83% | 1.44e-04 | [-2.55e-05, +2.63e-04] | ❌ |

## 逐臂门槛

### `extended`（❌ 不通过）
- ✅ 1_mean_delta_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ✅ 3_survives_drop_best_fold
- ✅ 4_relative_gain_at_least_0.03
- ✅ 5_two_delta_A_exceeds_delta_B
- ❌ 6_paired_bootstrap_ci_lower_bound_positive
- ❌ 7_exceeds_detection_floor

## 裁决：REJECT

没有臂通过预注册门禁。

> ab_featsweep 测过 feat323 但那是线性 Ridge 时代（+4.4%、7/10、未晋级）；joint_recalibration_plan 里 Ridge 有 200/323 两档、LGBM 那 9 格只变容量和轮数，没有 feature_count 档

