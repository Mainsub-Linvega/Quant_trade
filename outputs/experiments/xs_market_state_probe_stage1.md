# 截面块市场态交互探针（`xs_market_state_probe_stage1`）

> 预注册判据 sha256 `e0c61ceae943ecbbb22740e982cb5affb41dcbae9efcba3cece48d13757fd1d9`，先于结果落盘。
> 基准臂 `tree_base` = 生产截面块逐列相同的设计；主臂 `tree_market_pred` = base + market_pred_t（训练折内拟合的行级 LGBM 打 y 的截面均值）。

⚠️ **Stage 1 单折**（预注册降级路径），不构成五折裁决。

| pooled 配对增量 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 | 检出下限 | 判定 |
|--:|--:|--:|--:|--:|:--:|--:|--:|:--:|
| **+1.49%** | 1/1 | +nan% | +1.63% | +0.28% | ✅ | +nan% | 6.1% | ❌ |

### 逐门槛

- ❌ 1_pooled_relative_gain_at_least_3pct
- ✅ 2_at_least_4_of_5_folds_positive
- — 3_survives_drop_best_fold
- ✅ 4_two_delta_A_exceeds_delta_B
- — 5_paired_bootstrap_ci_lower_bound_positive
- ❌ 6_exceeds_detection_floor

### 逐折 IC

| fold | 生产 e_lgbm | tree_base | tree_market_pred | 增量 |
|---|--:|--:|--:|--:|
| 0 | +0.05097 | +0.05107 | +0.05183 | +1.49% |

## 裁决：REJECTED

