# 截面块窄 peer 对确认档（`xs_peer_pair_confirm_3s480`）

> 筛选档 `xs_peer_pair_probe.md` 的确认档：3 种子 × 480 轮，检出下限换成 3s480 的 8.7%。特征、门禁、peer 对、折版图与筛选档相同，未重新挑选。

| pooled 配对增量 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 | 检出下限 | 判定 |
|--:|--:|--:|--:|--:|:--:|--:|--:|:--:|
| **+3.29%** | 5/5 | +2.93% | +4.46% | +2.18% | ✅ | +2.30% | 8.7% | ❌ |

### 逐门槛

- ✅ 1_pooled_relative_gain_at_least_3pct
- ✅ 2_at_least_4_of_5_folds_positive
- ✅ 3_survives_drop_best_fold
- ✅ 4_two_delta_A_exceeds_delta_B
- ✅ 5_paired_bootstrap_ci_lower_bound_positive
- ❌ 6_exceeds_detection_floor

### 逐折 IC

| fold | 生产 e_lgbm | tree_base | tree_peer_pair | 增量 |
|---|--:|--:|--:|--:|
| 0 | +0.05097 | +0.05014 | +0.05100 | +1.71% |
| 1 | +0.05838 | +0.05733 | +0.05857 | +2.16% |
| 2 | +0.05861 | +0.05828 | +0.06038 | +3.60% |
| 3 | +0.04975 | +0.05023 | +0.05236 | +4.24% |
| 4 | +0.04473 | +0.04544 | +0.04773 | +5.04% |

## 裁决：REJECTED

