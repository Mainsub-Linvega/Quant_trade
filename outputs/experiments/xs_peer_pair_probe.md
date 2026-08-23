# 截面块窄 peer 对探针（`xs_peer_pair_probe`）

> 非正式预注册（用户要求'先简单试试'）；门禁与种子复用 `xs_market_state_probe.py`，以便两次探针互相对照。特征来源见 `asset_grouping_diagnostic.py` 挑出的 3 对：`(0,6) (2,14) (1,13)`。

| pooled 配对增量 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 | 检出下限 | 判定 |
|--:|--:|--:|--:|--:|:--:|--:|--:|:--:|
| **+2.39%** | 4/5 | +1.31% | +3.13% | +1.18% | ✅ | -0.49% | 6.1% | ❌ |

### 逐门槛

- ❌ 1_pooled_relative_gain_at_least_3pct
- ✅ 2_at_least_4_of_5_folds_positive
- ✅ 3_survives_drop_best_fold
- ✅ 4_two_delta_A_exceeds_delta_B
- ❌ 5_paired_bootstrap_ci_lower_bound_positive
- ❌ 6_exceeds_detection_floor

### 逐折 IC

| fold | 生产 e_lgbm | tree_base | tree_peer_pair | 增量 |
|---|--:|--:|--:|--:|
| 0 | +0.05097 | +0.05107 | +0.04975 | -2.59% |
| 1 | +0.05838 | +0.05439 | +0.05483 | +0.82% |
| 2 | +0.05861 | +0.05493 | +0.05676 | +3.33% |
| 3 | +0.04975 | +0.04632 | +0.04961 | +7.11% |
| 4 | +0.04473 | +0.04313 | +0.04485 | +3.98% |

## 裁决：REJECTED

