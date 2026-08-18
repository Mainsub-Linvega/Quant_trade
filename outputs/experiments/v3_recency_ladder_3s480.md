# P4：训练窗阶梯（recency）

基准 = 生产窗 78,960，逐折 peak 均值 **0.00160444**；1,461,732 行验证数据。

> **配对保证**：各臂与基准的 time_id/asset_id/fold/target/weight 逐位相同（已断言）；做法是固定 fold 版图、只截短训练段（--train-truncate），**不是**改 --train-window（那会把验证段一起挪走）。

> **混淆项**：窗口变短 ⟹ 行数变少 ⟹ min_data_in_leaf 自动变小 ⟹ 有效容量被动改变。`frozen` 臂把 min_data_in_leaf 冻结在 78,960 档，用来拆开这两件事。

| 臂 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 配对 CI | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|:--:|
| `w60000_scaled` | -1.524e-04 | -9.50% | 1/5 | -12.64% | -3.17% | +6.49% | [-3.18e-04, -1.04e-04] | ❌ |
| `w40000_scaled` | -3.937e-04 | -24.54% | 0/5 | -26.39% | -3.95% | +23.17% | [-5.59e-04, -2.43e-04] | ❌ |
| `w40000_frozen` | -3.882e-04 | -24.19% | 0/5 | -27.54% | -6.41% | +15.50% | [-5.41e-04, -2.38e-04] | ❌ |

检出下限（配对 bootstrap 半宽均值）≈ `1.39e-04`，相当于基准 peak 的 8.7% —— 效应没明显超过它就写「测不出来」，不写「没有效果」。

## 逐臂门槛

### `w60000_scaled`（❌ 不通过）
- ❌ 1_mean_delta_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ❌ 3_survives_drop_best_fold
- ❌ 4_relative_gain_at_least_1pct
- ❌ 5_two_delta_A_exceeds_delta_B
- ❌ 6_paired_bootstrap_ci_lower_bound_positive

### `w40000_scaled`（❌ 不通过）
- ❌ 1_mean_delta_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ❌ 3_survives_drop_best_fold
- ❌ 4_relative_gain_at_least_1pct
- ❌ 5_two_delta_A_exceeds_delta_B
- ❌ 6_paired_bootstrap_ci_lower_bound_positive

### `w40000_frozen`（❌ 不通过）
- ❌ 1_mean_delta_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ❌ 3_survives_drop_best_fold
- ❌ 4_relative_gain_at_least_1pct
- ❌ 5_two_delta_A_exceeds_delta_B
- ❌ 6_paired_bootstrap_ci_lower_bound_positive

## 晋级限制

第②类拟合紧密度轴，本项目已三次本地↔公榜量反 ⟹ 本地结果不单独晋级，须公榜或回补标签裁决（CLAUDE.md §8.1）。

