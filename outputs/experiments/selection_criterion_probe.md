# 选列准则探针（`selection_criterion_probe`）

> 预注册判据 sha256 `a86d4de02c255eae77a40394337f5df9282a074c27d09c1746d6b9c0c3ca3b84`，先于结果落盘。
> 评价器：LightGBM 1s x 160 rounds, cross block only。基准臂 `base` = 现状（当期单变量准则）。
> 检出下限 **6.1%**：落在 3%~6.1% 之间读作「过门槛但测不出来」，只能作为花公榜额度的理由，不能直接晋级。

| 臂 | pooled Δpeak | 正折 | 去最好折 | bootstrap CI 下界 | 超检出下限 | 判定 |
|---|--:|--:|--:|--:|:--:|:--:|
| `hist_lag1` | **-4.38%** | 0/5 | -4.88% | -7.10% | ❌ | ❌ |
| `hist_roll5` | **-10.39%** | 0/5 | -11.25% | -12.47% | ❌ | ❌ |
| `lasso200` | **-1.29%** | 2/5 | -2.59% | -6.47% | ❌ | ❌ |

> ⊘ **判据里没有 `2ΔA>ΔB`**：它混着两臂解的共同尺度，且是**两分量配比**的判别式，不适用于「同一模型换选列」。下表的 ΔA/ΔB **仅供参考，不是判据**。

| 臂 | ΔA（非判据） | ΔB（非判据） |
|---|--:|--:|
| `hist_lag1` | -2.97% | -1.67% |
| `hist_roll5` | -7.32% | -4.56% |
| `lasso200` | -1.82% | -2.54% |

### `hist_lag1` 逐门槛

- ❌ 1_mean_delta_peak_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ❌ 3_survives_drop_best_fold
- ❌ 4_relative_gain_at_least_3pct
- ❌ 5_paired_bootstrap_ci_lower_bound_positive

### `hist_roll5` 逐门槛

- ❌ 1_mean_delta_peak_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ❌ 3_survives_drop_best_fold
- ❌ 4_relative_gain_at_least_3pct
- ❌ 5_paired_bootstrap_ci_lower_bound_positive

### `lasso200` 逐门槛

- ❌ 1_mean_delta_peak_positive
- ❌ 2_at_least_4_of_5_folds_positive
- ❌ 3_survives_drop_best_fold
- ❌ 4_relative_gain_at_least_3pct
- ❌ 5_paired_bootstrap_ci_lower_bound_positive

### 逐折

| fold | peak(base) | peak(hist_lag1) | peak(hist_roll5) | peak(lasso200) | hist∩base(hist_lag1) | hist∩base(hist_roll5) | hist∩base(lasso200) | 200列 uni∩lasso |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 2.5263e-03 | 2.4720e-03 | 2.2149e-03 | 2.4958e-03 | 17 | 12 | 23 | 107 |
| 1 | 2.9779e-03 | 2.7113e-03 | 2.5773e-03 | 2.6871e-03 | 19 | 15 | 22 | 112 |
| 2 | 2.8497e-03 | 2.7642e-03 | 2.5641e-03 | 2.9622e-03 | 22 | 20 | 20 | 117 |
| 3 | 2.1639e-03 | 2.1195e-03 | 1.9780e-03 | 2.2681e-03 | 24 | 21 | 25 | 119 |
| 4 | 1.8691e-03 | 1.7770e-03 | 1.7658e-03 | 1.8135e-03 | 21 | 21 | 13 | 68 |

## 裁决：REJECTED

