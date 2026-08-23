# peer 搭档量：oracle vs 可部署（`xs_peer_deployable_probe`）

> 预注册判据 sha256 `6d8fd172b30afef63ba1339a4072dbc2a15b221e8687b76b357b9c92e445c1da`，先于结果落盘。
> 缓存 `v3_production_oof_confirm_3s480_phasebal_prodwindow.npz`，评估 1,461,732 行 / 98,697 个 time_id / 5 折（fold 0 被系数拟合消耗）。
> ê 覆盖 **100.00%**（不足 100% 即报错退出）。

| 臂 | 搭档量 | 滞后 | Δpeak 折均 | 相对 | 正折 | 去最好折 | CI 下界 | 减阴性对照 | 判定 |
|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| `oracle_lag1` | e_true | 1 | +7.289e-06 | **+0.69%** | 3/4 | -4.708e-06 | +2.595e-05 | +2.45pp | ❌ |
| `deployable_lag1` | e_lgbm | 1 | -3.395e-05 | **-3.21%** | 0/4 | -4.446e-05 | -6.285e-06 | -1.45pp | ❌ |
| `deployable_now` | e_lgbm | 0 | -2.549e-05 | **-2.41%** | 1/4 | -3.573e-05 | +1.869e-06 | -0.65pp | ❌ |
| `shuffled_lag1` | e_lgbm | 1 | -1.862e-05 | **-1.76%** | 1/4 | -3.366e-05 | +5.359e-06 | +0.00pp | ❌ |

> 「减阴性对照」列是**事后**旁证，不在预注册里：`posthoc_contrast_vs_negative_control_pp` 不在预注册里，只作旁证。读法：oracle 高出阴性对照才说明尺子有分辨力；可部署臂若落在对照同侧或更低，则它带来的不是 peer 信息。

### 逐门槛

**`oracle_lag1`**
- ✅ `1_mean_delta_positive`
- ✅ `2_at_least_3_of_4_folds_positive`
- ❌ `3_survives_drop_best_fold`
- ❌ `4_relative_gain_at_least_3pct`
- ✅ `5_two_delta_A_exceeds_delta_B`
- ✅ `6_paired_bootstrap_ci_lower_bound_positive`
- ❌ `7_exceeds_detection_floor`

**`deployable_lag1`**
- ❌ `1_mean_delta_positive`
- ❌ `2_at_least_3_of_4_folds_positive`
- ❌ `3_survives_drop_best_fold`
- ❌ `4_relative_gain_at_least_3pct`
- ❌ `5_two_delta_A_exceeds_delta_B`
- ❌ `6_paired_bootstrap_ci_lower_bound_positive`
- ❌ `7_exceeds_detection_floor`

**`deployable_now`**
- ❌ `1_mean_delta_positive`
- ❌ `2_at_least_3_of_4_folds_positive`
- ❌ `3_survives_drop_best_fold`
- ❌ `4_relative_gain_at_least_3pct`
- ✅ `5_two_delta_A_exceeds_delta_B`
- ✅ `6_paired_bootstrap_ci_lower_bound_positive`
- ❌ `7_exceeds_detection_floor`

**`shuffled_lag1`**
- ❌ `1_mean_delta_positive`
- ❌ `2_at_least_3_of_4_folds_positive`
- ❌ `3_survives_drop_best_fold`
- ❌ `4_relative_gain_at_least_3pct`
- ✅ `5_two_delta_A_exceeds_delta_B`
- ✅ `6_paired_bootstrap_ci_lower_bound_positive`
- ❌ `7_exceeds_detection_floor`

## 裁决：INCONCLUSIVE_NO_DETECTION_POWER

> 按预注册：阳性对照 `oracle_lag1` **没有过门禁** ⟹ **这把线性尺子对该机制检出力不足**，因此两个可部署臂的阴性结果不能升级为「没效果」。⭐ oracle 仍为正（+0.69%，3/4 折，bootstrap CI 下界为正），说明尺子并非全无分辨力，只是分辨不到 3% 门槛。 实操结论仍是不推进：见事后旁证与 2026-08-23 的前置测量。

⚠️ 缓存探针是线性配比、逐有向对 6 列只覆盖一阶交互；树是非线性用这个特征的 ⟹ 本探针的阴性结果弱于树的阴性结果。oracle_lag1 就是用来定量这条局限的。

