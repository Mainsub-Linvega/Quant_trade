# P1：预测里的「死方差」—— 慢分量该不该降权

OOF：`v3_production_oof_phasebal_prodwindow_exact.npz`；1,461,732 行 / 98,697 个 time_id。

**机制**：信号 m_t 的自相关在真实 lag 5 归零，而预测 m̂ 在 lag 50 仍有 0.324（证据 outputs/experiments/v3_temporal_smoothing_3s480.md）

⚠️ **口径**：本脚本的 trailing mean 走**采样格**（每 ~5 个真实 time_id 一个点），生产走全分辨率 ⟹ K=200 采样步 ≈ 1000 真实步。两者估计同一个慢分量，全分辨率点更多、估计更准（方向有利），但必须由全分辨率核对确认。

⚠️ **诚实声明**：K ∈ {10,25,50,100,200} 在写本脚本前已被探索过并看到 K 越大越好，所以 K 阶梯不是干净的预注册。抗选择保护 = K 连续段门槛 + 第二份 cache 复现 + 全分辨率口径核对。

## 模型阶梯（同一个 Gram 的投影，严格嵌套）

| 模型 | 系数 | 回答什么 |
|---|---:|---|
| `M0_global_scale` | 1 | 全局单 scale = **基线** |
| `M1_block_scales` | 2 | 两块各自 scale |
| `M2_slow_fast` | 2 | 死方差假设本身 |
| `M3_block_x_slow_fast` | 4 | 死方差在哪一块 |

## 臂：`production`

| 模型 | K | OOS 折均 Δ | 相对 | 正折 | 去最好折 | ΔA | ΔB |
|---|---|---:|---:|---:|---:|---:|---:|
| `M1_block_scales` | 25 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` | 50 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` | 100 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` ←选中 | 200 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` | 400 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` | 800 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` | 1600 | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M1_block_scales` | inf | +2.302e-06 | +0.16% | 2/4 | -2.98% | +9.95% | +24.73% |
| `M2_slow_fast` | 25 | +1.186e-05 | +0.81% | 2/4 | -2.24% | +7.30% | +17.09% |
| `M2_slow_fast` | 50 | +1.992e-05 | +1.36% | 2/4 | -1.80% | +7.02% | +15.55% |
| `M2_slow_fast` | 100 | +5.367e-05 | +3.67% | 3/4 | +2.19% | +6.66% | +11.17% |
| `M2_slow_fast` | 200 | +7.145e-05 | +4.89% | 3/4 | +3.58% | +7.32% | +10.99% |
| `M2_slow_fast` ←选中 | 400 | +8.590e-05 | +5.87% | 3/4 | +4.67% | +8.24% | +11.81% |
| `M2_slow_fast` | 800 | +5.326e-05 | +3.64% | 3/4 | +2.00% | +6.40% | +10.55% |
| `M2_slow_fast` | 1600 | +6.202e-05 | +4.24% | 3/4 | +2.68% | +7.26% | +11.82% |
| `M2_slow_fast` | inf | +4.494e-05 | +3.07% | 3/4 | +1.54% | +6.42% | +11.46% |
| `M3_block_x_slow_fast` | 25 | +1.170e-05 | +0.80% | 2/4 | -5.80% | +15.59% | +37.90% |
| `M3_block_x_slow_fast` | 50 | +2.154e-05 | +1.47% | 2/4 | -4.98% | +15.28% | +36.10% |
| `M3_block_x_slow_fast` | 100 | +5.450e-05 | +3.73% | 3/4 | -0.36% | +15.51% | +33.27% |
| `M3_block_x_slow_fast` | 200 | +7.464e-05 | +5.10% | 3/4 | +1.03% | +16.44% | +33.54% |
| `M3_block_x_slow_fast` ←选中 | 400 | +9.411e-05 | +6.43% | 3/4 | +2.49% | +17.55% | +34.31% |
| `M3_block_x_slow_fast` | 800 | +5.552e-05 | +3.80% | 3/4 | -0.95% | +15.55% | +33.28% |
| `M3_block_x_slow_fast` | 1600 | +6.612e-05 | +4.52% | 3/4 | -0.18% | +16.43% | +34.40% |
| `M3_block_x_slow_fast` | inf | +3.664e-05 | +2.51% | 3/4 | -2.38% | +15.04% | +33.94% |

### `M1_block_scales`（选中 K=200）

pooled 系数（对照基线单 scale 0.8396）：`m̂` = 0.7079、`ê` = 1.0443

block bootstrap 95% CI：[-7.946e-05, +8.276e-05]（中位数 -2.043e-06）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ❌ |
| 3_survives_drop_best_fold | ❌ |
| 4_relative_gain_at_least_1pct | ❌ |
| 5_K_plateau_all_positive | ✅ |
| 6_bootstrap_ci_lower_bound_positive | ❌ |

**❌ 不通过**

### `M2_slow_fast`（选中 K=400）

pooled 系数（对照基线单 scale 0.8396）：`slow` = 0.3348、`fast` = 0.9124

block bootstrap 95% CI：[+2.286e-05, +1.379e-04]（中位数 +8.159e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ✅ |
| 3_survives_drop_best_fold | ✅ |
| 4_relative_gain_at_least_1pct | ✅ |
| 5_K_plateau_all_positive | ✅ |
| 6_bootstrap_ci_lower_bound_positive | ✅ |

**✅ PASS**

### `M3_block_x_slow_fast`（选中 K=400）

pooled 系数（对照基线单 scale 0.8396）：`m_slow` = 0.1133、`m_fast` = 0.7822、`e_slow` = 0.5904、`e_fast` = 1.1237

block bootstrap 95% CI：[-3.279e-06, +1.850e-04]（中位数 +8.493e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ✅ |
| 3_survives_drop_best_fold | ✅ |
| 4_relative_gain_at_least_1pct | ✅ |
| 5_K_plateau_all_positive | ✅ |
| 6_bootstrap_ci_lower_bound_positive | ❌ |

**❌ 不通过**

## 臂：`asset_adapter`

> ⚠️ fold 0 无法因果适配（没有可用的过去），它不参与评估但参与系数拟合 ⟹ 该臂拟合集是「未适配 fold 0 + 已适配后续折」的混合。线上同样如此，所以是诚实的因果设置，但该臂系数比 production 臂多一层噪声。

| 模型 | K | OOS 折均 Δ | 相对 | 正折 | 去最好折 | ΔA | ΔB |
|---|---|---:|---:|---:|---:|---:|---:|
| `M1_block_scales` | 25 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` | 50 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` | 100 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` | 200 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` | 400 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` ←选中 | 800 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` | 1600 | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M1_block_scales` | inf | -6.651e-05 | -8.65% | 0/4 | -10.40% | -15.72% | -27.83% |
| `M2_slow_fast` | 25 | +1.105e-05 | +1.44% | 2/4 | -4.66% | +10.05% | +24.80% |
| `M2_slow_fast` | 50 | +1.035e-05 | +1.35% | 2/4 | -5.23% | +9.55% | +23.62% |
| `M2_slow_fast` | 100 | +3.605e-05 | +4.69% | 3/4 | +1.58% | +9.10% | +16.66% |
| `M2_slow_fast` | 200 | +4.394e-05 | +5.71% | 3/4 | +2.56% | +9.54% | +16.10% |
| `M2_slow_fast` ←选中 | 400 | +5.802e-05 | +7.54% | 3/4 | +4.90% | +10.86% | +16.54% |
| `M2_slow_fast` | 800 | +2.711e-05 | +3.52% | 2/4 | -0.01% | +7.63% | +14.68% |
| `M2_slow_fast` | 1600 | +4.003e-05 | +5.20% | 2/4 | +1.82% | +9.24% | +16.14% |
| `M2_slow_fast` | inf | +2.561e-05 | +3.33% | 2/4 | +0.38% | +7.39% | +14.33% |
| `M3_block_x_slow_fast` | 25 | -4.059e-05 | -5.28% | 2/4 | -11.21% | -1.38% | +5.31% |
| `M3_block_x_slow_fast` | 50 | -4.269e-05 | -5.55% | 2/4 | -9.91% | -3.47% | +0.09% |
| `M3_block_x_slow_fast` | 100 | -2.605e-05 | -3.39% | 2/4 | -6.91% | -5.50% | -9.13% |
| `M3_block_x_slow_fast` | 200 | -1.532e-05 | -1.99% | 1/4 | -5.89% | -4.73% | -9.42% |
| `M3_block_x_slow_fast` ←选中 | 400 | +6.131e-06 | +0.80% | 2/4 | -3.95% | -2.51% | -8.18% |
| `M3_block_x_slow_fast` | 800 | -2.547e-05 | -3.31% | 1/4 | -8.64% | -5.69% | -9.76% |
| `M3_block_x_slow_fast` | 1600 | -9.335e-06 | -1.21% | 2/4 | -6.84% | -3.92% | -8.54% |
| `M3_block_x_slow_fast` | inf | -2.083e-05 | -2.71% | 2/4 | -7.16% | -4.67% | -8.02% |

### `M1_block_scales`（选中 K=800）

pooled 系数（对照基线单 scale 0.7950）：`m̂` = 0.7592、`ê` = 1.0468

block bootstrap 95% CI：[-1.147e-04, -1.495e-05]（中位数 -6.770e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ❌ |
| 2_at_least_3_of_4_folds_positive | ❌ |
| 3_survives_drop_best_fold | ❌ |
| 4_relative_gain_at_least_1pct | ❌ |
| 5_K_plateau_all_positive | ❌ |
| 6_bootstrap_ci_lower_bound_positive | ❌ |

**❌ 不通过**

### `M2_slow_fast`（选中 K=400）

pooled 系数（对照基线单 scale 0.7950）：`slow` = 0.2459、`fast` = 0.8722

block bootstrap 95% CI：[+1.821e-06, +1.071e-04]（中位数 +5.499e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ✅ |
| 3_survives_drop_best_fold | ✅ |
| 4_relative_gain_at_least_1pct | ✅ |
| 5_K_plateau_all_positive | ✅ |
| 6_bootstrap_ci_lower_bound_positive | ✅ |

**✅ PASS**

### `M3_block_x_slow_fast`（选中 K=400）

pooled 系数（对照基线单 scale 0.7950）：`m_slow` = 0.1202、`m_fast` = 0.8504、`e_slow` = 0.9226、`e_fast` = 1.0307

block bootstrap 95% CI：[-6.964e-05, +7.704e-05]（中位数 +1.803e-06）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ❌ |
| 3_survives_drop_best_fold | ❌ |
| 4_relative_gain_at_least_1pct | ❌ |
| 5_K_plateau_all_positive | ❌ |
| 6_bootstrap_ci_lower_bound_positive | ❌ |

**❌ 不通过**

