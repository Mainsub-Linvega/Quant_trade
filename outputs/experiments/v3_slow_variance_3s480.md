# P1：预测里的「死方差」—— 慢分量该不该降权

OOF：`v3_production_oof_confirm_3s480_phasebal_prodwindow.npz`；1,461,732 行 / 98,697 个 time_id。

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
| `M1_block_scales` | 25 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` | 50 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` | 100 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` | 200 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` ←选中 | 400 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` | 800 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` | 1600 | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M1_block_scales` | inf | -3.881e-06 | -0.23% | 2/4 | -2.38% | +6.21% | +15.01% |
| `M2_slow_fast` | 25 | +1.513e-05 | +0.90% | 2/4 | -1.57% | +6.42% | +13.97% |
| `M2_slow_fast` | 50 | +2.382e-05 | +1.42% | 2/4 | -0.78% | +6.18% | +12.69% |
| `M2_slow_fast` | 100 | +6.093e-05 | +3.64% | 3/4 | +2.44% | +6.11% | +9.48% |
| `M2_slow_fast` | 200 | +8.046e-05 | +4.81% | 3/4 | +3.63% | +6.96% | +9.89% |
| `M2_slow_fast` ←选中 | 400 | +9.651e-05 | +5.77% | 3/4 | +4.66% | +8.07% | +11.21% |
| `M2_slow_fast` | 800 | +5.979e-05 | +3.58% | 3/4 | +1.96% | +6.23% | +9.87% |
| `M2_slow_fast` | 1600 | +6.346e-05 | +3.79% | 3/4 | +2.30% | +6.96% | +11.28% |
| `M2_slow_fast` | inf | +3.825e-05 | +2.29% | 3/4 | +0.71% | +5.88% | +10.79% |
| `M3_block_x_slow_fast` | 25 | +6.880e-06 | +0.41% | 2/4 | -4.49% | +11.10% | +25.72% |
| `M3_block_x_slow_fast` | 50 | +1.804e-05 | +1.08% | 2/4 | -3.46% | +10.92% | +24.38% |
| `M3_block_x_slow_fast` | 100 | +5.541e-05 | +3.31% | 3/4 | +0.49% | +11.23% | +22.05% |
| `M3_block_x_slow_fast` | 200 | +7.422e-05 | +4.44% | 3/4 | +1.39% | +12.12% | +22.63% |
| `M3_block_x_slow_fast` ←选中 | 400 | +9.721e-05 | +5.81% | 3/4 | +2.83% | +13.35% | +23.64% |
| `M3_block_x_slow_fast` | 800 | +5.272e-05 | +3.15% | 3/4 | -0.44% | +11.35% | +22.55% |
| `M3_block_x_slow_fast` | 1600 | +6.144e-05 | +3.67% | 3/4 | +0.06% | +12.07% | +23.55% |
| `M3_block_x_slow_fast` | inf | +2.233e-05 | +1.34% | 3/4 | -2.54% | +10.59% | +23.25% |

### `M1_block_scales`（选中 K=400）

pooled 系数（对照基线单 scale 0.7296）：`m̂` = 0.6335、`ê` = 0.8437

block bootstrap 95% CI：[-7.910e-05, +6.788e-05]（中位数 -9.951e-06）

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

pooled 系数（对照基线单 scale 0.7296）：`slow` = 0.2828、`fast` = 0.7881

block bootstrap 95% CI：[+2.514e-05, +1.504e-04]（中位数 +8.809e-05）

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

pooled 系数（对照基线单 scale 0.7296）：`m_slow` = 0.1066、`m_fast` = 0.6972、`e_slow` = 0.4541、`e_fast` = 0.8990

block bootstrap 95% CI：[+5.608e-08, +1.780e-04]（中位数 +8.514e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ✅ |
| 3_survives_drop_best_fold | ✅ |
| 4_relative_gain_at_least_1pct | ✅ |
| 5_K_plateau_all_positive | ✅ |
| 6_bootstrap_ci_lower_bound_positive | ✅ |

**✅ PASS**

## 臂：`asset_adapter`

> ⚠️ fold 0 无法因果适配（没有可用的过去），它不参与评估但参与系数拟合 ⟹ 该臂拟合集是「未适配 fold 0 + 已适配后续折」的混合。线上同样如此，所以是诚实的因果设置，但该臂系数比 production 臂多一层噪声。

| 模型 | K | OOS 折均 Δ | 相对 | 正折 | 去最好折 | ΔA | ΔB |
|---|---|---:|---:|---:|---:|---:|---:|
| `M1_block_scales` | 25 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` | 50 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` | 100 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` | 200 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` | 400 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` ←选中 | 800 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` | 1600 | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M1_block_scales` | inf | -5.728e-05 | -6.90% | 0/4 | -8.34% | -13.13% | -23.26% |
| `M2_slow_fast` | 25 | +1.121e-05 | +1.35% | 2/4 | -3.90% | +8.55% | +20.24% |
| `M2_slow_fast` | 50 | +1.030e-05 | +1.24% | 2/4 | -4.13% | +8.41% | +20.06% |
| `M2_slow_fast` | 100 | +3.982e-05 | +4.80% | 3/4 | +2.02% | +8.74% | +15.16% |
| `M2_slow_fast` | 200 | +4.717e-05 | +5.68% | 3/4 | +2.34% | +9.48% | +15.65% |
| `M2_slow_fast` ←选中 | 400 | +6.215e-05 | +7.49% | 3/4 | +4.88% | +10.95% | +16.58% |
| `M2_slow_fast` | 800 | +2.745e-05 | +3.31% | 2/4 | -0.09% | +7.46% | +14.20% |
| `M2_slow_fast` | 1600 | +3.997e-05 | +4.82% | 2/4 | +1.58% | +9.04% | +15.91% |
| `M2_slow_fast` | inf | +1.882e-05 | +2.27% | 2/4 | -0.75% | +6.68% | +13.85% |
| `M3_block_x_slow_fast` | 25 | -3.389e-05 | -4.08% | 2/4 | -8.72% | -0.04% | +6.54% |
| `M3_block_x_slow_fast` | 50 | -3.516e-05 | -4.24% | 2/4 | -7.31% | -1.53% | +2.88% |
| `M3_block_x_slow_fast` | 100 | -1.531e-05 | -1.84% | 2/4 | -4.56% | -3.09% | -5.11% |
| `M3_block_x_slow_fast` | 200 | -4.103e-06 | -0.49% | 2/4 | -3.02% | -1.71% | -3.69% |
| `M3_block_x_slow_fast` ←选中 | 400 | +2.136e-05 | +2.57% | 3/4 | -1.06% | +1.07% | -1.37% |
| `M3_block_x_slow_fast` | 800 | -1.337e-05 | -1.61% | 2/4 | -6.44% | -2.20% | -3.15% |
| `M3_block_x_slow_fast` | 1600 | +2.240e-06 | +0.27% | 2/4 | -4.75% | -0.60% | -2.02% |
| `M3_block_x_slow_fast` | inf | -1.696e-05 | -2.04% | 2/4 | -5.52% | -1.94% | -1.76% |

### `M1_block_scales`（选中 K=800）

pooled 系数（对照基线单 scale 0.7107）：`m̂` = 0.6906、`ê` = 0.8181

block bootstrap 95% CI：[-1.010e-04, -9.372e-06]（中位数 -5.855e-05）

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

pooled 系数（对照基线单 scale 0.7107）：`slow` = 0.2200、`fast` = 0.7756

block bootstrap 95% CI：[-3.476e-06, +1.135e-04]（中位数 +5.661e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ✅ |
| 3_survives_drop_best_fold | ✅ |
| 4_relative_gain_at_least_1pct | ✅ |
| 5_K_plateau_all_positive | ✅ |
| 6_bootstrap_ci_lower_bound_positive | ❌ |

**❌ 不通过**

### `M3_block_x_slow_fast`（选中 K=400）

pooled 系数（对照基线单 scale 0.7107）：`m_slow` = 0.1181、`m_fast` = 0.7693、`e_slow` = 0.6874、`e_fast` = 0.8107

block bootstrap 95% CI：[-5.996e-05, +9.190e-05]（中位数 +1.421e-05）

| 门槛 | 结果 |
|---|---|
| 1_oos_mean_delta_positive | ✅ |
| 2_at_least_3_of_4_folds_positive | ✅ |
| 3_survives_drop_best_fold | ❌ |
| 4_relative_gain_at_least_1pct | ✅ |
| 5_K_plateau_all_positive | ❌ |
| 6_bootstrap_ci_lower_bound_positive | ❌ |

**❌ 不通过**

