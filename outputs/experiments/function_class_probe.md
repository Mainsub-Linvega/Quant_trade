# 函数类探针（`function_class_probe`）

> 预注册判据 sha256 `7290281815e62701061ea8d65bee22f1b26a1758a03d6530090f364899a30877`，先于结果落盘。
> 基准：`v3_production_oof_confirm_3s480_phasebal_prodwindow.npz` 的 `e_lgbm`（3 种子 × 480 轮生产强度）。

| 臂 | pooled r | pooled ρ | 隐含集成增益 (IC) | r>ρ 折数 | 去最好折 | 判定 |
|---|--:|--:|--:|--:|--:|:--:|
| `rff_full` | +0.7983 | +0.7022 | **+0.91%** | 5/5 | +0.70% | ❌ |
| `rff_pca64` | +0.6982 | +0.6379 | **+0.31%** | 5/5 | +0.23% | ❌ |
| `linear` | +0.6105 | +0.5918 | **+0.03%** | 3/5 | +0.00% | 对照 |

> 零增益边界是 `r = ρ`。`r ≤ ρ` ⟹ 集成增益恰好为 0（CLAUDE.md §8.6 的精确形式）。

### `rff_full` 逐门槛

- ✅ 1_pooled_r_exceeds_rho_plus_margin
- ❌ 2_implied_blend_gain_at_least_3pct_ic
- ✅ 3_at_least_4_of_5_folds_r_exceeds_rho
- ❌ 4_survives_drop_best_fold
- ✅ 5_linear_control_does_not_pass_gate_1

### `rff_pca64` 逐门槛

- ✅ 1_pooled_r_exceeds_rho_plus_margin
- ❌ 2_implied_blend_gain_at_least_3pct_ic
- ✅ 3_at_least_4_of_5_folds_r_exceeds_rho
- ❌ 4_survives_drop_best_fold
- ✅ 5_linear_control_does_not_pass_gate_1

### 逐折

| fold | IC(e_lgbm) | r(linear) | r(rff_full) | r(rff_pca64) | ρ(linear) | ρ(rff_full) | ρ(rff_pca64) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | +0.05097 | +0.7680 | +0.8257 | +0.7073 | +0.6496 | +0.6808 | +0.6280 |
| 1 | +0.05838 | +0.6533 | +0.7895 | +0.6902 | +0.6465 | +0.7027 | +0.6133 |
| 2 | +0.05861 | +0.7171 | +0.8136 | +0.6818 | +0.6230 | +0.7219 | +0.6542 |
| 3 | +0.04975 | +0.5731 | +0.7927 | +0.7125 | +0.6628 | +0.7421 | +0.6885 |
| 4 | +0.04473 | +0.3410 | +0.7697 | +0.6993 | +0.3772 | +0.6634 | +0.6055 |

## 裁决：FAIL

