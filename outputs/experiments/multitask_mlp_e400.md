# 多任务辅助监督 MLP（`multitask_mlp_e400`）

λ=0.3（预注册，唯一超参）；辅助目标 ['responder_00', 'responder_02', 'responder_03', 'responder_04', 'responder_05']；配比口径 **oracle**

| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00105595 | `target_only` | 1.4% | 0.00108490 | +2.7407% | 14.1% | 0.248 |
| 0 | 0.00105595 | `multitask` | 1.2% | 0.00109037 | +3.2592% | 15.2% | 0.249 |

## 汇总

| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |
|---|---:|---:|---:|---:|---:|---:|
| `target_only` | +2.894e-05 | +2.7407% | 1/1 | +2.894e-05 | -1.800e-03 | 1.120e-01 |
| `multitask` | +3.442e-05 | +3.2592% | 1/1 | +3.442e-05 | -1.947e-03 | 1.102e-01 |

## Stage 1 门禁（跑前写死）

- ✅ `delta_positive`
- ✅ `relative_gain_at_least_3pct`
- ✅ `beats_target_only`

**判定：过 —— 可进 Stage 2 冻结系数五折终审**

⚠️ oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 −2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）
