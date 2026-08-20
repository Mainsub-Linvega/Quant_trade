# 多任务辅助监督 MLP（`multitask_mlp_e150`）

λ=0.3（预注册，唯一超参）；辅助目标 ['responder_00', 'responder_02', 'responder_03', 'responder_04', 'responder_05']；配比口径 **oracle**

| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00105595 | `target_only` | 6.5% | 0.00106106 | +0.4832% | 6.7% | 0.270 |
| 0 | 0.00105595 | `multitask` | 5.7% | 0.00106676 | +1.0228% | 9.4% | 0.277 |

## 汇总

| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |
|---|---:|---:|---:|---:|---:|---:|
| `target_only` | +5.103e-06 | +0.4832% | 1/1 | +5.103e-06 | -5.679e-04 | 6.319e-02 |
| `multitask` | +1.080e-05 | +1.0228% | 1/1 | +1.080e-05 | -8.186e-04 | 6.205e-02 |

## Stage 1 门禁（跑前写死）

- ✅ `delta_positive`
- ❌ `relative_gain_at_least_3pct`
- ✅ `beats_target_only`

**判定：不过 —— 停，不调参**

⚠️ oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 −2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）
