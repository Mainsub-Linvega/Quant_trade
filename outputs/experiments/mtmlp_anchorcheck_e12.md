# 多任务辅助监督 MLP（`mtmlp_anchorcheck_e12`）

λ=0.3（预注册，唯一超参）；辅助目标 ['responder_00', 'responder_02', 'responder_03', 'responder_04', 'responder_05']；配比口径 **oracle**

| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00105595 | `target_only` | 17.4% | 0.00105622 | +0.0248% | 1.7% | 0.414 |
| 0 | 0.00105595 | `multitask` | 20.3% | 0.00105623 | +0.0256% | 1.8% | 0.421 |

## 汇总

| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |
|---|---:|---:|---:|---:|---:|---:|
| `target_only` | +2.616e-07 | +0.0248% | 1/1 | +2.616e-07 | -7.168e-05 | 1.965e-02 |
| `multitask` | +2.701e-07 | +0.0256% | 1/1 | +2.701e-07 | +7.140e-05 | 1.887e-02 |

## Stage 1 门禁（跑前写死）

- ✅ `delta_positive`
- ❌ `relative_gain_at_least_3pct`
- ✅ `beats_target_only`

**判定：不过 —— 停，不调参**

⚠️ oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 −2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）
