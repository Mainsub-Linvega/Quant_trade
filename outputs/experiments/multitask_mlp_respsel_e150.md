# 多任务辅助监督 MLP（`multitask_mlp_respsel_e150`）

λ=0.3（预注册，唯一超参）；辅助目标 ['responder_00', 'responder_02', 'responder_03', 'responder_04', 'responder_05']；配比口径 **oracle**

| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00105595 | `target_only` | 4.4% | 0.00106797 | +1.1374% | 9.8% | 0.255 |
| 0 | 0.00105595 | `multitask` | 6.9% | 0.00106160 | +0.5350% | 7.0% | 0.278 |

## 汇总

| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |
|---|---:|---:|---:|---:|---:|---:|
| `target_only` | +1.201e-05 | +1.1374% | 1/1 | +1.201e-05 | -8.781e-04 | 6.420e-02 |
| `multitask` | +5.649e-06 | +0.5350% | 1/1 | +5.649e-06 | -5.992e-04 | 6.357e-02 |

## Stage 1 门禁（跑前写死）

- ✅ `delta_positive`
- ❌ `relative_gain_at_least_3pct`
- ❌ `beats_target_only`

**判定：不过 —— 停，不调参**

⚠️ oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 −2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）
