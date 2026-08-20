# 多任务辅助监督 MLP（`multitask_mlp_e50`）

λ=0.3（预注册，唯一超参）；辅助目标 ['responder_00', 'responder_02', 'responder_03', 'responder_04', 'responder_05']；配比口径 **oracle**

| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00105595 | `target_only` | 26.2% | 0.00111610 | +5.6955% | 21.1% | 0.290 |
| 0 | 0.00105595 | `multitask` | 28.8% | 0.00112422 | +6.4648% | 22.4% | 0.293 |

## 汇总

| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |
|---|---:|---:|---:|---:|---:|---:|
| `target_only` | +6.014e-05 | +5.6955% | 1/1 | +6.014e-05 | +1.277e-03 | 2.709e-02 |
| `multitask` | +6.827e-05 | +6.4648% | 1/1 | +6.827e-05 | +1.327e-03 | 2.580e-02 |

## Stage 1 门禁（跑前写死）

- ✅ `delta_positive`
- ✅ `relative_gain_at_least_3pct`
- ✅ `beats_target_only`

**判定：过 —— 可进 Stage 2 冻结系数五折终审**

⚠️ oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 −2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）
