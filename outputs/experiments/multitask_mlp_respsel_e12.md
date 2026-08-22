# 多任务辅助监督 MLP（`multitask_mlp_respsel_e12`）

λ=0.3（预注册，唯一超参）；辅助目标 ['responder_00', 'responder_02', 'responder_03', 'responder_04', 'responder_05']；配比口径 **oracle**

| fold | 基准 peak | 臂 | MLP/基准 | blend peak | 相对 | MLP 幅度占比 | corr |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00105595 | `target_only` | 17.5% | 0.00105597 | +0.0018% | 0.5% | 0.404 |
| 0 | 0.00105595 | `multitask` | 21.6% | 0.00105683 | +0.0825% | 3.1% | 0.419 |

## 汇总

| 臂 | 折均 Δ | 相对 | 正折 | 去最好折 | 残差信号 | 残差能量 |
|---|---:|---:|---:|---:|---:|---:|
| `target_only` | +1.852e-08 | +0.0018% | 1/1 | +1.852e-08 | -1.911e-05 | 1.971e-02 |
| `multitask` | +8.708e-07 | +0.0825% | 1/1 | +8.708e-07 | +1.287e-04 | 1.901e-02 |

## Stage 1 门禁（跑前写死）

- ✅ `delta_positive`
- ❌ `relative_gain_at_least_3pct`
- ✅ `beats_target_only`

**判定：不过 —— 停，不调参**

⚠️ oracle = 系数在评估折自身重解，是**上界**；仓库量过的冻结系数让步为 −2.54%~−3.84%（horizon_auxiliary_cache_probe 的 null_frozen_scale 臂）
