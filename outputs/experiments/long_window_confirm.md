# 长窗 w512 确认档（`long_window_confirm`）

> 预注册判据 sha256 `11eb76283c692018f88e88f1b35eab76698e2023c791023b6d243b8ffdd5c93a`，先于结果落盘。
> 档位：**confirmation 3s x 480**，只跑截面块。窗口 512（观测数）。
> ⚠️ 检出下限 **8.7%**（3s480），比筛选档的 6.1% **更高**。

> ✅ 线性对拍：10 次拟合全部与 `long_window_ladder` **逐位相同** ⟹ 两次运行的数据路径一致。

| 指标 | 值 |
|---|--:|
| pooled Δpeak | **+7.77%** |
| 正折 | 5/5 |
| 去最好折 | +6.49% |
| bootstrap CI 下界 | +4.18% |
| 检出下限倍数 | 0.89× |

### 逐门槛

- ✅ 1_mean_delta_peak_positive
- ✅ 2_at_least_4_of_5_folds_positive
- ✅ 3_survives_drop_best_fold
- ✅ 4_relative_gain_at_least_3pct
- ✅ 5_paired_bootstrap_ci_lower_bound_positive

### 逐折

| fold | base | w512 | Δpeak |
|---|--:|--:|--:|
| 0 | 2.5975e-03 | 2.9450e-03 | +13.38% |
| 1 | 3.4086e-03 | 3.6389e-03 | +6.76% |
| 2 | 3.4356e-03 | 3.5012e-03 | +1.91% |
| 3 | 2.4748e-03 | 2.6835e-03 | +8.43% |
| 4 | 2.0004e-03 | 2.2300e-03 | +11.48% |

## 裁决：PASS_BUT_BELOW_DETECTION_FLOOR

> 五道门槛全过，但幅度低于 3s480 的 8.7% 检出下限 ⟹ **方向可信、幅度测不出**。
> 按预注册：只够作为花一次公榜额度的理由，**不构成晋级依据**。

