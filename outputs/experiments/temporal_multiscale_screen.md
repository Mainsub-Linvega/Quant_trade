# 多尺度时间状态筛选（`temporal_multiscale_screen`）

配置：1 seed × 160 rounds，5 folds，modulo 10/phase_balanced。

| arm | baseline peak | candidate peak | relative | +folds | drop best | 2ΔA>ΔB | pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| t1_lags | 0.00139833 | 0.00135844 | -2.85% | 0/5 | -4.3e-05 | ❌ | ❌ |
| t2_state | 0.00139833 | 0.00140079 | +0.18% | 3/5 | -8.15e-06 | ✅ | ❌ |
| t3_full | 0.00139833 | 0.00133503 | -4.53% | 0/5 | -7.57e-05 | ❌ | ❌ |

**stop temporal expansion**
