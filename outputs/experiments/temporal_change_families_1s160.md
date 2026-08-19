# 多尺度时间状态筛选（`temporal_change_families_1s160`）

配置：1 seed × 160 rounds，5 folds，modulo 5/phase_balanced。

| arm | baseline peak | candidate peak | relative | +folds | drop best | 2ΔA>ΔB | pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| x1_rank | 0.00129672 | 0.00126397 | -2.53% | 1/5 | -4.1e-05 | ❌ | ❌ |
| f_lags | 0.00129672 | 0.00130159 | +0.38% | 3/5 | -5.64e-06 | ❌ | ❌ |
| f_changes | 0.00129672 | 0.00129236 | -0.34% | 3/5 | -1.68e-05 | ❌ | ❌ |
| f_volatility | 0.00129672 | 0.00128950 | -0.56% | 2/5 | -1.72e-05 | ❌ | ❌ |
| f_trend | 0.00129672 | 0.00127551 | -1.64% | 2/5 | -3.14e-05 | ❌ | ❌ |

**stop temporal expansion**
