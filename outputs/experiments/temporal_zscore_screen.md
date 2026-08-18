# 多尺度时间状态筛选（`temporal_zscore_screen`）

配置：1 seed × 160 rounds，5 folds，modulo 10/phase_balanced。

| arm | baseline peak | candidate peak | relative | +folds | drop best | 2ΔA>ΔB | pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| t5_zscore | 0.00139833 | 0.00140815 | +0.70% | 3/5 | -3.14e-06 | ✅ | ❌ |

**stop temporal expansion**
