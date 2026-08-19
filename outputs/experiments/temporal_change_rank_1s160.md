# 多尺度时间状态筛选（`temporal_change_rank_1s160`）

配置：1 seed × 160 rounds，5 folds，modulo 5/phase_balanced。

| arm | baseline peak | candidate peak | relative | +folds | drop best | 2ΔA>ΔB | pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| x2_change_rank | 0.00129672 | 0.00127657 | -1.55% | 2/5 | -3.19e-05 | ❌ | ❌ |

**stop temporal expansion**
