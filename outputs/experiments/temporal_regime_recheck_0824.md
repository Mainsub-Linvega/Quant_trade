# 多尺度时间状态筛选（`temporal_regime_recheck_0824`）

配置：1 seed × 160 rounds，5 folds，modulo 10/phase_balanced。

| arm | baseline peak | candidate peak | relative | +folds | drop best | 2ΔA>ΔB | pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| t4_regime | 0.00146333 | 0.00148350 | +1.38% | 3/5 | -8.47e-07 | ✅ | ❌ |

**stop temporal expansion**
