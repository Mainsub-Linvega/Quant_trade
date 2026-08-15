# Residual signal search — 2026-08-14

## Objective

After the 3-seed/480 OOF confirmed the current market capacity and the basic per-asset adapter, test whether the remaining gain can be recovered by low-capacity residual structure rather than by a larger generic tree.

All experiments use the strict 3-seed/480 OOF cache and fit second-stage parameters on one meta fold, freezing them on the other four.

## Results

### Asset × magnitude

The adapter fits two slopes per asset, split by the median absolute XS prediction magnitude from the meta fold.

| Meta fold | Peak gain | Positive | Drop-best | Verdict |
|---:|---:|---:|---:|:---:|
| 0 | +1.31% | 3/4 | +0.63% | PASS |
| 1 | +3.42% | 4/4 | +2.54% | PASS |
| 2 | +1.24% | 3/4 | +0.79% | PASS |
| 3 | +0.54% | 3/4 | -0.27% | FAIL |
| 4 | -1.47% | 1/4 | -2.42% | FAIL |

Conclusion: this is not stable across chronological meta folds. Do not build a public candidate from it.

### Asset × observable regime

The same low-capacity adapter split by predicted cross-sectional RMS was also time-dependent:

```text
meta0 +0.99%   meta1 +3.01%   meta2 +1.19%   meta3 +0.84%   meta4 -0.95%
```

It does not pass the 1% and stability gates. The previous prediction-only soft gate was strongly negative and remains rejected.

### Low-rank cross-sectional factors

PCA was fitted on the meta fold's 200 selected features, then factor scores were interacted with asset exposures and used as a frozen residual correction.

| Components | Peak gain | Positive | Drop-best | Frozen-scale gain | Verdict |
|---:|---:|---:|---:|---:|:---:|
| 4 | -20.99% | 3/4 | -29.81% | -193.63% | FAIL |
| 8 | -18.69% | 2/4 | -27.54% | -115.08% | FAIL |
| 16 | -21.42% | 3/4 | -31.88% | -8918.19% | FAIL |

Conclusion: a static PCA factor basis is not aligned with the residual and is rejected. Do not spend further compute on k/alpha grids for this formulation.

## Interpretation

The remaining residual is not explained by a stable static asset×magnitude rule, a simple predicted-regime gate, or a global low-rank factor basis. The useful asset calibration found earlier is likely a coarse average effect, while its conditional forms are regime-dependent.

The next signal search should move toward genuinely causal temporal structure or feature-space interactions learned inside each rolling fold:

1. sparse asset×feature residual interactions rather than global PCA;
2. multi-horizon causal feature changes and volatility summaries;
3. cross-sectional dispersion/coverage states computed from raw input features, with very low-capacity gates;
4. only then revisit narrowly selected peer lead-lag pairs, not a full peer matrix.

The currently best local candidate remains `outputs/candidates/v3_asset_cross_3s480_shrink500/`. This phase produced no new candidate and no public submission artifact.
