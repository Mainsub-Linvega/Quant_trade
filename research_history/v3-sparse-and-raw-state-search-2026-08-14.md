# Sparse asset-feature and raw-state residual search — 2026-08-14

## Objective

Continue beyond the rejected static PCA and prediction-state gates by testing two narrower, inference-available residual structures on the strict 3-seed/480 OOF cache:

1. sparse `asset × current feature` linear residual interactions;
2. an asset-specific gate driven by current raw-feature cross-sectional dispersion.

No production artifacts or public submission files were changed.

## Sparse asset × feature interactions

Protocol:

- use the 200 XS columns selected by the meta fold's strict base model;
- standardize and cross-sectionally demean features using meta-fold-only statistics;
- select top features by weighted correlation with the XS residual correction;
- fit a sparse Ridge correction with separate coefficients per asset;
- freeze on the four other folds;
- test nested capacities `k=4/8/16` with fixed alpha 1000.

| Capacity | Peak gain | Positive folds | Drop-best | Frozen-scale gain | Verdict |
|---:|---:|---:|---:|---:|:---:|
| 4 | -2.84% | 1/4 | -3.99% | -0.44% | FAIL |
| 8 | -5.58% | 0/4 | -6.98% | -1.39% | FAIL |
| 16 | -14.51% | 0/4 | -17.12% | -7.96% | FAIL |

Increasing capacity worsens the result monotonically. This formulation is rejected without rotating meta folds or searching alpha.

## Raw-feature cross-sectional dispersion state

For each time_id, the adapter computes the mean cross-sectional standard deviation of the 200 selected, meta-standardized current features. It then fits two asset-specific XS slopes split at the meta-fold median state.

Result:

```text
Peak gain: -17.87%
positive folds: 0/4
drop-best: -19.63%
```

The raw dispersion gate is decisively rejected. The failure is consistent with earlier conditional blend and prediction-state expert failures: simple regime partitions do not transfer across time.

## Decision

The currently best candidate remains:

```text
outputs/candidates/v3_asset_cross_3s480_shrink500/
```

This phase produced no new candidate. Do not continue parameter grids for sparse Ridge, PCA factors, or scalar regime gates. The next materially different direction, if exploration continues, is a fully causal multi-horizon feature-change model trained inside each rolling fold, not another static residual recalibration.
