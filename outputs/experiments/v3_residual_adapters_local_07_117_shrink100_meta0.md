# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`0.938, 0.907, 0.903, 0.946, 0.907, 0.603, 0.979, 0.969, 0.916, 0.955, 0.830, 0.492, 1.427, 0.967, 0.925`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -35.04% | 1/4 | -48.29% | FAIL | -0.00101286 | FAIL |
| `market_hgb` | -25.33% | 0/4 | -28.33% | FAIL | -0.00182424 | FAIL |
| `cross_asset` | +1.03% | 3/4 | +0.30% | PASS | +0.00004487 | PASS |
| `linear_plus_asset` | -35.68% | 1/4 | -49.24% | FAIL | -0.00134574 | FAIL |
| `hgb_plus_asset` | -23.92% | 0/4 | -27.80% | FAIL | -0.00260881 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00205554 | 0.00136802 | 0.00177336 | 0.00202669 | 0.00130870 | 0.00175744 |
| 2 | 0.00161870 | 0.00170013 | 0.00109674 | 0.00163985 | 0.00170519 | 0.00111227 |
| 3 | 0.00152239 | 0.00084905 | 0.00086968 | 0.00154579 | 0.00085643 | 0.00088365 |
| 4 | 0.00172495 | 0.00057890 | 0.00142887 | 0.00178052 | 0.00058183 | 0.00151277 |
