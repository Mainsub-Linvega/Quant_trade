# Frozen v3 residual adapters

Meta-train fold: `3`; evaluation folds: `[0, 1, 2, 4]`

## Asset slopes

`1.014, 0.879, 0.875, 0.901, 0.839, 0.994, 0.889, 1.024, 1.247, 0.824, 0.908, 0.940, 1.170, 0.984, 0.875`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -49.11% | 0/4 | -57.90% | FAIL | -0.00148089 | FAIL |
| `market_hgb` | -38.10% | 1/4 | -53.17% | FAIL | -0.00326679 | FAIL |
| `cross_asset` | +2.69% | 4/4 | +2.34% | PASS | +0.00004166 | PASS |
| `linear_plus_asset` | -46.72% | 0/4 | -55.11% | FAIL | -0.00143703 | FAIL |
| `hgb_plus_asset` | -34.82% | 1/4 | -50.10% | FAIL | -0.00306744 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00098176 | 0.00064218 | 0.00056468 | 0.00100673 | 0.00066027 | 0.00059128 |
| 1 | 0.00175666 | 0.00067961 | 0.00098913 | 0.00181285 | 0.00071929 | 0.00105082 |
| 2 | 0.00146308 | 0.00092486 | 0.00156900 | 0.00150613 | 0.00096769 | 0.00162758 |
| 4 | 0.00177166 | 0.00079310 | 0.00057430 | 0.00180827 | 0.00083544 | 0.00062374 |
