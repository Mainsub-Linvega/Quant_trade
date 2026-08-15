# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`1.005, 0.927, 0.896, 0.956, 1.030, 0.690, 1.403, 1.082, 1.268, 1.137, 0.636, -0.095, 2.551, 1.167, 1.257`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -44.51% | 0/4 | -59.02% | FAIL | -0.00160909 | FAIL |
| `market_hgb` | -42.96% | 0/4 | -44.25% | FAIL | -0.00737553 | FAIL |
| `cross_asset` | +1.71% | 3/4 | +1.06% | PASS | +0.00012397 | PASS |
| `linear_plus_asset` | -38.02% | 1/4 | -52.65% | FAIL | -0.00156699 | FAIL |
| `hgb_plus_asset` | -36.91% | 0/4 | -37.89% | FAIL | -0.00763612 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00175666 | 0.00100066 | 0.00112534 | 0.00179028 | 0.00110420 | 0.00120783 |
| 2 | 0.00146308 | 0.00144736 | 0.00072945 | 0.00152185 | 0.00155783 | 0.00084392 |
| 3 | 0.00147164 | 0.00064077 | 0.00081602 | 0.00144536 | 0.00072215 | 0.00088272 |
| 4 | 0.00177166 | 0.00049767 | 0.00101574 | 0.00181593 | 0.00062134 | 0.00114318 |
