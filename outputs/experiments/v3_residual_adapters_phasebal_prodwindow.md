# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`1.000, 1.000, 1.000, 1.000, 1.000, 0.999, 1.000, 1.000, 1.000, 1.000, 1.000, 0.999, 1.002, 1.000, 1.000`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -44.51% | 0/4 | -59.02% | FAIL | -0.00160909 | FAIL |
| `market_hgb` | -42.96% | 0/4 | -44.25% | FAIL | -0.00737553 | FAIL |
| `cross_asset` | +0.02% | 4/4 | +0.02% | PASS | +0.00000035 | PASS |
| `linear_plus_asset` | -44.50% | 0/4 | -59.01% | FAIL | -0.00160936 | FAIL |
| `hgb_plus_asset` | -42.94% | 0/4 | -44.24% | FAIL | -0.00737665 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00175666 | 0.00100066 | 0.00112534 | 0.00175692 | 0.00100083 | 0.00112558 |
| 2 | 0.00146308 | 0.00144736 | 0.00072945 | 0.00146337 | 0.00144763 | 0.00072970 |
| 3 | 0.00147164 | 0.00064077 | 0.00081602 | 0.00147190 | 0.00064099 | 0.00081625 |
| 4 | 0.00177166 | 0.00049767 | 0.00101574 | 0.00177198 | 0.00049782 | 0.00101605 |
