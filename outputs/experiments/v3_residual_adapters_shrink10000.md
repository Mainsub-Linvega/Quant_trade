# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`1.000, 1.000, 1.000, 1.000, 1.000, 0.995, 1.001, 1.000, 1.002, 1.000, 0.999, 0.994, 1.009, 1.000, 1.002`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -44.51% | 0/4 | -59.02% | FAIL | -0.00160909 | FAIL |
| `market_hgb` | -42.96% | 0/4 | -44.25% | FAIL | -0.00737553 | FAIL |
| `cross_asset` | +0.09% | 4/4 | +0.08% | PASS | +0.00000172 | PASS |
| `linear_plus_asset` | -44.45% | 0/4 | -58.96% | FAIL | -0.00161045 | FAIL |
| `hgb_plus_asset` | -42.88% | 0/4 | -44.18% | FAIL | -0.00738107 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00175666 | 0.00100066 | 0.00112534 | 0.00175794 | 0.00100152 | 0.00112653 |
| 2 | 0.00146308 | 0.00144736 | 0.00072945 | 0.00146455 | 0.00144870 | 0.00073067 |
| 3 | 0.00147164 | 0.00064077 | 0.00081602 | 0.00147291 | 0.00064186 | 0.00081717 |
| 4 | 0.00177166 | 0.00049767 | 0.00101574 | 0.00177327 | 0.00049846 | 0.00101724 |
