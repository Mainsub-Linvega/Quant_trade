# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`1.001, 0.981, 0.983, 0.995, 1.009, 0.813, 1.105, 1.005, 1.123, 1.025, 0.918, 0.630, 1.565, 1.022, 1.098`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -44.51% | 0/4 | -59.02% | FAIL | -0.00160909 | FAIL |
| `market_hgb` | -42.96% | 0/4 | -44.25% | FAIL | -0.00737553 | FAIL |
| `cross_asset` | +3.42% | 4/4 | +3.22% | PASS | +0.00008576 | PASS |
| `linear_plus_asset` | -41.04% | 1/4 | -55.81% | FAIL | -0.00164317 | FAIL |
| `hgb_plus_asset` | -39.06% | 0/4 | -40.26% | FAIL | -0.00760340 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00175666 | 0.00100066 | 0.00112534 | 0.00181014 | 0.00105263 | 0.00118413 |
| 2 | 0.00146308 | 0.00144736 | 0.00072945 | 0.00152571 | 0.00151601 | 0.00079370 |
| 3 | 0.00147164 | 0.00064077 | 0.00081602 | 0.00151149 | 0.00069453 | 0.00086901 |
| 4 | 0.00177166 | 0.00049767 | 0.00101574 | 0.00183648 | 0.00054762 | 0.00109203 |
