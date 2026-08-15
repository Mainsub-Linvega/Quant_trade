# Frozen v3 residual adapters

Meta-train fold: `2`; evaluation folds: `[0, 1, 3, 4]`

## Asset slopes

`1.028, 0.824, 1.060, 0.979, 1.056, 0.883, 1.048, 0.993, 1.203, 1.034, 1.052, 0.984, 1.228, 0.888, 1.016`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -39.79% | 0/4 | -51.17% | FAIL | -0.00118265 | FAIL |
| `market_hgb` | -41.61% | 0/4 | -54.57% | FAIL | -0.00272569 | FAIL |
| `cross_asset` | +3.00% | 4/4 | +2.81% | PASS | +0.00004764 | PASS |
| `linear_plus_asset` | -36.72% | 0/4 | -47.87% | FAIL | -0.00113593 | FAIL |
| `hgb_plus_asset` | -37.79% | 1/4 | -50.67% | FAIL | -0.00260044 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00098176 | 0.00089692 | 0.00094085 | 0.00102227 | 0.00093272 | 0.00099440 |
| 1 | 0.00175666 | 0.00084710 | 0.00103026 | 0.00181007 | 0.00089613 | 0.00109387 |
| 3 | 0.00147164 | 0.00121463 | 0.00113696 | 0.00151644 | 0.00126546 | 0.00119054 |
| 4 | 0.00177166 | 0.00064272 | 0.00038474 | 0.00181259 | 0.00069085 | 0.00044214 |
