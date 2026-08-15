# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`1.000, 0.998, 0.998, 1.000, 1.001, 0.959, 1.014, 1.001, 1.021, 1.003, 0.990, 0.947, 1.084, 1.003, 1.015`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -44.51% | 0/4 | -59.02% | FAIL | -0.00160909 | FAIL |
| `market_hgb` | -42.96% | 0/4 | -44.25% | FAIL | -0.00737553 | FAIL |
| `cross_asset` | +0.78% | 4/4 | +0.74% | PASS | +0.00001586 | PASS |
| `linear_plus_asset` | -43.92% | 0/4 | -58.49% | FAIL | -0.00162062 | FAIL |
| `hgb_plus_asset` | -42.24% | 0/4 | -43.52% | FAIL | -0.00742461 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00175666 | 0.00100066 | 0.00112534 | 0.00176814 | 0.00100889 | 0.00113627 |
| 2 | 0.00146308 | 0.00144736 | 0.00072945 | 0.00147637 | 0.00145979 | 0.00074079 |
| 3 | 0.00147164 | 0.00064077 | 0.00081602 | 0.00148275 | 0.00065079 | 0.00082646 |
| 4 | 0.00177166 | 0.00049767 | 0.00101574 | 0.00178608 | 0.00050521 | 0.00102960 |
