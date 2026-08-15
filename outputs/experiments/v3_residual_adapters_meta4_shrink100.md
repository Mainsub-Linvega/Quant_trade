# Frozen v3 residual adapters

Meta-train fold: `4`; evaluation folds: `[0, 1, 2, 3]`

## Asset slopes

`1.001, 0.991, 1.013, 0.999, 0.872, 0.806, 1.064, 0.996, 1.024, 0.953, 0.887, 0.967, 1.091, 0.926, 0.948`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -34.20% | 1/4 | -47.65% | FAIL | -0.00090831 | FAIL |
| `market_hgb` | -36.52% | 1/4 | -49.23% | FAIL | -0.00479674 | FAIL |
| `cross_asset` | +0.65% | 4/4 | +0.39% | PASS | +0.00000752 | PASS |
| `linear_plus_asset` | -34.15% | 1/4 | -47.66% | FAIL | -0.00091474 | FAIL |
| `hgb_plus_asset` | -36.35% | 1/4 | -49.17% | FAIL | -0.00485307 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00098176 | 0.00106906 | 0.00100481 | 0.00098349 | 0.00107227 | 0.00101184 |
| 1 | 0.00175666 | 0.00127807 | 0.00142963 | 0.00175786 | 0.00127286 | 0.00142780 |
| 2 | 0.00146308 | 0.00062203 | 0.00063800 | 0.00147677 | 0.00062111 | 0.00063826 |
| 3 | 0.00147164 | 0.00076382 | 0.00052910 | 0.00149215 | 0.00076968 | 0.00053318 |
