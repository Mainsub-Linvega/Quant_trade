# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`1.001, 0.981, 0.983, 0.995, 1.009, 0.813, 1.105, 1.005, 1.123, 1.025, 0.918, 0.630, 1.565, 1.022, 1.098`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -48.52% | 0/4 | -63.71% | FAIL | -0.00159929 | FAIL |
| `market_hgb` | -32.05% | 0/4 | -37.43% | FAIL | -0.00168637 | FAIL |
| `cross_asset` | +3.33% | 4/4 | +3.10% | PASS | +0.00009069 | PASS |
| `linear_plus_asset` | -44.89% | 1/4 | -60.37% | FAIL | -0.00163745 | FAIL |
| `hgb_plus_asset` | -28.38% | 0/4 | -33.51% | FAIL | -0.00186946 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00163381 | 0.00076366 | 0.00138812 | 0.00168516 | 0.00081473 | 0.00143315 |
| 2 | 0.00143389 | 0.00138787 | 0.00075059 | 0.00149482 | 0.00145793 | 0.00082098 |
| 3 | 0.00149302 | 0.00060606 | 0.00095873 | 0.00152419 | 0.00066095 | 0.00099522 |
| 4 | 0.00161617 | 0.00042196 | 0.00109983 | 0.00167840 | 0.00047056 | 0.00117447 |
