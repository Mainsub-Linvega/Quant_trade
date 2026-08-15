# Frozen v3 residual adapters

Meta-train fold: `0`; evaluation folds: `[1, 2, 3, 4]`

## Asset slopes

`0.948, 0.925, 0.913, 0.956, 0.944, 0.633, 1.019, 0.974, 0.975, 0.979, 0.836, 0.457, 1.567, 0.982, 0.972`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -38.37% | 1/4 | -51.83% | FAIL | -0.00119046 | FAIL |
| `market_hgb` | -26.69% | 0/4 | -30.76% | FAIL | -0.00208903 | FAIL |
| `cross_asset` | +1.99% | 3/4 | +1.34% | PASS | +0.00007401 | PASS |
| `linear_plus_asset` | -37.39% | 1/4 | -51.15% | FAIL | -0.00146524 | FAIL |
| `hgb_plus_asset` | -24.52% | 0/4 | -27.96% | FAIL | -0.00280069 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00200736 | 0.00129378 | 0.00175505 | 0.00200267 | 0.00127526 | 0.00175490 |
| 2 | 0.00162059 | 0.00165599 | 0.00106307 | 0.00166357 | 0.00168831 | 0.00110100 |
| 3 | 0.00155913 | 0.00081002 | 0.00083998 | 0.00159078 | 0.00083785 | 0.00087040 |
| 4 | 0.00177915 | 0.00053372 | 0.00144866 | 0.00184783 | 0.00056037 | 0.00153182 |
