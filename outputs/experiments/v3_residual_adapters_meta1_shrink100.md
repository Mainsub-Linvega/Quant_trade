# Frozen v3 residual adapters

Meta-train fold: `1`; evaluation folds: `[0, 2, 3, 4]`

## Asset slopes

`0.981, 0.943, 1.007, 0.945, 0.986, 0.951, 0.916, 0.991, 1.317, 1.017, 0.988, 1.061, 1.235, 1.009, 0.990`

## Honest later-fold results

| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `market_linear` | -50.81% | 0/4 | -57.82% | FAIL | -0.00237742 | FAIL |
| `market_hgb` | -44.97% | 0/4 | -51.71% | FAIL | -0.01034866 | FAIL |
| `cross_asset` | +3.80% | 4/4 | +3.49% | PASS | +0.00005388 | PASS |
| `linear_plus_asset` | -47.32% | 0/4 | -54.05% | FAIL | -0.00230626 | FAIL |
| `hgb_plus_asset` | -41.11% | 0/4 | -47.84% | FAIL | -0.01002088 | FAIL |

## Per-fold peak

| Fold | `baseline` | `market_linear` | `market_hgb` | `cross_asset` | `linear_plus_asset` | `hgb_plus_asset` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00098176 | 0.00055829 | 0.00055528 | 0.00102699 | 0.00059592 | 0.00059671 |
| 2 | 0.00146308 | 0.00055110 | 0.00059902 | 0.00153038 | 0.00060243 | 0.00065974 |
| 3 | 0.00147164 | 0.00051793 | 0.00055624 | 0.00153645 | 0.00057574 | 0.00061921 |
| 4 | 0.00177166 | 0.00117082 | 0.00141977 | 0.00181064 | 0.00122225 | 0.00147394 |
