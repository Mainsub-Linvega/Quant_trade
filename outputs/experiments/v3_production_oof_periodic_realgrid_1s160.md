# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/v3_production_oof_periodic_realgrid_1s160.npz`

## Configuration

```json
{
  "n_folds": 5,
  "train_window": 78960,
  "embargo": 6,
  "sample_modulo": 5,
  "sampling": "periodic",
  "history_window": 5,
  "num_iteration": 160,
  "n_seeds": 1,
  "seed": 2026,
  "num_threads": 4,
  "prediction_scale": 1.16,
  "prediction_clip": 0.5,
  "train_truncate": null,
  "freeze_min_data": false,
  "expanding_train": false,
  "expanding_cap": null,
  "phase_feature": false,
  "fold_grid": "outputs/experiments/v3_real_time_fold_grid.json"
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00129874 | 0.00142915 | 0.768008 |
| `prediction_raw` | 0.00140771 | 0.00142915 | 0.890889 |
| `market` | 0.00065550 | 0.00068847 | 0.820445 |
| `market_ridge` | 0.00058471 | 0.00058644 | 0.948504 |
| `market_lgbm` | 0.00044691 | 0.00067158 | 0.633555 |
| `e_lgbm` | 0.00086419 | 0.00087896 | 1.148916 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00006297 | 0.00075270 | 0.00075270 |
| 1 | 0.00186437 | 0.00188272 | 0.00188272 |
| 2 | 0.00165741 | 0.00169373 | 0.00169373 |
| 3 | 0.00153655 | 0.00157820 | 0.00157820 |
| 4 | 0.00156527 | 0.00166042 | 0.00166042 |
