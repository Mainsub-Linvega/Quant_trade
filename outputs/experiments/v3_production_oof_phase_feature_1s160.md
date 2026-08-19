# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/v3_production_oof_phase_feature_1s160.npz`

## Configuration

```json
{
  "n_folds": 5,
  "train_window": 78960,
  "embargo": 6,
  "sample_modulo": 5,
  "sampling": "phase_balanced",
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
  "phase_feature": true,
  "fold_grid": "outputs/experiments/v3_real_time_fold_grid.json"
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00117939 | 0.00136701 | 0.729677 |
| `prediction_raw` | 0.00132200 | 0.00136701 | 0.846425 |
| `market` | 0.00060792 | 0.00066242 | 0.777098 |
| `market_ridge` | 0.00059214 | 0.00059924 | 0.901801 |
| `market_lgbm` | 0.00035991 | 0.00062898 | 0.604575 |
| `e_lgbm` | 0.00084462 | 0.00085451 | 1.120513 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00012435 | 0.00085956 | 0.00085956 |
| 1 | 0.00152992 | 0.00162055 | 0.00162055 |
| 2 | 0.00132595 | 0.00145084 | 0.00145084 |
| 3 | 0.00143417 | 0.00151662 | 0.00151662 |
| 4 | 0.00162873 | 0.00168708 | 0.00168708 |
