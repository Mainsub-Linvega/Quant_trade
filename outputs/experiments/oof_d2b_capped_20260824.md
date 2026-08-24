# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/oof_d2b_capped_20260824.npz`

## Configuration

```json
{
  "n_folds": 5,
  "train_window": 78960,
  "embargo": 6,
  "sample_modulo": 10,
  "sampling": "phase_balanced",
  "history_window": 5,
  "num_iteration": 160,
  "n_seeds": 1,
  "seed": 2026,
  "num_threads": 8,
  "prediction_scale": 1.16,
  "prediction_clip": 0.5,
  "train_truncate": null,
  "freeze_min_data": false,
  "expanding_train": false,
  "expanding_cap": null,
  "train_time_id_max": 888479,
  "phase_feature": false,
  "fold_grid": "outputs/experiments/d2b_backfill_fold_grid.json",
  "disk_cache": "/tmp/claude-1000/-home-mainsub-Documents-Quant-trade/f0680a57-f2a7-4d67-abf2-06c25a25e6e1/scratchpad/d2b_disk/a",
  "design_cache_dir": null,
  "fixed_production_features": null
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00273645 | 0.00273786 | 0.977803 |
| `prediction_raw` | 0.00269950 | 0.00273786 | 1.134251 |
| `market` | 0.00164389 | 0.00166977 | 1.142215 |
| `market_ridge` | 0.00093040 | 0.00093042 | 0.996132 |
| `market_lgbm` | 0.00208027 | 0.00208459 | 1.047685 |
| `e_lgbm` | 0.00128158 | 0.00145620 | 1.529737 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00161601 | 0.00171999 | 0.00171999 |
| 1 | 0.00369372 | 0.00379897 | 0.00379897 |
| 2 | 0.00391652 | 0.00397024 | 0.00397024 |
| 3 | 0.00157705 | 0.00175903 | 0.00175903 |
