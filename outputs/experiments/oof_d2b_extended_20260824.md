# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/oof_d2b_extended_20260824.npz`

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
  "train_time_id_max": null,
  "phase_feature": false,
  "fold_grid": "outputs/experiments/d2b_backfill_fold_grid.json",
  "disk_cache": "/tmp/claude-1000/-home-mainsub-Documents-Quant-trade/f0680a57-f2a7-4d67-abf2-06c25a25e6e1/scratchpad/d2b_disk/b",
  "design_cache_dir": null,
  "fixed_production_features": null
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00284612 | 0.00284729 | 0.980161 |
| `prediction_raw` | 0.00280596 | 0.00284729 | 1.136987 |
| `market` | 0.00180987 | 0.00185438 | 1.183320 |
| `market_ridge` | 0.00113881 | 0.00115184 | 1.118994 |
| `market_lgbm` | 0.00218926 | 0.00219279 | 1.041786 |
| `e_lgbm` | 0.00124888 | 0.00140426 | 1.498453 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00161601 | 0.00171999 | 0.00171999 |
| 1 | 0.00381328 | 0.00394155 | 0.00394155 |
| 2 | 0.00390349 | 0.00395094 | 0.00395094 |
| 3 | 0.00194320 | 0.00211150 | 0.00211150 |
