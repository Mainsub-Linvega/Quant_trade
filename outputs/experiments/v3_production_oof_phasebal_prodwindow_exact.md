# v3 production architecture strict OOF

Cache: `outputs/cache/v3_production_oof_phasebal_prodwindow_exact.npz`

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
  "prediction_clip": 0.5
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00115482 | 0.00135167 | 0.723787 |
| `prediction_raw` | 0.00130233 | 0.00135167 | 0.839593 |
| `market` | 0.00059263 | 0.00065156 | 0.768786 |
| `market_ridge` | 0.00059214 | 0.00059924 | 0.901801 |
| `market_lgbm` | 0.00033317 | 0.00061462 | 0.596408 |
| `e_lgbm` | 0.00083833 | 0.00084673 | 1.110624 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00013011 | 0.00086514 | 0.00086514 |
| 1 | 0.00154458 | 0.00163381 | 0.00163381 |
| 2 | 0.00130756 | 0.00143389 | 0.00143389 |
| 3 | 0.00140067 | 0.00149302 | 0.00149302 |
| 4 | 0.00153749 | 0.00161617 | 0.00161617 |
