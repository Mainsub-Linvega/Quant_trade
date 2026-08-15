# v3 production architecture strict OOF

Cache: `outputs/cache/v3_production_oof_phasebal_prodwindow.npz`

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
| `prediction` | 0.00110409 | 0.00143744 | 0.674960 |
| `prediction_raw` | 0.00132698 | 0.00143744 | 0.782953 |
| `market` | 0.00063810 | 0.00076375 | 0.711436 |
| `market_ridge` | 0.00059214 | 0.00059924 | 0.901801 |
| `market_lgbm` | 0.00010285 | 0.00076792 | 0.517966 |
| `e_lgbm` | 0.00083833 | 0.00084673 | 1.110624 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00005188 | 0.00098176 | 0.00098176 |
| 1 | 0.00160377 | 0.00175666 | 0.00175666 |
| 2 | 0.00118937 | 0.00146308 | 0.00146308 |
| 3 | 0.00125272 | 0.00147164 | 0.00147164 |
| 4 | 0.00158651 | 0.00177166 | 0.00177166 |
