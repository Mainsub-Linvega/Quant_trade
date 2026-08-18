# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/v3_recency_expanding_1s160.npz`

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
  "expanding_train": true,
  "expanding_cap": null
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00115275 | 0.00136811 | 0.715944 |
| `prediction_raw` | 0.00131112 | 0.00136811 | 0.830495 |
| `market` | 0.00053834 | 0.00062561 | 0.728075 |
| `market_ridge` | 0.00058530 | 0.00059756 | 0.874740 |
| `market_lgbm` | 0.00024390 | 0.00058682 | 0.566753 |
| `e_lgbm` | 0.00092248 | 0.00095012 | 1.205610 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00013011 | 0.00086514 | 0.00086514 |
| 1 | 0.00143299 | 0.00156762 | 0.00156762 |
| 2 | 0.00162745 | 0.00172649 | 0.00172649 |
| 3 | 0.00144134 | 0.00153761 | 0.00153761 |
| 4 | 0.00128724 | 0.00142134 | 0.00142134 |
