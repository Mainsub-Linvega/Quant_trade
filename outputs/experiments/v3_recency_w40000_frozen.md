# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/v3_recency_w40000_frozen.npz`

## Configuration

```json
{
  "n_folds": 5,
  "train_window": 78960,
  "embargo": 6,
  "sample_modulo": 5,
  "sampling": "phase_balanced",
  "history_window": 5,
  "num_iteration": 480,
  "n_seeds": 3,
  "seed": 2026,
  "num_threads": 4,
  "prediction_scale": 1.16,
  "prediction_clip": 0.5,
  "train_truncate": 40000,
  "freeze_min_data": true
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00008209 | 0.00117323 | 0.509066 |
| `prediction_raw` | 0.00060908 | 0.00117323 | 0.590516 |
| `market` | 0.00014434 | 0.00047367 | 0.545308 |
| `market_ridge` | 0.00018882 | 0.00035171 | 0.595043 |
| `market_lgbm` | -0.00039068 | 0.00048965 | 0.427195 |
| `e_lgbm` | 0.00067219 | 0.00079934 | 0.714877 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | -0.00081807 | 0.00076132 | 0.00076132 |
| 1 | 0.00104521 | 0.00154650 | 0.00154650 |
| 2 | 0.00040835 | 0.00144692 | 0.00144692 |
| 3 | -0.00011187 | 0.00117249 | 0.00117249 |
| 4 | 0.00011698 | 0.00115415 | 0.00115415 |
