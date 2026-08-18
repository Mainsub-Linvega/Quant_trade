# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/v3_recency_w40000_scaled.npz`

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
  "freeze_min_data": false
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | -0.00009847 | 0.00115867 | 0.489806 |
| `prediction_raw` | 0.00048939 | 0.00115867 | 0.568175 |
| `market` | 0.00009978 | 0.00049124 | 0.528352 |
| `market_ridge` | 0.00018882 | 0.00035171 | 0.595043 |
| `market_lgbm` | -0.00062287 | 0.00050920 | 0.401438 |
| `e_lgbm` | 0.00058786 | 0.00075469 | 0.680191 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | -0.00104197 | 0.00073314 | 0.00073314 |
| 1 | 0.00122421 | 0.00166877 | 0.00166877 |
| 2 | 0.00000623 | 0.00134586 | 0.00134586 |
| 3 | -0.00029891 | 0.00118022 | 0.00118022 |
| 4 | -0.00013491 | 0.00112587 | 0.00112587 |
