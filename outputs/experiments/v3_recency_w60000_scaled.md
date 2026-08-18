# v3 production architecture strict OOF

Cache: `/home/mainsub/Documents/Quant_trade/outputs/cache/v3_recency_w60000_scaled.npz`

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
  "train_truncate": 60000,
  "freeze_min_data": false
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00056839 | 0.00134993 | 0.567896 |
| `prediction_raw` | 0.00098770 | 0.00134993 | 0.658759 |
| `market` | 0.00038990 | 0.00060701 | 0.625758 |
| `market_ridge` | 0.00045064 | 0.00048424 | 0.791501 |
| `market_lgbm` | -0.00023323 | 0.00061054 | 0.459646 |
| `e_lgbm` | 0.00080517 | 0.00086065 | 0.797521 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | -0.00122747 | 0.00066739 | 0.00066739 |
| 1 | 0.00182921 | 0.00205677 | 0.00205677 |
| 2 | 0.00105666 | 0.00160685 | 0.00160685 |
| 3 | 0.00052537 | 0.00130303 | 0.00130303 |
| 4 | 0.00102216 | 0.00162634 | 0.00162634 |
