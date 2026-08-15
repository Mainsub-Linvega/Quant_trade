# v3 production architecture strict OOF

Cache: `outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz`

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
  "prediction_clip": 0.5
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00101150 | 0.00155137 | 0.628966 |
| `prediction_raw` | 0.00133828 | 0.00155137 | 0.729600 |
| `market` | 0.00058584 | 0.00071809 | 0.699722 |
| `market_ridge` | 0.00059214 | 0.00059924 | 0.901801 |
| `market_lgbm` | 0.00008158 | 0.00071501 | 0.515138 |
| `e_lgbm` | 0.00098264 | 0.00099397 | 0.903542 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | -0.00016194 | 0.00105595 | 0.00105595 |
| 1 | 0.00171774 | 0.00200736 | 0.00200736 |
| 2 | 0.00114890 | 0.00162059 | 0.00162059 |
| 3 | 0.00111330 | 0.00155913 | 0.00155913 |
| 4 | 0.00144253 | 0.00177915 | 0.00177915 |
