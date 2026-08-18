# v3 production architecture strict OOF

Cache: `/tmp/claude-1000/-home-mainsub-Documents-Quant-trade/7aa0dc3d-31aa-4a0d-8f97-9c6b12e523e5/scratchpad/refactor_parity_check.npz`

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
  "num_threads": 8,
  "prediction_scale": 1.16,
  "prediction_clip": 0.5,
  "train_truncate": null,
  "freeze_min_data": false,
  "expanding_train": false,
  "expanding_cap": null
}
```

## Pooled metrics

| Component | Score | Peak | Optimal scale |
|---|---:|---:|---:|
| `prediction` | 0.00115481 | 0.00135167 | 0.723786 |
| `prediction_raw` | 0.00130233 | 0.00135167 | 0.839592 |
| `market` | 0.00059262 | 0.00065156 | 0.768785 |
| `market_ridge` | 0.00059214 | 0.00059925 | 0.901798 |
| `market_lgbm` | 0.00033317 | 0.00061462 | 0.596408 |
| `e_lgbm` | 0.00083833 | 0.00084673 | 1.110624 |

## Fold scores

| Fold | Score | Peak | Raw peak |
|---:|---:|---:|---:|
| 0 | 0.00013010 | 0.00086514 | 0.00086514 |
| 1 | 0.00154457 | 0.00163380 | 0.00163380 |
| 2 | 0.00130756 | 0.00143389 | 0.00143389 |
| 3 | 0.00140066 | 0.00149302 | 0.00149302 |
| 4 | 0.00153750 | 0.00161617 | 0.00161617 |
