# v3 residual atlas

OOF rows: `1,461,732`; time_ids: `98,697`

## Component metrics

| Component | Score | RMSE | Bias | Corr |
|---|---:|---:|---:|---:|
| `emitted_prediction` | 0.00110409 | 1.09562271 | -7.193e-03 | 0.0515 |
| `raw_prediction` | 0.00132698 | 1.09550047 | -6.078e-03 | 0.0515 |
| `market_only` | 0.00070490 | 0.92306874 | -9.739e-03 | 0.0532 |
| `cross_only` | 0.00238465 | 0.58807682 | +2.547e-03 | 0.0476 |
| `market_ridge` | 0.00084215 | 0.92300534 | -4.998e-03 | 0.0453 |
| `market_lgbm` | 0.00023459 | 0.92328593 | -1.161e-02 | 0.0541 |
| `e_lgbm` | 0.00243647 | 0.58806154 | +2.228e-03 | 0.0476 |

## Stable bucket candidates

Only buckets with model-vs-market positive delta in at least 3 observed folds are listed.

### phase
- `0`: mean delta=+9.5695e-04, positive folds=5
- `1`: mean delta=+2.6626e-04, positive folds=4
- `2`: mean delta=+4.5282e-04, positive folds=5
- `3`: mean delta=+6.4045e-04, positive folds=5
- `4`: mean delta=+5.0090e-04, positive folds=4
- `5`: mean delta=+7.6835e-04, positive folds=5
- `6`: mean delta=+7.9304e-04, positive folds=5
- `7`: mean delta=+6.3658e-04, positive folds=5
- `8`: mean delta=+1.0190e-03, positive folds=5
- `9`: mean delta=+4.6437e-04, positive folds=5

### asset
- `0`: mean delta=-3.0592e-05, positive folds=3
- `1`: mean delta=+1.6097e-04, positive folds=3
- `11`: mean delta=+3.3060e-04, positive folds=4
- `12`: mean delta=+6.5415e-03, positive folds=5
- `13`: mean delta=+4.1765e-04, positive folds=4
- `14`: mean delta=+2.2785e-04, positive folds=3
- `2`: mean delta=+1.4656e-04, positive folds=3
- `4`: mean delta=+5.5554e-04, positive folds=3
- `5`: mean delta=+5.8897e-04, positive folds=5
- `7`: mean delta=+9.1666e-05, positive folds=3
- `8`: mean delta=+1.2716e-03, positive folds=5
- `9`: mean delta=+5.5958e-04, positive folds=4

### market_vol_quartile
- `1`: mean delta=+3.9905e-04, positive folds=5
- `2`: mean delta=+8.1189e-04, positive folds=5
- `3`: mean delta=+1.8609e-03, positive folds=5

### market_abs_quartile
- `0`: mean delta=+1.4724e-03, positive folds=5
- `1`: mean delta=+1.1877e-03, positive folds=5
- `2`: mean delta=+7.6796e-04, positive folds=5
- `3`: mean delta=+2.6405e-04, positive folds=4

### prediction_rms_quartile
- `0`: mean delta=+2.3752e-04, positive folds=4
- `1`: mean delta=+4.6501e-04, positive folds=4
- `2`: mean delta=+8.4293e-04, positive folds=5
- `3`: mean delta=+1.0505e-03, positive folds=5

### ridge_lgbm_gap_quartile
- `0`: mean delta=+5.4535e-04, positive folds=5
- `1`: mean delta=+4.5235e-04, positive folds=5
- `2`: mean delta=+6.3151e-04, positive folds=5
- `3`: mean delta=+9.4487e-04, positive folds=5

## Worst time_id diagnostics

| time_id | fold | market residual | cross RMS | target RMS | ridge/LGBM gap |
|---:|---:|---:|---:|---:|---:|
| 716073 | 3 | -2.177e+00 | 6.702e-02 | 2.128e+00 | 2.798e-02 |
| 479619 | 0 | -2.175e+00 | 1.105e-01 | 2.106e+00 | 8.273e-02 |
| 511296 | 1 | -2.172e+00 | 8.895e-02 | 2.051e+00 | 4.560e-02 |
| 818196 | 4 | -2.164e+00 | 7.072e-02 | 2.095e+00 | 8.503e-02 |
| 469032 | 0 | -2.161e+00 | 1.268e-01 | 2.080e+00 | 6.176e-02 |
| 466228 | 0 | -2.147e+00 | 2.469e-01 | 2.035e+00 | 6.935e-02 |
| 564669 | 1 | -2.160e+00 | 5.080e-02 | 2.134e+00 | 4.698e-02 |
| 409350 | 0 | -2.143e+00 | 6.622e-02 | 2.112e+00 | 3.281e-02 |
| 444400 | 0 | -2.140e+00 | 9.068e-02 | 2.042e+00 | 5.736e-02 |
| 819991 | 4 | -2.136e+00 | 4.924e-02 | 2.130e+00 | 7.587e-03 |
| 570387 | 1 | +2.066e+00 | 5.334e-01 | 2.050e+00 | 5.840e-02 |
| 469455 | 0 | -2.131e+00 | 8.073e-02 | 2.049e+00 | 2.584e-02 |
| 664969 | 2 | -2.128e+00 | 1.149e-01 | 2.124e+00 | 2.100e-02 |
| 746841 | 3 | -2.126e+00 | 1.121e-01 | 2.044e+00 | 6.530e-02 |
| 717328 | 3 | -2.125e+00 | 1.267e-01 | 2.094e+00 | 4.499e-02 |
| 503878 | 1 | +2.124e+00 | 1.141e-01 | 2.114e+00 | 1.052e-02 |
| 624555 | 2 | -2.113e+00 | 2.307e-01 | 2.089e+00 | 3.123e-02 |
| 469532 | 0 | -2.106e+00 | 2.796e-01 | 2.003e+00 | 8.549e-03 |
| 501746 | 1 | +2.120e+00 | 6.847e-02 | 2.152e+00 | 8.544e-02 |
| 839037 | 4 | +2.116e+00 | 1.455e-01 | 2.025e+00 | 7.573e-02 |
