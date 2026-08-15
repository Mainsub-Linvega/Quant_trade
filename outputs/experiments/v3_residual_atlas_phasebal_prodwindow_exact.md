# v3 residual atlas

OOF rows: `1,461,732`; time_ids: `98,697`

## Component metrics

| Component | Score | RMSE | Bias | Corr |
|---|---:|---:|---:|---:|
| `emitted_prediction` | 0.00115482 | 1.09559489 | -6.827e-03 | 0.0499 |
| `raw_prediction` | 0.00130233 | 1.09551399 | -5.762e-03 | 0.0499 |
| `market_only` | 0.00072323 | 0.92306027 | -9.373e-03 | 0.0515 |
| `cross_only` | 0.00238465 | 0.58807682 | +2.547e-03 | 0.0476 |
| `market_ridge` | 0.00084215 | 0.92300534 | -4.998e-03 | 0.0453 |
| `market_lgbm` | 0.00053562 | 0.92314691 | -1.098e-02 | 0.0524 |
| `e_lgbm` | 0.00243647 | 0.58806154 | +2.228e-03 | 0.0476 |

## Stable bucket candidates

Only buckets with model-vs-market positive delta in at least 3 observed folds are listed.

### phase
- `0`: mean delta=+9.8438e-04, positive folds=5
- `1`: mean delta=+2.9506e-04, positive folds=4
- `2`: mean delta=+4.8244e-04, positive folds=5
- `3`: mean delta=+6.6914e-04, positive folds=5
- `4`: mean delta=+5.2995e-04, positive folds=4
- `5`: mean delta=+7.9729e-04, positive folds=5
- `6`: mean delta=+8.2093e-04, positive folds=5
- `7`: mean delta=+6.6574e-04, positive folds=5
- `8`: mean delta=+1.0479e-03, positive folds=5
- `9`: mean delta=+4.9385e-04, positive folds=5

### asset
- `0`: mean delta=-1.7670e-05, positive folds=3
- `1`: mean delta=+1.3438e-04, positive folds=3
- `11`: mean delta=+3.5396e-04, positive folds=4
- `12`: mean delta=+6.4472e-03, positive folds=5
- `13`: mean delta=+4.2134e-04, positive folds=4
- `14`: mean delta=+2.6748e-04, positive folds=4
- `2`: mean delta=+1.3282e-04, positive folds=3
- `4`: mean delta=+4.9284e-04, positive folds=3
- `5`: mean delta=+6.1345e-04, positive folds=5
- `7`: mean delta=+7.8339e-05, positive folds=3
- `8`: mean delta=+1.4260e-03, positive folds=5
- `9`: mean delta=+5.4566e-04, positive folds=4

### market_vol_quartile
- `1`: mean delta=+4.3011e-04, positive folds=5
- `2`: mean delta=+8.4986e-04, positive folds=5
- `3`: mean delta=+1.9001e-03, positive folds=5

### market_abs_quartile
- `0`: mean delta=+1.5514e-03, positive folds=5
- `1`: mean delta=+1.2380e-03, positive folds=5
- `2`: mean delta=+7.9732e-04, positive folds=5
- `3`: mean delta=+2.7646e-04, positive folds=4

### prediction_rms_quartile
- `0`: mean delta=+1.7203e-04, positive folds=4
- `1`: mean delta=+5.3759e-04, positive folds=5
- `2`: mean delta=+6.7456e-04, positive folds=5
- `3`: mean delta=+1.2928e-03, positive folds=5

### ridge_lgbm_gap_quartile
- `0`: mean delta=+6.0819e-04, positive folds=5
- `1`: mean delta=+4.9642e-04, positive folds=5
- `2`: mean delta=+7.2495e-04, positive folds=5
- `3`: mean delta=+8.8110e-04, positive folds=5

## Worst time_id diagnostics

| time_id | fold | market residual | cross RMS | target RMS | ridge/LGBM gap |
|---:|---:|---:|---:|---:|---:|
| 511296 | 1 | -2.164e+00 | 8.895e-02 | 2.051e+00 | 3.140e-02 |
| 716073 | 3 | -2.158e+00 | 6.702e-02 | 2.128e+00 | 4.476e-03 |
| 564669 | 1 | -2.150e+00 | 5.080e-02 | 2.134e+00 | 3.052e-02 |
| 479619 | 0 | -2.147e+00 | 1.105e-01 | 2.106e+00 | 3.496e-02 |
| 466228 | 0 | -2.132e+00 | 2.469e-01 | 2.035e+00 | 4.370e-02 |
| 818196 | 4 | -2.143e+00 | 7.072e-02 | 2.095e+00 | 4.847e-02 |
| 469032 | 0 | -2.140e+00 | 1.268e-01 | 2.080e+00 | 2.617e-02 |
| 409350 | 0 | -2.141e+00 | 6.622e-02 | 2.112e+00 | 2.863e-02 |
| 469455 | 0 | -2.136e+00 | 8.073e-02 | 2.049e+00 | 3.499e-02 |
| 664969 | 2 | -2.128e+00 | 1.149e-01 | 2.124e+00 | 2.107e-02 |
| 454473 | 0 | +2.127e+00 | 1.309e-01 | 2.049e+00 | 2.334e-02 |
| 819991 | 4 | -2.130e+00 | 4.924e-02 | 2.130e+00 | 1.823e-03 |
| 444400 | 0 | -2.123e+00 | 9.068e-02 | 2.042e+00 | 2.812e-02 |
| 624555 | 2 | -2.111e+00 | 2.307e-01 | 2.089e+00 | 2.882e-02 |
| 503878 | 1 | +2.119e+00 | 1.141e-01 | 2.114e+00 | 3.155e-03 |
| 570387 | 1 | +2.053e+00 | 5.334e-01 | 2.050e+00 | 3.626e-02 |
| 746841 | 3 | -2.109e+00 | 1.121e-01 | 2.044e+00 | 3.698e-02 |
| 404782 | 0 | +2.110e+00 | 6.240e-02 | 2.132e+00 | 9.254e-05 |
| 821946 | 4 | -2.101e+00 | 1.981e-01 | 2.067e+00 | 2.873e-02 |
| 469532 | 0 | -2.091e+00 | 2.796e-01 | 2.003e+00 | 1.764e-02 |
