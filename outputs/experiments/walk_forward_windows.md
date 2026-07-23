# Walk-forward training-window experiment

| Window | Valid p006 | Valid p007 | Valid p008 | Mean | Min | Positive folds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.00055748 | 0.00070715 | 0.00055436 | 0.00060633 | 0.00055436 | 3/3 |
| 3 | 0.00073612 | 0.00081079 | 0.00083852 | 0.00079514 | 0.00073612 | 3/3 |
| 4 | 0.00070249 | 0.00082732 | 0.00087310 | 0.00080097 | 0.00070249 | 3/3 |
| 6 | 0.00080599 | 0.00074873 | 0.00073481 | 0.00076318 | 0.00073481 | 3/3 |

Recommended window: **4 partitions**.

Ranking prioritises the number of positive folds before mean and worst-fold score.
