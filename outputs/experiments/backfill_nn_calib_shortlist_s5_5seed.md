# Backfill NN holdout (backfill_nn_calib_shortlist_s5_5seed)

## Distinction from prior NN records

- New data: 20260823 public backfill labels at /mnt/e/量化/public_release_20260823/data/train
- Prior NN difference: target_mlp_screen used only the original train rolling OOF setup and cached v3 OOF blend comparison; it did not train on the 20260823 public backfill labels.
- Training reopen reason: previous NN did not show stable signal on the old original-train rolling setup; 20260823 public backfill creates a new training distribution and a public-period holdout, so this is a fresh training experiment rather than a parameter tweak of the old NN.
- Scope boundary: not a continuation of the old target_mlp_screen verdict; it re-tests NN training only under the expanded label set, and current responder auxiliary results remain unstable across sample density
- Current holdout: public backfill tail reserved by real time_id; default is the last 60000 public-backfill time_id values.

Sample: 3,289,030 rows; train 2,825,517; holdout 179,943.
Holdout: public backfill time_id >= 948486 (948487..1008478).
Arms: both; seeds: 2026, 2027, 2028, 2029, 2030; responders: responder_04, responder_28, responder_05, responder_29, responder_06; aux lambda=0.3.
Artifacts: `outputs/experiments/backfill_nn_calib_shortlist_s5_5seed_artifacts`; replay max abs=9.708e-08; cold-start max abs=9.708e-08.

| individual seed | peak | optimal scale | score@unit |
|---|---:|---:|---:|
| `2026` | 0.00074709 | 0.2472 | -0.00617845 |
| `2027` | 0.00078001 | 0.2720 | -0.00481023 |
| `2028` | 0.00109266 | 0.2805 | -0.00609392 |
| `2029` | 0.00109036 | 0.3221 | -0.00373853 |
| `2030` | 0.00076913 | 0.2729 | -0.00468915 |

Ensemble target-only prediction is the rowwise mean across seeds.

| arm | peak | optimal scale | score@unit | A | B |
|---|---:|---:|---:|---:|---:|
| `zero` | 0.00000000 | n/a | 0.00000000 | +0.0000e+00 | 0.0000e+00 |
| `target_only` | 0.00185326 | 0.5796 | 0.00087814 | +3.1976e-03 | 5.5170e-03 |
| `aux_shortlist` | 0.00181567 | 0.5804 | 0.00086696 | +3.1281e-03 | 5.3893e-03 |

Aux delta vs target-only: -3.759e-05 (-2.03%).
