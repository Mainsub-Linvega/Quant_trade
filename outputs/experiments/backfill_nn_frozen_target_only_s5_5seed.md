# Backfill NN holdout (backfill_nn_frozen_target_only_s5_5seed)

## Distinction from prior NN records

- New data: 20260823 public backfill labels at /mnt/e/量化/public_release_20260823/data/train
- Prior NN difference: target_mlp_screen used only the original train rolling OOF setup and cached v3 OOF blend comparison; it did not train on the 20260823 public backfill labels.
- Training reopen reason: previous NN did not show stable signal on the old original-train rolling setup; 20260823 public backfill creates a new training distribution and a public-period holdout, so this is a fresh training experiment rather than a parameter tweak of the old NN.
- Scope boundary: not a continuation of the old target_mlp_screen verdict; it re-tests NN training only under the expanded label set, and current responder auxiliary results remain unstable across sample density
- Current holdout: public backfill tail reserved by real time_id; default is the last 60000 public-backfill time_id values.

Sample: 3,289,030 rows; train 3,005,475; holdout 112,282.
Holdout: public backfill time_id >= 1008486 (1008487..1045919).
Arms: target_only; seeds: 2026, 2027, 2028, 2029, 2030; responders: none; aux lambda=0.3.
Artifacts: `outputs/experiments/backfill_nn_frozen_target_only_s5_5seed_artifacts`; replay max abs=8.794e-08; cold-start max abs=8.794e-08.

| individual seed | peak | optimal scale | score@unit |
|---|---:|---:|---:|
| `2026` | 0.00017243 | 0.1304 | -0.00750213 |
| `2027` | 0.00014688 | 0.1175 | -0.00813668 |
| `2028` | 0.00129237 | 0.3204 | -0.00452214 |
| `2029` | 0.00018941 | 0.1358 | -0.00748452 |
| `2030` | 0.00010785 | 0.0929 | -0.01017672 |

Ensemble target-only prediction is the rowwise mean across seeds.

| arm | peak | optimal scale | score@unit | A | B |
|---|---:|---:|---:|---:|---:|
| `zero` | 0.00000000 | n/a | 0.00000000 | +0.0000e+00 | 0.0000e+00 |
| `target_only` | 0.00082915 | 0.4525 | -0.00038494 | +1.8325e-03 | 4.0499e-03 |

Auxiliary arm skipped for target-only training experiment.
