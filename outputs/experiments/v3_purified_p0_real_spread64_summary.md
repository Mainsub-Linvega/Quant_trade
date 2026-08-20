# Purified Interaction P0 Real-Data Summary

Date: 2026-08-20

## Frozen baseline

```text
market_lambda    = 0.7
blend_weight     = 1.17
prediction_scale = 1.16
asset adapter    = disabled
history          = frozen
```

This run is diagnostic only. It did not modify a production model, create a
candidate, or generate a submission CSV.

## Inputs

| Task | Residual | Rows | Input SHA-256 |
|---|---|---:|---|
| Ridge | `target - (market_ridge + e_ridge)` | 1,461,732 | `9d1fe24f05e06b0801ef6b8602fa91a6a04b32dd5ca781b852791500e507f9d3` |
| XS | `target_cross_unweighted - e_lgbm` | 1,461,732 | `77544e06d83d63b142666b50dbe571d7c850f4218f1fbf14e7b0512adde068b0` |
| Market | `target_mean_unweighted - market_lgbm` | 98,697 | `58f2f299d732c65ff2b25b45bf5c5a16b39e35a9b5da5dfdb17674babccf1b92` |

All inputs use strict OOF rows (`fold >= 0`) from
`v3_production_oof_local_07_117_rebuild_3s480.npz`. The source row identity was
confirmed for `time_id`, `asset_id`, `target`, and `weight`. The feature cache is
the raw 323-column spill created from the same sampled `load_rows` ordering.

## Pair budget

The manifest contains 64 fixed pairs at equally spaced lexical ranks over all
52,003 pairs. It was written before real pair scoring and covers source feature
indices from 0 through 322. This is a broad P0 distribution diagnostic, not an
exhaustive search.

## Results

| Task | Null q95 | Positive median | Positive drop-best | Above null | Accepted |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.00029710 | 2 / 64 | 1 / 64 | 0 / 64 | 0 / 64 |
| XS | -0.00001629 | 1 / 64 | 0 / 64 | 1 / 64 | 0 / 64 |
| Market | 0.00096446 | 7 / 64 | 1 / 64 | 0 / 64 | 0 / 64 |

Near-miss diagnostics:

- Ridge `[159, 318]`: median gain `0.00005159`, drop-best
  `0.00003822`, but below the Ridge null threshold.
- XS `[200, 274]`: median gain `0.00001675`, above the negative null threshold,
  but drop-best was `-0.00004686`.
- Market `[46, 302]`: median gain `0.00004413`, drop-best
  `0.00004361`, but far below the Market null threshold.

## Decision

All three tasks failed the frozen P0 gate. Do not enter P1 Ridge, P2 XS, or P3
Market with these pairs. Do not loosen the null or stability thresholds after
seeing the result, and do not generate a candidate or leaderboard CSV.

This result does not prove that every unscanned pair is null. It says the fixed,
coverage-oriented 64-pair sample provides no evidence strong and stable enough
to justify model retraining. At the measured runtime, an exhaustive row-level
scan would cost many hours per task, so it is not justified under the current
time budget without a separately pre-registered cheap proposal stage.
