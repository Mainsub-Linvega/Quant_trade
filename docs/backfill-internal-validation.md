# Backfill-Internal Validation Protocol

## Purpose

The public backfill restores labels for the chronological interval that followed the
original training data. It must be split by time_id, never randomly by row. This
protocol reserves earlier backfill intervals for selection and later intervals for
frozen validation. It applies to target-only NN, responder-assisted NN, and V3/NN
fusion decisions.

## Data Timeline

| Source | Time IDs | Labels |
|---|---:|---|
| Original training data | 0..888479 | target, weight, responders |
| Public backfill | 888480..1105919 | target, weight, responders |
| Competition test data | unlabeled | no local score available |

All assets of one time_id remain in the same split. Training may use
phase_balanced / modulo5; every validation run uses the same fixed sampling
rule so paired score comparisons use identical rows.

## Sequential Split

The configured embargo is six complete time_id values. Embargo rows do not
enter fitting or score calculation. For V3, their features may only advance
causal history state when sequential inference is needed.

| Stage | Training labels | Embargo | Scored interval | Allowed decisions |
|---|---|---|---|---|
| Calibration | original + backfill before 948480 | 948480..948485 | 948486..1008479 | choose model family, responder arm, NN seed count, fusion weight |
| Frozen validation | original + backfill before 1008480 | 1008480..1008485 | 1008486..1045919 | validate the already frozen decision |
| Tail confirmation | original + backfill before 1045920 | 1045920..1045925 | 1045926..1105919 | report only; do not retune |

The historical tail experiment used an equivalent boundary at 1045920 but has
already been consumed for NN density, seed count, V3 comparison, and fusion
diagnostics. It is therefore a confirmation record, not an independent
selection set.

## Calibration Implementation

experiments/backfill_nn_train.py accepts these three arguments as one atomic
split specification:

    --train-backfill-end-time-id
    --validation-start-time-id
    --validation-end-time-id

If any one is supplied, all three are required. The loader keeps every original
row in training, admits only backfill rows before the training end, excludes the
gap, and scores only the specified backfill validation window.

For the executed calibration fold:

    train_backfill_end_exclusive = 948480
    validation_start_inclusive   = 948486
    validation_end_exclusive     = 1008480
    sample_modulo                = 5
    sampling                     = phase_balanced

This produced 2,825,517 training rows and 179,943 scored rows. The first/last
sampled validation times are 948487 and 1008478 because sampling is fixed by
time phase.

## Fair Target and Responder Comparison

Target-only and responder-assisted models use the same:

- rows, feature count, preprocessing, selected features, optimizer settings,
  five seeds, and validation window;
- market head for each seed;
- per-time cross-residual projection.

The responder auxiliary arm now trains one multi-output cross head per seed and
pairs it with that seed's market prediction before averaging. This prevents the
old single-seed auxiliary arm from being compared with a five-seed target-only
ensemble.

Responder values are never inference features. They are training-only auxiliary
labels. The only question is whether this auxiliary target improves prediction
from deployable features.

## Executed Calibration Result

| Arm | Peak | Unit-scale score | Relative to target-only |
|---|---:|---:|---:|
| Target-only, 5 seeds | 0.00185326 | 0.00087814 | baseline |
| Ladder auxiliary | 0.00181214 | 0.00083651 | -2.22% |
| Shortlist auxiliary | 0.00181567 | 0.00086696 | -2.03% |

Both responder arms are negative on the new calibration window under a
seed-matched comparison. They do not advance to frozen validation. Target-only
NN remains eligible only as a separately calibrated, small complement to V3.

## Required Next Step

Freeze the responder decision as off. Before using NN in a fusion, select the
NN market-only weight on the calibration interval, then run the frozen
validation interval without changing:

    target-only architecture
    five fixed seeds
    NN market/cross choice
    normalization/projection
    fusion weight
    prediction scale

Only after a positive frozen validation may the tail confirmation be reported.
