# P4 Task-Aligned Feature Reselection Design

**Date:** 2026-08-20
**Branch:** `exp/adaptive-feature-search`
**Status:** approved for implementation

## 1. Goal

Complete ROADMAP P4 as the four-arm experiment pre-registered on 2026-08-17.
The experiment tests whether the current marginal-correlation feature selection
misses stable, task-aligned signal while preserving the established V3 hybrid
architecture.

This design does not evaluate the later adaptive selector. Every arm keeps the
direct feature counts fixed at:

```text
Ridge   200
XS      200
Market  200
History  40
```

The frozen comparison model keeps:

```text
market_lambda     = 0.7
blend_weight      = 1.17
prediction_scale  = 1.16
prediction_clip   = 0.5
history_window    = 5
```

The screen writes experiment evidence only. It does not modify production,
train a final candidate, or generate a leaderboard CSV.

## 2. Why This Is Separate From Adaptive Selection

The original P4 changes exactly one selection stage at a time and keeps model
width fixed. The adaptive selector added later changes feature counts, uses
shadow floors, redundancy clusters, and shallow-tree path evidence. Those are
different structural changes and cannot be interpreted as a P4 result.

Existing incomplete adaptive fold manifests remain untouched and are not used
as inputs, baselines, or evidence in this experiment.

## 3. Four Frozen Arms

### 3.1 `baseline_corr`

Reproduce the current production selection inside every outer training fold:

- Ridge200: highest absolute weighted pooled correlation with the full target.
- XS200: highest absolute unweighted pooled correlation with the
  cross-sectional target.
- Market200: reuse the XS200 indices.
- History40: within XS200, highest absolute contemporaneous association between
  cross-sectional deviations and the cross-sectional target.

This arm is the exact paired reference for all candidates.

### 3.2 `market_task_aligned`

Change only Market200. For each complete training `time_id`:

1. compute the unweighted mean of each robust-transformed source feature;
2. compute the unweighted mean target;
3. rank all 323 features by absolute training-only correlation between these
   two per-time series;
4. select exactly 200 features using global feature index as the deterministic
   tie breaker.

Ridge200, XS200, and History40 remain byte-for-byte equal to `baseline_corr`.

### 3.3 `xs_time_stable`

Change only XS200. Divide the outer training window into four contiguous,
complete-time blocks. For each source feature, calculate its unweighted
correlation with the cross-sectional target in every block, then rank by:

```text
median absolute block correlation
x sign-consistency fraction
```

Sort descending by the stability score, then median absolute correlation, then
global feature index. Select exactly 200 features. Ridge200, Market200, and
History40 remain equal to `baseline_corr`; in particular, Market does not reuse
the candidate XS set in this arm.

### 3.4 `history_lag_aligned`

Change only History40. Candidate bases are restricted to baseline XS200. Build
the four causal history families using window five:

```text
previous
difference
rolling_mean
rolling_deviation
```

For each base feature and family, measure training-only association with the
future cross-sectional target on four contiguous time blocks. Rank each base by
its strongest stable family using median absolute correlation and sign
consistency. Select exactly 40 unique base features with global index as the
tie breaker.

History state is causal and may use feature observations before a validation
row. It never uses historical or current target values as model inputs.
Ridge200, XS200, and Market200 remain equal to `baseline_corr`.

## 4. Validation Protocol

Use the existing strict V3 rolling protocol:

```text
outer folds       = 5
train window      = 78,960 time_ids
embargo           = 6 time_ids
sampling          = phase_balanced
sample modulo     = 5
screen seeds      = [2026]
screen rounds     = 160
confirmation      = 3 seeds x 480 rounds
```

All robust preprocessing, task targets, correlations, history evidence, and
selected indices are fitted independently inside each outer training window.
Validation rows cannot influence selection, preprocessing, thresholds, or
tie-breaking.

## 5. Paired Execution Architecture

A dedicated P4 runner processes one outer fold at a time:

1. load the common training and validation rows;
2. fit one common robust transform on training rows;
3. build `baseline_corr` selections;
4. derive the three candidate selections from the same training rows;
5. fit the frozen Ridge component once because no P4 arm changes Ridge200;
6. fit the baseline XS and Market components once;
7. for each candidate, retrain only the component whose design changes;
8. construct complete hybrid predictions with frozen fusion values;
9. record paired fold metrics and selection manifests;
10. release fold matrices before processing the next fold.

`history_lag_aligned` changes history columns consumed by both LightGBM
components, so it retrains both XS and Market forests. The other candidates
retrain only their named forest.

The runner may stop a candidate early when the number of remaining folds makes
the four-of-five positive-fold gate mathematically impossible. The baseline and
other viable candidates continue. A stopped candidate receives an explicit
`early_stopped` verdict and cannot advance to confirmation.

## 6. Metrics And Gates

For every arm and fold, report:

- weighted target alignment `A`;
- weighted prediction energy `B`;
- optimum unclipped Peak derived from `A` and `B`;
- score at frozen `prediction_scale=1.16` and clip `0.5`;
- runtime and peak RSS;
- selected indices and overlap with the baseline set.

Each candidate is compared with the exact fold-matched baseline. It passes the
one-seed screen only when all conditions hold:

```text
mean paired Peak delta          > 0
positive folds                 >= 4 / 5
drop-best mean Peak delta       > 0
2 * relative delta_A           > relative delta_B
all metrics finite              = true
```

Only a passing arm is rerun at three seeds and 480 rounds. Confirmation uses
the same gates without changing selectors, counts, folds, or fusion. Arms are
confirmed independently and are never combined after observing results.

## 7. Components And Interfaces

Create a pure selector module exposing deterministic functions for:

- market task-aligned ranking;
- XS time-stable ranking;
- causal lag-aligned history ranking;
- four-arm contract validation.
