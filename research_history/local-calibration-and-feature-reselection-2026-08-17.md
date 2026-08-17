# Fork local calibration and feature reselection - 2026-08-17

## Scope and model identity

This note records local work performed after upstream commit `21efa38`. It does not change the
upstream production model in `strategies/v3_hybrid/model/`.

The upstream production baseline remains:

```text
market_lambda=0.5
blend_weight=1.0
prediction_scale=1.16
public score=0.0039977510
```

The best unpromoted fork candidate is:

```text
market_lambda=0.7
blend_weight=1.17
prediction_scale=1.16
asset_cross_scales absent
public score=0.00407075
```

Only metadata scalars differ. Ridge and all six LightGBM model files are unchanged.

## Public calibration ladder

| market lambda | blend weight | public score | decision |
|---:|---:|---:|---|
| 0.5 | 1.0 | 0.0039977510 | upstream production reference |
| 0.6 | 1.0 | 0.0040362837 | direction confirmed |
| 0.6 | 1.1 | 0.00405916 | keep as calibration point |
| 0.6 | 1.17 | 0.00406362 | best point at lambda 0.6 |
| 0.6 | 1.2 | 0.00406261 | plateau, slightly below 1.17 |
| 0.7 | 1.17 | **0.00407075** | freeze as fork local best |
| 0.8 | 1.17 | 0.0040523419 | rollback; lambda axis passed its optimum |

The ladder supports a broad local optimum, not precision tuning. The fork therefore freezes
`0.7/1.17` and does not search additional decimal places.

## Strict OOF rebuild

The exact production-equivalent OOF was rebuilt with:

```text
5 rolling folds
train_window=78,960
embargo=6
sample_modulo=5 / phase_balanced
XS and market forests: 3 seeds x 480 rounds
history40 / window5
```

The cache adds the missing `e_ridge` component, allowing exact reconstruction of arbitrary
`blend_weight` values:

```text
market = (1 - market_lambda) * market_ridge + market_lambda * market_lgbm
cross  = (1 - blend_weight) * e_ridge + blend_weight * e_lgbm
```

All five fold peaks matched the committed upstream confirmation artifact exactly; maximum peak
difference was `0.0`. Both `e_ridge` and `e_lgbm` had maximum per-time group mean near `1.3e-17`.

## Per-asset adapter: local pass, public rejection

The adapter used only fold 0 for fitting and froze all parameters on folds 1--4:

```text
asset_shrink=100
market_lambda=0.7
blend_weight=1.17
```

Local gates passed. Peak improved by `+1.03%`, three of four later folds were positive, drop-best
remained `+0.30%`, and fold-0-frozen scale scores improved on all four folds.

The one-shot public result contradicted the local result:

```text
baseline 0.7/1.17: 0.00407075
with asset adapter: 0.0039613753
absolute delta:     -0.0001093747
relative delta:     -2.68684%
```

Runner integrity was not the cause: 3,217,458 rows, exact row alignment, zero timeouts, zero
non-finite values, and zero clipped rows. The likely failure is transfer instability in static
asset-specific slopes. The adapter is `REJECTED_PUBLIC`; it must not be combined with further
asset-conditioned leaderboard searches.

## Priority audit

| Priority | Status on 2026-08-17 | Decision |
|---|---|---|
| Delivery closure | still required for upstream production | keep separate from local research |
| Data refresh audit | waits for an actual changed release | audit before retraining |
| Extended-data retraining | blocked by the audit | use the frozen matrix only after a real change |
| Independent market rounds | completed | 3-seed confirmation keeps 480 rounds |
| Prediction-only market experts | rejected | large negative OOF deltas |
| Per-asset scaling | rejected by public score | restore no-adapter candidate |
| Asset magnitude/regime gates | unstable across rotating meta folds | no public submission |
| Low-rank and sparse residual interactions | rejected | negative strict OOF results |

This leaves feature selection as the next model-side structural question with a clear mechanism.

## Current feature-selection limitation

The current selector ranks each feature by absolute pooled Pearson correlation and keeps 200 of
323 anonymous columns. Ridge uses weighted correlation with `target`; XS LightGBM uses unweighted
correlation with the cross-sectional target `e`; history40 is selected again from the 200 XS
columns using contemporaneous correlation.

This method is fast and leakage-safe inside each fold, but it is univariate. It ignores redundancy,
time stability, nonlinear-only relevance, and whether a feature is intended for the market or XS
task. The strongest mismatch is that the market forest predicts row-level `y` and is reduced to a
per-time market mean, yet it reuses the 200 columns selected for XS target `e`.

## Frozen reselection screen

The next screen keeps feature counts and all model hyperparameters fixed. It compares only:

1. `baseline_corr`: current selection, used as the exact reference.
2. `market_task_aligned`: select the market forest's 200 raw columns from training-only per-time
   feature means against the unweighted market target; leave XS and history unchanged.
3. `xs_time_stable`: rank XS columns using contiguous training-time blocks, median absolute
   correlation, and sign consistency; leave market and history unchanged.
4. `history_lag_aligned`: keep the main 200 columns, but choose history40 from causal
   previous/difference/rolling transformations instead of contemporaneous deviation correlation.

No feature-count grid is allowed in this screen. A candidate must have positive paired mean Peak,
at least 4/5 positive folds, positive drop-best delta, and `2*delta_A > delta_B`. Screening success
only authorizes a 3-seed confirmation; it does not authorize a public CSV or expansion of the
candidate matrix.

The machine-readable preregistration is
`outputs/experiments/feature_reselection_plan_20260817.json`.
