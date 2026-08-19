# Additive Interaction Features Design

**Date:** 2026-08-18  
**Branch:** `exp/adaptive-feature-search`  
**Status:** revised after user design approval

## Goal

Preserve the established V3 correlation-selected feature contracts and add task-specific joint effects as derived training columns. Interaction discovery may inspect all 323 current anonymous features, including low-marginal-correlation columns outside the baseline top 200, but an outside feature may enter a model only through an accepted derived interaction column. The work must not add another prediction model, replace the existing Ridge/XS LightGBM/Market LightGBM components, or use interaction evidence to delete established baseline features.

The frozen base is:

- Ridge: the 200 current anonymous feature sources with highest absolute weighted marginal correlation to the full target, represented by 200 robust current-value columns plus their 200 cross-sectional-deviation columns.
- XS LightGBM: the 200 current features with highest absolute unweighted marginal correlation to the cross-sectional target deviation, plus 40 causal history bases selected inside those 200 and expanded to 160 columns.
- Market LightGBM: the production XS top 200 represented as raw values and cross-sectional deviations, plus the same 160 causal history columns.
- Existing `asset_id`, preprocessing, fusion, clipping, `market_lambda=0.7`, and `blend_weight=1.17` remain unchanged during interaction evaluation.

Interaction columns are appended to those matrices. A failed interaction experiment therefore rolls back by omitting the new columns; it does not require reconstructing the production baseline.

## Non-Goals

- Do not replace the production top-200 selectors with another global raw-feature selector.
- Do not reduce the established 200/200/200/40 feature sets.
- Do not append a feature outside the production top 200 as an ordinary raw model column.
- Do not generate four history blocks for all 323 anonymous features; causal history remains the established History40.
- Do not enumerate every polynomial pair or triple.
- Do not add a fourth model or a new fusion coefficient.
- Do not tune final fusion or prediction scale before the interaction feature gate passes.
- Do not use public leaderboard feedback to discover, select, or weight interactions.

## Approaches Considered

### 1. Stable tree-path conditional features (selected)

Use shallow trees only as a training-time interaction miner. Convert repeated split paths into deterministic columns, then retrain the existing models with the original columns plus those derived columns.

This directly represents range-dependent effects, supports interactions involving two to four source features, and avoids an exhaustive feature cross.

### 2. Exhaustive polynomial crosses

For 323 inputs, pair products alone produce 52,003 columns per task before raw/deviation variants or history interactions. This is expensive, obscures threshold effects, and creates a large multiple-testing surface. It is rejected.

### 3. History-only crosses

Current/history products are important but do not cover interactions among current features or market regimes. This is retained as one interaction family inside approach 1, not used as the whole design.

## Architecture

### 1. Frozen baseline contracts

The existing marginal-correlation selectors remain the authoritative source for ordinary model columns. Adaptive-selection manifests are not used to replace them. A new interaction manifest may reference any of the 323 current anonymous columns by semantic source and global feature index, but only the resulting derived column is appended to the model matrix.

For example, if global `feature_250` is outside a task's top 200 but conditions the effect of top-200 `feature_030`, the model receives:

```text
feature_030 * I(feature_250 > threshold)
```

It does not receive `feature_250` as a standalone column. Inference still loads and preprocesses `feature_250` because it is a source needed to compute the derived column.

Each task has an independent interaction namespace:

- `ridge`: interactions mined from all 323 robust current features for the full weighted target.
- `xs`: interactions mined from all 323 cross-sectional deviations plus the existing History40 for target deviation.
- `market`: interactions mined from all 323 raw values, all 323 cross-sectional deviations, and the existing History40 for the market target.

An interaction discovered for one task is not copied into another task.

### 2. Training-only residual path mining

Within each outer training window:

1. Split the training window into four chronological inner blocks.
2. Fit the existing task model on expanding inner-train blocks using only frozen baseline columns.
3. Predict the next inner block and form out-of-sample residuals.
4. Fit shallow deterministic LightGBM miners to the residuals using the task's full interaction-source universe: all 323 current sources in the appropriate view plus the existing causal History40 where that task already uses it.
5. Extract root-to-leaf paths containing two to four distinct source columns.

The miner is not serialized as a prediction component. It only proposes derived feature definitions.

### 3. Canonical interaction definitions

Raw tree thresholds vary slightly between blocks. Every condition is therefore stored in a canonical training-only form:

- source family: `current`, `xs_deviation`, `market_raw`, or one of the four existing History40 blocks;
- global anonymous feature index;
- comparison direction;
- threshold represented by its empirical training quantile bin;
- missing-value direction.

Paths are canonicalized by source and quantile bin. A path is accepted as a candidate when the same canonical condition set appears in at least two inner validation blocks. This gate applies only to newly derived columns; no baseline column is removed when it fails.

### 4. Derived column families

For every accepted path, create:

1. **Region indicator**

   `1` when all path conditions are true, otherwise `0`.

2. **Gated value**

   The final path source value multiplied by the indicator formed from the preceding conditions. This models a feature whose slope changes in a particular region.

3. **Current/history gated value**

   When a path contains both a current and a causal history source, multiply the history value by the current-feature region indicator. This captures effects such as a previous value mattering only under a current regime.

Pure products are not generated unless they are supported by an accepted path. Paths with duplicate semantic sources, a single distinct feature, or only `asset_id` are rejected.

A derived column may use one or more current sources outside the task's top 200. Those source columns are inputs to the interaction builder only and are never inserted directly into the final model matrix. Current/history paths may use any of the 323 current sources but only the established History40 historical sources.

The accepted count is evidence-driven rather than fixed. An operational cell budget guards memory: exceeding it aborts the experiment with a report instead of silently truncating or ranking interactions.

### 5. Model matrices

The final training matrices become:

```text
Ridge:
  base_current_200 + base_deviation_200 + ridge_interactions

XS LightGBM:
  xs_deviation_200 + history_160 + xs_interactions + asset_id

Market LightGBM:
  market_raw_200 + market_deviation_200 + history_160
  + market_interactions + asset_id
```

The same existing Ridge and LightGBM trainers fit these enlarged matrices. No prediction is produced by the path miner itself.

The direct base widths remain unchanged: Ridge keeps 400 direct columns from 200 selected sources, XS keeps 360 direct non-asset columns, and Market keeps 560 direct non-asset columns. An interaction referencing a top-200-external source increases only the interaction width, not any direct baseline width.

### 6. Metadata and inference

The candidate metadata stores, separately for each task:

- interaction schema version;
- source columns and source families;
- robust preprocessing statistics for every referenced current source outside the task's direct top 200;
- canonical conditions and resolved numeric thresholds;
- operation type;
- output column order;
- training protocol and discovery blocks.

Offline inference, NumPy inference, LightGBM inference, and the official sequential runner call one shared interaction builder. The runner loads the union of direct baseline columns and interaction source columns, applies training-frozen preprocessing, constructs only the declared derived columns, and then discards source-only columns before model prediction. History conditions use only the existing History40 `AssetHistory` values available before the current row prediction. No target or future row is needed at inference time.

Old metadata without interaction definitions resolves to zero added columns and remains readable.

## Leakage Controls

- Robust preprocessing and threshold quantiles are fitted on inner-train rows only.
- Baseline residuals used for path mining are predictions on the next chronological block, never in-sample residuals.
- History state advances through the full chronological stream and uses no future observation.
- Every outer OOF fold rediscovers its own interaction definitions from that fold's training window.
- The final all-training manifest is built only after the frozen OOF gate passes.

## Validation

### Unit and consistency tests

- Canonical paths are invariant to condition order and small threshold movement within one quantile bin.
- A range-dependent synthetic effect produces the expected region and gated-value columns.
- Current/history interactions never read current-row history after state advancement.
- Original baseline columns and their order are unchanged.
- A source outside the top 200 can produce an interaction column but never appears as a standalone model column.
- History state remains exactly 40 bases and is not expanded to all 323 sources.
- Train, offline inference, NumPy inference, and official sequential inference construct identical columns.
- Old model metadata remains backward compatible.
- Cell-budget overflow fails explicitly and does not truncate interactions.

### Strict OOF gate

Run the existing five rolling folds with `train_window=78960`, `embargo=6`, phase-balanced modulo 5 sampling, one seed, and 160 rounds. Compare the exact frozen baseline against baseline plus interaction columns.

The screening gate remains:

- positive mean Peak change;
- at least four of five folds positive;
- positive result after dropping the best fold;
- improvement is driven by target alignment rather than only prediction-energy growth.

Stop as soon as the four-of-five gate becomes mathematically impossible. Only a passing screen advances to three seeds and 480 rounds.

## Outputs and Rollback

Each fold writes an interaction manifest and counts by task, source family, path order, and operation. Reports include overlap across folds and the OOF effect of adding all interaction columns.

Production artifacts and the current `0.7/1.17` candidate remain untouched. Rollback is deleting the candidate interaction metadata and derived matrices; the frozen baseline remains directly runnable.

## Success Criteria

- All established direct base columns remain present and in their original order, including both Ridge blocks derived from its 200 selected sources.
- Interaction discovery covers all 323 current anonymous features while direct raw inputs remain the production correlation-selected top 200.
- At least one repeated two-to-four-source conditional interaction is represented, or the report proves none satisfies the frozen training-only stability gate.
- No additional prediction model or fusion parameter is introduced.
- The strict OOF screening gate passes before final retraining or CSV generation.
- All inference backends reproduce the same interaction columns and predictions within existing numerical tolerances.
