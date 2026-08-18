# Additive Interaction Features Design

**Date:** 2026-08-18  
**Branch:** `exp/adaptive-feature-search`  
**Status:** proposed after user design approval

## Goal

Preserve the established V3 feature contracts and add task-specific joint effects as derived training columns. The work must not add another prediction model, replace the existing Ridge/XS LightGBM/Market LightGBM components, or use interaction evidence to delete established baseline features.

The frozen base is:

- Ridge: the established 200 current anonymous features.
- XS LightGBM: the established 200 cross-sectional current features plus 40 causal history bases expanded to 160 columns.
- Market LightGBM: the established 200 raw current features, their 200 cross-sectional deviations, and the same 160 causal history columns.
- Existing `asset_id`, preprocessing, fusion, clipping, `market_lambda=0.7`, and `blend_weight=1.17` remain unchanged during interaction evaluation.

Interaction columns are appended to those matrices. A failed interaction experiment therefore rolls back by omitting the new columns; it does not require reconstructing the production baseline.

## Non-Goals

- Do not perform another global feature-selection pass over the 323 anonymous columns.
- Do not reduce the established 200/200/200/40 feature sets.
- Do not enumerate every polynomial pair or triple.
- Do not add a fourth model or a new fusion coefficient.
- Do not tune final fusion or prediction scale before the interaction feature gate passes.
- Do not use public leaderboard feedback to discover, select, or weight interactions.

## Approaches Considered

### 1. Stable tree-path conditional features (selected)

Use shallow trees only as a training-time interaction miner. Convert repeated split paths into deterministic columns, then retrain the existing models with the original columns plus those derived columns.

This directly represents range-dependent effects, supports interactions involving two to four source features, and avoids an exhaustive feature cross.

### 2. Exhaustive polynomial crosses

For 200 inputs, pair products alone produce 19,900 columns per task before history interactions. This is expensive, obscures threshold effects, and creates a large multiple-testing surface. It is rejected.

### 3. History-only crosses

Current/history products are important but do not cover interactions among current features or market regimes. This is retained as one interaction family inside approach 1, not used as the whole design.

## Architecture

### 1. Frozen baseline contracts

The existing baseline feature lists remain the authoritative source for each model. Adaptive-selection manifests are not used to replace them. A new interaction manifest references columns in the frozen base matrices by semantic source and global feature index.

Each task has an independent interaction namespace:

- `ridge`: current-feature interactions for the full weighted target.
- `xs`: cross-sectional current/current and current/history interactions for target deviation.
- `market`: raw/deviation/current-history interactions for the market target.

An interaction discovered for one task is not copied into another task.

### 2. Training-only residual path mining

Within each outer training window:

1. Split the training window into four chronological inner blocks.
2. Fit the existing task model on expanding inner-train blocks using only frozen baseline columns.
3. Predict the next inner block and form out-of-sample residuals.
4. Fit shallow deterministic LightGBM miners to the residuals using the same baseline columns, including causal history where that task already uses it.
5. Extract root-to-leaf paths containing two to four distinct source columns.

The miner is not serialized as a prediction component. It only proposes derived feature definitions.

### 3. Canonical interaction definitions

Raw tree thresholds vary slightly between blocks. Every condition is therefore stored in a canonical training-only form:

- source family: `current`, `xs_deviation`, `market_raw`, or one of the four history blocks;
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

The accepted count is evidence-driven rather than fixed. An operational cell budget guards memory: exceeding it aborts the experiment with a report instead of silently truncating or ranking interactions.

### 5. Model matrices

The final training matrices become:

```text
Ridge:
  base_current_200 + ridge_interactions

XS LightGBM:
  xs_deviation_200 + history_160 + xs_interactions + asset_id

Market LightGBM:
  market_raw_200 + market_deviation_200 + history_160
  + market_interactions + asset_id
```

The same existing Ridge and LightGBM trainers fit these enlarged matrices. No prediction is produced by the path miner itself.

### 6. Metadata and inference

The candidate metadata stores, separately for each task:

- interaction schema version;
- source columns and source families;
- canonical conditions and resolved numeric thresholds;
- operation type;
- output column order;
- training protocol and discovery blocks.

Offline inference, NumPy inference, LightGBM inference, and the official sequential runner call one shared interaction builder. History conditions use only `AssetHistory` values available before the current row prediction. No target or future row is needed at inference time.

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

- All established base features remain present and in their original order.
- At least one repeated two-to-four-source conditional interaction is represented, or the report proves none satisfies the frozen training-only stability gate.
- No additional prediction model or fusion parameter is introduced.
- The strict OOF screening gate passes before final retraining or CSV generation.
- All inference backends reproduce the same interaction columns and predictions within existing numerical tolerances.
