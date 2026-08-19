# Stable Subpath Interaction Discovery Design

**Date:** 2026-08-19  
**Branch:** `exp/adaptive-feature-search`  
**Status:** approved approach, frozen before implementation

## Context

The first additive-interaction screen completed two outer folds and stopped early. Each task and inner validation block produced roughly 1,100 complete LightGBM paths, but no exact complete path repeated in two inner time blocks. Consequently Ridge, XS LightGBM, and Market LightGBM each received zero derived columns, both paired OOF deltas were exactly zero, and the unchanged four-of-five-positive gate correctly rejected the experiment.

This result does not show that the 323 anonymous features contain no joint effects. It shows that exact matching of every condition in a complete path is too brittle: a stable two-feature conditional effect can be embedded in different three- or four-condition tree paths across time blocks.

## Goal

Moderately relax training-only interaction discovery by matching stable two-to-four-condition subpaths and coarsening threshold support from 32 to 16 quantile regions. Preserve every production model, direct feature set, fusion parameter, leakage control, and final OOF acceptance condition from the approved additive-interaction design.

## Frozen Boundaries

- All 323 current anonymous features remain eligible only as interaction sources.
- Production Top200 and History40 direct inputs remain unchanged.
- Ridge, XS LightGBM, and Market LightGBM remain the only prediction models.
- `market_lambda=0.7` and `blend_weight=1.17` remain fixed.
- A source outside Top200 may enter only through an accepted derived column.
- Inner support still requires at least two distinct chronological validation blocks.
- The outer gate still requires positive mean Peak delta, at least four of five positive folds, positive drop-best mean, and `2 * delta_A > delta_B`.
- No candidate, final retraining, or submission CSV is produced unless the screen and later confirmation pass.
- There is no fallback that relaxes the rules again after observing the new OOF result.

## Discovery Algorithm

### 1. Complete path extraction

Keep the existing deterministic residual miners and strict expanding inner OOS residuals. Extract valid root-to-leaf paths containing two to four distinct semantic sources. Split direction and missing-value direction remain part of every condition.

### 2. Subpath expansion

For a complete path with `n` conditions, emit every condition combination of sizes `2..n` while retaining the selected conditions' original tree order:

- two-condition path: one pair;
- three-condition path: three pairs and one triple;
- four-condition path: six pairs, four triples, and one quadruple.

Only combinations with distinct semantic sources are valid. Identical subpaths proposed by multiple leaves or trees inside one block are deduplicated for stability counting. Their repeated occurrence within one block never increases cross-block support.

The original condition order is retained in the deterministic representative because `gated_value` uses the final selected condition as its value source. Stability matching remains order-invariant so the same conditional region can match when trees split the sources in a different order.

### 3. Coarse threshold canonicalization

Fit source-specific quantile grids using inner-training rows only. The support grid has 16 regions, mathematically equivalent to fitting 32 quantile bins and merging adjacent pairs. A support token contains:

```text
semantic source + comparison direction + 16-bin region + missing direction
```

Conditions match only when all four fields match. Direction and missing behavior are not relaxed. The accepted representative stores the 16-bin index, and each outer-training window resolves that index again against its own source distribution. Validation data never determines a production threshold.

### 4. Cross-block aggregation

Group canonical subpaths by the order-invariant set of support tokens. Accept a subpath only when it appears in at least two distinct inner validation blocks. Support is the number of distinct blocks, not the number of trees, leaves, or duplicate occurrences.

Sort accepted paths deterministically by descending block support, then canonical support key and representative location. Generate the existing region, gated-value, and current/history-gated definitions without adding a new operation or model.

### 5. Diagnostics and budgets

Each task manifest records, per inner split:

- complete candidate path count;
- expanded subpath count before within-block deduplication;
- unique canonical subpath count after within-block deduplication.

It also records `support_quantile_bins=16`, the unchanged minimum support of two blocks, and accepted counts by order. Existing source-cell and interaction-cell budgets remain hard failures. They do not silently truncate or select a fixed number of interactions.

## OOF Protocol

Run a new experiment label so the exact-path report remains immutable. Use the same frozen screen:

```text
5 folds
train_window = 78960
embargo = 6
phase-balanced modulo 5
1 seed x 160 rounds
market_lambda = 0.7
blend_weight = 1.17
```

Each outer fold rediscovers subpaths from that fold's training window. Baseline A and augmented B share all preprocessing, direct columns, history state, targets, weights, seeds, and model parameters. Stop as soon as four positive folds become mathematically impossible.

## Tests

- A three-condition complete path expands to its three pairs and one triple.
- A four-condition path expands to 11 unique ordered representatives.
- Reversed tree order produces the same support key but preserves each representative's order.
- Thresholds in adjacent original 32-bin regions map to one 16-bin support region.
- Thresholds across a 16-bin boundary do not match.
- Duplicate proposals inside one block count as one block of support.
- The same subpath in two distinct blocks is accepted.
- Source, direction, or missing-direction disagreement prevents a match.
- Outer threshold resolution uses 16 bins.
- Existing direct matrices, interaction operations, memory gates, and strict OOF gates remain unchanged.

## Success and Rollback

The discovery revision succeeds only if at least one derived column is proposed and the unchanged strict OOF gate passes. Finding interactions is not itself success. A zero-interaction result, a memory-budget failure, or an OOF failure leaves the current `0.7/1.17` production candidate unchanged and produces no CSV.

