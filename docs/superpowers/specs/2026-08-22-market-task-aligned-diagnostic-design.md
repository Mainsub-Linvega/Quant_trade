# Market Task-Aligned Diagnostic Design

**Date:** 2026-08-22  
**Branch:** `exp/adaptive-feature-search`  
**Status:** approved for implementation

## 1. Goal

Explain why the ROADMAP P4 arm `market_task_aligned` improves the mean paired
Peak but fails the four-of-five positive-fold gate, with particular attention
to validation folds 1 and 2. The diagnostic must not modify production,
promote an arm, enter confirmation, or generate a leaderboard CSV.

## 2. Frozen Reference

Use only the completed, corrected P4 screen:

```text
label: v3_p4_task_aligned_screen_1s160_phasebal_prodwindow
folds: 5
seed: 2026
rounds: 160
train window: 78,960 time_ids
embargo: 6 time_ids
sampling: phase_balanced
sample modulo: 5
market_lambda: 0.7
blend_weight: 1.17
prediction_scale: 1.16
prediction_clip: 0.5
```

The corrected screen distinguishes baseline Market200, which is the baseline
XS200 set, from the candidate Market200 selected by per-time market-mean
association. Earlier screen artifacts with an identical History arm are not
valid evidence and must not be reused.

## 3. Diagnostic Questions

Answer four questions separately:

1. How many Market features are replaced in each fold, and is replacement
   concentrated in folds 1 and 2?
2. Are candidate market features stable across contiguous training periods,
   or are they selected by a full-window correlation that is regime-specific?
3. In folds 1 and 2, does the candidate lose Peak because target alignment
   `A` falls, prediction energy `B` grows, or both?
4. Can a low-cost one-factor retrain reproduce the direction of the effect
   while keeping Ridge, XS, History, and fusion constants unchanged?

## 4. Work Packages

### 4.1 Existing-evidence audit

Read the corrected P4 JSON and fold manifests. For each fold calculate:

- Market200 overlap count and Jaccard similarity between baseline and candidate;
- added and removed feature indices;
- paired `Peak`, `A`, and `B` deltas;
- whether the fold is positive or negative;
- whether the candidate's final fusion output is finite.

No model retraining is needed for this package.

### 4.2 Training-only stability scan

For each outer training window, reuse the same robust-transformed training
features and complete time groups. Split the training time IDs into four
contiguous blocks. For every feature calculate blockwise market-mean
correlation with the blockwise mean target. Report:

- full-window rank and absolute correlation;
- block ranks and absolute correlations;
- rank dispersion;
- sign consistency;
- top-200 membership frequency across blocks;
- membership in the candidate set but not baseline, and vice versa.

The scan is training-only. Validation rows must not be read for selection
statistics. Feature order and ties use the global feature index.

### 4.3 A/B/Peak attribution

For each fold and for the pooled OOF rows, report:

```text
delta_A = A_candidate - A_baseline
delta_B = B_candidate - B_baseline
delta_Peak = Peak_candidate - Peak_baseline
```

Also report the identity-level contribution under the frozen scale where
useful:

```text
Score(a) = 2*a*A - a^2*B
```

The report must identify whether negative folds are primarily alignment loss,
energy inflation, or a mixed effect. It must not infer organizer weights from
the local diagnostic.

### 4.4 Optional low-cost one-factor retrain

Only if packages 4.1 to 4.3 leave the cause ambiguous, run a diagnostic
retrain that changes Market200 only. Keep the original Ridge, XS, History,
fusion constants, fold protocol, and seed fixed. Use a registered diagnostic
label and record predictions/components, but do not evaluate it as a P4
promotion candidate and do not produce a CSV.

## 5. Acceptance And Outputs

The diagnostic is successful when it produces a machine-readable report with:

- all five fold replacement and stability summaries;
- complete A/B/Peak attribution;
- explicit conclusions for folds 1 and 2;
- finite-value checks;
- a statement that no promotion, confirmation, or submission was performed.

Expected artifacts:

```text
outputs/experiments/v3_market_task_aligned_diagnostic.json
outputs/experiments/v3_market_task_aligned_diagnostic.md
outputs/experiments/v3_market_task_aligned_diagnostic_folds/
```

The report may reference the existing P4 NPZ and JSON, but must not overwrite
them. Any optional retrain uses a separate label and separate cache.

## 6. Tests

Add focused synthetic tests for:

- overlap and Jaccard calculation;
- blockwise market rank stability and sign consistency;
- A/B/Peak delta attribution;
- rejection of validation rows in training-only stability inputs;
- atomic diagnostic artifact output without submission files.

Run the focused tests and the existing full regression suite before claiming
the diagnostic is complete.
