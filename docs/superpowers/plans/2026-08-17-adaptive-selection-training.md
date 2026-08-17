# Adaptive Selection And Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build task-specific adaptive feature sets from all 323 anonymous features, validate them with strict rolling OOF, retrain the V3 hybrid, jointly recalibrate fusion, and generate one official-runner leaderboard CSV.

**Architecture:** A pure selector combines four-block marginal stability, stable redundancy clusters, shadow-feature noise floors, and repeated shallow-tree paths of three to six features. The OOF runner fits selection inside each training window; the final manifest is fit once on all training rows and consumed by candidate training and inference with separate Ridge, XS, market, and history feature contracts.

**Tech Stack:** Python 3.13, NumPy, SciPy, scikit-learn Ridge, LightGBM, pytest, JSON/NPZ/Markdown, official sequential runner.

---

### Task 1: Adaptive Selector Core

**Files:**
- Create: `experiments/v3_adaptive_selection.py`
- Create: `tests/test_v3_adaptive_selection.py`

- [ ] Write failing tests for deterministic shadow targets, cluster representative selection, stable three-feature path support, and non-fixed stopping counts.
- [ ] Run `.venv/bin/python -m pytest tests/test_v3_adaptive_selection.py -q` and confirm import failures.
- [ ] Implement `linear_evidence`, `extract_tree_paths`, `aggregate_path_support`, `select_from_clusters`, and `select_task_features` as pure functions.
- [ ] Require a feature to pass at least one training-only gate: stable marginal evidence above the circular-shift shadow floor, stable shallow-tree gain above shadow gain, or repeated membership in a path containing 3-6 distinct features.
- [ ] Select one representative per stable redundancy cluster, then retain a supported alternate only when it contributes a distinct accepted high-order path.
- [ ] Stop without a fixed count when no remaining candidate clears its gate; return evidence, reasons, path hyperedges, and selected indices.
- [ ] Run focused tests and `git diff --check`.
- [ ] Commit as `feat: add adaptive task feature selector`.

### Task 2: Selection Manifest CLI

**Files:**
- Create: `experiments/v3_adaptive_selection_manifest.py`
- Modify: `tests/test_v3_adaptive_selection.py`

- [ ] Add a failing test that a manifest contains separate `ridge`, `xs`, `market`, and `history` arrays plus path evidence and training protocol.
- [ ] Implement a CLI that loads sampled rows, fits robust preprocessing on the selected training window, builds the three task views, runs the selector, and chooses history bases from causal lag-aligned evidence within the XS pool.
- [ ] Freeze shallow-tree search before results: three chronological inner validations, 80 rounds, max depth 4, 15 leaves, seeds 2026/2027/2028, row cap 150000, shadow count 32.
- [ ] Write JSON and Markdown reports with selected counts as outputs, not inputs.
- [ ] Run a one-fold smoke manifest and validate all selected indices are unique and between 0 and 322.
- [ ] Commit as `feat: generate adaptive feature manifests`.

### Task 3: Strict Rolling OOF Comparison

**Files:**
- Modify: `experiments/v3_production_oof.py`
- Modify: `tests/test_v3_adaptive_selection.py`

- [ ] Add failing tests for resolving baseline versus adaptive per-fold selections and for separate XS/market design shapes.
- [ ] Add `--selection-mode baseline|adaptive`, selector budget options, and per-fold selection artifacts.
- [ ] Fit every preprocessing statistic, shadow floor, path, threshold, and selected set only on each outer training window.
- [ ] Train Ridge, weighted XS LightGBM, unweighted market LightGBM, and history using their own selected sets.
- [ ] Save OOF components `m_R`, `m_L`, `e_R`, `e_L` and fold metrics needed for joint fusion.
- [ ] Run screening at 1 seed x 160 rounds, then compare paired fold Peak against the existing baseline under the gates: positive mean, at least 4/5 positive, positive drop-best, and `2*delta_A > delta_B`.
- [ ] Only if screening passes, run 3 seeds x 480 rounds confirmation.
- [ ] Commit as `feat: validate adaptive features in strict oof`.

### Task 4: Task-Specific Candidate Training And Inference

**Files:**
- Modify: `strategies/v1_ridge/train.py`
- Modify: `strategies/v3_hybrid/train.py`
- Modify: `strategies/v3_hybrid/main.py`
- Modify: `tests/test_v3_adaptive_selection.py`

- [ ] Add failing tests for explicit Ridge selections and backward-compatible optional `market_features` metadata.
- [ ] Allow Ridge fitting with explicit selected indices while preserving the old count-based default.
- [ ] Add `--selection-manifest` to V3 training; train a candidate Ridge artifact and separate XS/market forests from manifest sets.
- [ ] Store independent preprocessing arrays for XS and market features; keep old metadata readable without changes.
- [ ] Build history only from manifest history bases and preserve the online causal state contract.
- [ ] Update offline and sequential inference to construct separate XS and market matrices with exact train/inference column order.
- [ ] Run unit tests, LightGBM-vs-NumPy backend checks, and full train/inference consistency checks.
- [ ] Commit as `feat: train task-specific adaptive hybrid`.

### Task 5: Joint Recalibration And Final CSV

**Files:**
- Create: `experiments/v3_joint_fusion_fit.py`
- Modify: `experiments/variant_submission.py`
- Create: `outputs/experiments/<adaptive-label>_fusion.json`
- Create: `outputs/candidates/<adaptive-label>/`
- Create: `outputs/submissions/<adaptive-label>.csv`

- [ ] Add tests for the weighted least-squares solution of `f = b + lambda*u + blend*v` and bounded fallback behavior.
- [ ] Fit `market_lambda` and `blend_weight` jointly from strict OOF components; fit final scale after fusion and before clipping.
- [ ] Report the weighted Gram matrix, coefficient covariance, fold-wise coefficients, frozen pooled coefficients, and sensitivity around the solution.
- [ ] Train the full candidate from the all-training manifest with 3 seeds x 480 rounds, preserving production models untouched.
- [ ] Run full tests and official sequential-runner consistency checks.
- [ ] Generate exactly one CSV from the candidate with the frozen OOF fusion values.
- [ ] Validate row count, exact `row_id` order, finite predictions, clip count, file size, and SHA-256; report the Windows path.

## Completion Gate

- The public leaderboard is never used for feature or parameter selection.
- Feature counts are outputs of stable gates, not fixed at 200/200/200/40.
- At least one accepted path contains three or more distinct features, or the report explicitly proves none clears the frozen stability gate.
- Production model files and PR #1 remain unchanged.
- The final CSV is generated only from the newly trained adaptive candidate through the official sequential runner.
