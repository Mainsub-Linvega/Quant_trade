+# P4 Task-Aligned Feature Reselection Implementation Plan
+
+> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
+
+**Goal:** Implement and run the pre-registered ROADMAP P4 four-arm feature reselection screen with fixed model width, strict rolling OOF validation, paired metrics, and auditable fold artifacts.
+
+**Architecture:** Add a pure selector module for the three candidate selection rules and an explicit four-arm contract. Add a dedicated paired runner that shares each fold's preprocessing, target views, Ridge prediction, and baseline predictions, retraining only the LightGBM component affected by an arm. The runner writes atomic JSON/Markdown/manifest evidence and never changes production artifacts or creates a leaderboard submission.
+
+**Tech Stack:** Python 3, NumPy, SciPy/scikit-learn utilities already used by the repository, LightGBM, pytest/unittest-compatible tests, rolling time folds, JSON/Markdown experiment reports, and NumPy compressed artifacts.
+
+---
+
+## File Map
+
+- Create: `experiments/v3_task_aligned_reselection.py` - pure deterministic selectors, arm contracts, paired metrics, and the P4 CLI runner.
+- Create: `tests/test_v3_task_aligned_reselection.py` - synthetic selector, leakage, contract, metric, and artifact tests.
+- Create: `docs/superpowers/plans/2026-08-20-p4-task-aligned-feature-reselection-implementation.md` - this implementation plan.
+- Create at runtime: `outputs/experiments/<label>.json`, `outputs/experiments/<label>.md`, `outputs/experiments/<label>_folds/fold_<n>.json`, and `outputs/cache/<label>.npz`; these are evidence artifacts and are not production model files.
+- Read only: `experiments/v3_production_oof.py`, `experiments/v3_feature_structure.py`, `experiments/v3_causal_history_selection.py`, `experiments/history_peak.py`, `src.validation`, `train.py`, and `outputs/experiments/feature_reselection_plan_20260817.json`.
+- Do not modify: `strategies/v3_hybrid/model/`, `outputs/candidates/`, existing adaptive-selection artifacts, `ROADMAP.md`, or submission CSVs during the P4 screen.
+
+## Frozen Protocol And API
+
+The implementation must encode these values in one immutable configuration object or equivalent validated mapping:
+
+```python
+P4_ARMS = (
+    "baseline_corr",
+    "market_task_aligned",
+    "xs_time_stable",
+    "history_lag_aligned",
+)
+P4_COUNTS = {"ridge": 200, "xs": 200, "market": 200, "history": 40}
+P4_TRAIN_WINDOW = 78_960
+P4_EMBARGO = 6
+P4_SAMPLE_MODULO = 5
+P4_SAMPLING = "phase_balanced"
+P4_SCREEN_SEEDS = (2026,)
+P4_SCREEN_ROUNDS = 160
+P4_CONFIRM_SEEDS = (2026, 2027, 2028)
+P4_CONFIRM_ROUNDS = 480
+P4_MARKET_LAMBDA = 0.7
+P4_BLEND_WEIGHT = 1.17
+P4_PREDICTION_SCALE = 1.16
+P4_PREDICTION_CLIP = 0.5
+P4_HISTORY_WINDOW = 5
+```
+
+The pure selector API must be deterministic and independent of filesystem, global random state, validation rows, or LightGBM:
+
+```python
+def select_market_task_aligned(
+    transformed: np.ndarray,
+    target: np.ndarray,
+    time_ids: np.ndarray,
+    count: int = 200,
+) -> np.ndarray: ...
+
+def select_xs_time_stable(
+    transformed: np.ndarray,
+    cross_target: np.ndarray,
+    time_ids: np.ndarray,
+    count: int = 200,
+    n_blocks: int = 4,
+) -> np.ndarray: ...
+
+def select_history_lag_aligned(
+    transformed_xs: np.ndarray,
+    cross_target: np.ndarray,
+    time_ids: np.ndarray,
+    candidate_indices: np.ndarray,
+    count: int = 40,
+    window: int = 5,
+    n_blocks: int = 4,
+) -> dict[str, object]: ...
+
+def resolve_p4_arm(
+    arm: str,
+    baseline: Mapping[str, np.ndarray],
+    candidates: Mapping[str, np.ndarray],
+    counts: Mapping[str, int],
+) -> dict[str, np.ndarray]: ...
+
+def paired_gate(fold_rows: Sequence[Mapping[str, object]]) -> dict[str, object]: ...
+```
+
+Selectors must reject unsorted or mismatched arrays, non-finite training inputs, invalid counts, duplicate/out-of-range indices, and insufficient complete time blocks. Ties are resolved by ascending global feature index. All rankings select exactly the requested count; history selection selects whole base features and returns the chosen causal family evidence without changing the fixed model column contract.
+
+## Task 1: Add The Pure P4 Selector Module
+
+**Files:**
+- Create: `experiments/v3_task_aligned_reselection.py`
+- Test: `tests/test_v3_task_aligned_reselection.py`
+
+- [ ] **Step 1: Write failing validation and deterministic-ranking tests.**
+
+```python
+def test_market_selector_uses_per_time_unweighted_means_and_index_ties():
+    time_ids = np.repeat(np.arange(4), 3)
+    target = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
+    features = np.column_stack([
+        np.repeat([1.0, 2.0, 3.0, 4.0], 3),
+        np.repeat([4.0, 3.0, 2.0, 1.0], 3),
+        np.tile([0.0, 1.0, 2.0], 4),
+    ])
+    selected = select_market_task_aligned(features, target, time_ids, count=2)
+    np.testing.assert_array_equal(selected, [0, 1])
+
+def test_xs_stable_selector_prefers_sign_consistent_signal():
+    time_ids = np.repeat(np.arange(16), 2)
+    cross_target = np.tile([-1.0, 1.0], 16)
+    stable = np.tile([-1.0, 1.0], 16)
+    unstable = stable.copy()
+    unstable[time_ids >= 8] *= -1.0
+    selected = select_xs_time_stable(
+        np.column_stack([stable, unstable]), cross_target, time_ids, count=1
+    )
+    np.testing.assert_array_equal(selected, [0])
+
+def test_selectors_reject_validation_rows_and_unsorted_time_ids():
+    with pytest.raises(ValueError, match="sorted"):
+        select_market_task_aligned(
+            np.ones((4, 2)), np.ones(4), np.array([1, 0, 1, 2]), count=1
+        )
+
+def test_p4_arm_contract_keeps_single_stage_changes():
+    baseline = {name: np.arange(count) for name, count in P4_COUNTS.items()}
+    candidates = {
+        "market_task_aligned": {"market": np.arange(200, 400)},
+        "xs_time_stable": {"xs": np.arange(123, 323)},
+        "history_lag_aligned": {"history": np.arange(40)},
+    }
+    result = resolve_p4_arm("market_task_aligned", baseline, candidates, P4_COUNTS)
+    np.testing.assert_array_equal(result["xs"], baseline["xs"])
+    np.testing.assert_array_equal(result["market"], candidates["market_task_aligned"]["market"])
+```
+
+- [ ] **Step 2: Run the focused tests and verify the module is absent.**
+
+Run:
+
+```bash
+pytest -q tests/test_v3_task_aligned_reselection.py
+```
+
+Expected: collection fails with `ModuleNotFoundError` or the named selector imports are unavailable.
+
+- [ ] **Step 3: Implement validation, per-time aggregation, stable block scoring, and arm resolution.**
+
+Use only training arrays. For market ranking, aggregate every feature and target by complete `time_id`, compute Pearson correlation between the two unweighted per-time series, and sort by `(-abs(correlation), feature_index)`. For XS ranking, use complete contiguous time blocks, compute per-block cross-sectional Pearson correlations, score `median(abs(correlation)) * max(positive_fraction, negative_fraction)`, then sort by `(-score, -median_abs, feature_index)`. Implement causal history families `previous`, `difference`, `rolling_mean`, and `rolling_deviation` from prior observations only; score each candidate's strongest family by block median absolute correlation multiplied by sign consistency, and return exactly the top 40 bases plus evidence. Reuse existing history naming/order constants where compatible, but keep the module importable without loading data.
+
+- [ ] **Step 4: Run focused tests and add edge-case coverage.**
+
+Run:
+
+```bash
+pytest -q tests/test_v3_task_aligned_reselection.py
+```
+
+Expected: all pure selector and contract tests pass. Add tests for constant columns, insufficient blocks, non-finite values, duplicate candidate indices, history subset enforcement, causal first-row exclusion, and deterministic repeated calls.
+
+- [ ] **Step 5: Commit the pure selector boundary.**
+
+```bash
+git add experiments/v3_task_aligned_reselection.py tests/test_v3_task_aligned_reselection.py
+git commit -m "feat: add deterministic P4 feature selectors"
+```
+
+## Task 2: Implement Paired Metrics And The Four-Arm Runner
+
+**Files:**
+- Modify: `experiments/v3_task_aligned_reselection.py`
+- Test: `tests/test_v3_task_aligned_reselection.py`
+
+- [ ] **Step 1: Write failing paired-metric, early-stop, and artifact tests.**
+
+```python
+def test_paired_gate_requires_all_screen_conditions():
+    rows = [
+        {"fold": i, "baseline": {"peak": 1.0, "A": 2.0, "B": 4.0},
+         "candidate": {"peak": 1.1, "A": 2.2, "B": 4.0}}
+        for i in range(5)
+    ]
+    gate = paired_gate(rows)
+    assert gate["passed"] is True
+    assert gate["positive_folds"] == 5
+    assert gate["drop_best_mean_peak_delta"] > 0
+
+def test_paired_gate_rejects_candidate_that_improves_only_one_fold():
+    rows = [
+        {"fold": i, "baseline": {"peak": 1.0, "A": 2.0, "B": 4.0},
+         "candidate": {"peak": 2.0 if i == 0 else 0.99,
+                       "A": 2.2, "B": 4.0}}
+        for i in range(5)
+    ]
+    assert paired_gate(rows)["passed"] is False
+
+def test_atomic_bundle_contains_fold_manifest_and_no_submission_path(tmp_path):
+    paths = write_p4_bundle({"arms": {}, "folds": []}, tmp_path, "p4_test")
+    assert paths["json"].exists() and paths["markdown"].exists()
+    assert paths["fold_dir"].is_dir()
+    assert not any("submission" in path.name.lower() for path in tmp_path.iterdir())
+```
+
+- [ ] **Step 2: Run focused tests to verify the missing APIs fail.**
+
+Run:
+
+```bash
+pytest -q tests/test_v3_task_aligned_reselection.py -k 'gate or atomic'
+```
+
+Expected: import or attribute failures for `paired_gate` and `write_p4_bundle`.
+
+- [ ] **Step 3: Implement paired statistics and immutable arm evaluation.**
+
+For each candidate, compare fold-matched candidate and baseline payloads:
+
+```python
+delta_peak = candidate["peak"] - baseline["peak"]
+delta_A = candidate["A"] - baseline["A"]
+delta_B = candidate["B"] - baseline["B"]
+relative_delta_A = delta_A / max(abs(baseline["A"]), np.finfo(float).tiny)
+relative_delta_B = delta_B / max(abs(baseline["B"]), np.finfo(float).tiny)
+```
+
+Pass only when mean `delta_peak > 0`, at least four fold deltas are positive, drop-best mean peak delta is positive, `2 * relative_delta_A > relative_delta_B`, and every metric is finite. Record the exact formulas and fold deltas. Implement an early-stop predicate that marks an arm `early_stopped` only when even all remaining folds positive cannot reach four positive folds or cannot make the final mean peak delta positive.
+
+- [ ] **Step 4: Implement the dedicated fold runner.**
+
+Reuse `load_rows`, `rolling_time_folds`, `robust_transform_fit`, `apply_robust_transform`, `ridge_designs`, `fit_ridge`, `build_task_lgbm_designs`, `fit_predict_lgbm`, `fit_predict_lgbm_checkpoints`, `stream_history_blocks`, and `metric_payload` from the existing modules. Do not import or invoke adaptive manifests. The runner must:
+
+1. Parse `--arm-set` (default all four arms), `--data-root`, `--output-dir`, `--cache-dir`, `--label`, fold/window/embargo/sampling options, `--n-seeds`, `--num-iteration`, `--num-threads`, and `--force`.
+2. Reject any runtime value differing from the P4 frozen protocol unless an explicit `--mode confirmation` changes only seeds and rounds from screen to confirmation.
+3. Fit robust statistics once per fold and derive all four selectors from the same training window.
+4. Fit Ridge once and cache its validation prediction for every arm.
+5. Build baseline designs and predictions once. For `market_task_aligned`, rebuild/retrain only Market; for `xs_time_stable`, rebuild/retrain only XS; for `history_lag_aligned`, rebuild history and retrain both XS and Market because both consume history. Keep fusion at `market_lambda=0.7`, `blend_weight=1.17`, scale 1.16, clip 0.5.
+6. Write each fold manifest only after all arm metrics, selected indices, overlaps, runtime, peak RSS, and finite-value checks are complete. Write via temporary files followed by atomic rename.
+7. Release fold matrices and invoke garbage collection before the next fold. Never write to production or create a submission CSV.
+
+The final JSON must contain protocol, arm definitions, fold records, paired gates, baseline identity, selection hashes/indices, model specs, fusion constants, and artifact paths. The Markdown report must include a compact arm table, fold-level paired deltas, gate verdicts, rejected/early-stopped reasons, and the explicit statement that no arm was combined after observing results.
+
+- [ ] **Step 5: Run unit tests, then the real CLI help and dry validation.**
+
+Run:
+
+```bash
+pytest -q tests/test_v3_task_aligned_reselection.py
+python experiments/v3_task_aligned_reselection.py --help
+python experiments/v3_task_aligned_reselection.py --label p4_invalid --num-iteration 480
+```
+
+Expected: tests pass; help lists screen/confirmation and arm options; the invalid screen command exits before loading data because screen rounds are fixed at 160.
+
+- [ ] **Step 6: Commit the runner and tests.**
+
+```bash
+git add experiments/v3_task_aligned_reselection.py tests/test_v3_task_aligned_reselection.py
+git commit -m "feat: add paired ROADMAP P4 screen runner"
+```
+
+## Task 3: Synthetic Acceptance And Regression Verification
+
+**Files:**
+- Modify: `tests/test_v3_task_aligned_reselection.py`
+- Read: `tests/test_v3_feature_structure.py`, `tests/test_v3_adaptive_selection.py`, `tests/test_v3_interactions.py`, `tests/test_v3_asset_adapter.py`
+
+- [ ] **Step 1: Add a tiny synthetic end-to-end panel test.**
+
+Create a sorted panel with at least 8 complete time groups, 4 assets, 6 features, one market signal, one stable cross-sectional signal, one causal lag signal, duplicate noise columns, and a target generated only from past feature values plus a market component. Run the selector pipeline with counts below production counts and assert that selected indices are deterministic, no validation rows are read, history's first usable row is after the window, all four arm contracts have the same keys, and paired metrics are finite.
+
+- [ ] **Step 2: Run the synthetic test.**
+
+Run:
+
+```bash
+pytest -q tests/test_v3_task_aligned_reselection.py -k synthetic
+```
+
+Expected: PASS with no filesystem writes outside the pytest temporary directory.
+
+- [ ] **Step 3: Run the complete existing regression suite.**
+
+Run:
+
+```bash
+pytest -q
+```
+
+Expected: all existing tests and the new P4 tests pass. A pre-existing LightGBM/MLP convergence warning is acceptable only if the exit code remains zero and no new warning is introduced by P4.
+
+- [ ] **Step 4: Commit verification-only test additions.**
+
+```bash
+git add tests/test_v3_task_aligned_reselection.py
+git commit -m "test: verify P4 selector contracts end to end"
+```
+
+## Task 4: Run The Registered Five-Fold Screen
+
+**Files:**
+- Create at runtime only: `outputs/experiments/v3_p4_task_aligned_screen_1s160_phasebal_prodwindow.json`
+- Create at runtime only: `outputs/experiments/v3_p4_task_aligned_screen_1s160_phasebal_prodwindow.md`
+- Create at runtime only: `outputs/experiments/v3_p4_task_aligned_screen_1s160_phasebal_prodwindow_folds/`
+- Create at runtime only: `outputs/cache/v3_p4_task_aligned_screen_1s160_phasebal_prodwindow.npz`
+
+- [ ] **Step 1: Verify the data and branch boundary before running.**
+
+Run:
+
+```bash
+git status --short --branch
+test -f data/train.parquet || test -d data
+python -c 'import numpy, lightgbm; print("runtime_ok")'
+```
+
+Expected: current branch is `exp/adaptive-feature-search`; unrelated untracked user artifacts remain untouched; the data root and required runtime imports exist.
+
+- [ ] **Step 2: Execute the exact one-seed screen.**
+
+Run:
+
+```bash
+python experiments/v3_task_aligned_reselection.py \
+  --mode screen \
+  --label v3_p4_task_aligned_screen_1s160_phasebal_prodwindow \
+  --n-folds 5 --train-window 78960 --embargo 6 \
+  --sample-modulo 5 --sampling phase_balanced \
+  --n-seeds 1 --num-iteration 160 --num-threads 4 \
+  --market-lambda 0.7 --blend-weight 1.17 \
+  --prediction-scale 1.16 --prediction-clip 0.5
+```
+
+Expected: five strict rolling folds, fixed counts 200/200/200/40 in every arm, no validation-derived selector statistics, and a final report with one gate verdict per arm. The command must not generate a public CSV.
+
+- [ ] **Step 3: Audit the screen artifacts before interpreting results.**
+
+Run:
+
+```bash
+python -c 'import json; from pathlib import Path; p=Path("outputs/experiments/v3_p4_task_aligned_screen_1s160_phasebal_prodwindow.json"); x=json.loads(p.read_text()); assert x["protocol"]["embargo"]==6; assert x["protocol"]["train_window"]==78960; assert set(x["arms"])==set(("baseline_corr","market_task_aligned","xs_time_stable","history_lag_aligned")); print("p4_artifact_ok")'
+```
+
+Expected: `p4_artifact_ok`; every fold has finite metrics and selected counts; every candidate has five paired fold rows unless explicitly marked `early_stopped` with the mathematical reason recorded.
+
+- [ ] **Step 4: Review the screen result without expanding the matrix.**
+
+Only arms passing every screen gate may be scheduled for confirmation. Do not combine candidate selectors, alter counts, tune fusion, change rounds, or package a CSV based on the screen. If no arm passes, record P4 as screened and rejected while preserving the baseline and the existing `0.7/1.17` candidate.
+
+## Task 5: Conditional Three-Seed Confirmation
+
+**Files:**
+- Create at runtime only: `outputs/experiments/v3_p4_task_aligned_confirm_3s480_phasebal_prodwindow.json`
+- Create at runtime only: `outputs/experiments/v3_p4_task_aligned_confirm_3s480_phasebal_prodwindow.md`
+
+- [ ] **Step 1: Confirm only the exact screen-passing arm list.**
+
+Run the same runner with `--mode confirmation`, `--n-seeds 3`, `--num-iteration 480`, and an explicit `--arm-set` containing only screen-passing arms. The runner must refuse an arm absent from the screen-pass manifest and must preserve the same selectors, folds, counts, embargo, sampling, fusion, scale, and clip.
+
+- [ ] **Step 2: Audit confirmation gates and compare to the screen.**
+
+Run:
+
+```bash
+python -c 'import json; from pathlib import Path; x=json.loads(Path("outputs/experiments/v3_p4_task_aligned_confirm_3s480_phasebal_prodwindow.json").read_text()); assert x["protocol"]["n_seeds"]==3; assert x["protocol"]["num_iteration"]==480; print("p4_confirmation_ok")'
+```
+
+Expected: only pre-approved arms are confirmed, each with the same paired gate, finite artifacts, and a final accepted/rejected verdict. No production promotion or CSV generation occurs automatically.
+
+- [ ] **Step 3: Update research records in a separate follow-up change.**
+
+After reviewing the artifacts, update `ROADMAP.md`, `experiments/ledger.csv`, and a dated `research_history/` note together only if the user explicitly requests the record update. The record must state the arm, protocol, fold deltas, gate result, and whether the result is eligible for promotion. Keep the future candidate “state-conditioned multi-scale history experts” in a separate next-experiment section with a status of `OPTIONAL_AFTER_P4`, not as a P4 outcome.
+
+## Future Candidate Preserved Separately: State-Conditioned Multi-Scale History Experts
+
+This option remains available after P4 and is intentionally excluded from the four-arm screen. Its eventual design should be a new preregistered experiment with:
+
+1. causal history windows such as 1/3/5/10/20, computed only from feature observations available before the prediction row;
+2. compact state variables computed from the current market/cross-sectional panel, such as market dispersion, cross-sectional volatility, missingness rate, and recent history energy, with all thresholds fit inside the training fold;
+3. separate history experts for short and long horizons, trained on the same fixed fold protocol;
+4. a simple state-conditioned gate or convex blend whose parameters are learned only on the training portion, with a frozen fallback to the current history block when state support is insufficient;
+5. nested or purged validation for choosing horizon/gate settings, since state-conditioned gating can overfit regime labels;
+6. acceptance requiring improvement in paired Peak, `A/B` decomposition, cross-time stability, and inference cost, followed by three-seed confirmation.
+
+It must not reuse P4 results to select its state thresholds or history horizons after observing the P4 leaderboard-like evidence, and it must not be added to a P4 candidate arm after the screen.
+
+## Self-Review Checklist
+
+- [ ] Spec coverage: fixed four-arm definitions, strict folds, embargo, sampling, counts, causal history, paired metrics, early stopping, confirmation, and no CSV/promotion are all represented above.
+- [ ] Placeholder scan: no `TBD`, `TODO`, “implement later”, or unspecified test step remains.
+- [ ] Type consistency: selector names, arm names, protocol keys, output labels, and CLI flags are used consistently across all tasks.
+- [ ] Boundary check: adaptive selector artifacts remain excluded; production and user-generated untracked files are not modified.
+- [ ] Option A check: state-conditioned multi-scale history experts are recorded as a separate future experiment, not mixed into P4.
+
