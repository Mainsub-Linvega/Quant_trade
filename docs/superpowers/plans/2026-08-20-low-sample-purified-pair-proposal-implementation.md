# Low-Sample Purified Pair Proposal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scan all 52,003 pairs of the 323 source features on deterministic low-sample proposal folds, freeze at most 256 leakage-safe candidates, and leave the production model and submission artifacts untouched.

**Architecture:** Extend the existing strict-OOF task input with explicit fold labels, then add a standalone proposal module. Each chronological split fits per-feature quantile bins once, scores pair surfaces from compact pre-binned arrays using the existing purification mathematics, aggregates eligibility metrics, and selects a deterministic core/diversity shortlist. The CLI writes only experiment NPZ/JSON/Markdown artifacts and records source hashes, fold isolation, sampling, benchmark, and fallback decisions.

**Tech Stack:** Python 3.12, NumPy, pytest, existing `experiments/v3_purified_interactions.py` primitives, JSON/NPZ/Markdown artifacts.

---

## File Map

- Modify `experiments/v3_purified_interaction_input.py`: retain row-aligned OOF `fold` for Ridge/XS and validate one unique fold per Market `time_id`.
- Modify `experiments/v3_purified_interactions.py`: provide reusable pre-binned pair fitting/scoring with numerical parity to `score_pair_split`.
- Create `experiments/v3_low_sample_purified_proposal.py`: frozen protocol, sampling, chronological scans, selection, benchmark guard, CLI, and atomic reports.
- Modify `tests/test_v3_purified_interaction_input.py`: fold propagation and Market consistency tests.
- Modify `tests/test_v3_purified_interactions.py`: pre-binned scoring parity test.
- Create `tests/test_v3_low_sample_purified_proposal.py`: protocol, isolation, sampling, pair enumeration, eligibility, shortlist, determinism, benchmark, synthetic, and no-output-boundary tests.
- Create `outputs/experiments/v3_low_sample_purified_protocol_v1.json`: frozen machine-readable proposal protocol.

### Task 1: Preserve Explicit OOF Fold Labels

**Files:**
- Modify: `tests/test_v3_purified_interaction_input.py`
- Modify: `experiments/v3_purified_interaction_input.py`

- [ ] **Step 1: Write failing assertions for Ridge, XS, and Market folds**

Add assertions that Ridge and XS return `[0, 0, 1, 1]`, Market returns `[0, 1]`, and a Market group containing two fold values raises `ValueError("one fold")`.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_purified_interaction_input.py -q`
Expected: FAIL because builders do not return `fold` and Market does not validate group fold identity.

- [ ] **Step 3: Implement row-aligned fold propagation**

Keep `fold` in `_validated_component_rows`; return `np.int8` folds from Ridge/XS. In Market, compute fold at group starts and require:

```python
fold_by_time = fold[starts]
if not np.array_equal(fold, np.repeat(fold_by_time, counts)):
    raise ValueError("market input requires exactly one fold within each time_id")
```

Add manifest fields `fold_values`, `proposal_folds_present`, and `gate_folds_present` from the emitted array.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_v3_purified_interaction_input.py -q`
Expected: PASS.

Commit only the two files with message `feat: retain folds in purified task inputs`.

### Task 2: Freeze and Validate the Proposal Protocol

**Files:**
- Create: `tests/test_v3_low_sample_purified_proposal.py`
- Create: `experiments/v3_low_sample_purified_proposal.py`
- Create: `outputs/experiments/v3_low_sample_purified_protocol_v1.json`

- [ ] **Step 1: Write failing protocol tests**

Assert `default_proposal_protocol()` returns proposal folds `[0,1,2]`, gate folds `[3,4]`, 4 blocks, 40,000/20,000 row caps, 4 bins, minimum cell weight 32, first 1,024 lexical benchmark pairs, 30-minute/4-GiB ceilings, eligibility `2/3, >0, .80, .50`, and budgets 192+64. Mutations that overlap folds, alter frozen fusion, or make budgets exceed 256 must fail validation.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_low_sample_purified_proposal.py -q`
Expected: collection FAIL because the proposal module does not exist.

- [ ] **Step 3: Implement protocol API and checked-in JSON**

Expose:

```python
def default_proposal_protocol() -> dict[str, object]: ...
def validate_proposal_protocol(protocol: Mapping[str, object]) -> None: ...
```

Validation must compare frozen outer/fusion values to `default_purified_protocol()`, require disjoint exact fold sets, positive integral caps/budgets, `core + diversity <= 256`, and fixed eligibility thresholds. Serialize the validated default protocol with sorted keys.

- [ ] **Step 4: Verify GREEN and commit**

Run both proposal tests and `tests/test_v3_purified_interactions.py`; expect PASS. Commit the three files as `feat: freeze low-sample proposal protocol`.

### Task 3: Deterministic Complete-Time Sampling and Fold Isolation

**Files:**
- Modify: `tests/test_v3_low_sample_purified_proposal.py`
- Modify: `experiments/v3_low_sample_purified_proposal.py`

- [ ] **Step 1: Write failing tests for fail-closed input and sampling**

Test that `validate_proposal_arrays` rejects missing `fold`, mixed folds within one `time_id`, unordered time, and row misalignment. Test `split_proposal_gate_rows` returns only folds 0-2 versus 3-4 with no shared rows/time IDs. Test `sample_complete_time_groups` never returns part of a time group, is deterministic, stays under cap when one more full group would exceed it, and chooses approximately even group positions.

- [ ] **Step 2: Verify RED**

Run only these tests; expected missing-function failures.

- [ ] **Step 3: Implement validation, four time blocks, and sampling**

Expose:

```python
def validate_proposal_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]: ...
def split_proposal_gate_rows(fold: np.ndarray, protocol: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]: ...
def chronological_time_blocks(time_id: np.ndarray, n_blocks: int) -> list[np.ndarray]: ...
def sample_complete_time_groups(time_id: np.ndarray, candidate_rows: np.ndarray, row_cap: int) -> np.ndarray: ...
```

Use only ordered complete groups. Evenly spaced selection is deterministic and independent of features/residual/scores. Reject any time ID spanning multiple folds.

- [ ] **Step 4: Verify GREEN and commit**

Run proposal tests; expect PASS. Commit as `feat: isolate and sample proposal folds`.

### Task 4: Pre-Bin Features and Reuse Purification Math

**Files:**
- Modify: `tests/test_v3_purified_interactions.py`
- Modify: `experiments/v3_purified_interactions.py`
- Modify: `tests/test_v3_low_sample_purified_proposal.py`
- Modify: `experiments/v3_low_sample_purified_proposal.py`

- [ ] **Step 1: Write failing parity and train-only edge tests**

Fit edges on train rows, assign train/valid to `uint8` with sentinel `255`, and score a pair through the wished-for `score_prebinned_pair_split`. Assert gain, coverage, dominant-cell share, finite flag, and checksum match `score_pair_split` to `1e-12`. Include extreme validation values proving validation never changes edges.

- [ ] **Step 2: Verify RED**

Run the named parity tests; expected import/function failure.

- [ ] **Step 3: Implement compact binning and pre-binned scorer**

Add `PrebinnedFeatureSplit` plus:

```python
def prebin_feature_split(train_features, valid_features, *, bins) -> PrebinnedFeatureSplit: ...
def score_prebinned_pair_split(train_bins, valid_bins, train_residual, valid_residual,
                               train_weight, valid_weight, *, pair, bins,
                               min_cell_weight, max_surface_cells) -> dict[str, object]: ...
```

The scorer uses `np.bincount` for at most 16 cells, calls `purify_pair_surface`, maps missing/unsupported cells to zero, and shares checksum/diagnostic semantics with the existing scorer.

- [ ] **Step 4: Verify GREEN and commit**

Run both purified test files; expect PASS. Commit as `feat: score purified pairs from compact bins`.

### Task 5: Enumerate and Aggregate Every Pair Exactly Once

**Files:**
- Modify: `tests/test_v3_low_sample_purified_proposal.py`
- Modify: `experiments/v3_low_sample_purified_proposal.py`

- [ ] **Step 1: Write failing enumeration, split, and eligibility tests**

Assert `enumerate_lexical_pairs(323)` has length 52,003, starts `(0,1)`, ends `(321,322)`, and has no duplicates. For synthetic split scores, verify median/mean/drop-best, positive count, min coverage, max dominant share, finite aggregation, and exact eligibility boundaries.

- [ ] **Step 2: Verify RED**

Run the named tests; expected missing-function failures.

- [ ] **Step 3: Implement scanning**

Expose:

```python
def enumerate_lexical_pairs(n_features: int) -> list[tuple[int, int]]: ...
def proposal_split_rows(sampled_blocks: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]: ...
def aggregate_pair_scores(split_scores: Sequence[Mapping[str, object]]) -> dict[str, object]: ...
def proposal_eligible(summary: Mapping[str, object], protocol: Mapping[str, object]) -> bool: ...
def scan_prebinned_pairs(..., pairs: Sequence[tuple[int, int]] | None = None) -> dict[str, np.ndarray]: ...
```

The full scanner makes three expanding-train/next-block splits, pre-bins each feature once per split, and writes one row per lexical pair with pair indices and all aggregate/split metrics.

- [ ] **Step 4: Verify GREEN and commit**

Run proposal and purified tests; expect PASS. Commit as `feat: scan all purified pair proposals`.

### Task 6: Deterministic Core and Diversity Selection

**Files:**
- Modify: `tests/test_v3_low_sample_purified_proposal.py`
- Modify: `experiments/v3_low_sample_purified_proposal.py`

- [ ] **Step 1: Write failing shortlist tests**

Verify ranking uses drop-best, median, mean, then lexical pair. Verify core is the first 192 eligible rows without parent limits. Verify diversity uses only remaining eligible rows, prioritizes pairs with at least one parent outside baseline Top200, caps each parent at four diversity appearances, never changes core order, never duplicates core, and never fills from failed rows. Verify repeated selection returns identical lexical manifest bytes and SHA-256.

- [ ] **Step 2: Verify RED**

Run shortlist tests; expected missing-function failures.

- [ ] **Step 3: Implement baseline references and selection**

`load_baseline_indices(task, candidate_dir)` reads Ridge `baseline_model.json:selected_indices`; XS/Market parse `hybrid_meta.json:lgbm_features` as `feature_###`. Require exactly 200 unique indices in `[0,322]`, hash the source file, and return indices/path/hash. `select_proposal_candidates` returns core in score order, diversity in its deterministic priority order, and a final lexical-sorted manifest.

- [ ] **Step 4: Verify GREEN and commit**

Run proposal tests; expect PASS. Commit as `feat: freeze purified proposal shortlist`.

### Task 7: CLI, Atomic Reports, and Runtime Guard

**Files:**
- Modify: `tests/test_v3_low_sample_purified_proposal.py`
- Modify: `experiments/v3_low_sample_purified_proposal.py`

- [ ] **Step 1: Write failing CLI/report tests**

Test required `--input-npz`, `--candidate-dir`, `--label`, and `--output-dir`; absence of `fold` fails closed. Test benchmark extrapolation chooses 40,000 when <=1,800 seconds and <4 GiB, chooses exactly 20,000 only when the 40,000 estimate exceeds time, and stops when fallback still exceeds time or RSS exceeds budget. Ensure fallback receives only timing/RSS/pair-count inputs. Assert writes are atomic and output names are `<label>_pair_scores.npz`, `<label>_manifest.json`, `<label>.md`; assert no CSV and no `outputs/candidates` files.

- [ ] **Step 2: Verify RED**

Run CLI/report tests; expected failures.

- [ ] **Step 3: Implement bounded CLI and reports**

Benchmark only the first 1,024 lexical pairs on the same sampled blocks, measure wall time and `resource.getrusage(...).ru_maxrss`, linearly extrapolate 52,003 pairs, and apply the frozen fallback. Write score arrays, manifest, and Markdown through temporary siblings plus `Path.replace`. Include protocol/source/baseline hashes, input fold counts, selected time IDs and row indices checksum, benchmark metrics, fallback reason, exactly scanned pair count, shortlist metrics, `candidate_generated:false`, and `submission_generated:false`.

- [ ] **Step 4: Verify GREEN and commit**

Run proposal tests and CLI `--help`; expect PASS and no candidate/CSV flags. Commit as `feat: add bounded purified proposal cli`.

### Task 8: Synthetic End-to-End Acceptance

**Files:**
- Modify: `tests/test_v3_low_sample_purified_proposal.py`
- Modify: `experiments/v3_low_sample_purified_proposal.py`

- [ ] **Step 1: Write failing synthetic tests**

Build five ordered folds with repeated complete time groups and features containing noise, additive parents, and a zero-marginal XOR/range-switch pair. Assert the true interaction enters the shortlist, additive controls fail eligibility, every pair is scored exactly once per split, and two runs produce identical manifest SHA-256.

- [ ] **Step 2: Verify RED**

Run only synthetic tests; expected behavior failure.

- [ ] **Step 3: Add a synthetic-smoke CLI mode without changing thresholds**

The mode creates deterministic arrays only; it uses the production scanner, eligibility, and selector unchanged. It may reduce feature count for test speed but must preserve fold isolation and complete time groups.

- [ ] **Step 4: Verify GREEN and commit**

Run proposal tests plus `--synthetic-smoke`; expect XOR shortlisted and additive controls rejected. Commit as `test: cover purified proposal end to end`.

### Task 9: Full Verification and Ridge Resource Gate

**Files:**
- Generated only: `outputs/cache/v3_purified_p0_ridge_input.npz`
- Generated only: `outputs/cache/v3_purified_p0_ridge_input.json`
- Generated only when resource gate passes: `outputs/experiments/<ridge-label>*`

- [ ] **Step 1: Run focused and full regression suites**

Run: `.venv/bin/python -m pytest tests/test_v3_purified_interaction_input.py tests/test_v3_purified_interactions.py tests/test_v3_low_sample_purified_proposal.py -q`
Expected: PASS.

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass; only the existing MLP convergence warning is permitted.

- [ ] **Step 2: Rebuild the Ridge task input with explicit folds**

Run the input builder with `--task ridge --force`. Verify NPZ contains `features,residual,weight,time_id,fold,feature_indices`; folds are exactly 0-4; manifest hashes match.

- [ ] **Step 3: Run the fixed 1,024-pair Ridge benchmark**

Use the frozen `0.7/1.17` candidate directory as baseline reference. Record wall time, extrapolated full time, and peak RSS without inspecting pair ranks.

- [ ] **Step 4: Conditionally run the full Ridge proposal**

Proceed only if estimate <=30 minutes and RSS <=4 GiB. If 40,000 rows fails time, rerun benchmark once at 20,000; if that also fails, stop with a resource-blocked report. A completed scan must report exactly 52,003 pairs per split and must not create or modify production, candidate, or submission files.

- [ ] **Step 5: Verify repository boundaries and commit tracked experiment code only**

Run `git status --short`, compare tracked candidate/submission paths against the pre-run tree, and commit only planned source/test/protocol files. Leave generated caches/results untracked unless a later explicit decision records a small report.
