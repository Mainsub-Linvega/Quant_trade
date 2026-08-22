# Market Task-Aligned Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a training-only diagnostic that explains the fold instability of the completed P4 `market_task_aligned` arm without promoting it or creating a submission.

**Architecture:** Create a small diagnostic module that reads the completed P4 JSON/fold manifests, computes feature-set overlap and A/B/Peak attribution, then recreates only training-fold robust transforms to calculate four-block market-correlation stability. It writes atomic JSON/Markdown/fold evidence distinct from P4 artifacts. No LightGBM retrain is included in this implementation.

**Tech Stack:** Python 3, NumPy, PyArrow, existing P4 loaders/selectors, pytest, JSON/Markdown artifacts.

---

### Task 1: Pure Evidence Summaries

**Files:**
- Create: `experiments/v3_market_task_aligned_diagnostic.py`
- Modify: `tests/test_v3_market_task_aligned_diagnostic.py`

- [ ] **Step 1: Write failing overlap and attribution tests.**

```python
def test_market_overlap_reports_added_removed_and_jaccard():
    summary = market_overlap_summary(np.array([0, 1, 2]), np.array([1, 2, 3]))
    assert summary["overlap_count"] == 2
    assert summary["jaccard"] == pytest.approx(0.5)
    assert summary["added"] == [3]
    assert summary["removed"] == [0]

def test_peak_attribution_labels_alignment_loss():
    result = attribute_peak_delta(
        {"A": 1.0, "B": 1.0, "peak": 1.0},
        {"A": 0.8, "B": 0.9, "peak": 0.7},
        scale=1.16,
    )
    assert result["primary_cause"] == "alignment_loss"
```

- [ ] **Step 2: Run the tests and verify imports fail.**

Run: `.venv/bin/python -m pytest -q tests/test_v3_market_task_aligned_diagnostic.py`

Expected: import failure because the diagnostic module does not exist.

- [ ] **Step 3: Implement pure summaries.**

```python
def market_overlap_summary(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    base = set(np.asarray(baseline, dtype=np.int64).tolist())
    cand = set(np.asarray(candidate, dtype=np.int64).tolist())
    overlap = sorted(base & cand)
    union = base | cand
    return {
        "overlap_count": len(overlap),
        "jaccard": len(overlap) / len(union),
        "added": sorted(cand - base),
        "removed": sorted(base - cand),
    }
```

Implement `attribute_peak_delta` with `delta_A`, `delta_B`, `delta_peak`, and exactly one of `alignment_loss`, `energy_inflation`, `mixed`, or `improved`.

- [ ] **Step 4: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/test_v3_market_task_aligned_diagnostic.py`

Expected: PASS.

Commit:

```bash
git add experiments/v3_market_task_aligned_diagnostic.py tests/test_v3_market_task_aligned_diagnostic.py
git commit -m "feat: add market task diagnostic summaries"
```

### Task 2: Training-Only Stability Scan

**Files:**
- Modify: `experiments/v3_market_task_aligned_diagnostic.py`
- Modify: `tests/test_v3_market_task_aligned_diagnostic.py`

- [ ] **Step 1: Write failing stability test.**

```python
def test_market_block_stability_marks_sign_flip_as_unstable():
    time_ids = np.repeat(np.arange(8), 2)
    target = np.repeat(np.arange(8, dtype=float), 2)
    feature = target.copy()
    feature[time_ids >= 4] *= -1
    stability = market_block_stability(
        np.column_stack([feature, target]), target, time_ids, n_blocks=4,
    )
    assert stability["sign_consistency"][0] < 1.0
```

- [ ] **Step 2: Verify the test fails.**

Run: `.venv/bin/python -m pytest -q tests/test_v3_market_task_aligned_diagnostic.py -k stability`

Expected: missing `market_block_stability`.

- [ ] **Step 3: Implement training-only market aggregation and stability.**

Use existing `contiguous_time_blocks` and `select_market_task_aligned` conventions. Aggregate each feature and target by complete `time_id`, calculate blockwise correlations, then report full-window rank, block ranks, absolute correlations, rank dispersion, sign consistency, and top-200 block membership frequency.

- [ ] **Step 4: Add validation boundary checks.**

Pass only `train_slice` arrays to the scan. Include `train_time_range` and the number of training rows in every fold report. Do not accept validation arrays in the public scan function.

- [ ] **Step 5: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/test_v3_market_task_aligned_diagnostic.py`

Expected: PASS.

Commit:

```bash
git add experiments/v3_market_task_aligned_diagnostic.py tests/test_v3_market_task_aligned_diagnostic.py
git commit -m "feat: add market feature stability scan"
```

### Task 3: Artifact Runner And Real Audit

**Files:**
- Modify: `experiments/v3_market_task_aligned_diagnostic.py`
- Modify: `tests/test_v3_market_task_aligned_diagnostic.py`
- Create at runtime: `outputs/experiments/v3_market_task_aligned_diagnostic.{json,md}`
- Create at runtime: `outputs/experiments/v3_market_task_aligned_diagnostic_folds/`

- [ ] **Step 1: Write failing artifact test.**

```python
def test_write_diagnostic_bundle_is_atomic_and_contains_no_submission(tmp_path):
    paths = write_diagnostic_bundle({"folds": [{"fold": 0}]}, tmp_path, "market_diag")
    assert paths["json"].exists()
    assert (paths["fold_dir"] / "fold_0.json").exists()
    assert not any("submission" in path.name.lower() for path in tmp_path.iterdir())
```

- [ ] **Step 2: Verify it fails, implement atomic writer, then verify it passes.**

Run: `.venv/bin/python -m pytest -q tests/test_v3_market_task_aligned_diagnostic.py -k bundle`

Expected before implementation: import or attribute failure. Expected after implementation: PASS.

- [ ] **Step 3: Implement CLI runner.**

Read only the corrected P4 JSON/fold manifests and the existing feature memmap. Load sampled metadata, reproduce each outer training robust transform, run training-only stability scans, build the report, and assert `submission_generated=False`, `confirmation_run=False`, and `production_modified=False`.

- [ ] **Step 4: Run the real audit.**

Run:

```bash
.venv/bin/python experiments/v3_market_task_aligned_diagnostic.py \
  --p4-label v3_p4_task_aligned_screen_1s160_phasebal_prodwindow \
  --label v3_market_task_aligned_diagnostic
```

Expected: five fold artifacts, one JSON, one Markdown, no CSV and no model directories.

- [ ] **Step 5: Run full regression and commit.**

Run: `.venv/bin/python -m pytest -q`

Expected: exit code zero; existing MLP convergence warning is acceptable.

Commit:

```bash
git add experiments/v3_market_task_aligned_diagnostic.py tests/test_v3_market_task_aligned_diagnostic.py
git commit -m "feat: add market task-aligned diagnostic audit"
```
