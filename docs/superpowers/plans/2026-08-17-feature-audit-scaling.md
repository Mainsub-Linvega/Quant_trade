# Feature Audit Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the feature-structure audit practical at production sampling by bounding rank-correlation work and moving dense matrices from JSON into a compressed NPZ bundle.

**Architecture:** Keep statistical sampling in `experiments/v3_feature_structure.py`. Keep report extraction and artifact writing in `experiments/v3_feature_structure_audit.py`; JSON remains the machine-readable index while NPZ stores dense matrices referenced by stable keys.

**Tech Stack:** Python 3.13, NumPy, SciPy, pytest, compressed NPZ, JSON, Markdown.

---

### Task 1: Deterministic Redundancy Sampling

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Modify: `experiments/v3_feature_structure.py`

- [ ] **Step 1: Write failing tests**

```python
def test_evenly_spaced_rows_is_deterministic_and_keeps_endpoints():
    got = evenly_spaced_rows(10, 4)
    np.testing.assert_array_equal(got, [0, 3, 6, 9])

def test_stable_redundancy_row_cap_preserves_duplicate_cluster():
    base = np.arange(400, dtype=float)
    x = np.column_stack([base, base * 2, np.sin(base)])
    tid = np.repeat(np.arange(200), 2)
    result = stable_redundancy(x, tid, n_blocks=4, threshold=0.05,
                               max_rows_per_block=25)
    assert result.labels[0] == result.labels[1]
    assert result.sampled_rows_per_block == [25, 25, 25, 25]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: imports/signature fail because deterministic sampling is absent.

- [ ] **Step 3: Implement sampling**

```python
def evenly_spaced_rows(n_rows: int, limit: int | None) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if limit is None or limit >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    if limit < 2:
        raise ValueError("limit must be at least two")
    return np.rint(np.linspace(0, n_rows - 1, limit)).astype(np.int64)
```

Add `max_rows_per_block` to `stable_redundancy`. Apply sampling independently inside each contiguous time block before Pearson/Spearman calculation. Extend `RedundancyResult` with `sampled_rows_per_block` so reports can audit the approximation.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: all feature-structure tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/v3_feature_structure.py tests/test_v3_feature_structure.py
git commit -m "perf: bound feature redundancy sampling"
```

### Task 2: Split Dense Matrices from JSON

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Modify: `experiments/v3_feature_structure_audit.py`

- [ ] **Step 1: Write failing extraction test**

```python
def test_extract_dense_matrices_replaces_arrays_with_npz_references():
    report = {"folds": [{"fold": 0, "tasks": {"ridge": {
        "redundancy": {"stability": np.eye(2), "labels": np.array([1, 2])}
    }}}]}
    summary, matrices = extract_dense_matrices(report)
    assert "fold_0.ridge.redundancy.stability" in matrices
    assert summary["folds"][0]["tasks"]["ridge"]["redundancy"]["stability"] == {
        "npz_key": "fold_0.ridge.redundancy.stability", "shape": [2, 2]
    }
    assert summary["folds"][0]["tasks"]["ridge"]["redundancy"]["labels"] == [1, 2]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: import fails for `extract_dense_matrices`.

- [ ] **Step 3: Implement extraction**

Only two-dimensional redundancy arrays with both dimensions greater than 16 move to NPZ. Convert moved arrays to float32. Keep labels and quality vectors in JSON. Use stable keys `fold_<n>.<task>.redundancy.<name>`.

```python
def extract_dense_matrices(report):
    summary = copy.deepcopy(report)
    matrices = {}
    for fold in summary["folds"]:
        for task_name, task in fold["tasks"].items():
            for name, value in list(task.get("redundancy", {}).items()):
                array = np.asarray(value)
                if array.ndim == 2 and min(array.shape) > 16:
                    key = f"fold_{fold['fold']}.{task_name}.redundancy.{name}"
                    matrices[key] = array.astype(np.float32)
                    task["redundancy"][name] = {"npz_key": key, "shape": list(array.shape)}
    return summary, matrices
```

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: extraction test passes.

- [ ] **Step 5: Commit**

```bash
git add experiments/v3_feature_structure_audit.py tests/test_v3_feature_structure.py
git commit -m "feat: split feature audit matrix artifacts"
```

### Task 3: Write and Validate Report Bundle

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Modify: `experiments/v3_feature_structure_audit.py`

- [ ] **Step 1: Write failing bundle test**

```python
def test_write_report_bundle_round_trips_json_and_npz(tmp_path):
    report = {"config": {"n_blocks": 4}, "folds": [{"fold": 0, "tasks": {
        "ridge": {"status": "ok", "n_features": 2, "cluster_count": 1,
                  "redundancy": {"stability": np.eye(20)}}}}],
              "gram": {"status": "not_available"}}
    paths = write_report_bundle(report, tmp_path, "audit")
    loaded = json.loads(paths["json"].read_text())
    with np.load(paths["npz"]) as matrices:
        assert matrices["fold_0.ridge.redundancy.stability"].shape == (20, 20)
    assert loaded["matrix_artifact"] == "audit_matrices.npz"
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: import fails for `write_report_bundle`.

- [ ] **Step 3: Implement bundle writing and CLI options**

Add CLI option:

```text
--redundancy-rows-per-block 100000
```

Pass it through `_task_report` to `stable_redundancy`. `write_report_bundle` writes:

```text
<label>.json
<label>.md
<label>_matrices.npz
```

Use `np.savez_compressed`, `json.dumps(..., allow_nan=False)`, and refuse overwrite of all three files unless `--force` is provided.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: all focused tests pass.

- [ ] **Step 5: Run full regression suite**

Run: `.venv/bin/python -m pytest -q`
Expected: baseline plus new tests pass; only the existing sklearn convergence warning may remain.

- [ ] **Step 6: Commit**

```bash
git add experiments/v3_feature_structure.py experiments/v3_feature_structure_audit.py tests/test_v3_feature_structure.py
git commit -m "feat: write scalable feature audit bundles"
```

### Task 4: Production-Protocol Audit

**Files:**
- Create: `outputs/experiments/v3_feature_structure_prod_5fold.json`
- Create: `outputs/experiments/v3_feature_structure_prod_5fold.md`
- Create: `outputs/experiments/v3_feature_structure_prod_5fold_matrices.npz`

- [ ] **Step 1: Run production-protocol audit**

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  experiments/v3_feature_structure_audit.py \
  --sample-modulo 5 --sampling phase_balanced \
  --n-folds 5 --train-window 78960 --embargo 6 \
  --n-blocks 4 --cluster-threshold 0.15 \
  --redundancy-rows-per-block 100000 \
  --label v3_feature_structure_prod_5fold --force
```

- [ ] **Step 2: Validate artifacts**

Run: `.venv/bin/python -m json.tool outputs/experiments/v3_feature_structure_prod_5fold.json`
Expected: valid JSON with five folds and no NaN/Infinity.

Run: `.venv/bin/python -c "import numpy as np; d=np.load('outputs/experiments/v3_feature_structure_prod_5fold_matrices.npz'); print(len(d.files))"`
Expected: dense matrices exist for each completed task/fold.

- [ ] **Step 3: Record decision summary**

The Markdown must report per task: cluster counts, high-stability features, cross-fold cluster consistency and whether market/XS structures differ enough to justify task-specific selection. This artifact becomes the only input to the next conditional-feature plan.

## Completion Gate

- Production-model files remain unchanged.
- The public leaderboard is not used.
- Dense matrices are absent from JSON and present in NPZ.
- Redundancy sampling is deterministic and reported.
- Five folds complete under the production sampling protocol.
