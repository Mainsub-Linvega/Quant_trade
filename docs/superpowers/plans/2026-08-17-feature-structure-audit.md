# Feature Structure Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe first-stage audit that constructs model-specific task views, measures feature quality/stability/drift/redundancy, analyzes weighted fusion coupling, and writes reproducible JSON/Markdown reports.

**Architecture:** Put pure NumPy/SciPy algorithms in `experiments/v3_feature_structure.py` and keep parquet loading, rolling-fold orchestration, CLI validation, and report writing in `experiments/v3_feature_structure_audit.py`. Synthetic pytest coverage defines every algorithm before implementation; the CLI reuses `experiments/lgbm_xs.py::load_rows`, `src.validation.rolling_time_folds`, and production feature transforms without changing production code.

**Tech Stack:** Python 3.13, NumPy, SciPy hierarchical clustering, PyArrow through the existing loader, pytest, JSON, Markdown.

---

## File Map

- Create `experiments/v3_feature_structure.py`: pure task-view, block, drift, redundancy, clustering and Gram algorithms.
- Create `experiments/v3_feature_structure_audit.py`: CLI, data loading, outer/inner time slicing, orchestration and reports.
- Create `tests/test_v3_feature_structure.py`: synthetic unit tests for all pure behavior and a small report test.
- Do not modify `strategies/v3_hybrid/model/`, production training code, or PR #1.

### Task 1: Task Views

**Files:**
- Create: `tests/test_v3_feature_structure.py`
- Create: `experiments/v3_feature_structure.py`

- [ ] **Step 1: Write failing tests for complete cross sections and task decomposition**

```python
def test_build_task_views_separates_market_and_cross_section():
    time_id = np.repeat([10, 11], 3)
    features = np.array([[1, 2], [2, 4], [3, 6], [4, 3], [5, 6], [6, 9]], dtype=float)
    target = np.array([1, 2, 3, 4, 5, 9], dtype=float)
    weight = np.array([1, 2, 1, 1, 1, 2], dtype=float)
    views = build_task_views(features, target, weight, time_id)
    np.testing.assert_allclose(views.market_features, [[2, 4], [5, 6]])
    np.testing.assert_allclose(np.add.reduceat(weight * views.cross_target, [0, 3]), 0, atol=1e-12)
    np.testing.assert_allclose(views.cross_features.reshape(2, 3, 2).mean(axis=1), 0, atol=1e-12)

def test_build_task_views_rejects_unsorted_time_ids():
    with pytest.raises(ValueError, match="sorted"):
        build_task_views(np.ones((3, 2)), np.ones(3), np.ones(3), np.array([1, 0, 1]))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: collection fails because `experiments.v3_feature_structure` does not exist.

- [ ] **Step 3: Implement `TaskViews` and `build_task_views`**

```python
@dataclass(frozen=True)
class TaskViews:
    raw_features: np.ndarray
    full_target: np.ndarray
    market_features: np.ndarray
    market_target: np.ndarray
    cross_features: np.ndarray
    cross_target: np.ndarray
    unique_time_ids: np.ndarray
    starts: np.ndarray
    counts: np.ndarray

def build_task_views(features, target, weight, time_ids) -> TaskViews:
    # Validate finite dimensions and sorted groups.
    # Market features use the unweighted row mean.
    # Market target uses the official weighted row mean.
    # Cross views subtract their corresponding per-time means.
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: task-view tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/v3_feature_structure.py tests/test_v3_feature_structure.py
git commit -m "feat: add feature audit task views"
```

### Task 2: Four-Block Quality, Stability and Drift

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Modify: `experiments/v3_feature_structure.py`

- [ ] **Step 1: Write failing tests for contiguous blocks and stable effects**

```python
def test_contiguous_time_blocks_do_not_split_time_ids():
    blocks = contiguous_time_blocks(np.repeat(np.arange(8), 2), 4)
    assert [np.unique(np.repeat(np.arange(8), 2)[block]).tolist() for block in blocks] == [[0, 1], [2, 3], [4, 5], [6, 7]]

def test_feature_quality_marks_stable_and_drifting_columns():
    x = np.column_stack([np.arange(16), np.r_[np.arange(8), -np.arange(8)]])
    y = np.arange(16, dtype=float)
    report = feature_quality_by_blocks(x, y, np.ones(16), np.repeat(np.arange(8), 2), 4)
    assert report["direction_consistency"][0] == 1.0
    assert report["direction_consistency"][1] < 1.0
    assert report["block_correlation"].shape == (4, 2)
```

- [ ] **Step 2: Run the two tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: imports fail for `contiguous_time_blocks` and `feature_quality_by_blocks`.

- [ ] **Step 3: Implement block and quality functions**

```python
def contiguous_time_blocks(time_ids: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    # Split unique sorted time ids with np.array_split and return row-index arrays.

def feature_quality_by_blocks(features, target, weight, time_ids, n_blocks=4) -> dict[str, np.ndarray]:
    # Return finite_rate, std, iqr, pooled_correlation, block_correlation,
    # median_abs_correlation, correlation_mad, direction_consistency,
    # early_late_delta and standardized_mean_shift.
```

Use weighted Pearson for Ridge/full target and plain weights of one for unweighted task views. Replace non-finite feature cells only inside each statistic; never use validation rows to fit imputation values.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: all quality tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/v3_feature_structure.py tests/test_v3_feature_structure.py
git commit -m "feat: add feature stability diagnostics"
```

### Task 3: Stable Redundancy Matrices and Clusters

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Modify: `experiments/v3_feature_structure.py`

- [ ] **Step 1: Write failing tests for duplicate and sign-flipped features**

```python
def test_stable_redundancy_clusters_duplicate_columns():
    base = np.arange(24, dtype=float)
    x = np.column_stack([base, base * 2, -base, np.tile([0, 1, 0], 8)])
    tid = np.repeat(np.arange(12), 2)
    result = stable_redundancy(x, tid, n_blocks=4, threshold=0.05)
    assert result.pearson.shape == (4, 4)
    assert result.spearman.shape == (4, 4)
    assert result.labels[0] == result.labels[1] == result.labels[2]
    assert result.labels[3] != result.labels[0]
```

- [ ] **Step 2: Run the redundancy test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: import fails for `stable_redundancy`.

- [ ] **Step 3: Implement stable correlation aggregation and clustering**

```python
@dataclass(frozen=True)
class RedundancyResult:
    pearson: np.ndarray
    spearman: np.ndarray
    stability: np.ndarray
    distance: np.ndarray
    labels: np.ndarray

def stable_redundancy(features, time_ids, n_blocks=4, threshold=0.15) -> RedundancyResult:
    # Compute Pearson and rank-based Spearman per block.
    # Aggregate median absolute correlation and minimum cross-block stability.
    # Cluster scipy linkage(squareform(distance), method="average").
```

Constant columns must have zero off-diagonal similarity and unit diagonal. Distance must be symmetric, finite, and have a zero diagonal before `squareform`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: redundancy tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/v3_feature_structure.py tests/test_v3_feature_structure.py
git commit -m "feat: cluster stable redundant features"
```

### Task 4: Weighted Component Gram Analysis

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Modify: `experiments/v3_feature_structure.py`

- [ ] **Step 1: Write a failing test for lambda/blend coupling**

```python
def test_weighted_component_gram_exposes_market_cross_coupling():
    y = np.array([1., -1., 2., -2.])
    b = np.array([.2, -.2, .3, -.3])
    u = np.array([1., 0., 1., 0.])
    v = np.array([1., 0., 1., 0.])
    result = weighted_component_gram(y, np.ones(4), b, u, v)
    assert result["labels"] == ["b", "u", "v", "y"]
    assert result["gram"][1, 2] > 0
    np.testing.assert_allclose(result["gram"], result["gram"].T)
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: import fails for `weighted_component_gram`.

- [ ] **Step 3: Implement normalized weighted Gram output**

```python
def weighted_component_gram(target, weight, baseline, market_delta, cross_delta) -> dict[str, object]:
    z = np.column_stack([baseline, market_delta, cross_delta, target]).astype(np.float64)
    denominator = float(np.dot(np.maximum(weight, 0), target * target))
    gram = z.T @ (np.maximum(weight, 0)[:, None] * z) / denominator
    return {"labels": ["b", "u", "v", "y"], "gram": gram, "uv_coupling": float(gram[1, 2])}
```

Reject mismatched, non-1D or non-finite component arrays and zero target energy.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_v3_feature_structure.py -q`
Expected: Gram tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/v3_feature_structure.py tests/test_v3_feature_structure.py
git commit -m "feat: add fusion component gram audit"
```

### Task 5: CLI and Reports

**Files:**
- Modify: `tests/test_v3_feature_structure.py`
- Create: `experiments/v3_feature_structure_audit.py`

- [ ] **Step 1: Write failing tests for serializable report summaries**

```python
