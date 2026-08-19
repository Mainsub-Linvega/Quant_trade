# Stable Subpath Interactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace brittle exact complete-path matching with deterministic 2-4-condition subpath matching over 16 support quantile regions, while preserving the frozen baseline and strict OOF gate.

**Architecture:** Keep complete LightGBM path extraction unchanged, expand each valid path into all distinct-source condition combinations, then canonicalize those subpaths with a two-fine-bin support width. Store coarse 0-15 bins in definitions so outer-fold threshold resolution uses the same 16-region meaning. The OOF runner receives definitions only; models and fusion remain unchanged.

**Tech Stack:** Python 3.13, NumPy, LightGBM, pytest, existing WSL project environment.

---

### Task 1: Expand Complete Paths Into Stable Subpaths

**Files:**
- Modify: `experiments/v3_interaction_features.py`
- Test: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing subpath tests**

Import `expand_candidate_subpaths` and add tests proving a three-condition path emits three pairs plus one triple and a four-condition path emits six pairs, four triples, and one quadruple. Assert every emitted representative retains original condition order and source uniqueness.

```python
expanded = expand_candidate_subpaths([candidate])
assert [len(path.conditions) for path in expanded].count(2) == 6
assert [len(path.conditions) for path in expanded].count(3) == 4
assert [len(path.conditions) for path in expanded].count(4) == 1
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -k 'expand_candidate_subpaths' -q
```

Expected: collection fails because `expand_candidate_subpaths` is not defined.

- [ ] **Step 3: Implement deterministic expansion**

Use `itertools.combinations` over condition positions. Emit sizes from two through `min(4, path length)`, reject combinations with duplicate semantic sources, and preserve block/tree/leaf metadata.

```python
def expand_candidate_subpaths(paths, *, min_sources=2, max_sources=4):
    expanded = []
    for path in paths:
        for width in range(min_sources, min(max_sources, len(path.conditions)) + 1):
            for positions in combinations(range(len(path.conditions)), width):
                conditions = tuple(path.conditions[index] for index in positions)
                if len({item.source for item in conditions}) != width:
                    continue
                expanded.append(PathCandidate(
                    conditions=conditions,
                    block_index=path.block_index,
                    tree_index=path.tree_index,
                    leaf_index=path.leaf_index,
                ))
    return expanded
```

- [ ] **Step 4: Verify GREEN and commit**

Run the focused tests and `git diff --check`, then commit:

```bash
git commit -m "feat: expand interaction path candidates"
```

### Task 2: Canonicalize Support Into 16 Quantile Regions

**Files:**
- Modify: `experiments/v3_interaction_features.py`
- Test: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing coarse-bin tests**

Add one test where thresholds in neighboring fine 32-bin regions produce the same support key with `support_bin_width=2`, and one where thresholds on opposite sides of a paired-region boundary differ. Also assert the stored `quantile_bin` is the coarse 0-15 index and the stored threshold is the upper boundary of the paired region.

```python
left = canonicalize_path(first, grids, support_bin_width=2)
right = canonicalize_path(second, grids, support_bin_width=2)
assert left.support_key == right.support_key
assert left.ordered_conditions[0]["quantile_bin"] == 0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -k 'paired_quantile' -q
```

Expected: `canonicalize_path` rejects the new keyword or neighboring bins remain distinct.

- [ ] **Step 3: Implement fine-to-support bin mapping**

Extend `_canonical_condition` and `canonicalize_path` with `support_bin_width`. Require the fine bin count to be divisible by the width, map `fine_bin // support_bin_width`, and resolve the representative threshold at `(support_bin + 1) * support_bin_width` in the fine grid.

```python
support_bin = fine_bin // support_bin_width
resolved_threshold = float(grid[(support_bin + 1) * support_bin_width])
```

- [ ] **Step 4: Verify GREEN and commit**

Run focused canonicalization and aggregation tests, then commit:

```bash
git commit -m "feat: coarsen interaction support bins"
```

### Task 3: Integrate Subpaths With Training-Only Mining

**Files:**
- Modify: `experiments/v3_interaction_features.py`
- Test: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing mining-protocol test**

Extend the strict OOS mining test to assert:

```python
assert result["protocol"]["quantile_bins"] == 32
assert result["protocol"]["support_quantile_bins"] == 16
assert result["protocol"]["support_bin_width"] == 2
assert all("expanded_subpaths" in split for split in result["protocol"]["splits"])
assert all("unique_canonical_subpaths" in split for split in result["protocol"]["splits"])
```

- [ ] **Step 2: Verify RED**

Run the single mining test and confirm the diagnostic keys are missing.

- [ ] **Step 3: Expand and deduplicate candidates per block**

Inside each inner split, expand complete candidates, canonicalize with width two, deduplicate support keys for block diagnostics, and append deterministic canonical representatives to the cross-block pool. Keep cross-block acceptance at `min_blocks=2`.

Record complete, expanded, and unique counts. Add protocol fields for 32 fine bins, width two, and 16 support bins. Do not rank or truncate accepted paths.

- [ ] **Step 4: Verify GREEN and commit**

Run the mining tests and commit:

```bash
git commit -m "feat: mine stable interaction subpaths"
```

### Task 4: Resolve Outer Thresholds With Manifest Support Bins

**Files:**
- Modify: `experiments/v3_interaction_oof.py`
- Test: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing runner contract test**

Extract a small helper that validates all three task protocols have the same positive `support_quantile_bins` and returns that value. Test `16` succeeds and mismatched or missing values fail explicitly.

```python
assert manifest_support_quantile_bins(manifest) == 16
with pytest.raises(ValueError, match="support_quantile_bins"):
    manifest_support_quantile_bins(mismatched)
```

- [ ] **Step 2: Verify RED**

Run the helper test and confirm import failure.

- [ ] **Step 3: Use the manifest value for outer resolution**

Replace hard-coded `bins=32` with the validated manifest value. Change the default experiment label to `v3_interactions_subpaths16_screen_1s160_07_117`; do not alter frozen model or OOF parameters.

- [ ] **Step 4: Verify GREEN and commit**

Run runner tests and commit:

```bash
git commit -m "feat: evaluate stable subpath interactions"
```

### Task 5: Verification and Frozen Screen

**Files:**
- Generate: `outputs/experiments/v3_interactions_subpaths16_screen_1s160_07_117.json`
- Generate: `outputs/experiments/v3_interactions_subpaths16_screen_1s160_07_117.md`
- Generate: `outputs/experiments/v3_interactions_subpaths16_screen_1s160_07_117_manifests/`

- [ ] **Step 1: Run focused and full verification**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -q
.venv/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass; the existing MLP convergence warning may remain.

- [ ] **Step 2: Run the frozen OOF screen**

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  experiments/v3_interaction_oof.py \
  --data-root /mnt/e/量化/public_release_20260630/public_release_20260630/data \
  --label v3_interactions_subpaths16_screen_1s160_07_117 \
  --n-folds 5 --train-window 78960 --embargo 6 \
  --sample-modulo 5 --sampling phase_balanced \
  --n-seeds 1 --num-iteration 160 \
  --market-lambda 0.7 --blend-weight 1.17 \
  --num-threads 4 --force
```

- [ ] **Step 3: Enforce the outcome gate**

Confirm report status, per-fold interaction counts and deltas, no temporary feature spill, and `candidate_generated=false` plus `submission_generated=false` unless all frozen screen checks pass. Do not produce a CSV from a failed screen.

