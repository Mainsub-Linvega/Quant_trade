# Purified Interaction P0 Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-safe P0 diagnostic that scores purified pairwise residual interactions against task-preserving empirical nulls without modifying any production model.

**Architecture:** Add one NumPy-only module for frozen protocol validation, weighted quantile surfaces, functional-ANOVA purification, chronological candidate scoring, and null generation. Add a separate CLI that consumes an explicit diagnostic NPZ, scans a pre-registered pair budget, and writes JSON/Markdown evidence; it cannot write model candidates or submissions.

**Tech Stack:** Python 3.13, NumPy, scikit-learn, pytest, JSON/NPZ/Markdown.

---

## File Structure

- Create `experiments/v3_purified_interactions.py`: pure validation, binning, purification, scoring, null, and selection primitives.
- Create `experiments/v3_purified_interaction_diagnostic.py`: P0 CLI, bounded pair scan, manifests, and reports.
- Create `tests/test_v3_purified_interactions.py`: synthetic mathematical, leakage, null, determinism, and CLI tests.
- Create `outputs/experiments/v3_purified_interaction_protocol_v1.json`: frozen numeric P0 protocol, generated only after tests pass.

### Task 1: Frozen Protocol Contract

**Files:**
- Create: `experiments/v3_purified_interactions.py`
- Create: `tests/test_v3_purified_interactions.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_default_protocol_matches_frozen_design() -> None:
    protocol = default_purified_protocol()
    assert protocol["schema_version"] == 1
    assert protocol["outer"] == {
        "n_folds": 5, "train_window": 78_960, "embargo": 6,
        "sample_modulo": 5, "sampling": "phase_balanced",
    }
    assert protocol["tasks"]["ridge"]["bins"] == 8
    assert protocol["tasks"]["xs"]["bins"] == 8
    assert protocol["tasks"]["market"]["bins"] == 4
    assert protocol["fusion"] == {
        "market_lambda": 0.7, "blend_weight": 1.17,
        "prediction_scale": 1.16,
    }


def test_protocol_rejects_result_dependent_search() -> None:
    protocol = default_purified_protocol()
    protocol["tasks"]["xs"]["choose_best_bins"] = [8, 16]
    with pytest.raises(ValueError, match="one primary bin count"):
        validate_purified_protocol(protocol)
```

Also reject nonpositive support, fewer than four inner blocks, null quantiles outside `(0.5, 1)`, duplicate seeds, nonpositive memory/pair budgets, history enabled in P0, or changed `0.7/1.17/1.16`.

- [ ] **Step 2: Run the tests and confirm failure**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q
```

Expected: import fails because `experiments.v3_purified_interactions` does not exist.

- [ ] **Step 3: Implement the protocol**

Expose:

```python
def default_purified_protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "outer": {"n_folds": 5, "train_window": 78960, "embargo": 6,
                  "sample_modulo": 5, "sampling": "phase_balanced"},
        "inner_blocks": 4,
        "tasks": {
            "ridge": {"bins": 8, "min_cell_weight": 64.0},
            "xs": {"bins": 8, "min_cell_weight": 64.0},
            "market": {"bins": 4, "min_cell_weight": 16.0},
        },
        "null": {"quantile": 0.95, "seeds": [2026, 2027, 2028, 2029],
                 "minimum_time_shift": 7},
        "stability": {"minimum_positive_blocks": 2,
                      "minimum_coverage": 0.80,
                      "maximum_single_cell_gain_share": 0.50},
        "budgets": {"max_pairs": 52003, "max_output_candidates": 256,
                    "max_surface_cells": 1000000},
        "history_enabled": False,
        "fusion": {"market_lambda": 0.7, "blend_weight": 1.17,
                   "prediction_scale": 1.16},
    }


def validate_purified_protocol(protocol: Mapping[str, object]) -> None:
    """Fail closed on missing, changed, ambiguous, or result-dependent settings."""
```

- [ ] **Step 4: Run tests and commit**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q
git diff --check
git add experiments/v3_purified_interactions.py tests/test_v3_purified_interactions.py
git commit -m "feat: freeze purified interaction protocol"
```

Expected: protocol tests pass.

### Task 2: Weighted Binning and Functional-ANOVA Purification

**Files:**
- Modify: `experiments/v3_purified_interactions.py`
- Modify: `tests/test_v3_purified_interactions.py`

- [ ] **Step 1: Write failing mathematical tests**

```python
def test_purification_removes_both_parent_main_effects() -> None:
    scores = np.array([[1., 2., 4.], [3., 5., 8.]])
    weights = np.array([[2., 1., 3.], [1., 4., 2.]])
    pure, impurities, intercept = purify_pair_surface(scores, weights)
    np.testing.assert_allclose(np.sum(pure * weights, axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.sum(pure * weights, axis=1), 0.0, atol=1e-10)
    reconstructed = pure + impurities[0][:, None] + impurities[1][None, :] + intercept
    np.testing.assert_allclose(reconstructed, scores, atol=1e-10)


def test_additive_surface_purifies_to_zero() -> None:
    left = np.array([-2., 1., 5.])
    right = np.array([3., -1.])
    scores = left[:, None] + right[None, :]
    pure, _, _ = purify_pair_surface(scores, np.ones_like(scores))
    np.testing.assert_allclose(pure, 0.0, atol=1e-10)
```

Also test training-only quantile edges, repeated edges, NaN mapping, low-support shrinkage, unseen validation cells, all-finite output, deterministic transform, and memory budget rejection before allocation.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q -k "purif or bin or surface"
```

Expected: missing binning and purification symbols.

- [ ] **Step 3: Implement the pure surface API**

```python
@dataclass(frozen=True)
class PurifiedPairSurface:
    left_feature: int
    right_feature: int
    edges_left: np.ndarray
    edges_right: np.ndarray
    values: np.ndarray
    cell_weights: np.ndarray
    coverage: float


def fit_quantile_edges(values: np.ndarray, bins: int) -> np.ndarray: ...
def assign_quantile_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray: ...
def fit_weighted_residual_surface(left, right, residual, weight, *, bins,
                                  min_cell_weight, max_surface_cells) -> PurifiedPairSurface: ...
def purify_pair_surface(scores, weights, *, tolerance=1e-10,
                        max_iterations=1000) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]: ...
def transform_purified_surface(surface, left, right) -> np.ndarray: ...
```

Fit edges, cell residual means, support masks, shrinkage, and purification on training rows only. Missing inputs and unseen cells map to zero.

- [ ] **Step 4: Verify and commit**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q
git diff --check
git add experiments/v3_purified_interactions.py tests/test_v3_purified_interactions.py
git commit -m "feat: purify weighted pair surfaces"
```

### Task 3: Chronological Gain, Stability, and Null Controls

**Files:**
- Modify: `experiments/v3_purified_interactions.py`
- Modify: `tests/test_v3_purified_interactions.py`

- [ ] **Step 1: Write failing signal and null tests**

```python
def test_pair_score_detects_nonadditive_signal_not_additive_signal() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(800, 2))
    weight = np.ones(800)
    joint = (x[:, 0] > 0) * (x[:, 1] > 0)
    nonlinear = score_pair_split(x[:600], x[600:], joint[:600], joint[600:],
                                 weight[:600], weight[600:], pair=(0, 1), bins=4)
    additive_y = x[:, 0] + 2 * x[:, 1]
    additive = score_pair_split(x[:600], x[600:], additive_y[:600], additive_y[600:],
                                weight[:600], weight[600:], pair=(0, 1), bins=4)
    assert nonlinear["gain"] > 0
    assert abs(additive["gain"]) < nonlinear["gain"]


def test_xs_null_preserves_time_groups_but_breaks_asset_alignment() -> None:
    residual = np.arange(12.0)
    time_id = np.repeat(np.arange(4), 3)
    got = make_task_null("xs", residual, time_id, seed=2026, embargo=6)
    for value in np.unique(time_id):
        rows = time_id == value
        np.testing.assert_array_equal(np.sort(got[rows]), np.sort(residual[rows]))
    assert not np.array_equal(got, residual)
```

Also test Market shift distance, Ridge two-stage null, deterministic seeds, invalid unordered time ids, empirical quantile calculation, positive-block counting, tail-cell concentration rejection, and drop-best-block gain.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q -k "score or null or stability"
```

- [ ] **Step 3: Implement scoring and gates**

```python
def weighted_residual_gain(residual, prediction, weight) -> float:
    """Return normalized SSE reduction against zero residual prediction."""

def score_pair_split(train_features, valid_features, train_residual, valid_residual,
                     train_weight, valid_weight, *, pair, bins,
                     min_cell_weight, max_surface_cells) -> dict[str, object]: ...

def make_task_null(task, residual, time_ids, *, seed, embargo) -> np.ndarray: ...
def empirical_null_threshold(null_gains, quantile) -> float: ...
def interaction_stability_gate(block_scores, *, null_threshold,
                               minimum_positive_blocks,
                               minimum_coverage,
                               maximum_single_cell_gain_share) -> dict[str, object]: ...
```

The pair score compares the purified joint surface with zero on already-main-effect-residualized input. It records gain, coverage, dominant-cell share, finite status, and surface checksum. Null transforms never cross the data boundary used to fit a surface.

- [ ] **Step 4: Verify and commit**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q
git diff --check
git add experiments/v3_purified_interactions.py tests/test_v3_purified_interactions.py
git commit -m "feat: gate purified interactions against nulls"
```

### Task 4: P0 Diagnostic CLI and Smoke Evidence

**Files:**
- Create: `experiments/v3_purified_interaction_diagnostic.py`
- Modify: `tests/test_v3_purified_interactions.py`
- Create: `outputs/experiments/v3_purified_interaction_protocol_v1.json`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_diagnostic_cli_defaults_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["v3_purified_interaction_diagnostic.py"])
    args = parse_args()
    assert args.task == "ridge"
    assert args.max_pairs == 256
    assert args.smoke is False
    assert args.write_candidate is False


def test_diagnostic_report_never_generates_candidate(tmp_path) -> None:
    paths = write_diagnostic_report(tmp_path, "smoke", payload={"status": "passed_p0"})
    assert set(paths) == {"json", "markdown"}
    assert not list(tmp_path.glob("*candidate*"))
    assert not list(tmp_path.glob("*.csv"))
```

Also test required NPZ keys, aligned row counts, exactly 323 features, ordered time ids, pair list determinism, exclusion of self-pairs, atomic report behavior, `--force`, and `--help`.

- [ ] **Step 2: Confirm failure**

```bash
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py -q -k "diagnostic or cli or report"
```

- [ ] **Step 3: Implement the P0 runner**

The explicit input NPZ contract is:

```text
features        float32 [rows, 323]
residual        float32/float64 [rows]
weight          float32/float64 [rows]
time_id         int64 [rows]
feature_indices int64 [323] (optional; defaults to 0..322)
```

CLI:

```bash
.venv/bin/python experiments/v3_purified_interaction_diagnostic.py \
  --input-npz outputs/cache/purified_p0_input.npz \
  --task ridge --max-pairs 256 \
  --protocol outputs/experiments/v3_purified_interaction_protocol_v1.json \
  --label v3_purified_p0_ridge_smoke
```

The runner uses `chronological_inner_splits`, scans pairs in lexical order unless an explicit pre-registered pair manifest is supplied, computes real and null block scores, applies the stability gate, retains at most the frozen output budget, and writes only `outputs/experiments/<label>.json` and `.md`.

- [ ] **Step 4: Generate the frozen protocol and run tests**

```bash
.venv/bin/python experiments/v3_purified_interaction_diagnostic.py --write-default-protocol \
  outputs/experiments/v3_purified_interaction_protocol_v1.json
.venv/bin/python -m pytest tests/test_v3_purified_interactions.py tests/test_v3_interactions.py -q
.venv/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass; the protocol validates; no candidate directory or CSV is generated.

- [ ] **Step 5: Run a deterministic synthetic smoke**

```bash
.venv/bin/python experiments/v3_purified_interaction_diagnostic.py \
  --synthetic-smoke --task ridge --max-pairs 32 \
  --protocol outputs/experiments/v3_purified_interaction_protocol_v1.json \
  --label v3_purified_p0_synthetic_smoke --force
```

Expected: the planted pair is ranked above its empirical null, additive-only controls do not pass, outputs contain no candidate/CSV, and rerunning produces identical candidate ordering and checksums.

- [ ] **Step 6: Commit**

```bash
git add experiments/v3_purified_interaction_diagnostic.py \
  experiments/v3_purified_interactions.py tests/test_v3_purified_interactions.py \
  outputs/experiments/v3_purified_interaction_protocol_v1.json
git commit -m "feat: add purified interaction P0 diagnostic"
```

## Completion Gate

P0 is complete only when the full test suite passes, the synthetic interaction clears its null, additive controls do not, repeated smoke output is deterministic, and no production/candidate/submission file changes. Real-data P0 scanning is a separate, explicitly recorded run after this implementation gate.
