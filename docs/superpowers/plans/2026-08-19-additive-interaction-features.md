# Additive Interaction Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover stable conditional interactions across all 323 anonymous features and append only accepted derived columns to the frozen production Ridge, XS LightGBM, and Market LightGBM inputs.

**Architecture:** A training-only miner fits shallow deterministic LightGBM trees to strict chronological out-of-sample residuals and converts repeated two-to-four-source paths into versioned, deterministic interaction definitions. One NumPy-only builder constructs those columns for training, offline prediction, LightGBM prediction, NumPy forest prediction, and official sequential inference; the original Top200/History40 direct columns, `market_lambda=0.7`, and `blend_weight=1.17` stay fixed throughout the A/B gate.

**Tech Stack:** Python 3.13, NumPy, LightGBM, scikit-learn Ridge, pytest, JSON/NPZ/Markdown, official sequential runner.

---

## File Structure

- Create `strategies/v3_hybrid/interactions.py`: validate versioned interaction metadata and build deterministic derived columns without LightGBM or pandas dependencies.
- Create `experiments/v3_interaction_features.py`: fit inner chronological residual miners, extract paths, canonicalize thresholds, aggregate support, and write fold manifests.
- Create `experiments/v3_interaction_oof.py`: run paired baseline-versus-augmented rolling OOF and enforce the frozen acceptance gate.
- Create `tests/test_v3_interactions.py`: cover schema, construction, path stability, leakage controls, unchanged base widths, and OOF verdict logic.
- Modify `strategies/v1_ridge/train.py`: accept optional appended training columns while preserving the current no-interaction API and artifact behavior.
- Modify `strategies/v3_hybrid/train.py`: consume an accepted interaction manifest, train enlarged matrices, and serialize source-only preprocessing metadata.
- Modify `strategies/v3_hybrid/main.py`: load source-only columns and append declared interactions before existing model prediction.
- Modify `strategies/v3_hybrid/features.py`: resolve old and new metadata into one backward-compatible feature contract.
- Modify `scripts/check_consistency.py`: report interaction column parity in addition to final prediction parity.

### Task 1: Versioned Interaction Contract And Pure Builder

**Files:**
- Create: `strategies/v3_hybrid/interactions.py`
- Create: `tests/test_v3_interactions.py`
- Modify: `strategies/v3_hybrid/features.py`

- [ ] **Step 1: Write failing schema and construction tests**

Add tests for region, gated value, current/history gated value, and source-only behavior:

```python
def test_build_interactions_supports_all_operations() -> None:
    sources = {
        "current:30": np.array([-1.0, 2.0, 3.0], dtype=np.float32),
        "current:250": np.array([0.1, 0.7, 0.9], dtype=np.float32),
        "history_previous:7": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }
    definitions = [
        {"name": "ridge_region_0000", "operation": "region",
         "conditions": [
             {"source": "current:30", "direction": "gt", "threshold": 0.0,
              "missing_matches": False},
             {"source": "current:250", "direction": "gt", "threshold": 0.5,
              "missing_matches": False}]},
        {"name": "ridge_gated_0000", "operation": "gated_value",
         "value_source": "current:30",
         "conditions": [{"source": "current:250", "direction": "gt", "threshold": 0.5,
                         "missing_matches": False}]},
        {"name": "xs_current_history_0000", "operation": "current_history_gated",
         "value_source": "history_previous:7",
         "conditions": [{"source": "current:250", "direction": "gt", "threshold": 0.5,
                         "missing_matches": False}]},
    ]
    got = build_interaction_columns(sources, definitions, max_cells=9)
    assert got.dtype == np.float32
    assert np.array_equal(got[:, 0], np.array([0.0, 1.0, 1.0]))
    assert np.array_equal(got[:, 1], np.array([0.0, 2.0, 3.0]))
    assert np.array_equal(got[:, 2], np.array([0.0, 5.0, 6.0]))


def test_external_source_is_not_a_direct_column() -> None:
    contract = resolve_interaction_contract(
        {"interaction_schema_version": 1, "interactions": manifest_using_feature_250()},
        direct_features={"ridge": [f"feature_{i:03d}" for i in range(200)]},
    )
    assert "feature_250" in contract["source_columns"]
    assert "feature_250" not in contract["direct_features"]["ridge"]
```

Also reject duplicate output names, unknown source families, one-source region paths, non-finite thresholds, unsupported operations, duplicate semantic sources, and cell-budget overflow.

- [ ] **Step 2: Confirm the tests fail before implementation**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -q
```

Expected: collection fails because `strategies.v3_hybrid.interactions` does not exist.

- [ ] **Step 3: Implement the NumPy-only contract and builder**

Create these public APIs:

```python
INTERACTION_SCHEMA_VERSION = 1
SOURCE_FAMILIES = frozenset({
    "current", "xs_deviation", "market_raw", "market_deviation",
    "history_previous", "history_difference", "history_rolling_mean",
    "history_rolling_deviation",
})
OPERATIONS = frozenset({"region", "gated_value", "current_history_gated"})


def source_key(family: str, feature_index: int) -> str:
    if family not in SOURCE_FAMILIES or feature_index < 0:
        raise ValueError("invalid interaction source")
    return f"{family}:{feature_index}"


def validate_interaction_definitions(definitions: list[dict[str, object]]) -> None:
    """Reject ambiguous, non-finite, duplicated, or unsupported definitions."""


def interaction_source_keys(definitions: list[dict[str, object]]) -> list[str]:
    """Return unique source keys in deterministic first-use order."""


def build_interaction_columns(
    sources: dict[str, np.ndarray],
    definitions: list[dict[str, object]],
    *,
    max_cells: int,
) -> np.ndarray:
    """Build float32 columns in manifest order and fail before exceeding max_cells."""
```

Condition semantics are exact: `gt` means `value > threshold`, `le` means `value <= threshold`, and `missing_matches` records the miner's missing-value direction even though robust current preprocessing normally removes non-finite values. Check `rows * definitions <= max_cells` before allocating and never silently truncate.

- [ ] **Step 4: Extend backward-compatible metadata resolution**

Old metadata resolves to schema version 0, empty task definitions, no source-only columns, and no source-only statistics. Version 1 requires complete `lower/upper/center/scale` arrays for source-only current columns and keeps those names out of every direct task feature list.

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py tests/test_v3_adaptive_selection.py -q
git diff --check
git add strategies/v3_hybrid/interactions.py strategies/v3_hybrid/features.py tests/test_v3_interactions.py
git commit -m "feat: add interaction feature contract"
```

Expected: focused tests pass and old metadata fallback remains green.

### Task 2: Training-Only Residual Path Miner

**Files:**
- Create: `experiments/v3_interaction_features.py`
- Modify: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing extraction and canonicalization tests**

```python
def test_canonical_path_is_order_invariant_within_quantile_bin() -> None:
    catalog = [Source("current", 2), Source("current", 250), Source("current", 17)]
    quantiles = quantiles_for(catalog, bins=32)
    left = canonicalize_path(path_a(), catalog, quantiles)
    right = canonicalize_path(same_conditions_reordered_and_shifted(), catalog, quantiles)
    assert left.support_key == right.support_key


def test_acceptance_requires_two_inner_validation_blocks() -> None:
    accepted = aggregate_repeated_paths(
        {0: [path("a")], 1: [path("a")], 2: [path("b")]}, min_blocks=2
    )
    assert [item.support_key for item in accepted] == ["a"]
```

Also test rejection of duplicate sources and paths outside two-to-four distinct sources. A synthetic response `x0 * I(x250 > q)` must produce a repeated path; permuted residuals must not.

- [ ] **Step 2: Confirm the new tests fail**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -q
```

Expected: missing miner symbols.

- [ ] **Step 3: Implement deterministic path records and extraction**

```python
@dataclass(frozen=True, order=True)
class Source:
    family: str
    feature_index: int


@dataclass(frozen=True)
class CanonicalPath:
    support_key: str
    ordered_conditions: tuple[dict[str, object], ...]
    blocks: tuple[int, ...]


def extract_candidate_paths(model_dump, catalog) -> list[list[dict]]:
    """Extract root-to-leaf paths with 2-4 distinct non-asset sources."""


def training_quantile_grid(values: np.ndarray, bins: int = 32) -> np.ndarray:
    """Fit finite monotone threshold values on inner-train rows only."""


def canonicalize_path(path, catalog, quantile_grids) -> CanonicalPath:
    """Map thresholds to training quantile bins and sort only the support key."""


def aggregate_repeated_paths(paths_by_block, min_blocks: int = 2) -> list[CanonicalPath]:
    """Keep paths supported by at least min_blocks in deterministic order."""
```

Support matching uses sorted canonical conditions including comparison direction, quantile bin, and missing-value direction. Gated-value construction preserves one representative path order selected by lowest block, tree, then lexical encoding.

- [ ] **Step 4: Implement strict expanding residual mining**

Use four chronological inner blocks. For each validation block, consume baseline predictions from a model fitted only on earlier blocks, form OOS residuals, fit a deterministic shallow LightGBM miner, and extract paths. Freeze `max_depth=4`, `num_leaves=15`, `learning_rate=0.03`, 80 rounds, seed 2026, row-wise deterministic training, and time-preserving row caps.

For each accepted path emit, in order: region indicator; final-source gated value; and, only for mixed current/history paths, history value gated by current conditions. Do not emit pure products or serialize the miner as a predictor.

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -q
git diff --check
git add experiments/v3_interaction_features.py tests/test_v3_interactions.py
git commit -m "feat: mine stable residual interactions"
```

Expected: deterministic synthetic interaction tests pass and shuffled controls do not clear support.

### Task 3: Full-323 Task Sources With Frozen History40

**Files:**
- Modify: `experiments/v3_interaction_features.py`
- Modify: `experiments/v3_production_oof.py`
- Modify: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing source-universe and base-width tests**

```python
def test_task_sources_cover_all_current_but_only_history40() -> None:
    views = build_interaction_source_views(
        transformed=np.zeros((3, 323), dtype=np.float32),
        time_ids=np.array([1, 1, 1]),
        history_indices=np.arange(40),
        history_blocks=[np.zeros((3, 40), dtype=np.float32) for _ in range(4)],
    )
    assert len(views["ridge"].catalog) == 323
    assert len(views["xs"].catalog) == 323 + 4 * 40
    assert len(views["market"].catalog) == 2 * 323 + 4 * 40
```

Also assert unchanged direct widths: Ridge 400, XS 360 plus `asset_id`, and Market 560 plus `asset_id`, including when an interaction uses global feature 250.

- [ ] **Step 2: Confirm failures**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -q
```

Expected: missing source-view builder and unchanged-width behavior.

- [ ] **Step 3: Implement task source views**

```python
def build_interaction_source_views(
    transformed: np.ndarray,
    time_ids: np.ndarray,
    history_indices: np.ndarray,
    history_blocks: list[np.ndarray],
) -> dict[str, TaskSourceView]:
    """Return Ridge current323, XS deviation323+History40x4,
    and Market raw323+deviation323+History40x4."""
```

Global current indices stay `0..322`. History identities use global anonymous feature indices, while only the established 40 bases receive four causal blocks.

- [ ] **Step 4: Add fold-local discovery**

Add `build_interaction_fold_manifest()` to `experiments/v3_production_oof.py`. It receives only the outer training slice, fits preprocessing and inner baseline models there, generates task-specific strict OOS residuals, and calls each task miner independently. Store source counts, History40 indices, inner block ranges, quantile grids, definitions, and source-only robust stats.

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py tests/test_v3_adaptive_selection.py -q
git diff --check
git add experiments/v3_interaction_features.py experiments/v3_production_oof.py tests/test_v3_interactions.py
git commit -m "feat: discover interactions across all features"
```

Expected: all current sources are discoverable, feature 250 can remain source-only, and history width remains 160.

### Task 4: Paired Strict OOF A/B Gate

**Files:**
- Create: `experiments/v3_interaction_oof.py`
- Modify: `tests/test_v3_interactions.py`

- [ ] **Step 1: Write failing A/B and verdict tests**

```python
def test_interaction_gate_requires_all_conditions() -> None:
    verdict = interaction_gate(
        deltas=np.array([0.01, 0.02, 0.03, 0.01, -0.001]),
        delta_a=0.05,
        delta_b=0.04,
    )
    assert verdict["passed"] is True
    assert interaction_gate(np.array([0.1, 0.1, -0.01, -0.01, -0.01]), 1.0, 0.1)["passed"] is False
    assert interaction_gate(np.array([0.01] * 5), 0.01, 0.03)["passed"] is False
```

Add prefix tests proving A columns are unchanged in B and an early-stop test proving no later fold runs once four positive folds are impossible.

- [ ] **Step 2: Confirm the missing runner failure**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py -q
```

Expected: import failure for `experiments.v3_interaction_oof`.

- [ ] **Step 3: Implement paired fold execution**

Freeze `train_window=78960`, `embargo=6`, phase-balanced modulo 5, the same Top200/History40, preprocessing, rows, seeds, model hyperparameters, `market_lambda=0.7`, and `blend_weight=1.17`.

A designs remain:
- Ridge: `[current200, deviation200]`.
- XS: `[deviation200, history160, asset_id]`.
- Market: `[raw200, deviation200, history160, asset_id]`.

B appends task interactions after the non-asset columns and keeps `asset_id` last. Recalculate the categorical index after appending.

- [ ] **Step 4: Implement reports and gate**

Write `outputs/experiments/v3_interactions_screen_1s160_07_117.json` and `.md`, plus one fold manifest per completed fold. Report A/B Peak, A, B, prediction standard deviation, interaction counts, source-family counts, path orders, fold overlap, and source-only features.

Pass only if mean Peak delta is positive, at least four folds are positive, drop-best mean delta is positive, and `2 * delta_A > delta_B`. Stop when four positive folds become impossible. A failed screen writes diagnostics but no final model or CSV.

- [ ] **Step 5: Verify and commit before expensive execution**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py tests/test_v3_adaptive_selection.py tests/test_metric.py -q
git diff --check
git add experiments/v3_interaction_oof.py tests/test_v3_interactions.py
git commit -m "feat: compare additive interactions in strict oof"
```

- [ ] **Step 6: Run the one-seed screen**

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_interaction_oof.py \
  --data-root /mnt/e/量化/public_release_20260630/public_release_20260630/data \
  --label v3_interactions_screen_1s160_07_117 \
  --n-folds 5 --train-window 78960 --embargo 6 \
  --sample-modulo 5 --sampling phase_balanced \
  --n-seeds 1 --num-iteration 160 --market-lambda 0.7 --blend-weight 1.17
```

Expected: complete or mathematically early-stopped paired report. Score direction is not assumed.

### Task 5: Candidate Training, Conditional On Screen Pass

**Files:**
- Modify: `strategies/v1_ridge/train.py`
- Modify: `strategies/v3_hybrid/train.py`
- Modify: `tests/test_ridge_infrastructure.py`
- Modify: `tests/test_v3_interactions.py`

- [ ] **Step 1: Execute this task only when Task 4 says `passed: true`**

If screening fails, record it in the experiment ledger and stop. Do not weaken gates, select paths from validation folds, or generate a CSV.

- [ ] **Step 2: Write failing optional-Ridge-column tests**

```python
def test_fit_model_appends_interactions_after_ridge_blocks() -> None:
    artifact = fit_model(
        features=features(), target=target(), weight=np.ones(8), time_ids=time_ids(),
        selected_indices=np.array([0, 2]),
        appended_design=np.arange(16, dtype=np.float32).reshape(8, 2),
        appended_names=["ridge_region_0000", "ridge_gated_0000"],
        alpha=10.0,
    )
    assert len(artifact["coef"]) == 6
    assert artifact["appended_feature_names"] == [
        "ridge_region_0000", "ridge_gated_0000"
    ]
```

Also assert omitted appended columns preserve old artifact keys and coefficient width exactly.

- [ ] **Step 3: Implement optional appended Ridge design**

Add keyword-only `appended_design` and `appended_names` to `fit_model()`. Validate finite, row-aligned float32 columns and append after `[raw, deviation]`. The omitted path remains behaviorally unchanged.

- [ ] **Step 4: Add V3 metadata tests and training support**

Add `--interaction-manifest` to V3 training. Build each task's interactions with the shared builder, append them in manifest order, and serialize schema version, definitions, source-only features and robust stats, output order, and discovery protocol. Direct feature lists and History40 remain unchanged. Production overwrite protection remains active.

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/python -m pytest tests/test_ridge_infrastructure.py tests/test_v3_interactions.py -q
git diff --check
git add strategies/v1_ridge/train.py strategies/v3_hybrid/train.py tests/test_ridge_infrastructure.py tests/test_v3_interactions.py
git commit -m "feat: train hybrid with additive interactions"
```

Expected: no-interaction training remains green and interaction candidates serialize complete metadata.

### Task 6: All Inference Paths Use One Builder

**Files:**
- Modify: `strategies/v3_hybrid/main.py`
- Modify: `strategies/v3_hybrid/train.py`
- Modify: `scripts/check_consistency.py`
- Modify: `tests/test_v3_interactions.py`
- Modify: `tests/test_asset_history_online.py`

- [ ] **Step 1: Write failing source-only, parity, and causal-history tests**

Use a tiny candidate with direct `feature_030`, source-only `feature_250`, and one `history_previous` source:

```python
assert "feature_250" in model.feature_columns
assert "feature_250" not in model.ridge_features
assert model.history.feature_count == 40
np.testing.assert_allclose(offline_interactions, sequential_interactions, atol=0.0, rtol=0.0)
```

Feed two time IDs and verify the second batch reads history from the first before state advancement.

- [ ] **Step 2: Confirm inference failures**

```bash
.venv/bin/python -m pytest tests/test_v3_interactions.py tests/test_asset_history_online.py -q
```

- [ ] **Step 3: Implement shared source preparation**

Cache positions for the union of direct and source-only columns. Transform source-only current columns with frozen metadata, compute task deviations, expose the same four History40 blocks by global feature identity, call `build_interaction_columns()`, and append results before `asset_id`. Ridge appends after its 400 direct columns.

Use identical calls in `train.predict_array()`. Schema version 0 produces zero-width interaction matrices and preserves existing predictions.

- [ ] **Step 4: Extend consistency diagnostics**

Report per-task interaction-column differences and final prediction difference. Pure current interaction columns must match exactly; complete online/offline prediction must remain within `1e-6`.

- [ ] **Step 5: Run all tests and commit**

```bash
.venv/bin/python -m pytest -q
git diff --check
git add strategies/v3_hybrid/main.py strategies/v3_hybrid/train.py scripts/check_consistency.py tests/test_v3_interactions.py tests/test_asset_history_online.py
git commit -m "feat: infer additive interactions consistently"
```

Expected: full suite passes with only the known existing MLP warning.

### Task 7: Confirmation, Candidate, And CSV Only After Both Gates

**Files:**
- Create: `outputs/experiments/v3_interactions_confirm_3s480_07_117.json`
- Create: `outputs/experiments/v3_interactions_confirm_3s480_07_117.md`
- Create: `outputs/candidates/v3_hybrid_interactions_07_117/`
- Create: `outputs/submissions/v3_hybrid_interactions_07_117.csv`
- Modify: `experiments/ledger.csv`

- [ ] **Step 1: Run three-seed, 480-round confirmation only after screen pass**

Use the same paired OOF protocol and require the same four gates. Every fold rediscovers interactions from its own training window.

- [ ] **Step 2: Stop without a candidate if confirmation fails**

Record both experiments in `experiments/ledger.csv`, preserve reports, and leave production plus the existing `0.7/1.17` candidate untouched.

- [ ] **Step 3: Build the all-training manifest only after confirmation pass**

Use the frozen discovery protocol on all training rows. The final manifest fits models only; it cannot revise gates or hyperparameters.

- [ ] **Step 4: Train with frozen fusion**

Train into `outputs/candidates/v3_hybrid_interactions_07_117/` using three seeds, 480 rounds, `market_lambda=0.7`, and `blend_weight=1.17`. Do not overwrite `strategies/v3_hybrid/model`.

- [ ] **Step 5: Verify LightGBM, NumPy, offline, and sequential paths**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_consistency.py \
  --strategy v3_hybrid --backend lightgbm \
  --data-root /mnt/e/量化/public_release_20260630/public_release_20260630/data \
  --model-path outputs/candidates/v3_hybrid_interactions_07_117/baseline_model.json \
  --n-time-ids 50
.venv/bin/python scripts/check_consistency.py \
  --strategy v3_hybrid --backend numpy \
  --data-root /mnt/e/量化/public_release_20260630/public_release_20260630/data \
  --model-path outputs/candidates/v3_hybrid_interactions_07_117/baseline_model.json \
  --n-time-ids 50
```

Expected: all tests pass and both backends meet existing tolerances.

- [ ] **Step 6: Generate and audit one CSV**

Use the official sequential runner. Verify exact test row count and `row_id` order, finite targets, clip count, file size, and SHA-256. Report:

```text
E:\\量化\\Quant_trade\\outputs\\submissions\\v3_hybrid_interactions_07_117.csv
```

No CSV is created when either OOF gate fails.

## Completion Gate

- Production model files, the existing `0.7/1.17` candidate, and PR #1 remain unchanged.
- Direct inputs remain Ridge 400 columns from Top200 sources, XS 360 columns from Top200 plus History40, and Market 560 columns from Top200 plus History40.
- Discovery can use all 323 current anonymous features, but a Top200-external source enters prediction only through an accepted derived column.
- History remains 40 bases expanded into the existing 160 causal columns.
- No fourth model, new fusion coefficient, leaderboard-driven selection, or pre-gate CSV is introduced.
- A final candidate exists only after both strict paired OOF gates pass.
