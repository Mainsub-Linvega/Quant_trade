from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

import experiments.v3_interaction_features as interaction_features_module
from experiments.history_peak import retained_lag_rows
from experiments.v3_interaction_features import (
    PathCandidate,
    PathCondition,
    Source,
    aggregate_repeated_paths,
    build_interaction_source_views,
    interaction_source_arrays,
    canonicalize_path,
    expand_candidate_subpaths,
    extract_candidate_paths,
    mine_task_interactions,
    resolve_quantile_thresholds,
    training_quantile_grids,
)
from experiments.v3_production_oof import (
    build_interaction_fold_manifest,
    build_task_lgbm_designs,
)
from experiments.v3_interaction_oof import (
    append_interactions_before_asset,
    compose_hybrid_raw,
    interaction_gate,
    manifest_support_quantile_bins,
    parse_args as parse_interaction_oof_args,
    paired_component_predictions,
    positive_fold_gate_impossible,
    run_paired_fold_sequence,
    spill_interaction_features,
)
from strategies.v3_hybrid.interactions import (
    build_interaction_columns,
    interaction_source_keys,
    resolve_interaction_contract,
    validate_interaction_definitions,
)
from strategies.v3_hybrid.features import resolve_feature_contract


def _condition(source: str, threshold: float = 0.5) -> dict[str, object]:
    return {
        "source": source,
        "direction": "gt",
        "threshold": threshold,
        "missing_matches": False,
    }


def _definitions() -> list[dict[str, object]]:
    return [
        {
            "name": "ridge_region_0000",
            "operation": "region",
            "conditions": [
                _condition("current:30", 0.0),
                _condition("current:250"),
            ],
        },
        {
            "name": "ridge_gated_0000",
            "operation": "gated_value",
            "value_source": "current:30",
            "conditions": [_condition("current:250")],
        },
        {
            "name": "xs_current_history_0000",
            "operation": "current_history_gated",
            "value_source": "history_previous:7",
            "conditions": [_condition("xs_deviation:250")],
        },
    ]


def test_build_interactions_supports_all_operations() -> None:
    sources = {
        "current:30": np.array([-1.0, 2.0, 3.0], dtype=np.float32),
        "current:250": np.array([0.1, 0.7, 0.9], dtype=np.float32),
        "xs_deviation:250": np.array([0.1, 0.7, 0.9], dtype=np.float32),
        "history_previous:7": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }

    got = build_interaction_columns(sources, _definitions(), max_cells=9)

    assert got.dtype == np.float32
    assert np.array_equal(got[:, 0], np.array([0.0, 1.0, 1.0]))
    assert np.array_equal(got[:, 1], np.array([0.0, 2.0, 3.0]))
    assert np.array_equal(got[:, 2], np.array([0.0, 5.0, 6.0]))


def test_build_interactions_honors_missing_direction() -> None:
    definitions = [{
        "name": "region",
        "operation": "region",
        "conditions": [
            {**_condition("current:1"), "missing_matches": True},
            _condition("current:2", 0.0),
        ],
    }]
    sources = {
        "current:1": np.array([np.nan, 1.0, -1.0], dtype=np.float32),
        "current:2": np.ones(3, dtype=np.float32),
    }

    got = build_interaction_columns(sources, definitions, max_cells=3)

    assert np.array_equal(got[:, 0], np.array([1.0, 1.0, 0.0]))


def test_build_interactions_rejects_cell_budget_instead_of_truncating() -> None:
    sources = {
        "current:30": np.ones(3, dtype=np.float32),
        "current:250": np.ones(3, dtype=np.float32),
        "xs_deviation:250": np.ones(3, dtype=np.float32),
        "history_previous:7": np.ones(3, dtype=np.float32),
    }

    with pytest.raises(MemoryError, match="9 cells exceeds max_cells=8"):
        build_interaction_columns(sources, _definitions(), max_cells=8)


@pytest.mark.parametrize(
    "definitions, message",
    [
        (
            [
                {"name": "same", "operation": "region", "conditions": [
                    _condition("current:1"), _condition("current:2")],},
                {"name": "same", "operation": "region", "conditions": [
                    _condition("current:3"), _condition("current:4")],},
            ],
            "duplicate interaction output name",
        ),
        (
            [{"name": "one", "operation": "region",
              "conditions": [_condition("current:1")]}],
            "between 2 and 4 distinct sources",
        ),
        (
            [{"name": "duplicate", "operation": "region", "conditions": [
                _condition("current:1"), _condition("current:1", 0.9)]}],
            "duplicate semantic source",
        ),
        (
            [{"name": "bad_threshold", "operation": "region", "conditions": [
                _condition("current:1", np.inf), _condition("current:2")]}],
            "finite threshold",
        ),
        (
            [{"name": "bad_family", "operation": "region", "conditions": [
                _condition("unknown:1"), _condition("current:2")]}],
            "unknown source family",
        ),
        (
            [{"name": "bad_operation", "operation": "product", "conditions": [
                _condition("current:1"), _condition("current:2")]}],
            "unsupported interaction operation",
        ),
    ],
)
def test_validate_interactions_rejects_ambiguous_definitions(
    definitions: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_interaction_definitions(definitions)


def test_interaction_source_keys_preserve_first_use_order() -> None:
    assert interaction_source_keys(_definitions()) == [
        "current:30",
        "current:250",
        "xs_deviation:250",
        "history_previous:7",
    ]


def test_contract_keeps_external_source_out_of_direct_features() -> None:
    direct = {
        "ridge": ["feature_030"],
        "xs": ["feature_030"],
        "market": ["feature_030"],
    }
    meta = {
        "interaction_schema_version": 1,
        "interactions": {
            "ridge": [{
                "name": "ridge_gated_0000",
                "operation": "gated_value",
                "value_source": "current:30",
                "conditions": [_condition("current:250")],
            }],
            "xs": [],
            "market": [],
        },
        "interaction_source_stats": {
            "ridge": {
                "features": ["feature_250"],
                "lower": [-2.0],
                "upper": [2.0],
                "center": [0.0],
                "scale": [1.0],
            },
            "xs": {"features": [], "lower": [], "upper": [], "center": [], "scale": []},
            "market": {"features": [], "lower": [], "upper": [], "center": [], "scale": []},
        },
    }

    contract = resolve_interaction_contract(meta, direct_features=direct)

    assert contract["source_columns"] == ["feature_250"]
    assert contract["direct_features"] == direct
    assert "feature_250" not in contract["direct_features"]["ridge"]


def test_contract_defaults_old_metadata_to_zero_interactions() -> None:
    direct = {"ridge": ["feature_001"], "xs": [], "market": []}

    contract = resolve_interaction_contract({}, direct_features=direct)

    assert contract["schema_version"] == 0
    assert contract["definitions"] == {"ridge": [], "xs": [], "market": []}
    assert contract["source_columns"] == []


def test_feature_contract_exposes_zero_interactions_for_old_metadata() -> None:
    meta = {
        "lgbm_features": ["feature_001"],
        "lower": [-2.0],
        "upper": [2.0],
        "center": [0.0],
        "scale": [1.0],
    }

    contract = resolve_feature_contract(meta)

    assert contract["interaction_schema_version"] == 0
    assert contract["interactions"] == {"ridge": [], "xs": [], "market": []}
    assert contract["interaction_source_columns"] == []


def test_contract_requires_stats_for_top200_external_source() -> None:
    meta = {
        "interaction_schema_version": 1,
        "interactions": {
            "ridge": [{
                "name": "ridge_gated_0000",
                "operation": "gated_value",
                "value_source": "current:30",
                "conditions": [_condition("current:250")],
            }],
            "xs": [],
            "market": [],
        },
        "interaction_source_stats": {
            task: {"features": [], "lower": [], "upper": [], "center": [], "scale": []}
            for task in ("ridge", "xs", "market")
        },
    }

    with pytest.raises(ValueError, match="ridge source-only preprocessing is missing feature_250"):
        resolve_interaction_contract(
            meta,
            direct_features={"ridge": ["feature_030"], "xs": [], "market": []},
        )


def _two_source_tree(
    *, first_threshold: float = 0.5, second_threshold: float = 1.5
) -> dict[str, object]:
    return {
        "tree_info": [{
            "tree_structure": {
                "split_feature": 0,
                "threshold": first_threshold,
                "decision_type": "<=",
                "default_left": True,
                "left_child": {
                    "split_feature": 1,
                    "threshold": second_threshold,
                    "decision_type": "<=",
                    "default_left": False,
                    "left_child": {"leaf_value": 1.0},
                    "right_child": {"leaf_value": -1.0},
                },
                "right_child": {"leaf_value": 0.0},
            },
        }],
    }


def test_extract_candidate_paths_preserves_branch_and_missing_direction() -> None:
    catalog = [Source("current", 30), Source("current", 250)]

    paths = extract_candidate_paths(_two_source_tree(), catalog, block_index=2)

    assert len(paths) == 2
    left_left = paths[0]
    assert [(item.direction, item.missing_matches) for item in left_left.conditions] == [
        ("le", True),
        ("le", False),
    ]
    left_right = paths[1]
    assert [(item.direction, item.missing_matches) for item in left_right.conditions] == [
        ("le", True),
        ("gt", True),
    ]
    assert left_left.block_index == 2


def test_extract_candidate_paths_rejects_duplicate_and_five_source_paths() -> None:
    duplicate = _two_source_tree()
    duplicate["tree_info"][0]["tree_structure"]["left_child"]["split_feature"] = 0
    assert extract_candidate_paths(
        duplicate, [Source("current", 1), Source("current", 2)], block_index=0
    ) == []

    node: dict[str, object] = {"leaf_value": 1.0}
    for feature_index in reversed(range(5)):
        node = {
            "split_feature": feature_index,
            "threshold": 0.0,
            "decision_type": "<=",
            "default_left": True,
            "left_child": node,
            "right_child": {"leaf_value": 0.0},
        }
    paths = extract_candidate_paths(
        {"tree_info": [{"tree_structure": node}]},
        [Source("current", index) for index in range(5)],
        block_index=0,
    )
    assert all(len(path.conditions) <= 4 for path in paths)
    assert all(len(path.conditions) >= 2 for path in paths)


def _path_candidate_with_sources(count: int) -> PathCandidate:
    return PathCandidate(
        conditions=tuple(
            PathCondition(Source("current", index), "le", float(index), False)
            for index in range(count)
        ),
        block_index=1,
        tree_index=2,
        leaf_index=3,
    )


def test_expand_candidate_subpaths_emits_all_three_condition_combinations() -> None:
    expanded = expand_candidate_subpaths([_path_candidate_with_sources(3)])

    assert [
        tuple(condition.source.feature_index for condition in path.conditions)
        for path in expanded
    ] == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    assert all((path.block_index, path.tree_index, path.leaf_index) == (1, 2, 3)
               for path in expanded)


def test_expand_candidate_subpaths_emits_eleven_four_condition_combinations() -> None:
    expanded = expand_candidate_subpaths([_path_candidate_with_sources(4)])
    widths = [len(path.conditions) for path in expanded]

    assert widths.count(2) == 6
    assert widths.count(3) == 4
    assert widths.count(4) == 1
    assert all(
        list(condition.source.feature_index for condition in path.conditions)
        == sorted(condition.source.feature_index for condition in path.conditions)
        for path in expanded
    )


def test_canonical_path_is_order_invariant_within_quantile_bin() -> None:
    catalog = [Source("current", 30), Source("current", 250)]
    source_values = np.column_stack([
        np.linspace(-2.0, 2.0, 101),
        np.linspace(-3.0, 3.0, 101),
    ])
    grids = training_quantile_grids(source_values, catalog, bins=8)
    first = extract_candidate_paths(
        _two_source_tree(first_threshold=0.11, second_threshold=0.21),
        catalog,
        block_index=0,
    )[0]
    shifted = extract_candidate_paths(
        _two_source_tree(first_threshold=0.14, second_threshold=0.24),
        catalog,
        block_index=1,
    )[0]
    second = PathCandidate(
        conditions=tuple(reversed(shifted.conditions)),
        block_index=shifted.block_index,
        tree_index=shifted.tree_index,
        leaf_index=shifted.leaf_index,
    )

    left = canonicalize_path(first, grids)
    right = canonicalize_path(second, grids)

    assert left.support_key == right.support_key


def _paired_quantile_candidate(
    threshold: float, *, block_index: int
) -> PathCandidate:
    return PathCandidate(
        conditions=(
            PathCondition(Source("current", 1), "le", threshold, False),
            PathCondition(Source("current", 2), "gt", 4.25, True),
        ),
        block_index=block_index,
        tree_index=0,
        leaf_index=0,
    )


def test_paired_quantile_bins_merge_adjacent_fine_regions() -> None:
    grids = {
        Source("current", 1): np.arange(33, dtype=np.float64),
        Source("current", 2): np.arange(33, dtype=np.float64),
    }

    first = canonicalize_path(
        _paired_quantile_candidate(0.25, block_index=0),
        grids,
        support_bin_width=2,
    )
    adjacent = canonicalize_path(
        _paired_quantile_candidate(1.25, block_index=1),
        grids,
        support_bin_width=2,
    )

    assert first.support_key == adjacent.support_key
    assert first.ordered_conditions[0]["quantile_bin"] == 0
    assert first.ordered_conditions[0]["threshold"] == 2.0


def test_paired_quantile_bins_keep_coarse_boundaries_distinct() -> None:
    grids = {
        Source("current", 1): np.arange(33, dtype=np.float64),
        Source("current", 2): np.arange(33, dtype=np.float64),
    }

    left = canonicalize_path(
        _paired_quantile_candidate(1.25, block_index=0),
        grids,
        support_bin_width=2,
    )
    right = canonicalize_path(
        _paired_quantile_candidate(2.25, block_index=1),
        grids,
        support_bin_width=2,
    )

    assert left.support_key != right.support_key
    assert right.ordered_conditions[0]["quantile_bin"] == 1
    assert right.ordered_conditions[0]["threshold"] == 4.0


@pytest.mark.parametrize(
    "family, index, message",
    [("unknown", 1, "unknown source family"), ("current", -1, "non-negative")],
)
def test_source_rejects_invalid_identity(
    family: str, index: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Source(family, index)


def test_aggregate_repeated_paths_requires_two_blocks() -> None:
    catalog = [Source("current", 30), Source("current", 250)]
    values = np.column_stack([np.linspace(-2.0, 2.0, 101)] * 2)
    grids = training_quantile_grids(values, catalog, bins=8)
    repeated = [
        canonicalize_path(
            extract_candidate_paths(_two_source_tree(), catalog, block_index=block)[0],
            grids,
        )
        for block in (0, 1)
    ]
    singleton = canonicalize_path(
        extract_candidate_paths(
            _two_source_tree(first_threshold=-1.5), catalog, block_index=2
        )[0],
        grids,
    )

    accepted = aggregate_repeated_paths([*repeated, singleton], min_blocks=2)

    assert len(accepted) == 1
    assert accepted[0].blocks == (0, 1)


def test_mine_task_interactions_uses_strict_oos_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_residuals: list[np.ndarray] = []

    class FakeBooster:
        def dump_model(self) -> dict[str, object]:
            return _two_source_tree()

    def fake_train(
        features: np.ndarray,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        params: dict[str, object],
        num_boost_round: int,
    ) -> FakeBooster:
        assert len(features) == len(residual) == len(weight)
        assert params["max_depth"] == 4
        assert params["num_leaves"] == 15
        assert num_boost_round == 80
        captured_residuals.append(residual.copy())
        return FakeBooster()

    monkeypatch.setattr(
        interaction_features_module, "_train_residual_booster", fake_train
    )
    time_ids = np.repeat(np.arange(8), 2)
    sources = np.column_stack([
        np.linspace(-2.0, 2.0, len(time_ids)),
        np.linspace(-3.0, 3.0, len(time_ids)),
    ]).astype(np.float32)
    target = np.linspace(-1.0, 1.0, len(time_ids))
    seen_splits: list[tuple[int, int]] = []

    def baseline_predictor(train_rows: np.ndarray, valid_rows: np.ndarray) -> np.ndarray:
        assert train_rows.max() < valid_rows.min()
        seen_splits.append((len(train_rows), len(valid_rows)))
        return target[valid_rows] - 2.0

    result = mine_task_interactions(
        task="ridge",
        source_values=sources,
        catalog=[Source("current", 30), Source("current", 250)],
        target=target,
        weight=np.ones(len(target)),
        time_ids=time_ids,
        baseline_predictor=baseline_predictor,
        n_blocks=4,
        min_blocks=2,
        row_cap=100,
        num_threads=1,
    )

    assert len(seen_splits) == 3
    assert len(captured_residuals) == 3
    assert all(np.allclose(residual, np.full(len(residual), 2.0), atol=1e-15)
               for residual in captured_residuals)
    assert result["definitions"]
    assert result["protocol"]["strict_oos_residuals"] is True
    assert result["protocol"]["quantile_bins"] == 32
    assert result["protocol"]["support_quantile_bins"] == 16
    assert result["protocol"]["support_bin_width"] == 2
    assert all(
        split["expanded_subpaths"] == split["candidate_paths"]
        for split in result["protocol"]["splits"]
    )
    assert all(
        split["unique_canonical_subpaths"] == split["expanded_subpaths"]
        for split in result["protocol"]["splits"]
    )


def test_real_miner_finds_range_effect_but_not_constant_residual() -> None:
    rng = np.random.default_rng(20260819)
    rows_per_block = 160
    base = rng.normal(size=(rows_per_block, 2)).astype(np.float32)
    sources = np.tile(base, (4, 1))
    time_ids = np.repeat(np.arange(8), rows_per_block // 2)
    target = sources[:, 0] * (sources[:, 1] > 0.0)
    kwargs = {
        "task": "ridge",
        "source_values": sources,
        "catalog": [Source("current", 30), Source("current", 250)],
        "weight": np.ones(len(target)),
        "time_ids": time_ids,
        "baseline_predictor": lambda train, valid: np.zeros(len(valid)),
        "n_blocks": 4,
        "min_blocks": 2,
        "row_cap": 1_000,
        "num_threads": 1,
    }

    signal = mine_task_interactions(target=target, **kwargs)
    null = mine_task_interactions(target=np.zeros(len(target)), **kwargs)

    assert any(item["operation"] == "gated_value" for item in signal["definitions"])
    assert null["definitions"] == []


def test_miner_builds_sources_after_time_preserving_split_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_rows: list[tuple[int, int]] = []
    trained_values: list[np.ndarray] = []
    catalog = [Source("current", 30), Source("current", 250)]

    class FakeBooster:
        def dump_model(self) -> dict[str, object]:
            return _two_source_tree()

    def source_builder(
        train_rows: np.ndarray, valid_rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[Source]]:
        built_rows.append((len(train_rows), len(valid_rows)))
        return (
            np.column_stack([train_rows, train_rows]).astype(np.float32),
            np.column_stack([valid_rows, valid_rows]).astype(np.float32),
            catalog,
        )

    def fake_train(
        features: np.ndarray,
        residual: np.ndarray,
        weight: np.ndarray,
        **kwargs: object,
    ) -> FakeBooster:
        trained_values.append(features.copy())
        return FakeBooster()

    monkeypatch.setattr(
        interaction_features_module, "_train_residual_booster", fake_train
    )
    time_ids = np.repeat(np.arange(12), 2)

    mine_task_interactions(
        task="ridge",
        source_values=None,
        catalog=None,
        source_builder=source_builder,
        target=np.zeros(len(time_ids)),
        weight=np.ones(len(time_ids)),
        time_ids=time_ids,
        baseline_predictor=lambda train, valid: np.zeros(len(valid)),
        n_blocks=4,
        row_cap=8,
        num_threads=1,
    )

    assert built_rows == [(6, 6), (8, 6), (8, 6)]
    assert all(len(values) <= 8 for values in trained_values)


def test_task_source_views_cover_all_current_but_only_history40() -> None:
    transformed = np.arange(4 * 323, dtype=np.float32).reshape(4, 323)
    time_ids = np.array([1, 1, 2, 2])
    history_indices = np.arange(40, dtype=np.int64)
    history_blocks = [
        np.full((4, 40), fill_value=index, dtype=np.float32)
        for index in range(4)
    ]

    views = build_interaction_source_views(
        transformed,
        time_ids,
        history_indices,
        history_blocks,
        max_cells=4 * 806,
    )

    assert views["ridge"].values.shape == (4, 323)
    assert views["xs"].values.shape == (4, 323 + 4 * 40)
    assert views["market"].values.shape == (4, 2 * 323 + 4 * 40)
    assert views["ridge"].catalog[250] == Source("current", 250)
    assert views["xs"].catalog[250] == Source("xs_deviation", 250)
    assert views["market"].catalog[250] == Source("market_raw", 250)
    assert views["market"].catalog[323 + 250] == Source("market_deviation", 250)
    history_sources = [
        source for source in views["market"].catalog
        if source.family.startswith("history_")
    ]
    assert len(history_sources) == 160
    assert {source.feature_index for source in history_sources} == set(range(40))


def test_task_source_views_fail_instead_of_exceeding_cell_budget() -> None:
    with pytest.raises(MemoryError, match="market source matrix"):
        build_interaction_source_views(
            np.zeros((4, 323), dtype=np.float32),
            np.array([1, 1, 2, 2]),
            np.arange(40),
            [np.zeros((4, 40), dtype=np.float32) for _ in range(4)],
            max_cells=4 * 806 - 1,
        )


def test_sparse_source_arrays_build_only_manifest_references() -> None:
    rows = 4
    transformed = np.arange(rows * 323, dtype=np.float32).reshape(rows, 323)
    time_ids = np.array([1, 1, 2, 2])
    history_indices = np.arange(40, dtype=np.int64)
    history_blocks = [
        np.full((rows, 40), block, dtype=np.float32) for block in range(4)
    ]
    definitions = [{
        "name": "xs_current_history_0000",
        "operation": "current_history_gated",
        "value_source": "history_previous:7",
        "conditions": [_condition("xs_deviation:250")],
    }]

    sources = interaction_source_arrays(
        "xs",
        definitions,
        transformed,
        time_ids,
        history_indices,
        history_blocks,
        max_cells=rows * 2,
    )

    assert list(sources) == ["xs_deviation:250", "history_previous:7"]
    assert all(values.shape == (rows,) for values in sources.values())
    assert np.shares_memory(sources["history_previous:7"], history_blocks[0])


def test_fold_definitions_resolve_bins_on_outer_training_sources() -> None:
    definitions = [{
        "name": "ridge_region_0000",
        "operation": "region",
        "conditions": [
            {**_condition("current:30", -999.0), "quantile_bin": 1},
            {**_condition("current:250", 999.0), "quantile_bin": 2},
        ],
    }]
    sources = {
        "current:30": np.arange(9, dtype=np.float32),
        "current:250": np.arange(0, 18, 2, dtype=np.float32),
    }

    resolved = resolve_quantile_thresholds(definitions, sources, bins=4)

    assert resolved is not definitions
    assert resolved[0]["conditions"][0]["threshold"] == pytest.approx(4.0)
    assert resolved[0]["conditions"][1]["threshold"] == pytest.approx(12.0)
    assert definitions[0]["conditions"][0]["threshold"] == -999.0


def test_added_interactions_preserve_direct_lgbm_prefix_and_asset_tail() -> None:
    rows = 4
    transformed = np.arange(rows * 323, dtype=np.float32).reshape(rows, 323)
    time_ids = np.array([1, 1, 2, 2])
    asset_ids = np.array([0, 1, 0, 1])
    history = [np.zeros((rows, 40), dtype=np.float32) for _ in range(4)]
    baseline = build_task_lgbm_designs(
        transformed,
        time_ids,
        asset_ids,
        xs_indices=np.arange(200),
        market_indices=np.arange(200),
        history_blocks=history,
    )
    augmented = build_task_lgbm_designs(
        transformed,
        time_ids,
        asset_ids,
        xs_indices=np.arange(200),
        market_indices=np.arange(200),
        history_blocks=history,
        xs_interactions=np.ones((rows, 2), dtype=np.float32),
        market_interactions=np.ones((rows, 3), dtype=np.float32),
    )

    assert baseline["xs"].shape == (rows, 360 + 1)
    assert baseline["market"].shape == (rows, 560 + 1)
    np.testing.assert_array_equal(
        augmented["xs"][:, :360], baseline["xs"][:, :360]
    )
    np.testing.assert_array_equal(
        augmented["market"][:, :560], baseline["market"][:, :560]
    )
    np.testing.assert_array_equal(augmented["xs"][:, -1], asset_ids)
    np.testing.assert_array_equal(augmented["market"][:, -1], asset_ids)
    assert augmented["xs"].shape == (rows, 360 + 2 + 1)
    assert augmented["market"].shape == (rows, 560 + 3 + 1)


def test_fold_manifest_mines_three_tasks_with_history40(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    def fake_mine(**kwargs: object) -> dict[str, object]:
        task = str(kwargs["task"])
        source_values = np.asarray(kwargs["source_values"])
        catalog = list(kwargs["catalog"])
        calls.append((task, source_values.shape[1], len(catalog)))
        return {
            "task": task,
            "definitions": [],
            "accepted_paths": [],
            "protocol": {"strict_oos_residuals": True},
        }

    monkeypatch.setattr(
        interaction_features_module, "mine_task_interactions", fake_mine
    )
    rows = 16
    transformed = np.arange(rows * 323, dtype=np.float32).reshape(rows, 323)
    time_ids = np.repeat(np.arange(8), 2)
    history_indices = np.arange(40, dtype=np.int64)
    history_blocks = [
        np.zeros((rows, 40), dtype=np.float32) for _ in range(4)
    ]
    predictors = {
        task: (lambda train, valid: np.zeros(len(valid), dtype=np.float64))
        for task in ("ridge", "xs", "market")
    }

    manifest = build_interaction_fold_manifest(
        transformed=transformed,
        target=np.linspace(-1.0, 1.0, rows),
        weight=np.ones(rows),
        time_ids=time_ids,
        history_indices=history_indices,
        history_blocks=history_blocks,
        baseline_predictors=predictors,
        max_source_cells=rows * 806,
        miner_kwargs={"n_blocks": 4, "row_cap": 100, "num_threads": 1},
    )

    assert calls == [
        ("ridge", 323, 323),
        ("xs", 483, 483),
        ("market", 806, 806),
    ]
    assert manifest["schema_version"] == 1
    assert manifest["history_indices"] == list(range(40))
    assert manifest["training_window"] == {
        "time_start": 0,
        "time_end": 7,
        "time_ids": 8,
        "rows": 16,
    }
    assert set(manifest["tasks"]) == {"ridge", "xs", "market"}


def test_interaction_gate_requires_all_four_conditions() -> None:
    passing = interaction_gate(
        np.array([0.01, 0.02, 0.03, 0.01, -0.001]),
        delta_a=0.05,
        delta_b=0.04,
    )
    assert passing["passed"] is True
    assert passing["positive_folds"] == 4
    assert passing["drop_best_mean"] > 0.0

    too_few = interaction_gate(
        np.array([0.1, 0.1, -0.01, -0.01, -0.01]),
        delta_a=1.0,
        delta_b=0.1,
    )
    assert too_few["passed"] is False
    assert too_few["checks"]["four_of_five_positive"] is False

    energy_only = interaction_gate(
        np.array([0.01] * 5), delta_a=0.01, delta_b=0.03
    )
    assert energy_only["passed"] is False
    assert energy_only["checks"]["target_alignment"] is False


def _manifest_with_support_bins(*values: int) -> dict[str, object]:
    return {
        "tasks": {
            task: {"protocol": {"support_quantile_bins": value}}
            for task, value in zip(("ridge", "xs", "market"), values)
        }
    }


def test_manifest_support_quantile_bins_requires_one_shared_value() -> None:
    assert manifest_support_quantile_bins(
        _manifest_with_support_bins(16, 16, 16)
    ) == 16

    with pytest.raises(ValueError, match="support_quantile_bins"):
        manifest_support_quantile_bins(_manifest_with_support_bins(16, 8, 16))

    with pytest.raises(ValueError, match="three interaction tasks"):
        manifest_support_quantile_bins(_manifest_with_support_bins(16, 16))


def test_positive_fold_gate_stops_after_two_nonpositive_folds() -> None:
    assert positive_fold_gate_impossible([-0.01], total_folds=5, required_positive=4) is False
    assert positive_fold_gate_impossible(
        [-0.01, 0.0], total_folds=5, required_positive=4
    ) is True
    assert positive_fold_gate_impossible(
        [0.01, -0.01, 0.02], total_folds=5, required_positive=4
    ) is False


def test_append_interactions_keeps_direct_prefix_and_asset_last() -> None:
    base = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 1.0],
    ], dtype=np.float32)
    interactions = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    got = append_interactions_before_asset(base, interactions)

    np.testing.assert_array_equal(got[:, :2], base[:, :2])
    np.testing.assert_array_equal(got[:, 2:4], interactions)
    np.testing.assert_array_equal(got[:, -1], base[:, -1])


def test_paired_fold_sequence_stops_when_four_positive_becomes_impossible() -> None:
    calls: list[int] = []

    def run_fold(index: int) -> dict[str, float]:
        calls.append(index)
        return {"peak_delta": -0.01, "delta_a": -0.1, "delta_b": 0.0}

    result = run_paired_fold_sequence(5, run_fold, required_positive=4)

    assert calls == [0, 1]
    assert result["stopped_early"] is True
    assert result["stop_reason"] == "four_of_five_positive_is_impossible"
    assert len(result["folds"]) == 2


def test_paired_component_reuses_baseline_when_no_interactions() -> None:
    calls: list[tuple[int, int]] = []
    base_train = np.ones((4, 3), dtype=np.float32)
    base_valid = np.ones((2, 3), dtype=np.float32)

    def fit_predict(train: np.ndarray, valid: np.ndarray) -> np.ndarray:
        calls.append((train.shape[1], valid.shape[1]))
        return np.arange(len(valid), dtype=np.float64)

    baseline, interaction = paired_component_predictions(
        base_train,
        base_valid,
        np.empty((4, 0), dtype=np.float32),
        np.empty((2, 0), dtype=np.float32),
        fit_predict,
        asset_last=True,
    )

    assert calls == [(3, 3)]
    np.testing.assert_array_equal(interaction, baseline)


def test_interaction_screen_spills_loaded_features_to_read_only_memmap(
    tmp_path: Path,
) -> None:
    expected = np.arange(30, dtype=np.float32).reshape(6, 5)
    loaded = {
        "features": expected.copy(),
        "target": np.arange(6, dtype=np.float32),
    }

    mapped = spill_interaction_features(
        loaded, tmp_path / "features.npy", chunk_rows=2
    )

    assert "features" not in loaded
    assert isinstance(mapped, np.memmap)
    assert mapped.mode == "r"
    np.testing.assert_array_equal(mapped, expected)


def test_compose_hybrid_raw_uses_frozen_07_117_weights() -> None:
    time_ids = np.array([1, 1, 2, 2])
    ridge = np.array([1.0, 3.0, 2.0, 6.0])
    xs = np.array([-2.0, 4.0, -1.0, 3.0])
    market = np.array([5.0, 7.0, 10.0, 14.0])

    got = compose_hybrid_raw(
        ridge,
        xs,
        market,
        time_ids,
        market_lambda=0.7,
        blend_weight=1.17,
    )

    ridge_market = np.array([2.0, 2.0, 4.0, 4.0])
    market_lgbm = np.array([6.0, 6.0, 12.0, 12.0])
    e_ridge = ridge - ridge_market
    e_lgbm = xs - np.array([1.0, 1.0, 1.0, 1.0])
    expected = (
        0.3 * ridge_market + 0.7 * market_lgbm
        - 0.17 * e_ridge + 1.17 * e_lgbm
    )
    np.testing.assert_allclose(got, expected)


def test_interaction_oof_cli_defaults_to_frozen_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["v3_interaction_oof.py"])

    args = parse_interaction_oof_args()

    assert (args.n_folds, args.train_window, args.embargo) == (5, 78_960, 6)
    assert (args.sample_modulo, args.sampling) == (5, "phase_balanced")
    assert (args.n_seeds, args.num_iteration) == (1, 160)
    assert (args.market_lambda, args.blend_weight) == (0.7, 1.17)


def test_interaction_oof_script_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "experiments/v3_interaction_oof.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--market-lambda" in result.stdout


def test_lag_cache_retains_sampled_rows_only_inside_requested_time_range() -> None:
    time_ids = np.arange(20, dtype=np.int64)

    mask = retained_lag_rows(
        time_ids,
        sample_modulo=2,
        sampling="periodic",
        minimum_time_id=5,
        maximum_time_id=12,
    )

    np.testing.assert_array_equal(np.flatnonzero(mask), np.array([6, 8, 10, 12]))
