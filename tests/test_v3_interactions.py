from __future__ import annotations

import numpy as np
import pytest

import experiments.v3_interaction_features as interaction_features_module
from experiments.v3_interaction_features import (
    PathCandidate,
    Source,
    aggregate_repeated_paths,
    canonicalize_path,
    extract_candidate_paths,
    mine_task_interactions,
    training_quantile_grids,
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
