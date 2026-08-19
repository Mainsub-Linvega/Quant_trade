from __future__ import annotations

import numpy as np
import pytest

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
