from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]

from backfill_nn_train import split_backfill_holdout, selected_responders


def test_split_backfill_holdout_reserves_tail_public_times_only() -> None:
    time_id = np.array([1, 2, 10, 11, 12, 13, 14], dtype=np.int64)
    is_backfill = np.array([False, False, True, True, True, True, True])

    train, holdout, cutoff = split_backfill_holdout(time_id, is_backfill, reserve_time_ids=2)

    np.testing.assert_array_equal(train, np.array([True, True, True, True, True, False, False]))
    np.testing.assert_array_equal(holdout, np.array([False, False, False, False, False, True, True]))
    assert cutoff == 13


def test_selected_responders_accepts_named_sets_and_custom_values() -> None:
    assert selected_responders('ladder', None) == [
        'responder_00',
        'responder_02',
        'responder_03',
        'responder_04',
        'responder_05',
    ]
    assert selected_responders('shortlist', None) == [
        'responder_04',
        'responder_28',
        'responder_05',
        'responder_29',
        'responder_06',
    ]
    assert selected_responders('custom', 'responder_04,responder_28') == [
        'responder_04',
        'responder_28',
    ]


def test_arm_score_serialises_zero_prediction_without_nan() -> None:
    import json

    from backfill_nn_train import arm_score

    payload = arm_score(np.array([1.0, -1.0]), np.zeros(2), np.ones(2))

    json.dumps(payload, allow_nan=False)
    assert payload['peak'] == 0.0
    assert payload['optimal_scale'] is None


def test_format_score_cell_handles_missing_values() -> None:
    from backfill_nn_train import format_score_cell

    assert format_score_cell(None, '.4f') == 'n/a'
    assert format_score_cell(1.23456, '.2f') == '1.23'


def test_experiment_distinction_names_public_backfill_and_prior_mlp() -> None:
    from backfill_nn_train import experiment_distinction

    block = experiment_distinction()

    assert block['uses_public_backfill_labels'] is True
    assert '20260823' in block['new_data']
    assert 'target_mlp_screen' in block['different_from'][0]
    assert 'cached v3 OOF' in block['different_from'][0]
    assert 'public backfill tail' in block['holdout']


def test_experiment_distinction_focuses_training_reopen_not_old_signal() -> None:
    from backfill_nn_train import experiment_distinction

    block = experiment_distinction()

    assert 'previous NN did not show stable signal' in block['training_reopen_reason']
    assert 'new training distribution' in block['training_reopen_reason']
    assert 'not a continuation of the old target_mlp_screen verdict' in block['scope_boundary']


def test_arm_mode_controls_auxiliary_work() -> None:
    from backfill_nn_train import arm_mode

    assert arm_mode('target_only') == {'target_only': True, 'aux': False}
    assert arm_mode('both') == {'target_only': True, 'aux': True}


def test_resolve_seeds_prefers_multi_seed_list() -> None:
    import argparse
    from backfill_nn_train import resolve_seeds

    assert resolve_seeds(argparse.Namespace(seed=2026, seeds=None)) == [2026]
    assert resolve_seeds(argparse.Namespace(seed=2026, seeds=[2027, 2028])) == [2027, 2028]


def test_prediction_ensemble_average_is_rowwise_mean() -> None:
    from backfill_nn_train import ensemble_prediction

    preds = [np.array([1.0, 3.0]), np.array([3.0, 5.0]), np.array([5.0, 7.0])]

    np.testing.assert_allclose(ensemble_prediction(preds), np.array([3.0, 5.0]))


def test_loaded_seed_models_replay_target_prediction() -> None:
    from backfill_nn_train import replay_target_only_from_models
    from mlp_numpy import NumpyMLP

    market = NumpyMLP([np.array([[2.0]])], [np.array([0.5])])
    cross = NumpyMLP([np.array([[1.5]])], [np.array([-0.25])])
    market_design = np.array([[1.0], [3.0]])
    cross_design = np.array([[2.0], [4.0], [6.0]])
    counts = np.array([1, 2])

    pred = replay_target_only_from_models(
        market,
        cross,
        market_design,
        cross_design,
        counts,
        market_mean=10.0,
        market_std=0.1,
        cross_mean=1.0,
        cross_std=0.2,
    )

    market_part = np.array([10.25, 10.65])
    resid = np.array([1.55, 2.15, 2.75])
    resid -= np.array([1.55, (2.15 + 2.75) / 2, (2.15 + 2.75) / 2])
    np.testing.assert_allclose(pred, np.repeat(market_part, counts) + resid)


def test_metadata_requires_transform_stats_for_cold_start_prediction() -> None:
    from backfill_nn_train import transform_selected_features_from_metadata

    features = np.array([[1.0, 2.0]], dtype=np.float32)

    try:
        transform_selected_features_from_metadata(features, {"selected": [0]})
    except ValueError as exc:
        assert "transform stats" in str(exc)
    else:
        raise AssertionError("missing transform stats should fail before cold-start inference")


def test_artifact_rows_predict_from_raw_features_matches_replay_path(tmp_path) -> None:
    from backfill_nn_train import predict_target_only_from_artifacts, replay_target_only_from_models
    from mlp_numpy import NumpyMLP

    market = NumpyMLP([np.array([[2.0]])], [np.array([0.5])])
    cross = NumpyMLP([np.array([[1.5], [0.0], [0.0]])], [np.array([-0.25])])
    market_path = tmp_path / "seed_1_market.npz"
    cross_path = tmp_path / "seed_1_cross.npz"
    stats = {"lower": [0.0], "upper": [10.0], "center": [1.0], "scale": [2.0]}
    market.save(market_path, {"selected": [0], "mean": 10.0, "std": 0.1, **stats})
    cross.save(cross_path, {"selected": [0], "mean": 1.0, "std": 0.2, "asset_one_hot": 2, **stats})
    features = np.array([[3.0], [5.0], [7.0]], dtype=np.float32)
    time_id = np.array([10, 11, 11], dtype=np.int64)
    asset_id = np.array([0, 0, 1], dtype=np.int64)
    counts = np.array([1, 2], dtype=np.int64)
    market_design = np.array([[1.0], [2.5]], dtype=np.float32)
    cross_design = np.column_stack([
        np.array([[0.0], [-0.5], [0.5]], dtype=np.float32),
        np.eye(2, dtype=np.float32)[asset_id],
    ])

    cold = predict_target_only_from_artifacts(
        [{"seed": 1, "market": str(market_path), "cross": str(cross_path)}],
        features,
        time_id,
        asset_id,
    )
    replay = replay_target_only_from_models(
        market,
        cross,
        market_design,
        cross_design,
        counts,
        market_mean=10.0,
        market_std=0.1,
        cross_mean=1.0,
        cross_std=0.2,
    )

    np.testing.assert_allclose(cold, replay)
