from __future__ import annotations

import numpy as np
import pytest

from experiments.v3_residual_adapters import compose_hybrid_prediction, fit_asset_slopes
from strategies.v3_hybrid.features import asset_scaled_zero_mean


def test_asset_scaled_zero_mean_preserves_each_group_mean() -> None:
    values = np.array([-1.0, 0.0, 1.0, 2.0, -2.0, 0.0])
    assets = np.array([0, 1, 2, 0, 1, 2])
    time_ids = np.array([10, 10, 10, 11, 11, 11])
    got = asset_scaled_zero_mean(values, assets, np.array([0.5, 1.0, 2.0]), time_ids)
    assert np.allclose(got[:3].mean(), 0.0, atol=1e-15)
    assert np.allclose(got[3:].mean(), 0.0, atol=1e-15)
    assert np.allclose(got[:3], np.array([-1.0, -0.5, 1.5]))


def test_unit_asset_scales_are_identity_for_zero_mean_groups() -> None:
    values = np.array([-2.0, 0.5, 1.5, -1.0, 1.0])
    assets = np.array([0, 1, 2, 0, 1])
    time_ids = np.array([20, 20, 20, 21, 21])
    got = asset_scaled_zero_mean(values, assets, np.ones(3), time_ids)
    assert np.array_equal(got, values)


def test_asset_scaled_zero_mean_rejects_unknown_asset() -> None:
    with pytest.raises(ValueError, match="outside scales"):
        asset_scaled_zero_mean(np.array([1.0]), np.array([3]), np.ones(3))


def test_compose_hybrid_prediction_uses_local_market_and_cross_weights() -> None:
    market_ridge = np.array([1.0, 1.0])
    market_lgbm = np.array([3.0, 3.0])
    e_ridge = np.array([-2.0, 2.0])
    e_lgbm = np.array([-4.0, 4.0])

    market, cross, prediction = compose_hybrid_prediction(
        market_ridge, market_lgbm, e_ridge, e_lgbm,
        market_lambda=0.7, blend_weight=1.17,
    )

    assert np.allclose(market, np.array([2.4, 2.4]))
    assert np.allclose(cross, -0.17 * e_ridge + 1.17 * e_lgbm)
    assert np.allclose(prediction, market + cross)


def test_fit_asset_slopes_calibrates_lgbm_after_fixed_ridge_cross() -> None:
    time_id = np.repeat(np.array([10, 11, 12]), 2)
    asset_id = np.tile(np.array([0, 1]), 3)
    e_lgbm = np.array([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0])
    e_ridge = np.array([-0.5, 0.5, -1.0, 1.0, -1.5, 1.5])
    blend_weight = 1.17
    fixed_cross = (1.0 - blend_weight) * e_ridge
    target = fixed_cross + blend_weight * 1.5 * e_lgbm

    slopes = fit_asset_slopes(
        time_id, target, np.ones_like(target), asset_id, e_lgbm, 0.0,
        fixed_cross=fixed_cross, variable_weight=blend_weight,
    )

    assert np.allclose(slopes, np.array([1.5, 1.5]))
