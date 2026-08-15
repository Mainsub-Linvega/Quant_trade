from __future__ import annotations

import numpy as np
import pytest

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
