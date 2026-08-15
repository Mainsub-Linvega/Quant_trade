from __future__ import annotations

import numpy as np

from experiments.v3_residual_signal_search import apply_binned, fit_binned, project


def test_binned_asset_adapter_is_reprojected_per_time_id() -> None:
    time_id = np.array([10, 10, 10, 11, 11, 11])
    asset_id = np.array([0, 1, 2, 0, 1, 2])
    values = np.array([1.0, -2.0, 0.5, 2.0, -1.0, 0.25])
    bins = np.array([0, 1, 0, 1, 0, 1])
    slopes = np.array([[1.0, 2.0], [0.5, 1.5], [1.2, 0.8]])
    got = apply_binned(values, asset_id, bins, slopes, time_id)
    assert np.allclose(got[:3].mean(), 0.0)
    assert np.allclose(got[3:].mean(), 0.0)


def test_fit_binned_returns_finite_shrunk_slopes() -> None:
    target = np.array([1.0, 2.0, -1.0, -2.0])
    pred = np.array([0.5, 1.0, -0.5, -1.0])
    weight = np.ones(4)
    assets = np.array([0, 0, 1, 1])
    bins = np.array([0, 1, 0, 1])
    slopes = fit_binned(target, pred, weight, assets, bins, 2, 10.0)
    assert slopes.shape == (2, 2)
    assert np.all(np.isfinite(slopes))


def test_project_removes_group_means() -> None:
    values = np.array([1.0, 3.0, -2.0, 4.0])
    time_id = np.array([1, 1, 2, 2])
    got = project(values, time_id)
    assert np.allclose(got[:2].mean(), 0.0)
    assert np.allclose(got[2:].mean(), 0.0)
