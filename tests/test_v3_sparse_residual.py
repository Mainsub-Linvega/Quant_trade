from __future__ import annotations

import numpy as np

from experiments.v3_sparse_asset_feature_residual import project, sparse_asset_design


def test_sparse_asset_design_places_features_in_asset_block() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    asset = np.array([0, 2, 1])
    got = sparse_asset_design(x, asset, 2).toarray()
    assert got.shape == (3, 6)
    assert np.array_equal(got[0], np.array([1, 2, 0, 0, 0, 0]))
    assert np.array_equal(got[1], np.array([0, 0, 0, 0, 3, 4]))
    assert np.array_equal(got[2], np.array([0, 0, 5, 6, 0, 0]))


def test_sparse_correction_projection_is_zero_mean() -> None:
    values = np.array([1.0, -2.0, 4.0, 3.0, 0.0, -1.0])
    time_id = np.array([10, 10, 10, 11, 11, 11])
    got = project(values, time_id)
    assert np.allclose(got[:3].mean(), 0.0)
    assert np.allclose(got[3:].mean(), 0.0)
