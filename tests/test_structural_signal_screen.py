from __future__ import annotations

import numpy as np

from experiments.structural_signal_screen import (
    asset_panel_features,
    cross_sectional_rank_features,
    group_mean,
    market_summary_features,
    stable_top_correlated,
)


def test_group_mean_is_constant_inside_cross_section() -> None:
    time_id = np.array([1, 1, 1, 2, 2, 2])
    values = np.array([1., 2., 6., -1., 2., 5.])
    got = group_mean(values, time_id)
    assert np.allclose(got[:3], 3.0)
    assert np.allclose(got[3:], 2.0)



def test_asset_panel_preserves_identity_and_is_prefix_causal() -> None:
    tid = np.array([10, 10, 11, 11, 12, 12])
    aid = np.array([1, 0, 0, 1, 1, 0])
    x = np.array([[2.], [1.], [3.], [4.], [6.], [5.]], dtype=np.float32)
    prefix = asset_panel_features(x[:4], tid[:4], aid[:4], n_assets=2)
    full = asset_panel_features(x, tid, aid, n_assets=2)
    assert np.array_equal(prefix, full[:2])
    # current panel is asset-major even when input row order changes
    assert np.array_equal(full[0, :2], [1., 2.])
    # third panel delta = [5, 6] - [3, 4]
    assert np.array_equal(full[2, 4:6], [2., 2.])

def test_rank_features_are_cross_sectional_and_shift_invariant() -> None:
    time_id = np.array([1, 1, 1, 2, 2, 2])
    x = np.array([[1., 4.], [3., 2.], [2., 3.], [10., 8.], [12., 6.], [11., 7.]], dtype=np.float32)
    got = cross_sectional_rank_features(x, time_id)
    shifted = cross_sectional_rank_features(x + np.array([[100., -30.]], dtype=np.float32), time_id)
    assert got.shape == (6, 6)
    assert np.allclose(got, shifted)
    assert np.allclose(got[:3, :2].mean(axis=0), 0.0)


def test_market_summary_dynamics_are_prefix_causal() -> None:
    tid = np.repeat(np.arange(10, 14), 3)
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    prefix = market_summary_features(x[:9], tid[:9])
    full = market_summary_features(x, tid)
    assert np.array_equal(prefix, full[:3])


def test_residual_selector_finds_known_signal() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2000, 5))
    y = 3.0 * x[:, 3] + rng.normal(scale=0.1, size=2000)
    chosen = stable_top_correlated(x, y, 2)
    assert chosen[0] == 3
