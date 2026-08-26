from __future__ import annotations

import numpy as np


def test_orthogonal_market_residual_removes_reference_projection() -> None:
    from backfill_nn_market_residual_search import orthogonal_residual

    reference = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = 2.0 * reference + np.array([1.0, -2.0, 1.0, 0.0])
    weight = np.ones(4)

    residual, gamma = orthogonal_residual(candidate, reference, weight)

    assert gamma == 2.0
    np.testing.assert_allclose(np.dot(weight, residual * reference), 0.0)
