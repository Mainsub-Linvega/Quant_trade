from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_compose_market_cross_prediction_projects_cross_within_time() -> None:
    from backfill_nn_train import compose_market_cross_prediction

    market = np.array([10.0, 20.0])
    cross = np.array([1.0, 3.0, -2.0, 4.0, 7.0])
    counts = np.array([2, 3])

    result = compose_market_cross_prediction(market, cross, counts)

    np.testing.assert_allclose(result, np.array([9.0, 11.0, 15.0, 21.0, 24.0]))
