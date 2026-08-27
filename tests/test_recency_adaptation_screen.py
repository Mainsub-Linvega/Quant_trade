from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_effective_fit_weight_does_not_change_metric_weight() -> None:
    from recency_adaptation_screen import effective_fit_weight

    metric_weight = np.array([1.0, 2.0, 3.0])
    multiplier = np.array([1.0, 2.0, 0.5])
    fit_weight = effective_fit_weight(metric_weight, multiplier)
    np.testing.assert_allclose(fit_weight, [1.0, 4.0, 1.5])
    np.testing.assert_allclose(metric_weight, [1.0, 2.0, 3.0])


def test_market_fit_weight_stays_unweighted() -> None:
    from recency_adaptation_screen import market_fit_weight

    assert market_fit_weight(np.array([1.0, 2.0]), "frozen_unweighted") is None


def test_policy_multiplier_only_changes_backfill_fit_rows() -> None:
    from recency_adaptation_screen import policy_multiplier

    ids = np.array([10, 888480, 928000, 948000])
    backfill = np.array([False, True, True, True])
    got = policy_multiplier(ids, backfill, "backfill_x2", fit_end=948480)
    np.testing.assert_allclose(got, [1.0, 2.0, 2.0, 2.0])


def test_paired_gate_requires_drop_best_and_four_positive_blocks() -> None:
    from recency_adaptation_screen import paired_gate

    baseline = {"blocks": [{"peak": 1.0, "A": 1.0, "B": 1.0}] * 5}
    candidate = {"blocks": [{"peak": 1.1, "A": 1.1, "B": 1.0}] * 5}
    assert paired_gate(baseline, candidate)["passed_gate"] is True
