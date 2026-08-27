from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_window_masks_keep_original_and_select_backfill_interval() -> None:
    from model_upgrade_protocol import split_labeled_window

    ids = np.array([
        10, 888479, 888480, 948479, 948480, 948481, 948482,
        948483, 948484, 948485, 948486, 1008480,
    ])
    is_backfill = np.array([
        False, False, True, True, True, True, True, True,
        True, True, True, True,
    ])

    train, valid, embargo = split_labeled_window(
        ids, is_backfill,
        train_backfill_end=948480,
        valid_start=948486,
        valid_end=1008480,
        embargo=6,
    )

    assert train.tolist() == [
        True, True, True, True, False, False, False, False,
        False, False, False, False,
    ]
    assert valid.tolist() == [
        False, False, False, False, False, False, False, False,
        False, False, True, False,
    ]
    assert embargo.tolist() == [
        False, False, False, False, True, True, True, True,
        True, True, False, False,
    ]


def test_half_life_multiplier_preserves_original_rows() -> None:
    from model_upgrade_protocol import recency_multiplier

    ids = np.array([0, 888480, 927960, 967440])
    is_backfill = np.array([False, True, True, True])

    result = recency_multiplier(
        ids, is_backfill, mode="half_life",
        backfill_origin=888480, half_life=39480,
    )

    np.testing.assert_allclose(result, [1.0, 1.0, 0.5, 0.25])


@pytest.mark.parametrize(
    ("mode", "half_life"),
    [("unknown", 39480), ("half_life", 0)],
)
def test_recency_multiplier_rejects_invalid_policy(
    mode: str, half_life: int,
) -> None:
    from model_upgrade_protocol import recency_multiplier

    with pytest.raises(ValueError):
        recency_multiplier(
            np.array([0, 888480]),
            np.array([False, True]),
            mode=mode,
            backfill_origin=888480,
            half_life=half_life,
        )


def test_split_rejects_missing_embargo_time_id() -> None:
    from model_upgrade_protocol import split_labeled_window

    with pytest.raises(ValueError, match="embargo"):
        split_labeled_window(
            np.array([10, 948480, 948481, 948483, 948484, 948485, 948486]),
            np.array([False, True, True, True, True, True, True]),
            train_backfill_end=948480,
            valid_start=948486,
            valid_end=948487,
            embargo=6,
        )
