from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_split_backfill_window_keeps_original_and_early_backfill_for_training() -> None:
    from backfill_nn_train import split_backfill_window

    time_id = np.array([5, 6, 888479, 888480, 948479, 948480, 948485, 948486, 1008479, 1008480])
    is_backfill = np.array([False, False, False, True, True, True, True, True, True, True])

    train, valid = split_backfill_window(
        time_id,
        is_backfill,
        train_backfill_end_exclusive=948480,
        valid_start_inclusive=948486,
        valid_end_exclusive=1008480,
    )

    np.testing.assert_array_equal(train, [True, True, True, True, True, False, False, False, False, False])
    np.testing.assert_array_equal(valid, [False, False, False, False, False, False, False, True, True, False])
