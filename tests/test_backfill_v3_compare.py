from __future__ import annotations

import numpy as np


def test_time_window_mask_selects_only_requested_interval() -> None:
    from backfill_v3_compare import time_window_mask

    ids = np.array([1, 5, 6, 10], dtype=np.int64)

    np.testing.assert_array_equal(
        time_window_mask(ids, start=5, end=10),
        np.array([False, True, True, False]),
    )
