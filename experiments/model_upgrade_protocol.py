from __future__ import annotations

from typing import Literal

import numpy as np


RecencyMode = Literal["none", "backfill_x2", "half_life"]


def _validate_arrays(time_id: np.ndarray, is_backfill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(time_id)
    backfill = np.asarray(is_backfill, dtype=bool)
    if ids.ndim != 1 or backfill.ndim != 1 or ids.shape != backfill.shape:
        raise ValueError("time_id and is_backfill must be one-dimensional with equal shapes")
    if not np.issubdtype(ids.dtype, np.number) or not np.all(np.isfinite(ids)):
        raise ValueError("time_id must contain finite numeric values")
    if np.any(ids != np.floor(ids)):
        raise ValueError("time_id must contain integer-valued values")
    ids = ids.astype(np.int64, copy=False)
    if np.any(np.diff(ids) < 0):
        raise ValueError("time_id must be nondecreasing")
    return ids, backfill


def split_labeled_window(
    time_id: np.ndarray,
    is_backfill: np.ndarray,
    *,
    train_backfill_end: int,
    valid_start: int,
    valid_end: int,
    embargo: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, backfill = _validate_arrays(time_id, is_backfill)
    boundaries = (train_backfill_end, valid_start, valid_end, embargo)
    if not all(np.isfinite(value) for value in boundaries):
        raise ValueError("split boundaries must be finite")
    if any(float(value) != np.floor(value) for value in boundaries):
        raise ValueError("split boundaries must be integer-valued")
    train_end = int(train_backfill_end)
    valid_begin = int(valid_start)
    valid_stop = int(valid_end)
    gap = int(embargo)
    if gap < 0 or train_end > valid_begin or valid_begin >= valid_stop:
        raise ValueError("invalid split boundaries")
    if not np.any(backfill):
        raise ValueError("no backfill rows are available")
    train = (~backfill) | (backfill & (ids < train_end))
    valid = backfill & (ids >= valid_begin) & (ids < valid_stop)
    embargo_mask = backfill & (ids >= train_end) & (ids < valid_begin)
    if gap:
        if valid_begin - train_end != gap:
            raise ValueError("embargo width does not match boundary gap")
        expected = np.arange(train_end, valid_begin, dtype=np.int64)
        observed = np.unique(ids[embargo_mask])
        if not np.array_equal(observed, expected):
            raise ValueError("embargo time_ids are not complete")
    if not train.any() or not valid.any():
        raise ValueError("training or validation window is empty")
    if np.any(train & valid) or np.any(train & embargo_mask) or np.any(valid & embargo_mask):
        raise ValueError("split masks overlap")
    return train, valid, embargo_mask


def recency_multiplier(
    time_id: np.ndarray,
    is_backfill: np.ndarray,
    *,
    mode: RecencyMode | str,
    backfill_origin: int,
    half_life: int,
) -> np.ndarray:
    ids, backfill = _validate_arrays(time_id, is_backfill)
    if mode not in ("none", "backfill_x2", "half_life"):
        raise ValueError(f"unsupported recency mode: {mode}")
    if not np.isfinite(backfill_origin) or float(backfill_origin) != np.floor(backfill_origin):
        raise ValueError("backfill_origin must be an integer")
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive")
    result = np.ones(len(ids), dtype=np.float64)
    if mode == "backfill_x2":
        result[backfill] = 2.0
    elif mode == "half_life":
        age = np.maximum(ids.astype(np.float64) - float(backfill_origin), 0.0)
        result[backfill] = np.power(2.0, -age[backfill] / float(half_life))
    return result
