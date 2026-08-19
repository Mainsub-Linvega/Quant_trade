from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.oof_cache import assert_row_alignment, load_oof_bundle, validate_oof_arrays


def sample_arrays() -> dict[str, np.ndarray]:
    n = 6
    arrays = {
        "target": np.linspace(-1, 1, n),
        "weight": np.ones(n),
        "time_id": np.array([10, 10, 10, 11, 11, 11]),
        "asset_id": np.array([0, 1, 2, 0, 1, 2]),
        "fold": np.array([0, 0, 0, 1, 1, 1], dtype=np.int16),
        "prediction_raw": np.zeros(n),
    }
    for name in ("market_ridge", "market_lgbm", "market", "e_lgbm", "e_target", "xs_lgbm"):
        arrays[name] = np.zeros(n)
    return arrays


def test_validate_oof_rejects_split_cross_section() -> None:
    arrays = sample_arrays()
    arrays["fold"][1] = 1
    with pytest.raises(ValueError, match="spans multiple folds"):
        validate_oof_arrays(arrays)


def test_load_bundle_checks_report_identity(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    report_dir = tmp_path / "experiments"
    cache_dir.mkdir(); report_dir.mkdir()
    path = cache_dir / "demo.npz"
    np.savez_compressed(path, **sample_arrays())
    (report_dir / "demo.json").write_text(json.dumps({"cache": str(path)}))
    bundle = load_oof_bundle(path, report_dir=report_dir)
    assert bundle.sha256
    assert bundle.valid_mask.all()


def test_assert_row_alignment_detects_drift() -> None:
    left = sample_arrays(); right = sample_arrays()
    right["asset_id"] = right["asset_id"].copy(); right["asset_id"][0] = 9
    with pytest.raises(ValueError, match="asset_id"):
        assert_row_alignment(left, right)
