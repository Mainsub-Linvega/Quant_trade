from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.oof_cache import (UNREPRODUCIBLE_CACHES, VERIFIED_CURRENT_CODE_CACHE,
                           assert_reproducible_cache, assert_row_alignment,
                           load_oof_bundle, validate_oof_arrays)


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


def test_quarantined_caches_are_refused(tmp_path: Path) -> None:
    """2026-08-18 INCIDENT 的回归（08-20 扩大范围）。

    这几份缓存出自 `experiments/v3_production_oof.py` **首次提交之前**的脚本版本 ⟹
    无法复现。`_exact` 那份实测与当前代码差 max|Δ(market_ridge)| = 3.37e-05，
    约折均 peak 的 2.4%，**与被测效应同量级** —— 拿它做配对基准会得出无法归因的结论。
    ⚠️ 光把文件改名不够：四个实验脚本原本把它写死成 `--oof` 的默认值。
    """
    assert "v3_production_oof_phasebal_prodwindow_exact.npz" in UNREPRODUCIBLE_CACHES
    # 08-20 新增：同签名（13 数组、无 checkpoint、mtime 更早）的那份此前没被点名
    assert "v3_production_oof_phasebal_prodwindow.npz" in UNREPRODUCIBLE_CACHES
    # 改名封存后的路径也必须挡住 —— 否则「把名字改回去」就绕过了
    assert "v3_production_oof_phasebal_prodwindow_exact.STALE-DO-NOT-USE.npz" in UNREPRODUCIBLE_CACHES

    for name in UNREPRODUCIBLE_CACHES:
        path = tmp_path / name
        path.write_bytes(b"")                      # 文件**存在**也必须拒绝
        with pytest.raises(SystemExit, match="从未入库"):
            assert_reproducible_cache(path)


def test_missing_cache_points_at_the_verified_one(tmp_path: Path) -> None:
    """缺文件时的报错要能指路 —— 否则 8/23 赶工时最省事的「修法」就是把毒缓存改回名字。"""
    with pytest.raises(SystemExit) as caught:
        assert_reproducible_cache(tmp_path / "nope.npz")
    message = str(caught.value)
    assert VERIFIED_CURRENT_CODE_CACHE in message
    assert "STALE-DO-NOT-USE" in message


def test_load_bundle_enforces_quarantine(tmp_path: Path) -> None:
    """`load_oof_bundle` 是带校验的入口，隔离必须在它里面生效，不能只靠调用方自觉。"""
    path = tmp_path / "v3_production_oof_phasebal_prodwindow.npz"
    np.savez_compressed(path, **sample_arrays())
    with pytest.raises(SystemExit, match="从未入库"):
        load_oof_bundle(path)
