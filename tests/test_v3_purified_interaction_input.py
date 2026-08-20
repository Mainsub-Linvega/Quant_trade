from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.v3_purified_interaction_input import (
    build_market_input_arrays,
    build_ridge_input_arrays,
    build_xs_input_arrays,
    write_ridge_input_artifacts,
    write_task_input_artifacts,
)


def _oof_arrays() -> dict[str, np.ndarray]:
    return {
        "target": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        "weight": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "time_id": np.array([0, 0, 1, 1, 2, 2]),
        "fold": np.array([-1, -1, 0, 0, 1, 1]),
        "market_ridge": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        "market_lgbm": np.array([12.0, 12.0, 32.0, 32.0, 52.0, 52.0]),
        "e_ridge": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "e_lgbm": np.array([-4.0, 4.0, -3.0, 3.0, -2.0, 2.0]),
    }


def test_ridge_input_keeps_only_strict_oof_rows_and_exact_residual() -> None:
    features = np.arange(6 * 323, dtype=np.float32).reshape(6, 323)

    got = build_ridge_input_arrays(features, _oof_arrays())

    np.testing.assert_array_equal(got["features"], features[2:])
    np.testing.assert_allclose(
        got["residual"],
        np.array([27.7, 37.6, 46.5, 56.4]),
    )
    np.testing.assert_array_equal(got["weight"], [3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(got["time_id"], [1, 1, 2, 2])
    np.testing.assert_array_equal(got["feature_indices"], np.arange(323))


def test_ridge_input_rejects_misaligned_feature_cache() -> None:
    features = np.zeros((5, 323), dtype=np.float32)

    with pytest.raises(ValueError, match="row-aligned"):
        build_ridge_input_arrays(features, _oof_arrays())


def test_xs_input_uses_unweighted_cross_target_and_strict_oof_component() -> None:
    features = np.arange(6 * 323, dtype=np.float32).reshape(6, 323)

    got = build_xs_input_arrays(features, _oof_arrays())

    np.testing.assert_array_equal(got["features"], features[2:])
    np.testing.assert_allclose(got["residual"], [-2.0, 2.0, -3.0, 3.0])
    np.testing.assert_array_equal(got["weight"], [3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(got["time_id"], [1, 1, 2, 2])


def test_market_input_aggregates_one_row_per_time_id() -> None:
    features = np.arange(6 * 323, dtype=np.float32).reshape(6, 323)

    got = build_market_input_arrays(features, _oof_arrays())

    np.testing.assert_allclose(got["features"], (features[2::2] + features[3::2]) / 2)
    np.testing.assert_allclose(got["residual"], [3.0, 3.0])
    np.testing.assert_array_equal(got["weight"], [7.0, 11.0])
    np.testing.assert_array_equal(got["time_id"], [1, 2])


def test_ridge_input_rejects_missing_or_nonfinite_oof_components() -> None:
    features = np.zeros((6, 323), dtype=np.float32)
    missing = _oof_arrays()
    del missing["e_ridge"]
    with pytest.raises(ValueError, match="missing"):
        build_ridge_input_arrays(features, missing)

    nonfinite = _oof_arrays()
    nonfinite["market_ridge"][3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_ridge_input_arrays(features, nonfinite)


def test_ridge_input_artifacts_round_trip_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    features_path = tmp_path / "features.npy"
    oof_path = tmp_path / "oof.npz"
    output_path = tmp_path / "ridge_input.npz"
    np.save(
        features_path,
        np.arange(6 * 323, dtype=np.float32).reshape(6, 323),
    )
    np.savez(oof_path, **_oof_arrays())

    paths = write_ridge_input_artifacts(
        features_path,
        oof_path,
        output_path,
    )

    assert set(paths) == {"npz", "manifest"}
    with np.load(paths["npz"], allow_pickle=False) as loaded:
        assert loaded["features"].shape == (4, 323)
        np.testing.assert_allclose(loaded["residual"], [27.7, 37.6, 46.5, 56.4])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["task"] == "ridge"
    assert manifest["residual_formula"] == "target - (market_ridge + e_ridge)"
    assert manifest["rows"] == 4
    assert len(manifest["npz_sha256"]) == 64
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*candidate*"))
    with pytest.raises(FileExistsError, match="force"):
        write_ridge_input_artifacts(features_path, oof_path, output_path)


@pytest.mark.parametrize(
    ("task", "formula", "rows"),
    [
        ("xs", "target_cross_unweighted - e_lgbm", 4),
        ("market", "target_mean_unweighted - market_lgbm", 2),
    ],
)
def test_component_input_artifacts_record_task_formula(
    tmp_path: Path,
    task: str,
    formula: str,
    rows: int,
) -> None:
    features_path = tmp_path / "features.npy"
    oof_path = tmp_path / "oof.npz"
    output_path = tmp_path / f"{task}_input.npz"
    np.save(features_path, np.zeros((6, 323), dtype=np.float32))
    np.savez(oof_path, **_oof_arrays())

    paths = write_task_input_artifacts(
        task,
        features_path,
        oof_path,
        output_path,
    )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["task"] == task
    assert manifest["residual_formula"] == formula
    assert manifest["rows"] == rows


def test_ridge_input_script_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "experiments/v3_purified_interaction_input.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--features-npy" in result.stdout
    assert "--oof-npz" in result.stdout
    assert "--output-npz" in result.stdout
