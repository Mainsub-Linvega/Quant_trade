"""Build row-aligned real-data inputs for purified interaction diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FEATURES = (
    _REPO_ROOT
    / "outputs"
    / "cache"
    / ".v3_adaptive_selection_screen_residual_1s160_phasebal_prodwindow_features.npy"
)
_DEFAULT_OOF = (
    _REPO_ROOT
    / "outputs"
    / "cache"
    / "v3_production_oof_local_07_117_rebuild_3s480.npz"
)
_DEFAULT_OUTPUT = (
    _REPO_ROOT / "outputs" / "cache" / "v3_purified_p0_ridge_input.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ridge", "xs", "market"], default="ridge")
    parser.add_argument("--features-npy", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--oof-npz", default=str(_DEFAULT_OOF))
    parser.add_argument("--output-npz")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_ridge_input_arrays(
    features: np.ndarray,
    oof_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return strict-OOF Ridge residual rows aligned with all 323 raw features."""
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != 323:
        raise ValueError("features must have exactly 323 columns")
    required = {
        "target",
        "weight",
        "time_id",
        "fold",
        "market_ridge",
        "e_ridge",
    }
    missing = sorted(required.difference(oof_arrays))
    if missing:
        raise ValueError(f"OOF cache is missing arrays: {missing}")
    arrays = {name: np.asarray(oof_arrays[name]) for name in required}
    if any(value.shape != (len(x),) for value in arrays.values()):
        raise ValueError("features and OOF cache must be row-aligned")
    valid = arrays["fold"] >= 0
    if not np.any(valid):
        raise ValueError("OOF cache contains no validation rows")
    target = np.asarray(arrays["target"][valid], dtype=np.float64)
    weight = np.asarray(arrays["weight"][valid], dtype=np.float64)
    market_ridge = np.asarray(arrays["market_ridge"][valid], dtype=np.float64)
    e_ridge = np.asarray(arrays["e_ridge"][valid], dtype=np.float64)
    time_id = np.asarray(arrays["time_id"][valid], dtype=np.int64)
    fold = np.asarray(arrays["fold"][valid], dtype=np.int8)
    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(market_ridge))
        or not np.all(np.isfinite(e_ridge))
        or np.any(weight <= 0.0)
    ):
        raise ValueError("strict OOF targets, weights, and Ridge components must be finite")
    if np.any(np.diff(time_id) < 0):
        raise ValueError("strict OOF time_id must be nondecreasing")
    return {
        "features": np.asarray(x[valid], dtype=np.float32),
        "residual": target - (market_ridge + e_ridge),
        "weight": weight,
        "time_id": time_id,
        "fold": fold,
        "feature_indices": np.arange(323, dtype=np.int64),
    }


def _group_bounds(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    return starts, counts


def _validated_component_rows(
    features: np.ndarray,
    oof_arrays: Mapping[str, np.ndarray],
    component: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != 323:
        raise ValueError("features must have exactly 323 columns")
    required = {"target", "weight", "time_id", "fold", component}
    missing = sorted(required.difference(oof_arrays))
    if missing:
        raise ValueError(f"OOF cache is missing arrays: {missing}")
    arrays = {name: np.asarray(oof_arrays[name]) for name in required}
    if any(value.shape != (len(x),) for value in arrays.values()):
        raise ValueError("features and OOF cache must be row-aligned")
    valid = arrays["fold"] >= 0
    if not np.any(valid):
        raise ValueError("OOF cache contains no validation rows")
    selected = {name: np.asarray(value[valid]) for name, value in arrays.items()}
    if (
        not np.all(np.isfinite(selected["target"]))
        or not np.all(np.isfinite(selected["weight"]))
        or not np.all(np.isfinite(selected[component]))
        or np.any(selected["weight"] <= 0.0)
    ):
        raise ValueError("strict OOF target, weight, and component must be finite")
    if np.any(np.diff(selected["time_id"]) < 0):
        raise ValueError("strict OOF time_id must be nondecreasing")
    return np.asarray(x[valid], dtype=np.float32), selected


def build_xs_input_arrays(
    features: np.ndarray,
    oof_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return strict-OOF cross-sectional LGBM residual rows."""
    x, rows = _validated_component_rows(features, oof_arrays, "e_lgbm")
    target = np.asarray(rows["target"], dtype=np.float64)
    time_id = np.asarray(rows["time_id"], dtype=np.int64)
    starts, counts = _group_bounds(time_id)
    means = np.add.reduceat(target, starts) / counts
    residual = target - np.repeat(means, counts) - rows["e_lgbm"]
    return {
        "features": x,
        "residual": residual,
        "weight": np.asarray(rows["weight"], dtype=np.float64),
        "time_id": time_id,
        "fold": np.asarray(rows["fold"], dtype=np.int8),
        "feature_indices": np.arange(323, dtype=np.int64),
    }


def _group_feature_means(
    features: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    *,
    column_chunk: int = 16,
) -> np.ndarray:
    output = np.empty((len(starts), features.shape[1]), dtype=np.float32)
    for left in range(0, features.shape[1], column_chunk):
        right = min(left + column_chunk, features.shape[1])
        block = np.asarray(features[:, left:right], dtype=np.float32).copy()
        finite = np.isfinite(block)
        block[~finite] = 0.0
        sums = np.add.reduceat(block, starts, axis=0)
        support = np.add.reduceat(finite, starts, axis=0)
        output[:, left:right] = np.divide(
            sums,
            support,
            out=np.full_like(sums, np.nan),
            where=support > 0,
        )
    return output


def build_market_input_arrays(
    features: np.ndarray,
    oof_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Aggregate strict-OOF market LGBM residuals to one row per time_id."""
    x, rows = _validated_component_rows(features, oof_arrays, "market_lgbm")
    target = np.asarray(rows["target"], dtype=np.float64)
    weight = np.asarray(rows["weight"], dtype=np.float64)
    time_id = np.asarray(rows["time_id"], dtype=np.int64)
    component = np.asarray(rows["market_lgbm"], dtype=np.float64)
    fold = np.asarray(rows["fold"], dtype=np.int8)
    starts, counts = _group_bounds(time_id)
    component_by_time = component[starts]
    fold_by_time = fold[starts]
    if not np.array_equal(fold, np.repeat(fold_by_time, counts)):
        raise ValueError("market input requires exactly one fold within each time_id")
    if not np.allclose(component, np.repeat(component_by_time, counts), rtol=0.0, atol=1e-12):
        raise ValueError("market_lgbm must be constant within each time_id")
    target_by_time = np.add.reduceat(target, starts) / counts
    return {
        "features": _group_feature_means(x, starts, counts),
        "residual": target_by_time - component_by_time,
        "weight": np.add.reduceat(weight, starts),
        "time_id": time_id[starts],
        "fold": fold_by_time,
        "feature_indices": np.arange(323, dtype=np.int64),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_TASK_BUILDERS = {
    "ridge": build_ridge_input_arrays,
    "xs": build_xs_input_arrays,
    "market": build_market_input_arrays,
}
_RESIDUAL_FORMULAS = {
    "ridge": "target - (market_ridge + e_ridge)",
    "xs": "target_cross_unweighted - e_lgbm",
    "market": "target_mean_unweighted - market_lgbm",
}


def write_task_input_artifacts(
    task: str,
    features_path: str | Path,
    oof_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Validate sources and atomically write one task P0 NPZ plus manifest."""
    if task not in _TASK_BUILDERS:
        raise ValueError(f"unknown P0 input task: {task}")
    feature_source = Path(features_path).resolve()
    oof_source = Path(oof_path).resolve()
    output = Path(output_path).resolve()
    manifest_path = output.with_suffix(".json")
    paths = {"npz": output, "manifest": manifest_path}
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError("P0 input artifact exists; use force to overwrite")
    features = np.load(feature_source, mmap_mode="r", allow_pickle=False)
    with np.load(oof_source, allow_pickle=False) as loaded:
        oof_arrays = {name: loaded[name] for name in loaded.files}
    arrays = _TASK_BUILDERS[task](features, oof_arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = output.with_name(output.name + ".tmp.npz")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        np.savez(temporary_npz, **arrays)
        temporary_npz.replace(output)
        unique_times = np.unique(arrays["time_id"])
        fold_values = sorted(int(value) for value in np.unique(arrays["fold"]))
        proposal_folds = [value for value in fold_values if value in {0, 1, 2}]
        manifest = {
            "schema_version": 1,
            "task": task,
            "residual_formula": _RESIDUAL_FORMULAS[task],
            "features_source": str(feature_source),
            "oof_source": str(oof_source),
            "rows": int(len(arrays["residual"])),
            "feature_count": int(arrays["features"].shape[1]),
            "time_id_range": [int(unique_times[0]), int(unique_times[-1])],
            "time_id_count": int(len(unique_times)),
            "weight_sum": float(np.sum(arrays["weight"])),
            "npz": str(output),
            "npz_sha256": _sha256_file(output),
            "candidate_generated": False,
            "fold_values": fold_values,
            "proposal_folds_present": proposal_folds,
            "gate_folds_present": [value for value in fold_values if value in {3, 4}],
            "submission_generated": False,
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return paths


def write_ridge_input_artifacts(
    features_path: str | Path,
    oof_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Backward-compatible Ridge-specific P0 artifact writer."""
    return write_task_input_artifacts(
        "ridge",
        features_path,
        oof_path,
        output_path,
        force=force,
    )


def main() -> None:
    args = parse_args()
    output = (
        Path(args.output_npz)
        if args.output_npz
        else _REPO_ROOT / "outputs" / "cache" / f"v3_purified_p0_{args.task}_input.npz"
    )
    paths = write_task_input_artifacts(
        args.task,
        args.features_npy,
        args.oof_npz,
        output,
        force=args.force,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
