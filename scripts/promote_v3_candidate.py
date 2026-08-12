"""Stage, validate, and optionally atomically promote a v3_hybrid model candidate.

The default action is safe: build a separate staging directory and write an auditable manifest. Production
is changed only with the explicit pair ``--activate --allow-production-overwrite``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "strategies" / "v3_hybrid"
PRODUCTION = STRATEGY / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage/validate/promote a v3_hybrid candidate.")
    parser.add_argument("--candidate", default=str(ROOT / "outputs" / "candidates" /
                                                    "v3_hybrid_r480_phasebal_hist"))
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--scale", type=float, default=1.16)
    parser.add_argument("--n-seeds", type=int, choices=[2, 3], default=3)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--allow-production-overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_files(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.iterdir() if path.is_file() and path.name != "promotion_manifest.json")


def validate_meta(meta: dict[str, Any], *, scale: float, n_seeds: int) -> None:
    checks = {
        "num_iteration_is_480": meta.get("num_iteration") == 480,
        "history_window_is_5": meta.get("history_window") == 5,
        "history_positions_is_40": len(meta.get("history_positions") or []) == 40,
        "prediction_scale_matches": abs(float(meta.get("prediction_scale", float("nan"))) - scale) < 1e-12,
        "model_file_count_matches": len(meta.get("lgbm_model_files") or []) == n_seeds,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"candidate metadata failed: {', '.join(failed)}")


def stage_candidate(candidate: Path, destination: Path, *, scale: float, n_seeds: int,
                    force: bool = False) -> dict[str, Any]:
    required = [candidate / "baseline_model.json", candidate / "hybrid_meta.json"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"candidate is incomplete: {candidate}")
    if destination.exists():
        if not force:
            raise FileExistsError(f"staging exists: {destination}; pass --force")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    original_meta = json.loads((candidate / "hybrid_meta.json").read_text(encoding="utf-8"))
    selected_models = list(original_meta["lgbm_model_files"])[:n_seeds]
    for name in selected_models:
        if not (candidate / name).is_file():
            raise FileNotFoundError(candidate / name)
    shutil.copy2(candidate / "baseline_model.json", destination / "baseline_model.json")
    for name in selected_models:
        shutil.copy2(candidate / name, destination / name)
    meta = dict(original_meta)
    meta["prediction_scale"] = float(scale)
    meta["lgbm_model_files"] = selected_models
    meta["promotion_note"] = ("Staged by scripts/promote_v3_candidate.py; source artifacts are unchanged. "
                              f"scale={scale}, seeds={n_seeds}")
    validate_meta(meta, scale=scale, n_seeds=n_seeds)
    (destination / "hybrid_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(candidate.resolve()), "destination": str(destination.resolve()),
        "configuration": {"prediction_scale": scale, "n_seeds": n_seeds},
        "source_files": {path.name: sha256_file(path) for path in model_files(candidate)},
        "staged_files": {path.name: sha256_file(path) for path in model_files(destination)},
    }
    (destination / "promotion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_model_class():
    sys.path.insert(0, str(STRATEGY))
    for name in ("main", "features", "history", "lgbm_numpy"):
        sys.modules.pop(name, None)
    try:
        return importlib.import_module("main").Model
    finally:
        sys.path.remove(str(STRATEGY))


def validate_staging(model_dir: Path) -> dict[str, Any]:
    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    validate_meta(meta, scale=float(meta["prediction_scale"]), n_seeds=len(meta["lgbm_model_files"]))
    Model = load_model_class()
    backends: dict[str, Any] = {}
    for backend in ("numpy", "lightgbm"):
        model = Model(model_dir, backend=backend)
        rows = 15
        predictions = []
        for time_id in (100, 101):
            frame = pd.DataFrame({
                "row_id": np.arange(rows) + time_id * rows,
                "time_id": np.full(rows, time_id, dtype=np.int64),
                "asset_id": np.arange(rows, dtype=np.int64),
                **{column: np.zeros(rows, dtype=np.float32) for column in model.feature_columns},
            })
            predictions.append(np.asarray(model.predict(frame), dtype=np.float64))
        values = np.concatenate(predictions)
        if values.shape != (30,) or not np.all(np.isfinite(values)):
            raise AssertionError(f"{backend} smoke returned invalid predictions")
        backends[backend] = {"rows": int(len(values)), "max_abs": float(np.abs(values).max()),
                             "mean": float(values.mean())}
    if abs(backends["numpy"]["mean"] - backends["lightgbm"]["mean"]) > 1e-10:
        raise AssertionError("numpy/lightgbm staging smoke mismatch")
    result = {"passed": True, "backends": backends,
              "files": {path.name: sha256_file(path) for path in model_files(model_dir)}}
    manifest_path = model_dir / "promotion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation"] = result
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def activate_staging(staging: Path, production: Path, backup_root: Path) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"model_before_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    incoming = production.parent / f".{production.name}.incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    shutil.copytree(staging, incoming, ignore=shutil.ignore_patterns("promotion_manifest.json"))
    try:
        os.replace(production, backup)
        os.replace(incoming, production)
    except Exception:
        if not production.exists() and backup.exists():
            os.replace(backup, production)
        if incoming.exists():
            shutil.rmtree(incoming)
        raise
    return backup


def main() -> None:
    args = parse_args()
    if args.activate and not args.allow_production_overwrite:
        raise SystemExit("--activate requires --allow-production-overwrite")
    candidate = Path(args.candidate)
    stage_dir = Path(args.stage_dir) if args.stage_dir else (
        ROOT / "outputs" / "promotions" / f"v3_hybrid_s{args.scale:g}_{args.n_seeds}seed"
    )
    stage_candidate(candidate, stage_dir, scale=args.scale, n_seeds=args.n_seeds, force=args.force)
    validation = validate_staging(stage_dir)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"staged and validated: {stage_dir}")
    if args.activate:
        backup = activate_staging(stage_dir, PRODUCTION, ROOT / "outputs" / "promotions" / "backups")
        validate_staging(PRODUCTION)
        print(f"production activated; backup: {backup}")


if __name__ == "__main__":
    main()
