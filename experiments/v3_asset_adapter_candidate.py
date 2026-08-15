"""Build a local v3 candidate with OOF-fitted per-asset cross-section scales.

The script copies the current production artifacts into outputs/candidates and patches only
``hybrid_meta.json``. It never touches the production model and never writes a submission CSV.
The calibration is fit on strict OOF predictions; with five folds, ``--asset-shrink 500`` keeps
the same per-fold shrink strength as the accepted screening value 100.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from v3_residual_adapters import fit_asset_slopes

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(_REPO_ROOT / "outputs" / "cache" /
                                         "v3_production_oof_phasebal_prodwindow_exact.npz"))
    p.add_argument("--source", default=str(_REPO_ROOT / "strategies" / "v3_hybrid" / "model"))
    p.add_argument("--destination", default=str(_REPO_ROOT / "outputs" / "candidates" /
                                                 "v3_asset_cross_shrink500"))
    p.add_argument("--asset-shrink", type=float, default=500.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    destination = Path(args.destination)
    if destination.exists():
        raise SystemExit(f"candidate already exists: {destination}; choose a new destination")
    meta_path = source / "hybrid_meta.json"
    baseline_path = source / "baseline_model.json"
    if not meta_path.is_file() or not baseline_path.is_file():
        raise SystemExit(f"incomplete source model: {source}")

    with np.load(args.oof, allow_pickle=False) as d:
        valid = d["fold"] >= 0
        slopes = fit_asset_slopes(
            d["time_id"][valid], d["target"][valid], np.maximum(d["weight"][valid], 0.0),
            d["asset_id"][valid], d["e_lgbm"][valid], args.asset_shrink)
        n_folds = int(len(np.unique(d["fold"][valid])))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_names = list(meta["lgbm_model_files"]) + list(meta.get("market_model_files") or [])
    required = [baseline_path, meta_path, *(source / name for name in model_names)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"source model files missing: {missing}")

    destination.mkdir(parents=True)
    shutil.copy2(baseline_path, destination / baseline_path.name)
    for name in model_names:
        shutil.copy2(source / name, destination / name)
    meta["market_num_iteration"] = int(meta.get("market_num_iteration", meta["num_iteration"]))
    meta["asset_cross_scales"] = [float(v) for v in slopes]
    meta["asset_cross_scale_note"] = (
        "OOF-only per-asset scaling of e_lgbm, followed by per-time_id zero-mean projection. "
        f"Fit from {n_folds} strict folds with shrink={args.asset_shrink}; "
        "market component and production artifacts are unchanged."
    )
    meta["asset_cross_scale_oof"] = str(Path(args.oof))
    (destination / "hybrid_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "source": str(source), "destination": str(destination), "oof": str(args.oof),
        "asset_shrink": float(args.asset_shrink), "n_folds": n_folds,
        "cross_num_iteration": int(meta["num_iteration"]),
        "market_num_iteration": int(meta["market_num_iteration"]),
        "asset_cross_scales": [float(v) for v in slopes],
        "status": "local candidate only; no submission generated",
    }
    (destination / "candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
