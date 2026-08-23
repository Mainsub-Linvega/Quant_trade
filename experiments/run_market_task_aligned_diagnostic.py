"""Run a training-only diagnostic for the completed P4 Market200 candidate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.v3_market_task_aligned_diagnostic import (
    attribute_peak_delta,
    market_block_stability,
    market_overlap_summary,
    write_diagnostic_bundle,
)

P4_FROZEN = {
    "n_folds": 5,
    "train_window": 78_960,
    "embargo": 6,
    "sample_modulo": 5,
    "sampling": "phase_balanced",
    "market_lambda": 0.7,
    "blend_weight": 1.17,
    "prediction_scale": 1.16,
    "prediction_clip": 0.5,
    "history_window": 5,
    "n_seeds": 1,
    "num_iteration": 160,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4-label", required=True)
    parser.add_argument("--label", default="v3_market_task_aligned_diagnostic")
    parser.add_argument("--data-root", default=str(_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_ROOT / "outputs" / "experiments"))
    parser.add_argument("--cache-dir", default=str(_ROOT / "outputs" / "cache"))
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced")
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--top-count", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _time_range_ids(time_ids: np.ndarray, time_range: list[int]) -> np.ndarray:
    start, stop = (int(value) for value in time_range)
    result = time_ids[(time_ids >= start) & (time_ids <= stop)]
    if len(result) == 0:
        raise ValueError(f"no sampled time IDs in range {time_range}")
    return result


def _row_slice(time_ids: np.ndarray, selected_ids: np.ndarray) -> slice:
    """Map a contiguous time-id interval to its sorted row slice."""
    ids = np.asarray(time_ids, dtype=np.int64)
    selected = np.asarray(selected_ids, dtype=np.int64)
    if ids.ndim != 1 or selected.ndim != 1 or len(selected) == 0:
        raise ValueError("time IDs and selected IDs must be non-empty vectors")
    left = int(np.searchsorted(ids, selected[0], side="left"))
    right = int(np.searchsorted(ids, selected[-1], side="right"))
    if left == right or ids[left] != selected[0] or ids[right - 1] != selected[-1]:
        raise ValueError("selected time IDs are not a contiguous row interval")
    return slice(left, right)


def _stability_rows(stability: dict[str, np.ndarray], indices: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in indices:
        rows.append({
            "feature": int(feature),
            "full_rank": int(stability["full_rank"][feature]),
            "full_abs_correlation": float(abs(stability["full_correlation"][feature])),
            "block_ranks": [int(value) for value in stability["block_ranks"][:, feature]],
            "block_abs_correlations": [float(abs(value)) for value in stability["block_correlation"][:, feature]],
            "rank_std": float(stability["rank_std"][feature]),
            "sign_consistency": float(stability["sign_consistency"][feature]),
            "top_count_frequency": int(stability["top_count_frequency"][feature]),
        })
    return rows


def _cause(attribution: dict[str, object]) -> str:
    kind = str(attribution["primary_cause"])
    return {
        "alignment_loss": "candidate loses target alignment despite non-increasing energy",
        "energy_inflation": "candidate increases prediction energy faster than alignment",
        "mixed": "alignment and energy changes pull in opposite directions",
        "improved": "candidate improves Peak on this fold",
    }[kind]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    p4_path = output_dir / f"{args.p4_label}.json"
    output_path = output_dir / f"{args.label}.json"
    if output_path.exists() and not args.force:
        raise SystemExit(f"output exists; use --force: {output_path}")
    p4 = json.loads(p4_path.read_text(encoding="utf-8"))
    if p4.get("status") != "completed" or len(p4.get("folds", [])) != 5:
        raise ValueError("requires the completed corrected P4 five-fold screen")
    for name, expected in P4_FROZEN.items():
        if p4["protocol"].get(name) != expected:
            raise ValueError(f"P4 protocol drift: {name}")
    legacy_path = str(_ROOT / "strategies" / "v1_ridge")
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)

    from experiments.v3_task_aligned_reselection import (
        load_p4_metadata,
        load_p4_rows,
        reuse_p4_feature_memmap,
    )
    from src.io import FEATURE_COLUMNS
    from train import robust_transform_fit

    feature_path = cache_dir / f".{args.p4_label}_features.npy"
    if feature_path.exists():
        cached = np.load(feature_path, mmap_mode="r")
        features = reuse_p4_feature_memmap(feature_path, cached.shape[0], len(FEATURE_COLUMNS))
        metadata = load_p4_metadata(Path(args.data_root), args.sample_modulo, args.sampling, cached.shape[0])
        del cached
    else:
        print("rebuilding diagnostic feature cache", flush=True)
        metadata = load_p4_rows(
            Path(args.data_root), args.sample_modulo, args.sampling, feature_path,
        )
        features = metadata["features"]
    time_ids = np.asarray(metadata["time_id"], dtype=np.int64)
    unique_time_ids = np.unique(time_ids)
    started = time.perf_counter()
    fold_reports: list[dict[str, object]] = []
    for record in p4["folds"]:
        fold = int(record["fold"])
        train_ids = _time_range_ids(unique_time_ids, list(record["train_time_range"]))
        train_rows = _row_slice(time_ids, train_ids)
        transformed = np.array(features[train_rows], dtype=np.float32, copy=True)
        transformed, _ = robust_transform_fit(transformed)
        stability = market_block_stability(
            transformed, metadata["target"][train_rows], time_ids[train_rows],
            n_blocks=args.n_blocks, top_count=args.top_count,
        )
        baseline = record["arms"]["baseline_corr"]
        candidate = record["arms"]["market_task_aligned"]
        overlap = market_overlap_summary(np.asarray(baseline["market"]), np.asarray(candidate["market"]))
        attribution = attribute_peak_delta(
            baseline, candidate, scale=float(p4["protocol"]["prediction_scale"]),
        )
        added = overlap["added"]
        removed = overlap["removed"]
        fold_reports.append({
            "fold": fold,
            "train_time_range": record["train_time_range"],
            "train_rows": int(len(transformed)),
            "overlap": overlap,
            "attribution": {**attribution, "conclusion": _cause(attribution)},
            "added_stability": _stability_rows(stability, added),
            "removed_stability": _stability_rows(stability, removed),
            "candidate_stability_summary": {
                "mean_rank_std": float(np.mean(stability["rank_std"][candidate["market"]])),
                "mean_sign_consistency": float(np.mean(stability["sign_consistency"][candidate["market"]])),
                "mean_top_count_frequency": float(np.mean(stability["top_count_frequency"][candidate["market"]])),
            },
            "all_finite": bool(np.all(np.isfinite(transformed))),
        })
        print(f"fold {fold}: overlap={overlap['overlap_count']}, cause={attribution['primary_cause']}", flush=True)
        del transformed, stability
        gc.collect()

    negative = [row for row in fold_reports if row["attribution"]["delta_peak"] < 0.0]
    payload = {
        "experiment": "market_task_aligned_diagnostic",
        "status": "completed",
        "p4_reference": str(p4_path),
        "protocol": p4["protocol"],
        "folds": fold_reports,
        "negative_fold_conclusions": {str(row["fold"]): row["attribution"]["conclusion"] for row in negative},
        "pooled_attribution": attribute_peak_delta(
            p4["pooled"]["baseline_corr"], p4["pooled"]["market_task_aligned"],
            scale=float(p4["protocol"]["prediction_scale"]),
        ),
        "confirmation_run": False,
        "production_modified": False,
        "submission_generated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_diagnostic_bundle(payload, output_dir, args.label)


if __name__ == "__main__":
    main()
