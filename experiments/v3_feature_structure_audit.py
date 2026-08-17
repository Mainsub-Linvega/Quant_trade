"""Run the first-stage V3 feature structure audit on strict rolling training windows."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "strategies" / "v1_ridge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.lgbm_xs import load_rows
from experiments.v3_feature_structure import (
    build_task_views,
    feature_quality_by_blocks,
    stable_redundancy,
    weighted_component_gram,
)
from src.io import FEATURE_COLUMNS
from src.validation import rolling_time_folds
from train import robust_transform_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V3 feature structure inside rolling folds.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="v3_feature_structure_audit")
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument(
        "--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=78_960)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--cluster-threshold", type=float, default=0.15)
    parser.add_argument(
        "--redundancy-rows-per-block", type=int, default=100_000,
        help="Deterministic row cap for each block's Pearson/Spearman matrices.",
    )
    parser.add_argument("--components-npz", default=None)
    parser.add_argument("--smoke-folds", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def extract_dense_matrices(
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Move dense redundancy matrices into a stable-keyed NPZ payload."""
    summary = copy.deepcopy(report)
    matrices: dict[str, np.ndarray] = {}
    for fold in summary.get("folds", []):
        fold_index = fold.get("fold")
        for task_name, task in fold.get("tasks", {}).items():
            redundancy = task.get("redundancy")
            if not isinstance(redundancy, dict):
                continue
            for name, value in list(redundancy.items()):
                array = np.asarray(value)
                if array.ndim != 2 or min(array.shape) <= 16:
                    continue
                key = f"fold_{fold_index}.{task_name}.redundancy.{name}"
                matrices[key] = array.astype(np.float32, copy=False)
                redundancy[name] = {
                    "npz_key": key,
                    "shape": [int(size) for size in array.shape],
                }
    return summary, matrices


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# V3 Feature Structure Audit", ""]
    config = report.get("config", {})
    lines.extend(
        [
            "## Configuration",
            "",
            f"- blocks: `{config.get('n_blocks', 'n/a')}`",
            f"- sample modulo: `{config.get('sample_modulo', 'n/a')}`",
            f"- train window: `{config.get('effective_train_window', config.get('train_window', 'n/a'))}`",
            f"- embargo: `{config.get('embargo', 'n/a')}`",
            "",
            "## Fold Summary",
            "",
        ]
    )
    for fold in report.get("folds", []):
        lines.append(f"### Fold {fold.get('fold', 'n/a')}")
        lines.append("")
        for task_name, task in fold.get("tasks", {}).items():
            status = task.get("status", "ok")
            if status != "ok":
                lines.append(f"- `{task_name}`: {status}")
                continue
            lines.append(
                f"- `{task_name}`: {task.get('n_rows', 'n/a')} rows, "
                f"{task.get('n_features', 'n/a')} features, "
                f"{task.get('cluster_count', 'n/a')} stable redundancy clusters"
            )
        lines.append("")

    gram = report.get("gram", {})
    lines.extend(["## Fusion Coupling", ""])
    if gram.get("status") == "ok":
        lines.append(f"- weighted `<u,v>`: `{_format_number(gram.get('uv_coupling'))}`")
        lines.append(
            "- interpretation: non-zero coupling means market lambda and blend weight must be "
            "recalibrated jointly after feature changes."
        )
    else:
        lines.append(f"- status: `{gram.get('status', 'not_available')}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "History transformations are intentionally not audited in this phase. No production "
            "model, fusion metadata, or leaderboard submission is changed by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_bundle(
    report: dict[str, Any], output_dir: Path, label: str
) -> dict[str, Path]:
    """Write a JSON index, Markdown summary and compressed dense-matrix artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    markdown_path = output_dir / f"{label}.md"
    npz_path = output_dir / f"{label}_matrices.npz"

    summary, matrices = extract_dense_matrices(report)
    summary["matrix_artifact"] = npz_path.name
    safe_summary = _json_safe(summary)
    np.savez_compressed(npz_path, **matrices)
    json_path.write_text(
        json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(safe_summary), encoding="utf-8")
    return {
        "json": json_path,
        "markdown": markdown_path,
        "npz": npz_path,
    }


def _task_report(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
    n_blocks: int,
    cluster_threshold: float,
    redundancy_rows_per_block: int | None,
) -> dict[str, Any]:
    quality = feature_quality_by_blocks(
        features, target, weight, time_ids, n_blocks=n_blocks
    )
    redundancy = stable_redundancy(
        features,
        time_ids,
        n_blocks=n_blocks,
        threshold=cluster_threshold,
        max_rows_per_block=redundancy_rows_per_block,
    )
    return {
        "status": "ok",
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
        "cluster_count": int(len(np.unique(redundancy.labels))),
        "quality": quality,
        "redundancy": {
            "pearson": redundancy.pearson,
            "spearman": redundancy.spearman,
            "stability": redundancy.stability,
            "distance": redundancy.distance,
            "labels": redundancy.labels,
            "sampled_rows_per_block": redundancy.sampled_rows_per_block,
        },
    }


def _component_report(path: str | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_available"}
    required = {"target", "weight", "m_ridge", "m_lgbm", "e_ridge", "e_lgbm"}
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"components NPZ is missing arrays: {missing}")
        target = payload["target"]
        weight = payload["weight"]
        m_ridge = payload["m_ridge"]
        m_lgbm = payload["m_lgbm"]
        e_ridge = payload["e_ridge"]
        e_lgbm = payload["e_lgbm"]
    result = weighted_component_gram(
        target,
        weight,
        m_ridge + e_ridge,
        m_lgbm - m_ridge,
        e_lgbm - e_ridge,
    )
    return {"status": "ok", **result, "source": str(Path(path).resolve())}


def _row_mask(time_ids: np.ndarray, selected_time_ids: np.ndarray) -> np.ndarray:
    left = int(np.searchsorted(time_ids, selected_time_ids[0], side="left"))
    right = int(np.searchsorted(time_ids, selected_time_ids[-1], side="right"))
    mask = np.zeros(len(time_ids), dtype=bool)
    mask[left:right] = True
    return mask


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_modulo <= 0:
        raise ValueError("sample_modulo must be positive")
    if args.n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if args.redundancy_rows_per_block is not None and args.redundancy_rows_per_block < 2:
        raise ValueError("redundancy_rows_per_block must be at least two")
    if args.smoke_folds is not None and args.smoke_folds <= 0:
        raise ValueError("smoke_folds must be positive")

    started = time.perf_counter()
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features = data["features"]
    target = data["target"]
    weight = data["weight"]
    time_ids = data["time_id"]
    if np.any(np.diff(time_ids) < 0):
        raise ValueError("loaded time_ids must be sorted")
    unique_time_ids = time_ids[np.r_[True, time_ids[1:] != time_ids[:-1]]]

    effective_train_window = args.train_window
    if effective_train_window + args.embargo >= len(unique_time_ids):
        if args.smoke_folds is None:
            raise ValueError("train_window and embargo exceed sampled time coverage")
        effective_train_window = max(args.n_blocks, int(len(unique_time_ids) * 4 / 9))
    folds = rolling_time_folds(
        unique_time_ids,
        args.n_folds,
        effective_train_window,
        args.embargo,
    )
    if args.smoke_folds is not None:
        folds = folds[: args.smoke_folds]

    report: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "data_root": str(Path(args.data_root).resolve()),
            "sample_modulo": args.sample_modulo,
            "sampling": args.sampling,
            "n_folds": args.n_folds,
            "audited_folds": len(folds),
            "train_window": args.train_window,
            "effective_train_window": effective_train_window,
            "embargo": args.embargo,
            "n_blocks": args.n_blocks,
            "cluster_threshold": args.cluster_threshold,
            "redundancy_rows_per_block": args.redundancy_rows_per_block,
        },
        "data": {
            "rows": int(len(time_ids)),
            "time_ids": int(len(unique_time_ids)),
            "features": int(features.shape[1]),
            "feature_names": list(FEATURE_COLUMNS),
        },
        "folds": [],
        "gram": _component_report(args.components_npz),
    }

    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        train_mask = _row_mask(time_ids, train_ids)
        transformed, _ = robust_transform_fit(features[train_mask].copy())
        train_target = target[train_mask]
        train_weight = weight[train_mask]
        train_time_ids = time_ids[train_mask]
        views = build_task_views(
            transformed, train_target, train_weight, train_time_ids
        )
        tasks = {
            "ridge": _task_report(
                views.raw_features,
                views.full_target,
                train_weight,
                train_time_ids,
                args.n_blocks,
                args.cluster_threshold,
                args.redundancy_rows_per_block,
            ),
            "xs": _task_report(
                views.cross_features,
                views.cross_target,
                np.ones(len(views.cross_target), dtype=np.float64),
                train_time_ids,
                args.n_blocks,
                args.cluster_threshold,
                args.redundancy_rows_per_block,
            ),
            "market": _task_report(
                views.market_features,
                views.market_target,
                np.ones(len(views.market_target), dtype=np.float64),
                views.unique_time_ids,
                args.n_blocks,
                args.cluster_threshold,
                args.redundancy_rows_per_block,
            ),
            "history": {
                "status": "not_run",
                "reason": "causal history transforms are scheduled for a later implementation plan",
            },
        }
        report["folds"].append(
            {
                "fold": fold_index,
                "train_time_start": int(train_ids[0]),
                "train_time_end": int(train_ids[-1]),
                "valid_time_start": int(valid_ids[0]),
                "valid_time_end": int(valid_ids[-1]),
                "train_rows": int(train_mask.sum()),
                "elapsed_seconds": time.perf_counter() - fold_started,
                "tasks": tasks,
            }
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"{args.label}.json"
    markdown_path = output_dir / f"{args.label}.md"
    npz_path = output_dir / f"{args.label}_matrices.npz"
    if not args.force and any(path.exists() for path in (json_path, markdown_path, npz_path)):
        raise SystemExit(
            f"report exists: {json_path} / {markdown_path} / {npz_path}; pass --force"
        )

    paths = write_report_bundle(run_audit(args), output_dir, args.label)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
