"""P0 diagnostic for purified residual interactions.

This command writes experiment evidence only. It cannot create model candidates
or leaderboard submissions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.v3_adaptive_selection_manifest import chronological_inner_splits
from experiments.v3_purified_interactions import (
    default_purified_protocol,
    empirical_null_threshold,
    interaction_stability_gate,
    make_split_task_nulls,
    score_pair_split,
    validate_purified_protocol,
)


DEFAULT_PROTOCOL = (
    _REPO_ROOT
    / "outputs"
    / "experiments"
    / "v3_purified_interaction_protocol_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz")
    parser.add_argument("--task", choices=["ridge", "xs", "market"], default="ridge")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--pair-manifest")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "outputs" / "experiments"),
    )
    parser.add_argument("--label", default="v3_purified_p0_diagnostic")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--write-default-protocol", metavar="PATH")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(write_candidate=False)
    return parser.parse_args()


def deterministic_pairs(
    n_features: int,
    *,
    max_pairs: int,
) -> list[tuple[int, int]]:
    """Return a bounded lexical prefix of all distinct feature pairs."""
    if n_features < 2 or max_pairs <= 0:
        raise ValueError("pair dimensions and max_pairs must be positive")
    pairs: list[tuple[int, int]] = []
    for pair in combinations(range(n_features), 2):
        pairs.append(pair)
        if len(pairs) == max_pairs:
            break
    return pairs


def validate_pair_manifest(
    pairs: object,
    *,
    n_features: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    """Validate an ordered, result-independent list of source feature pairs."""
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair manifest must contain a nonempty list")
    if len(pairs) > max_pairs:
        raise ValueError("pair manifest exceeds the requested pair budget")
    validated: list[tuple[int, int]] = []
    for item in pairs:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
        ):
            raise ValueError("pair manifest entries must be integer pairs")
        left, right = int(item[0]), int(item[1])
        if left < 0 or right >= n_features or left >= right:
            raise ValueError("pair manifest entries must be ordered valid feature indices")
        validated.append((left, right))
    if len(set(validated)) != len(validated):
        raise ValueError("pair manifest contains duplicate pairs")
    return validated


def load_pair_manifest(
    path: str | Path,
    *,
    n_features: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    """Load a JSON pair manifest without permitting result-dependent settings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else payload
    return validate_pair_manifest(
        pairs,
        n_features=n_features,
        max_pairs=max_pairs,
    )


def validate_diagnostic_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and normalize the explicit P0 NPZ contract."""
    required = {"features", "residual", "weight", "time_id"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"diagnostic input is missing arrays: {missing}")
    features = np.asarray(arrays["features"], dtype=np.float32)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    weight = np.asarray(arrays["weight"], dtype=np.float64)
    time_id = np.asarray(arrays["time_id"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 323:
        raise ValueError("diagnostic features must have exactly 323 columns")
    rows = len(features)
    if (
        residual.shape != (rows,)
        or weight.shape != (rows,)
        or time_id.shape != (rows,)
    ):
        raise ValueError("diagnostic arrays must be row-aligned")
    if (
        not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(weight))
        or np.any(weight <= 0.0)
    ):
        raise ValueError("residual and positive weight must be finite")
    if np.any(np.diff(time_id) < 0):
        raise ValueError("time_id must be nondecreasing")
    feature_indices = np.asarray(
        arrays.get("feature_indices", np.arange(323)), dtype=np.int64
    )
    if (
        feature_indices.shape != (323,)
        or len(np.unique(feature_indices)) != 323
        or np.any(feature_indices < 0)
    ):
        raise ValueError("feature_indices must contain 323 unique nonnegative indices")
    return {
        "features": features,
        "residual": residual,
        "weight": weight,
        "time_id": time_id,
        "feature_indices": feature_indices,
    }


def _json_score(score: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in score.items()
        if key != "surface"
    }


def _markdown_report(payload: Mapping[str, object]) -> str:
    rows = [
        "# Purified Interaction P0 Diagnostic",
        "",
        f"- status: `{payload.get('status', 'unknown')}`",
        f"- task: `{payload.get('task', 'unknown')}`",
        f"- scanned pairs: `{payload.get('scanned_pairs', 0)}`",
        f"- null threshold: `{float(payload.get('null_threshold', 0.0)):.8f}`",
        f"- accepted pairs: `{payload.get('accepted_pairs', 0)}`",
        "",
        "| Rank | Pair | Median gain | Drop-best | Passed |",
        "|---:|---|---:|---:|:---:|",
    ]
    for rank, item in enumerate(payload.get("pairs", []), start=1):
        gate = item["gate"]
        rows.append(
            f"| {rank} | {item['pair']} | {gate['median_gain']:.8f} | "
            f"{gate['drop_best_mean_gain']:.8f} | "
            f"{'yes' if gate['passed'] else 'no'} |"
        )
    rows.extend([
        "",
        "This P0 report does not create or promote a model candidate.",
        "",
    ])
    return "\n".join(rows)


def write_diagnostic_report(
    output_dir: str | Path,
    label: str,
    *,
    payload: Mapping[str, object],
    force: bool = False,
) -> dict[str, Path]:
    """Atomically write JSON and Markdown P0 evidence only."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": directory / f"{label}.json",
        "markdown": directory / f"{label}.md",
    }
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError("diagnostic report exists; use force to overwrite")
    content = {
        "json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
        "markdown": _markdown_report(payload),
    }
    for name, path in paths.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content[name], encoding="utf-8")
        temporary.replace(path)
    return paths


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return validate_diagnostic_arrays({name: loaded[name] for name in loaded.files})


def _synthetic_arrays(seed: int = 17) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    time_count = 400
    assets = 3
    rows = time_count * assets
    features = rng.normal(size=(rows, 323)).astype(np.float32)
    interaction = np.sign(features[:, 0]) * np.sign(features[:, 1])
    residual = interaction + rng.normal(scale=0.05, size=rows)
    return validate_diagnostic_arrays({
        "features": features,
        "residual": residual,
        # Dense support makes this a mathematical smoke for purification and
        # null gating, not a test of sparse-cell rejection.
        "weight": np.full(rows, 100.0),
        "time_id": np.repeat(np.arange(time_count), assets),
        "feature_indices": np.arange(323),
    })


def run_diagnostic(
    arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    protocol: Mapping[str, object],
    max_pairs: int,
    pair_manifest: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Run bounded chronological real/null scoring for one task."""
    validate_purified_protocol(protocol)
    data = validate_diagnostic_arrays(arrays)
    if task not in protocol["tasks"]:
        raise ValueError(f"unknown diagnostic task: {task}")
    _, time_counts = np.unique(data["time_id"], return_counts=True)
    if task == "market" and np.any(time_counts != 1):
        raise ValueError("market diagnostic requires exactly one row per time_id")
    if task in {"ridge", "xs"} and np.any(time_counts < 2):
        raise ValueError(
            f"{task} diagnostic requires multiple assets per time_id"
        )
    budget = int(protocol["budgets"]["max_pairs"])
    if max_pairs <= 0 or max_pairs > budget:
        raise ValueError(f"max_pairs must be between 1 and frozen budget {budget}")
    task_settings = protocol["tasks"][task]
    pairs = (
        deterministic_pairs(323, max_pairs=max_pairs)
        if pair_manifest is None
        else validate_pair_manifest(
            pair_manifest,
            n_features=323,
            max_pairs=max_pairs,
        )
    )
    splits = chronological_inner_splits(
        data["time_id"], n_blocks=int(protocol["inner_blocks"])
    )
    split_payloads = [
        (
            train_rows,
            valid_rows,
            make_split_task_nulls(
                task,
                data["residual"][train_rows],
                data["residual"][valid_rows],
                data["time_id"][train_rows],
                data["time_id"][valid_rows],
                seeds=[int(seed) for seed in protocol["null"]["seeds"]],
                embargo=int(protocol["outer"]["embargo"]),
            ),
        )
        for train_rows, valid_rows in splits
    ]

    scored: list[dict[str, object]] = []
    all_null_gains: list[float] = []
    for pair in pairs:
        pair_features = np.ascontiguousarray(data["features"][:, pair])
        local_pair = (0, 1)
        block_scores: list[dict[str, object]] = []
        for train_rows, valid_rows, split_nulls in split_payloads:
            train_pair_features = pair_features[train_rows]
            valid_pair_features = pair_features[valid_rows]
            score = score_pair_split(
                train_pair_features,
                valid_pair_features,
                data["residual"][train_rows],
                data["residual"][valid_rows],
                data["weight"][train_rows],
                data["weight"][valid_rows],
                pair=local_pair,
                bins=int(task_settings["bins"]),
                min_cell_weight=float(task_settings["min_cell_weight"]),
                max_surface_cells=int(protocol["budgets"]["max_surface_cells"]),
            )
            score["pair"] = [int(pair[0]), int(pair[1])]
            block_scores.append(_json_score(score))
            for null_train, null_valid in split_nulls:
                null_score = score_pair_split(
                    train_pair_features,
                    valid_pair_features,
                    null_train,
                    null_valid,
                    data["weight"][train_rows],
                    data["weight"][valid_rows],
                    pair=local_pair,
                    bins=int(task_settings["bins"]),
                    min_cell_weight=float(task_settings["min_cell_weight"]),
                    max_surface_cells=int(protocol["budgets"]["max_surface_cells"]),
                )
                all_null_gains.append(float(null_score["gain"]))
        scored.append({"pair": list(pair), "blocks": block_scores})

    null_threshold = empirical_null_threshold(
        np.asarray(all_null_gains), float(protocol["null"]["quantile"])
    )
    for item in scored:
        item["gate"] = interaction_stability_gate(
            item["blocks"],
            null_threshold=null_threshold,
            minimum_positive_blocks=int(
                protocol["stability"]["minimum_positive_blocks"]
            ),
            minimum_coverage=float(protocol["stability"]["minimum_coverage"]),
            maximum_single_cell_gain_share=float(
                protocol["stability"]["maximum_single_cell_gain_share"]
            ),
        )
    scored.sort(
        key=lambda item: (
            -float(item["gate"]["median_gain"]),
            item["pair"],
        )
    )
    output_limit = int(protocol["budgets"]["max_output_candidates"])
    retained = scored[:output_limit]
    accepted = sum(bool(item["gate"]["passed"]) for item in retained)
    return {
        "experiment": "v3_purified_interaction_p0",
        "status": "passed_p0" if accepted else "failed_p0",
        "task": task,
        "pair_source": "lexical" if pair_manifest is None else "manifest",
        "scanned_pairs": len(pairs),
        "null_samples": len(all_null_gains),
        "null_threshold": float(null_threshold),
        "accepted_pairs": int(accepted),
        "protocol": protocol,
        "feature_indices": data["feature_indices"].tolist(),
        "pairs": retained,
        "candidate_generated": False,
        "submission_generated": False,
    }


def _write_protocol(path: str | Path) -> Path:
    protocol = default_purified_protocol()
    validate_purified_protocol(protocol)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def main() -> None:
    args = parse_args()
    if args.write_default_protocol:
        output = Path(args.write_default_protocol)
        if output.exists() and not args.force:
            raise SystemExit(f"{output} exists; use --force to overwrite")
        print(_write_protocol(output))
        return
    protocol_path = Path(args.protocol)
    if not protocol_path.exists():
        raise SystemExit(
            f"{protocol_path} does not exist; use --write-default-protocol first"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_purified_protocol(protocol)
    if args.synthetic_smoke:
        arrays = _synthetic_arrays()
    elif args.input_npz:
        arrays = _load_npz(args.input_npz)
    else:
        raise SystemExit("--input-npz or --synthetic-smoke is required")
    payload = run_diagnostic(
        arrays,
        task=args.task,
        protocol=protocol,
        max_pairs=args.max_pairs,
        pair_manifest=(
            load_pair_manifest(
                args.pair_manifest,
                n_features=323,
                max_pairs=args.max_pairs,
            )
            if args.pair_manifest
            else None
        ),
    )
    paths = write_diagnostic_report(
        args.output_dir,
        args.label,
        payload=payload,
        force=args.force,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
