from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from train import (
    FEATURE_COLUMNS,
    fit_model,
    predict_array,
    train_files,
    weighted_zero_mean_r2,
)


READ_COLUMNS = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare chronological training windows over three folds.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "experiments"))
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--validation-partitions", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--train-sample-modulo", type=int, default=10)
    parser.add_argument("--valid-sample-modulo", type=int, default=5)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--reference-window", type=int, default=3)
    parser.add_argument("--final-sample-modulo", type=int, default=5)
    parser.add_argument("--final-ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=0.5)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    return parser.parse_args()


def load_partition_sample(path: Path, sample_modulo: int) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "features": [],
        "target": [],
        "weight": [],
        "time_id": [],
        "asset_id": [],
    }
    for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=READ_COLUMNS):
        frame = batch.to_pandas()
        mask = frame["time_id"].to_numpy(copy=False) % sample_modulo == 0
        if not mask.any():
            continue
        parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
        parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float32, copy=True))
        parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float32, copy=True))
        parts["time_id"].append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
        parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int8, copy=True))
    return {name: np.concatenate(values) for name, values in parts.items()}


def concatenate(cached: dict[int, dict[str, np.ndarray]], indices: range | list[int]) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate([cached[index][name] for index in indices])
        for name in ["features", "target", "weight", "time_id", "asset_id"]
    }


def asset_scores(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray, asset_id: np.ndarray
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for asset in np.unique(asset_id):
        mask = asset_id == asset
        scores[str(int(asset))] = weighted_zero_mean_r2(target[mask], prediction[mask], weight[mask])
    return scores


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = train_files(data_root)

    windows = sorted(set(args.windows))
    validation_indices = sorted(set(args.validation_partitions))
    if min(windows) <= 0:
        raise ValueError("windows must be positive")
    if min(validation_indices) - max(windows) < 0 or max(validation_indices) >= len(files):
        raise ValueError("fold configuration does not fit available partitions")

    required_train_indices = sorted(
        {index for valid_index in validation_indices for index in range(valid_index - max(windows), valid_index)}
    )
    print("loading reusable chronological samples", flush=True)
    train_cache: dict[int, dict[str, np.ndarray]] = {}
    for index in required_train_indices:
        train_cache[index] = load_partition_sample(files[index], args.train_sample_modulo)
        print(
            f"train cache p{index:03d}: {len(train_cache[index]['target']):,} rows",
            flush=True,
        )
    valid_cache: dict[int, dict[str, np.ndarray]] = {}
    for index in validation_indices:
        valid_cache[index] = load_partition_sample(files[index], args.valid_sample_modulo)
        print(
            f"valid cache p{index:03d}: {len(valid_cache[index]['target']):,} rows",
            flush=True,
        )

    experiments: list[dict[str, object]] = []
    for window in windows:
        fold_results: list[dict[str, object]] = []
        for valid_index in validation_indices:
            started = time.perf_counter()
            train_indices = list(range(valid_index - window, valid_index))
            train = concatenate(train_cache, train_indices)
            valid = valid_cache[valid_index]
            # Ridge minimises a sum of losses. Preserve the same regularisation
            # per sampled row as the v1 final fit while changing sample density/window.
            ridge_alpha = (
                args.final_ridge_alpha
                * args.final_sample_modulo
                / args.train_sample_modulo
                * window
                / args.reference_window
            )
            artifact, selected = fit_model(
                train["features"],
                train["target"],
                train["weight"],
                train["time_id"],
                args.feature_count,
                ridge_alpha,
            )
            prediction = predict_array(
                artifact,
                valid["features"],
                valid["time_id"],
                selected,
                args.prediction_scale,
                args.prediction_clip,
            )
            score = weighted_zero_mean_r2(valid["target"], prediction, valid["weight"])
            fold = {
                "valid_partition": int(valid_index),
                "train_partitions": train_indices,
                "train_rows": int(len(train["target"])),
                "valid_rows": int(len(valid["target"])),
                "ridge_alpha": float(ridge_alpha),
                "score": float(score),
                "prediction_mean": float(np.mean(prediction)),
                "prediction_std": float(np.std(prediction)),
                "prediction_min": float(np.min(prediction)),
                "prediction_max": float(np.max(prediction)),
                "asset_scores": asset_scores(
                    valid["target"], prediction, valid["weight"], valid["asset_id"]
                ),
                "selected_features": [FEATURE_COLUMNS[index] for index in selected],
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            fold_results.append(fold)
            print(
                f"window={window} fold=p{valid_index:03d} score={score:.8f} "
                f"rows={len(train['target']):,}",
                flush=True,
            )
            del train, artifact, selected, prediction
            gc.collect()

        scores = np.asarray([float(fold["score"]) for fold in fold_results])
        experiments.append(
            {
                "window": int(window),
                "mean_score": float(scores.mean()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
                "std_score": float(scores.std()),
                "positive_folds": int(np.sum(scores > 0)),
                "folds": fold_results,
            }
        )

    ranked = sorted(
        experiments,
        key=lambda item: (int(item["positive_folds"]), float(item["mean_score"]), float(item["min_score"])),
        reverse=True,
    )
    payload = {
        "metric": "weighted_zero_mean_r2",
        "public_baseline_score": 0.00119088,
        "configuration": {
            "windows": windows,
            "validation_partitions": validation_indices,
            "train_sample_modulo": args.train_sample_modulo,
            "valid_sample_modulo": args.valid_sample_modulo,
            "feature_count": args.feature_count,
            "prediction_scale": args.prediction_scale,
            "prediction_clip": args.prediction_clip,
        },
        "ranking_rule": "positive_folds, then mean_score, then min_score",
        "recommended_window": int(ranked[0]["window"]),
        "experiments": experiments,
    }
    (output_dir / "walk_forward_windows.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "walk_forward_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["window", *[f"fold_p{index:03d}" for index in validation_indices], "mean", "min", "std", "positive_folds"]
        )
        for experiment in experiments:
            writer.writerow(
                [
                    experiment["window"],
                    *[fold["score"] for fold in experiment["folds"]],
                    experiment["mean_score"],
                    experiment["min_score"],
                    experiment["std_score"],
                    experiment["positive_folds"],
                ]
            )

    markdown = [
        "# Walk-forward training-window experiment",
        "",
        "| Window | " + " | ".join(f"Valid p{index:03d}" for index in validation_indices) + " | Mean | Min | Positive folds |",
        "|---:|" + "---:|" * (len(validation_indices) + 4),
    ]
    for experiment in experiments:
        markdown.append(
            f"| {experiment['window']} | "
            + " | ".join(f"{float(fold['score']):.8f}" for fold in experiment["folds"])
            + f" | {float(experiment['mean_score']):.8f} | {float(experiment['min_score']):.8f} | {experiment['positive_folds']}/{len(validation_indices)} |"
        )
    markdown.extend(
        [
            "",
            f"Recommended window: **{payload['recommended_window']} partitions**.",
            "",
            "Ranking prioritises the number of positive folds before mean and worst-fold score.",
        ]
    )
    (output_dir / "walk_forward_windows.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"recommended_window": payload["recommended_window"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
