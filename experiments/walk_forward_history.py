from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from history_features import AssetHistory, build_history_design
from src.io import FEATURE_COLUMNS, train_files
from src.metric import weighted_zero_mean_r2
from train import robust_transform_fit, select_features
from walk_forward import asset_scores, concatenate, load_partition_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate causal per-asset history features.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--validation-partitions", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--train-sample-modulo", type=int, default=10)
    parser.add_argument("--valid-sample-modulo", type=int, default=5)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--history-feature-count", type=int, default=40)
    parser.add_argument("--final-ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--reference-window", type=int, default=3)
    parser.add_argument("--final-sample-modulo", type=int, default=5)
    parser.add_argument("--prediction-scale", type=float, default=0.5)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    return parser.parse_args()


def feature_pipeline(
    features: np.ndarray, target: np.ndarray, weight: np.ndarray, feature_count: int, history_feature_count: int
) -> tuple[dict[str, object], np.ndarray]:
    features, preprocessing = robust_transform_fit(features)
    selected = select_features(features, target, weight, feature_count)
    history_positions = select_features(features[:, selected], target, weight, history_feature_count)
    artifact: dict[str, object] = {
        "selected_features": [FEATURE_COLUMNS[index] for index in selected],
        "selected_indices": selected.tolist(),
        "lower": preprocessing["lower"][selected].tolist(),
        "upper": preprocessing["upper"][selected].tolist(),
        "center": preprocessing["center"][selected].tolist(),
        "scale": preprocessing["scale"][selected].tolist(),
    }
    return artifact, history_positions


def score_prediction(target: np.ndarray, raw_prediction: np.ndarray, weight: np.ndarray, scale: float, clip: float) -> tuple[float, np.ndarray]:
    prediction = np.clip(raw_prediction * scale, -clip, clip)
    return weighted_zero_mean_r2(target, prediction, weight), prediction


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = train_files(data_root)
    baseline_payload = json.loads((output_dir / "walk_forward_windows.json").read_text(encoding="utf-8"))
    baseline_experiment = next(
        experiment for experiment in baseline_payload["experiments"] if experiment["window"] == args.window
    )
    baseline_scores = {
        int(fold["valid_partition"]): float(fold["score"]) for fold in baseline_experiment["folds"]
    }

    results: list[dict[str, object]] = []
    for valid_index in args.validation_partitions:
        train_indices = list(range(valid_index - args.window, valid_index))
        print(f"history fold p{valid_index:03d}: feature selection", flush=True)
        cached = {
            index: load_partition_sample(files[index], args.train_sample_modulo) for index in train_indices
        }
        sampled = concatenate(cached, train_indices)
        del cached
        artifact, history_positions = feature_pipeline(
            sampled["features"],
            sampled["target"],
            sampled["weight"],
            args.feature_count,
            args.history_feature_count,
        )
        del sampled
        gc.collect()

        train_design, train_target, train_weight, _, history = build_history_design(
            [files[index] for index in train_indices],
            artifact,
            history_positions,
            args.train_sample_modulo,
            AssetHistory(feature_count=len(history_positions), window_size=5),
        )
        valid_design, valid_target, valid_weight, valid_asset, _ = build_history_design(
            [files[valid_index]],
            artifact,
            history_positions,
            args.valid_sample_modulo,
            history,
        )
        ridge_alpha = (
            args.final_ridge_alpha
            * args.final_sample_modulo
            / args.train_sample_modulo
            * args.window
            / args.reference_window
        )
        estimator = Ridge(
            alpha=ridge_alpha, solver="lsqr", tol=1e-4, max_iter=100, fit_intercept=True, copy_X=False
        )
        estimator.fit(train_design, train_target, sample_weight=np.maximum(train_weight, 0.0))
        raw_prediction = estimator.predict(valid_design)
        score, prediction = score_prediction(
            valid_target, raw_prediction, valid_weight, args.prediction_scale, args.prediction_clip
        )
        result = {
            "valid_partition": int(valid_index),
            "train_partitions": train_indices,
            "train_rows": int(len(train_target)),
            "valid_rows": int(len(valid_target)),
            "design_columns": int(train_design.shape[1]),
            "feature_count": int(args.feature_count),
            "history_feature_count": int(args.history_feature_count),
            "history_window": 5,
            "ridge_alpha": float(ridge_alpha),
            "baseline_score": baseline_scores[valid_index],
            "history_score": float(score),
            "improvement": float(score - baseline_scores[valid_index]),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
            "asset_scores": asset_scores(valid_target, prediction, valid_weight, valid_asset),
        }
        results.append(result)
        print(
            f"history fold=p{valid_index:03d} baseline={baseline_scores[valid_index]:.8f} "
            f"history={score:.8f} delta={score - baseline_scores[valid_index]:+.8f}",
            flush=True,
        )
        del train_design, train_target, train_weight, valid_design, valid_target, valid_weight
        del estimator, raw_prediction, prediction, history, artifact
        gc.collect()

    scores = np.asarray([result["history_score"] for result in results], dtype=float)
    baseline = np.asarray([result["baseline_score"] for result in results], dtype=float)
    accepted = bool(
        np.sum(scores > 0) >= np.sum(baseline > 0)
        and scores.mean() > baseline.mean()
        and scores.min() >= baseline.min() - 0.00005
    )
    payload = {
        "metric": "weighted_zero_mean_r2",
        "acceptance_rule": "no fewer positive folds, higher mean, worst fold no more than 0.00005 below baseline",
        "accepted": accepted,
        "baseline_mean": float(baseline.mean()),
        "history_mean": float(scores.mean()),
        "mean_improvement": float(scores.mean() - baseline.mean()),
        "baseline_min": float(baseline.min()),
        "history_min": float(scores.min()),
        "folds": results,
    }
    (output_dir / "walk_forward_history.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# Walk-forward causal history experiment",
        "",
        "| Fold | Baseline | History | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for result in results:
        markdown.append(
            f"| p{result['valid_partition']:03d} | {result['baseline_score']:.8f} | "
            f"{result['history_score']:.8f} | {result['improvement']:+.8f} |"
        )
    markdown.extend(
        [
            "",
            f"Baseline mean: `{payload['baseline_mean']:.8f}`",
            f"History mean: `{payload['history_mean']:.8f}`",
            f"Accepted for final training: **{accepted}**",
        ]
    )
    (output_dir / "walk_forward_history.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["accepted", "baseline_mean", "history_mean", "mean_improvement"]}, indent=2))


if __name__ == "__main__":
    main()
