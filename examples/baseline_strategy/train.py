from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge


FEATURE_COLUMNS = [f"feature_{index:03d}" for index in range(323)]
READ_COLUMNS = ["time_id", "weight", *FEATURE_COLUMNS, "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a robust, CPU-friendly contest baseline.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    parser.add_argument("--train-partitions", type=int, default=3)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--validation-sample-modulo", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=0.5)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def train_files(data_root: Path) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_paths = manifest.get("files", {}).get("train", [])
        if relative_paths:
            return [data_root / str(path) for path in relative_paths]
    return sorted((data_root / "train").glob("*.parquet"))


def load_time_sample(paths: list[Path], sample_modulo: int) -> tuple[np.ndarray, ...]:
    if sample_modulo <= 0:
        raise ValueError("sample modulo must be positive")

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []

    for path in paths:
        kept_rows = 0
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=120_000, columns=READ_COLUMNS):
            frame = batch.to_pandas()
            mask = frame["time_id"].to_numpy(copy=False) % sample_modulo == 0
            if not mask.any():
                continue
            feature_parts.append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            target_parts.append(frame.loc[mask, "target"].to_numpy(dtype=np.float32, copy=True))
            weight_parts.append(frame.loc[mask, "weight"].to_numpy(dtype=np.float32, copy=True))
            time_parts.append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
            kept_rows += int(mask.sum())
        print(f"loaded {path.name}: {kept_rows:,} sampled rows", flush=True)

    if not feature_parts:
        raise ValueError("training sample is empty")
    return (
        np.concatenate(feature_parts),
        np.concatenate(target_parts),
        np.concatenate(weight_parts),
        np.concatenate(time_parts),
    )


def robust_transform_fit(features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Anonymous market features occasionally contain NaN/inf and regime-specific
    # extremes. Learn every preprocessing statistic on the training period only.
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = np.quantile(features, [0.001, 0.25, 0.5, 0.75, 0.999], axis=0).astype(np.float32)
    lower, q25, center, q75, upper = quantiles
    scale = np.maximum(q75 - q25, np.float32(1e-4))
    np.clip(features, lower, upper, out=features)
    features -= center
    features /= scale
    np.clip(features, -10.0, 10.0, out=features)
    return features, {"lower": lower, "upper": upper, "center": center, "scale": scale}


def robust_transform_apply(features: np.ndarray, preprocessing: dict[str, np.ndarray]) -> np.ndarray:
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(features, preprocessing["lower"], preprocessing["upper"], out=features)
    features -= preprocessing["center"]
    features /= preprocessing["scale"]
    np.clip(features, -10.0, 10.0, out=features)
    return features


def select_features(features: np.ndarray, target: np.ndarray, weight: np.ndarray, count: int) -> np.ndarray:
    count = min(max(1, count), features.shape[1])
    weight64 = np.maximum(weight.astype(np.float64), 0.0)
    target64 = target.astype(np.float64)
    total_weight = float(weight64.sum())
    weighted_target_sum = float(np.dot(weight64, target64))
    target_variance = float(np.dot(weight64, target64 * target64) - weighted_target_sum**2 / total_weight)
    correlations = np.zeros(features.shape[1], dtype=np.float64)

    # Work in column blocks to avoid materialising several full float64 matrices.
    for start in range(0, features.shape[1], 64):
        stop = min(start + 64, features.shape[1])
        block = features[:, start:stop].astype(np.float64)
        weighted_feature_sum = block.T @ weight64
        weighted_square_sum = (block * block).T @ weight64
        weighted_cross_sum = block.T @ (weight64 * target64)
        covariance = weighted_cross_sum - weighted_feature_sum * weighted_target_sum / total_weight
        variance = weighted_square_sum - weighted_feature_sum**2 / total_weight
        correlations[start:stop] = covariance / np.sqrt(np.maximum(variance * target_variance, 1e-30))

    selected = np.argsort(np.abs(correlations), kind="stable")[-count:]
    # Keeping original column order makes inference indexing and artifacts easier to inspect.
    return np.sort(selected)


def cross_sectional_deviation(features: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    means = np.add.reduceat(features, starts, axis=0) / counts[:, None]
    deviation = features.copy()
    # Chunk the expansion so the temporary repeated-mean matrix stays small.
    for group_start in range(0, len(starts), 20_000):
        group_stop = min(group_start + 20_000, len(starts))
        row_start = int(starts[group_start])
        row_stop = int(starts[group_stop]) if group_stop < len(starts) else len(features)
        deviation[row_start:row_stop] -= np.repeat(
            means[group_start:group_stop], counts[group_start:group_stop], axis=0
        )
    return deviation


def make_design(features: np.ndarray, time_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    raw = features[:, selected].copy()
    deviation = cross_sectional_deviation(raw, time_ids)
    return np.column_stack([raw, deviation]).astype(np.float32, copy=False)


def weighted_zero_mean_r2(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    target64 = target.astype(np.float64)
    prediction64 = prediction.astype(np.float64)
    weight64 = np.maximum(weight.astype(np.float64), 0.0)
    denominator = float(np.dot(weight64, target64 * target64))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 - np.dot(weight64, (target64 - prediction64) ** 2) / denominator)


def fit_model(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
    feature_count: int,
    ridge_alpha: float,
) -> tuple[dict[str, object], np.ndarray]:
    features, preprocessing = robust_transform_fit(features)
    selected = select_features(features, target, weight, feature_count)
    design = make_design(features, time_ids, selected)
    del features

    estimator = Ridge(
        alpha=ridge_alpha,
        solver="lsqr",
        tol=1e-4,
        max_iter=100,
        fit_intercept=True,
        copy_X=False,
    )
    estimator.fit(design, target, sample_weight=np.maximum(weight, 0.0))
    artifact: dict[str, object] = {
        "selected_indices": selected.tolist(),
        "selected_features": [FEATURE_COLUMNS[index] for index in selected],
        "lower": preprocessing["lower"][selected].tolist(),
        "upper": preprocessing["upper"][selected].tolist(),
        "center": preprocessing["center"][selected].tolist(),
        "scale": preprocessing["scale"][selected].tolist(),
        "intercept": float(estimator.intercept_),
        "coef": estimator.coef_.astype(np.float64).tolist(),
        "ridge_alpha": float(ridge_alpha),
    }
    return artifact, selected


def predict_array(
    artifact: dict[str, object],
    full_features: np.ndarray,
    time_ids: np.ndarray,
    selected: np.ndarray,
    prediction_scale: float,
    prediction_clip: float,
) -> np.ndarray:
    selected_features = full_features[:, selected].copy()
    preprocessing = {
        name: np.asarray(artifact[name], dtype=np.float32)
        for name in ["lower", "upper", "center", "scale"]
    }
    selected_features = robust_transform_apply(selected_features, preprocessing)
    deviation = cross_sectional_deviation(selected_features, time_ids)
    design = np.column_stack([selected_features, deviation])
    prediction = float(artifact["intercept"]) + design @ np.asarray(artifact["coef"], dtype=np.float32)
    prediction *= prediction_scale
    return np.clip(prediction, -prediction_clip, prediction_clip)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    files = train_files(data_root)
    if len(files) < args.train_partitions + 1:
        raise ValueError("not enough chronological train partitions")
    if args.prediction_scale <= 0 or args.prediction_clip <= 0:
        raise ValueError("prediction scale and clip must be positive")

    validation_score: float | None = None
    validation_train_paths = files[-(args.train_partitions + 1) : -1]
    validation_path = files[-1]
    if not args.skip_validation:
        print("validation: train on earlier partitions and score the final partition", flush=True)
        x_train, y_train, w_train, t_train = load_time_sample(
            validation_train_paths, args.validation_sample_modulo
        )
        # Ridge's loss is a sum over rows, so keep alpha proportional to the
        # sampled row count when validation uses a sparser training sample.
        validation_alpha = args.ridge_alpha * args.sample_modulo / args.validation_sample_modulo
        validation_artifact, selected = fit_model(
            x_train, y_train, w_train, t_train, args.feature_count, validation_alpha
        )
        del x_train, y_train, w_train, t_train
        x_valid, y_valid, w_valid, t_valid = load_time_sample([validation_path], args.sample_modulo)
        valid_prediction = predict_array(
            validation_artifact,
            x_valid,
            t_valid,
            selected,
            args.prediction_scale,
            args.prediction_clip,
        )
        validation_score = weighted_zero_mean_r2(y_valid, valid_prediction, w_valid)
        print(f"validation weighted zero-mean R2: {validation_score:.8f}", flush=True)
        del x_valid, y_valid, w_valid, t_valid, valid_prediction, validation_artifact

    final_paths = files[-args.train_partitions :]
    print("final fit: " + ", ".join(path.name for path in final_paths), flush=True)
    features, target, weight, time_ids = load_time_sample(final_paths, args.sample_modulo)
    artifact, _ = fit_model(
        features, target, weight, time_ids, args.feature_count, args.ridge_alpha
    )
    artifact.update(
        {
            "strategy": "robust_ridge_cross_section_baseline",
            "train_files": [path.name for path in final_paths],
            "sample_modulo": int(args.sample_modulo),
            "train_rows": int(len(target)),
            "feature_count": int(args.feature_count),
            "prediction_scale": float(args.prediction_scale),
            "prediction_clip": float(args.prediction_clip),
            "validation_score": validation_score,
            "validation_metric": "weighted_zero_mean_r2",
        }
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "baseline_model.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_path": str(output_path),
                "train_rows": artifact["train_rows"],
                "selected_features": artifact["feature_count"],
                "validation_score": validation_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
