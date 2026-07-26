from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

# 离线训练专用：把仓库根加入 sys.path 以复用 src/ 下的公共实现。
# 注意 main.py（提交件）绝不允许这样做 —— 提交包里没有 src/。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io import FEATURE_COLUMNS, load_time_sample, train_files
from src.metric import weighted_zero_mean_r2

# 同目录的预处理 / 推理唯一实现（main.py 也 import 它，两侧口径由此保持一致）。
from features import apply_robust_transform, cross_sectional_deviation, linear_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a robust, CPU-friendly contest baseline.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
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


def robust_transform_fit(features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Anonymous market features occasionally contain NaN/inf and regime-specific
    # extremes. Learn every preprocessing statistic on the training period only.
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = np.quantile(features, [0.001, 0.25, 0.5, 0.75, 0.999], axis=0).astype(np.float32)
    lower, q25, center, q75, upper = quantiles
    scale = np.maximum(q75 - q25, np.float32(1e-4))
    features = apply_robust_transform(features, lower, upper, center, scale)
    return features, {"lower": lower, "upper": upper, "center": center, "scale": scale}


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


def make_design(features: np.ndarray, time_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    raw = features[:, selected].copy()
    deviation = cross_sectional_deviation(raw, time_ids)
    return np.column_stack([raw, deviation]).astype(np.float32, copy=False)


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
    selected_features = apply_robust_transform(
        selected_features,
        preprocessing["lower"],
        preprocessing["upper"],
        preprocessing["center"],
        preprocessing["scale"],
    )
    deviation = cross_sectional_deviation(selected_features, time_ids)
    return linear_predict(
        selected_features,
        deviation,
        float(artifact["intercept"]),
        np.asarray(artifact["coef"], dtype=np.float32),
        prediction_scale,
        prediction_clip,
    )


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
