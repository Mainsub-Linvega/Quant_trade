from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "strategies" / "v3_hybrid")]

from backfill_nn_submission import load_target_only_artifacts, predict_target_only_batch
from src.io import FEATURE_COLUMNS, time_sample_mask
from src.metric import scale_invariant_score, weighted_zero_mean_r2
from strategies.v3_hybrid.main import Model


def orthogonal_residual(candidate: np.ndarray, reference: np.ndarray,
                         weight: np.ndarray) -> tuple[np.ndarray, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    denominator = float(np.dot(weight, reference * reference))
    gamma = float(np.dot(weight, candidate * reference) / denominator) if denominator > 0 else 0.0
    return candidate - gamma * reference, gamma


def weighted_rms(values: np.ndarray, weight: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    total = float(weight.sum())
    return float(np.sqrt(np.dot(weight, values * values) / total)) if total > 0 else 0.0


def normalize_to_reference(values: np.ndarray, reference: np.ndarray,
                           weight: np.ndarray) -> tuple[np.ndarray, float]:
    value_rms = weighted_rms(values, weight)
    reference_rms = weighted_rms(reference, weight)
    if value_rms <= 0:
        return np.zeros_like(values, dtype=np.float64), 0.0
    return np.asarray(values, dtype=np.float64) * (reference_rms / value_rms), reference_rms / value_rms

def load_frozen_selection(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selection = payload.get("selection", payload)
    required = ("gamma", "normalization", "scale", "beta")
    missing = [key for key in required if key not in selection]
    if missing:
        raise ValueError(f"selection JSON missing keys: {missing}")
    return {key: float(selection[key]) for key in required}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search scale and orthogonal NN market residual.")
    parser.add_argument("--data-root", default="/mnt/e/量化/public_release_20260823/data")
    parser.add_argument("--v3-model-dir", required=True)
    parser.add_argument("--nn-experiment-json", required=True)
    parser.add_argument("--start-cutoff", type=int, required=True)
    parser.add_argument("--end-cutoff", type=int, required=True)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--sampling", default="phase_balanced")
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def split_prediction(values: np.ndarray, time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(time_id, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)])
    market = np.repeat(np.add.reduceat(values, starts) / counts, counts)
    return market, values - market


def score_at_scale(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray,
                   scale: float) -> float:
    return float(2.0 * scale * np.dot(weight, target * prediction) -
                 scale * scale * np.dot(weight, prediction * prediction)) / float(
                     np.dot(weight, target * target)
                 )


def flush_groups(model: Model, loaded_nn: list[dict[str, object]], frame: pd.DataFrame,
                 start_cutoff: int, end_cutoff: int, sample_modulo: int, sampling: str,
                 state: dict[str, list[np.ndarray]]) -> None:
    ids = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    for start, stop in zip(starts, stops):
        group = frame.iloc[start:stop]
        v3_scaled = model.predict(group.loc[:, ["time_id", "asset_id", *FEATURE_COLUMNS]])
        time_value = int(ids[start])
        if time_value < start_cutoff or time_value >= end_cutoff:
            continue
        if not time_sample_mask(np.asarray([time_value]), sample_modulo, sampling=sampling)[0]:
            continue
        features = group.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        nn = predict_target_only_batch(
            loaded_nn,
            features,
            group["time_id"].to_numpy(dtype=np.int64, copy=False),
            group["asset_id"].to_numpy(dtype=np.int64, copy=False),
        )
        v3_raw = np.asarray(v3_scaled, dtype=np.float64) / float(model.prediction_scale)
        v3_market, _ = split_prediction(v3_raw, group["time_id"].to_numpy(dtype=np.int64))
        nn_market, _ = split_prediction(nn, group["time_id"].to_numpy(dtype=np.int64))
        state["target"].append(group["target"].to_numpy(dtype=np.float64, copy=True))
        state["weight"].append(group["weight"].to_numpy(dtype=np.float64, copy=True))
        state["time_id"].append(group["time_id"].to_numpy(dtype=np.int64, copy=True))
        state["v3"].append(v3_raw)
        state["v3_market"].append(v3_market)
        state["nn_market"].append(nn_market)


def load_window(args: argparse.Namespace) -> dict[str, np.ndarray]:
    model = Model(args.v3_model_dir, backend="lightgbm")
    payload = json.loads(Path(args.nn_experiment_json).read_text(encoding="utf-8"))
    loaded_nn = load_target_only_artifacts(list(payload["artifacts"]["seeds"]))
    state: dict[str, list[np.ndarray]] = {
        key: [] for key in ("target", "weight", "time_id", "v3", "v3_market", "nn_market")
    }
    carry: pd.DataFrame | None = None
    columns = ["time_id", "asset_id", "weight", "target", *FEATURE_COLUMNS]
    for path in sorted((Path(args.data_root) / "train").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            if carry is not None:
                frame = pd.concat([carry, frame], ignore_index=True)
                carry = None
            ids = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            final_start = int(np.flatnonzero(ids != ids[-1])[-1] + 1) if np.any(ids != ids[-1]) else 0
            if final_start:
                flush_groups(model, loaded_nn, frame.iloc[:final_start],
                             args.start_cutoff, args.end_cutoff,
                             args.sample_modulo, args.sampling, state)
                carry = frame.iloc[final_start:].copy()
            else:
                carry = frame
    if carry is not None and len(carry):
        flush_groups(model, loaded_nn, carry, args.start_cutoff, args.end_cutoff,
                     args.sample_modulo, args.sampling, state)
    return {key: np.concatenate(value) for key, value in state.items()}


def evaluate(prediction: np.ndarray, target: np.ndarray, weight: np.ndarray,
             scale: float) -> dict[str, float]:
    metric = scale_invariant_score(target, prediction, weight)
    return {
        "peak": float(metric["peak"]),
        "score_at_unit_scale": float(metric["score_at_unit_scale"]),
        "score_at_scale": score_at_scale(target, prediction, weight, scale),
        "optimal_scale": float(metric["optimal_scale"]),
    }


def main() -> None:
    args = parse_args()
    data = load_window(args)
    target, weight = data["target"], np.maximum(data["weight"], 0.0)
    v3, v3_market, nn_market = data["v3"], data["v3_market"], data["nn_market"]
    if args.selection_json:
        selection = load_frozen_selection(args.selection_json)
        gamma = selection["gamma"]
        normalization = selection["normalization"]
        scale = selection["scale"]
        beta = selection["beta"]
        selection_source = str(args.selection_json)
    else:
        residual, gamma = orthogonal_residual(nn_market, v3_market, weight)
        _, normalization = normalize_to_reference(residual, v3_market, weight)
        q = residual * normalization
        scales = [0.80, 0.90, 1.00, 1.16, 1.30, 1.50]
        scale = max(scales, key=lambda value: score_at_scale(target, v3, weight, value))
        betas = [0.0, 0.05, 0.10, 0.20]
        beta = max(betas, key=lambda value: scale_invariant_score(
            target, v3 + value * q, weight
        )["peak"])
        selection_source = "calibration_search"
    residual, _ = orthogonal_residual(nn_market, v3_market, weight)
    q = residual * normalization
    candidates = {
        "v3": evaluate(v3, target, weight, scale),
        "v3_plus_orthogonal_nn_market": evaluate(v3 + beta * q, target, weight, scale),
        "nn_market_residual": evaluate(q, target, weight, scale),
    }
    output = {
        "protocol": {
            "data_root": str(args.data_root),
            "v3_model_dir": str(args.v3_model_dir),
            "nn_experiment_json": str(args.nn_experiment_json),
            "start_cutoff": args.start_cutoff,
            "end_cutoff": args.end_cutoff,
            "rows": int(len(target)),
            "sample_modulo": args.sample_modulo,
            "sampling": args.sampling,
            "selection_source": selection_source,
        },
        "selection": {
            "scale": scale,
            "gamma": gamma,
            "normalization": normalization,
            "beta": beta,
            "beta_grid": [0.0, 0.05, 0.10, 0.20],
            "scale_grid": [0.80, 0.90, 1.00, 1.16, 1.30, 1.50],
        },
        "candidates": candidates,
        "correlation": float(np.corrcoef(v3, nn_market)[0, 1]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
