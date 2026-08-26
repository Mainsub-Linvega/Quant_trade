from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge", ROOT / "strategies" / "v4_mlp"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.io import FEATURE_COLUMNS
from features import cross_sectional_deviation
from mt_predictability import group_starts
from history_features import iter_complete_time_batches
from mlp_numpy import NumpyMLP
from backfill_nn_train import (
    ensemble_prediction,
    replay_target_only_from_models,
    transform_selected_features_from_metadata,
)

DEFAULT_EXPERIMENT_JSON = ROOT / "outputs" / "experiments" / "backfill_nn_target_only_s5_5seed_coldstart.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cold-start CSV from backfill NN artifacts.")
    parser.add_argument("--experiment-json", default=str(DEFAULT_EXPERIMENT_JSON))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=120_000)
    parser.add_argument("--decimals", type=int, default=8)
    return parser.parse_args()


def load_target_only_artifacts(seed_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for row in seed_rows:
        market_model, market_meta = NumpyMLP.load(row["market"])
        cross_model, cross_meta = NumpyMLP.load(row["cross"])
        loaded.append({
            "seed": row.get("seed"),
            "market_model": market_model,
            "market_meta": market_meta,
            "cross_model": cross_model,
            "cross_meta": cross_meta,
        })
    if not loaded:
        raise ValueError("experiment JSON does not contain any artifact rows")
    return loaded


def predict_target_only_batch(loaded_artifacts: list[dict[str, object]], features: np.ndarray, time_id: np.ndarray,
                      asset_id: np.ndarray) -> np.ndarray:
    tid = np.asarray(time_id, dtype=np.int64)
    aid = np.asarray(asset_id, dtype=np.int64)
    if len(tid) != len(aid) or len(features) != len(tid):
        raise ValueError("features, time_id, and asset_id must have identical row counts")
    starts = group_starts(tid)
    counts = np.diff(np.r_[starts, len(tid)])
    predictions: list[np.ndarray] = []
    for artifact in loaded_artifacts:
        market_meta = artifact["market_meta"]
        cross_meta = artifact["cross_meta"]
        market_features = transform_selected_features_from_metadata(features, market_meta)
        cross_features = transform_selected_features_from_metadata(features, cross_meta)
        market_design = np.add.reduceat(market_features, starts, axis=0) / counts[:, None]
        cross_dev = cross_sectional_deviation(cross_features, tid)
        asset_one_hot = int(cross_meta.get("asset_one_hot", 15))
        if asset_one_hot <= 0:
            raise ValueError("asset_one_hot must be positive")
        if aid.min() < 0 or aid.max() >= asset_one_hot:
            raise ValueError("asset_id is outside the saved asset_one_hot width")
        cross_design = np.ascontiguousarray(np.column_stack([
            cross_dev,
            np.eye(asset_one_hot, dtype=np.float32)[aid],
        ]))
        predictions.append(replay_target_only_from_models(
            artifact["market_model"],
            artifact["cross_model"],
            market_design,
            cross_design,
            counts,
            market_mean=float(market_meta["mean"]),
            market_std=float(market_meta["std"]),
            cross_mean=float(cross_meta["mean"]),
            cross_std=float(cross_meta["std"]),
        ))
    return ensemble_prediction(predictions)


def write_submission_csv(experiment_json: Path, data_root: Path, output: Path, *, batch_size: int = 120_000,
                         decimals: int = 8) -> Path:
    payload = json.loads(Path(experiment_json).read_text(encoding="utf-8"))
    artifacts = load_target_only_artifacts(list(payload["artifacts"]["seeds"]))
    sample = pd.read_csv(data_root / "sample_submission.csv", usecols=["row_id"])
    row_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    columns = ["row_id", "time_id", "asset_id", *FEATURE_COLUMNS]
    for path in sorted((data_root / "test").glob("*.parquet")):
        for frame in iter_complete_time_batches(path, columns, batch_size=batch_size):
            row_parts.append(frame["row_id"].to_numpy(dtype=np.int64, copy=False))
            prediction_parts.append(predict_target_only_batch(
                artifacts,
                frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True),
                frame["time_id"].to_numpy(dtype=np.int64, copy=False),
                frame["asset_id"].to_numpy(dtype=np.int64, copy=False),
            ))
    if not row_parts:
        raise ValueError("no test rows were loaded")
    row_id = np.concatenate(row_parts)
    prediction = np.concatenate(prediction_parts)
    if not np.array_equal(row_id, sample["row_id"].to_numpy(dtype=np.int64, copy=False)):
        raise ValueError("test row_id order does not match sample_submission.csv")
    frame = pd.DataFrame({"row_id": row_id, "target": prediction})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format=f"%.{decimals}f")
    return output


def main() -> None:
    args = parse_args()
    path = write_submission_csv(
        Path(args.experiment_json),
        Path(args.data_root),
        Path(args.output),
        batch_size=args.batch_size,
        decimals=args.decimals,
    )
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
