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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score retrained v3 and NN on the shared public-backfill holdout.")
    parser.add_argument("--data-root", default="/mnt/e/量化/public_release_20260823/data")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--nn-experiment-json", default=str(ROOT / "outputs" / "experiments" / "backfill_nn_target_only_s5_5seed_coldstart.json"))
    parser.add_argument("--cutoff", type=int, default=1045920)
    parser.add_argument("--end-cutoff", type=int, default=None)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--sampling", default="phase_balanced")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def time_window_mask(time_id: np.ndarray, *, start: int, end: int | None) -> np.ndarray:
    ids = np.asarray(time_id, dtype=np.int64)
    mask = ids >= int(start)
    if end is not None:
        mask &= ids < int(end)
    return mask


def flush_groups(model: Model, loaded_nn: list[dict[str, object]], frame: pd.DataFrame,
                 cutoff: int, end_cutoff: int | None, sample_modulo: int, sampling: str,
                 state: dict[str, list[np.ndarray]]) -> None:
    time_id = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    for start, stop in zip(starts, stops):
        group = frame.iloc[start:stop]
        v3_scaled = model.predict(group.loc[:, ["time_id", "asset_id", *FEATURE_COLUMNS]])
        tid = int(time_id[start])
        if not time_window_mask(np.asarray([tid]), start=cutoff, end=end_cutoff)[0] or not time_sample_mask(
                np.asarray([tid], dtype=np.int64), sample_modulo, sampling=sampling)[0]:
            continue
        features = group.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        nn = predict_target_only_batch(
            loaded_nn,
            features,
            group["time_id"].to_numpy(dtype=np.int64, copy=False),
            group["asset_id"].to_numpy(dtype=np.int64, copy=False),
        )
        state["target"].append(group["target"].to_numpy(dtype=np.float64, copy=True))
        state["weight"].append(group["weight"].to_numpy(dtype=np.float64, copy=True))
        state["time_id"].append(group["time_id"].to_numpy(dtype=np.int64, copy=True))
        state["v3_scaled"].append(np.asarray(v3_scaled, dtype=np.float64))
        state["nn"].append(np.asarray(nn, dtype=np.float64))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    payload = json.loads(Path(args.nn_experiment_json).read_text(encoding="utf-8"))
    loaded_nn = load_target_only_artifacts(list(payload["artifacts"]["seeds"]))
    model = Model(args.model_dir, backend="lightgbm")
    state: dict[str, list[np.ndarray]] = {key: [] for key in ("target", "weight", "time_id", "v3_scaled", "nn")}
    carry: pd.DataFrame | None = None
    columns = ["time_id", "asset_id", "weight", "target", *FEATURE_COLUMNS]
    for path in sorted((data_root / "train").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            if carry is not None:
                frame = pd.concat([carry, frame], ignore_index=True)
                carry = None
            ids = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            final_start = int(np.flatnonzero(ids != ids[-1])[-1] + 1) if np.any(ids != ids[-1]) else 0
            if final_start:
                flush_groups(model, loaded_nn, frame.iloc[:final_start], args.cutoff, args.end_cutoff,
                             args.sample_modulo, args.sampling, state)
                carry = frame.iloc[final_start:].copy()
            else:
                carry = frame
    if carry is not None and len(carry):
        flush_groups(model, loaded_nn, carry, args.cutoff, args.end_cutoff, args.sample_modulo, args.sampling, state)

    target = np.concatenate(state["target"])
    weight = np.concatenate(state["weight"])
    time_id = np.concatenate(state["time_id"])
    v3_scaled = np.concatenate(state["v3_scaled"])
    nn = np.concatenate(state["nn"])
    v3_scale = float(model.prediction_scale)
    v3_raw = v3_scaled / v3_scale
    rows = {
        "v3_raw": scale_invariant_score(target, v3_raw, weight),
        "v3_deployed": {"score": weighted_zero_mean_r2(target, v3_scaled, weight),
                        "scale": v3_scale, "clipped_rows": int(np.count_nonzero(np.abs(v3_scaled) >= 0.5))},
        "nn": scale_invariant_score(target, nn, weight),
    }
    alpha_rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        prediction = (1.0 - alpha) * v3_raw + alpha * nn
        score = scale_invariant_score(target, prediction, weight)
        alpha_rows.append({"nn_weight": float(alpha), **score})
    by_time = []
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    for block in np.array_split(np.arange(len(starts)), 5):
        row_start = starts[block[0]]
        row_stop = starts[block[-1] + 1] if block[-1] + 1 < len(starts) else len(target)
        by_time.append({
            "time_id_start": int(time_id[row_start]),
            "time_id_stop": int(time_id[row_stop - 1]),
            "v3_raw_peak": scale_invariant_score(target[row_start:row_stop], v3_raw[row_start:row_stop], weight[row_start:row_stop])["peak"],
            "nn_peak": scale_invariant_score(target[row_start:row_stop], nn[row_start:row_stop], weight[row_start:row_stop])["peak"],
        })
    report = {
        "protocol": {
            "model": str(Path(args.model_dir)),
            "nn_experiment": str(Path(args.nn_experiment_json)),
            "data_root": str(data_root),
            "cutoff": args.cutoff,
            "end_cutoff": args.end_cutoff,
            "sample_modulo": args.sample_modulo,
            "sampling": args.sampling,
            "rows": int(len(target)),
            "time_id_range": [int(time_id.min()), int(time_id.max())],
            "v3_history_warmed_from_first_backfill_time": True,
            "v3_retrained_on_original_plus_pre_cutoff_backfill": True,
        },
        "models": rows,
        "blend_grid_oracle_on_holdout": alpha_rows,
        "time_blocks": by_time,
        "correlation": float(np.corrcoef(v3_raw, nn)[0, 1]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
