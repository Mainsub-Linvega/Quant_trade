"""P0 细粒度验证：按 time_id 滚动切 fold + embargo，压低标准误。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward_rolling.py
输出：outputs/experiments/walk_forward_rolling.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import fit_model, predict_array

READ_COLUMNS = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling time_id-level walk-forward validation.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--train-window", type=int, default=None,
                        help="Training window in unique sampled time_ids. Default: ~4/9 of total (≈4 partitions).")
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=0.5)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    return parser.parse_args()


def load_all_sampled(files: list[Path], sample_modulo: int) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "features": [], "target": [], "weight": [], "time_id": [], "asset_id": [],
    }
    for path in files:
        kept = 0
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
            kept += int(mask.sum())
        print(f"loaded {path.name}: {kept:,} sampled rows", flush=True)
    return {name: np.concatenate(values) for name, values in parts.items()}


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = train_files(data_root)

    print("loading all partitions...", flush=True)
    data = load_all_sampled(files, args.sample_modulo)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    print(f"total rows: {len(all_time_ids):,}, unique time_ids: {len(unique_time_ids):,}", flush=True)

    train_window = args.train_window
    if train_window is None:
        train_window = int(len(unique_time_ids) * 4 / 9)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, train_window, args.embargo)
    print(f"generated {len(folds)} folds (embargo={args.embargo})", flush=True)

    # Scale alpha: match per-row regularisation to the production fit.
    # Production: modulo=5, ~400K raw time_ids → 80K sampled → ~1.2M rows.
    # train_window is already in sampled time_ids, so compare directly.
    prod_sampled_window = 400_000 // 5  # 80K sampled time_ids in production
    ridge_alpha = args.ridge_alpha * (train_window / prod_sampled_window)

    fold_results: list[dict[str, object]] = []
    for i, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        train_set = np.isin(all_time_ids, train_ids)
        valid_set = np.isin(all_time_ids, valid_ids)

        t_features = data["features"][train_set]
        t_target = data["target"][train_set]
        t_weight = data["weight"][train_set]
        t_time = data["time_id"][train_set]

        v_features = data["features"][valid_set]
        v_target = data["target"][valid_set]
        v_weight = data["weight"][valid_set]
        v_time = data["time_id"][valid_set]

        artifact, selected = fit_model(
            t_features, t_target, t_weight, t_time,
            args.feature_count, ridge_alpha,
        )
        prediction = predict_array(
            artifact, v_features, v_time, selected,
            args.prediction_scale, args.prediction_clip,
        )
        score = weighted_zero_mean_r2(v_target, prediction, v_weight)

        fold = {
            "fold": i,
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "embargo_gap": int(valid_ids[0] - train_ids[-1]),
            "train_rows": int(train_set.sum()),
            "valid_rows": int(valid_set.sum()),
            "score": float(score),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        fold_results.append(fold)
        print(
            f"fold {i:2d}: train=[{train_ids[0]},{train_ids[-1]}] "
            f"valid=[{valid_ids[0]},{valid_ids[-1]}] "
            f"embargo={fold['embargo_gap']} score={score:.8f} "
            f"({fold['elapsed_seconds']:.1f}s)",
            flush=True,
        )
        del t_features, t_target, t_weight, t_time
        del v_features, v_target, v_weight, v_time, artifact, selected, prediction
        gc.collect()

    scores = np.array([f["score"] for f in fold_results])
    n = len(scores)
    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=1)) if n > 1 else 0.0
    se_score = std_score / np.sqrt(n) if n > 1 else 0.0

    payload = {
        "metric": "weighted_zero_mean_r2",
        "configuration": {
            "n_folds": args.n_folds,
            "train_window": train_window,
            "embargo": args.embargo,
            "sample_modulo": args.sample_modulo,
            "feature_count": args.feature_count,
            "ridge_alpha": float(ridge_alpha),
            "prediction_scale": args.prediction_scale,
            "prediction_clip": args.prediction_clip,
        },
        "summary": {
            "mean_score": mean_score,
            "std_score": std_score,
            "se_score": se_score,
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "positive_folds": int((scores > 0).sum()),
            "total_folds": n,
        },
        "folds": fold_results,
    }
    (output_dir / "walk_forward_rolling.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Rolling time_id-level walk-forward validation",
        "",
        f"train_window={train_window:,}, embargo={args.embargo}, "
        f"sample_modulo={args.sample_modulo}, n_folds={args.n_folds}",
        "",
        "| Fold | Train range | Valid range | Embargo | Score |",
        "|---:|---|---|---:|---:|",
    ]
    for f in fold_results:
        tr = f["train_time_range"]
        vr = f["valid_time_range"]
        lines.append(
            f"| {f['fold']} | [{tr[0]:,}, {tr[1]:,}] | [{vr[0]:,}, {vr[1]:,}] "
            f"| {f['embargo_gap']} | {f['score']:.8f} |"
        )
    lines += [
        "",
        f"**Mean**: {mean_score:.8f}, **SE**: {se_score:.8f}, "
        f"**Positive folds**: {int((scores > 0).sum())}/{n}",
    ]
    (output_dir / "walk_forward_rolling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "mean_score": mean_score,
        "se_score": se_score,
        "positive_folds": int((scores > 0).sum()),
        "total_folds": n,
    }, indent=2))


if __name__ == "__main__":
    main()
