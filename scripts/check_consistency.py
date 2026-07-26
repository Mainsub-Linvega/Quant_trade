"""断言训练侧（train.predict_array）与推理侧（main.Model.predict）口径一致。

两侧现在都 import strategies/v1_ridge/features.py，但截面去均值仍是两条浮点
路径（分组 reduceat vs 单批 mean）。本脚本从真实分区抽若干 time_id，分别走
两条路径，逐元素断言在 1e-6 内一致。任何人改了预处理却只改了一侧，这里会响。

用法：
    .venv/bin/python scripts/check_consistency.py [--partition-index 8] [--n-time-ids 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, train_files
import main as strategy_main
import train as strategy_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert train/inference prediction parity.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--partition-index", type=int, default=8)
    parser.add_argument("--n-time-ids", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def load_leading_time_ids(path: Path, n_time_ids: int):
    """读取分区开头 n 个完整 time_id 的行（丢掉可能被批边界切开的最后一个）。"""
    import pandas as pd

    columns = ["row_id", "time_id", "asset_id", *FEATURE_COLUMNS]
    frames = []
    seen: set[int] = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
        frame = batch.to_pandas()
        frames.append(frame)
        seen.update(frame["time_id"].unique().tolist())
        if len(seen) > n_time_ids + 1:
            break
    data = pd.concat(frames, ignore_index=True)
    kept = sorted(seen)[: n_time_ids]
    return data[data["time_id"].isin(kept)].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    import json

    artifact = json.loads(
        (_REPO_ROOT / "strategies" / "v1_ridge" / "model" / "baseline_model.json").read_text(
            encoding="utf-8"
        )
    )
    selected = np.asarray(artifact["selected_indices"], dtype=np.int64)
    files = train_files(Path(args.data_root))
    frame = load_leading_time_ids(files[args.partition_index], args.n_time_ids)
    print(f"loaded {len(frame):,} rows / {frame['time_id'].nunique()} time_ids from {files[args.partition_index].name}")

    # 训练侧路径：整块矩阵 + 分组截面去均值
    full = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    time_ids = frame["time_id"].to_numpy(dtype=np.int64)
    pred_train = strategy_train.predict_array(
        artifact,
        full,
        time_ids,
        selected,
        float(artifact["prediction_scale"]),
        float(artifact["prediction_clip"]),
    )

    # 推理侧路径：官方 API 语义，按 time_id 递增逐批喂 Model
    model = strategy_main.Model()
    parts = []
    for time_id in sorted(frame["time_id"].unique()):
        parts.append(model.predict(frame[frame["time_id"] == time_id]))
    pred_infer = np.concatenate(parts)

    assert pred_infer.shape == pred_train.shape, "两侧输出长度不一致"
    max_diff = float(np.max(np.abs(pred_train.astype(np.float64) - pred_infer.astype(np.float64))))
    print(f"max |train - infer| = {max_diff:.3e}")
    assert np.allclose(pred_train, pred_infer, atol=args.atol), (
        f"训练与推理口径不一致（max diff {max_diff:.3e} > atol {args.atol:g}）——"
        "有人改了一侧的预处理没改另一侧"
    )
    print("OK: 训练与推理口径一致")


if __name__ == "__main__":
    main()
