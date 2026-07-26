"""数据读取 —— 离线训练 / 实验共用（提交包不含本文件，main.py 不得 import）。

FEATURE_COLUMNS 是数据 schema 常量（323 个匿名特征列名），以这里为唯一定义。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


FEATURE_COLUMNS = [f"feature_{index:03d}" for index in range(323)]
READ_COLUMNS = ["time_id", "weight", *FEATURE_COLUMNS, "target"]


def train_files(data_root: Path) -> list[Path]:
    """按 manifest.json 的顺序返回训练分区（分区顺序 = 时间顺序）。"""
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_paths = manifest.get("files", {}).get("train", [])
        if relative_paths:
            return [data_root / str(path) for path in relative_paths]
    return sorted((data_root / "train").glob("*.parquet"))


def load_time_sample(paths: list[Path], sample_modulo: int) -> tuple[np.ndarray, ...]:
    """按 time_id % sample_modulo == 0 抽样加载若干分区，返回 (features, target, weight, time_id)。"""
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
