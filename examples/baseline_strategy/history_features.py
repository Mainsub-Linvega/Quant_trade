from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class AssetHistory:
    feature_count: int
    window_size: int = 5
    values: dict[int, np.ndarray] = field(default_factory=dict)

    def transform(self, current: np.ndarray, asset_ids: np.ndarray) -> tuple[np.ndarray, ...]:
        previous = np.zeros_like(current, dtype=np.float32)
        rolling_mean = np.zeros_like(current, dtype=np.float32)

        for asset in np.unique(asset_ids):
            indices = np.flatnonzero(asset_ids == asset)
            asset_current = current[indices]
            history = self.values.get(int(asset))
            if history is None:
                history = np.empty((0, self.feature_count), dtype=np.float32)
            combined = np.vstack([history, asset_current])
            history_count = len(history)

            if history_count:
                previous[indices[0]] = history[-1]
            if len(indices) > 1:
                previous[indices[1:]] = asset_current[:-1]

            cumulative = np.vstack(
                [np.zeros((1, self.feature_count), dtype=np.float64), np.cumsum(combined, axis=0, dtype=np.float64)]
            )
            positions = history_count + np.arange(len(asset_current))
            starts = np.maximum(0, positions - self.window_size)
            counts = positions - starts
            sums = cumulative[positions] - cumulative[starts]
            rolling_mean[indices] = np.divide(
                sums,
                counts[:, None],
                out=np.zeros_like(sums),
                where=counts[:, None] > 0,
            ).astype(np.float32)
            self.values[int(asset)] = combined[-self.window_size :].astype(np.float32, copy=True)

        return previous, current - previous, rolling_mean, current - rolling_mean

    def as_payload(self) -> dict[str, list[list[float]]]:
        return {str(asset): values.astype(float).tolist() for asset, values in sorted(self.values.items())}

    @classmethod
    def from_payload(
        cls, payload: dict[str, list[list[float]]], feature_count: int, window_size: int
    ) -> "AssetHistory":
        history = cls(feature_count=feature_count, window_size=window_size)
        history.values = {
            int(asset): np.asarray(values, dtype=np.float32) for asset, values in payload.items()
        }
        return history


def iter_complete_time_batches(
    path: Path, columns: list[str], batch_size: int = 120_000
):
    carry: pd.DataFrame | None = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
            carry = None
        if frame.empty:
            continue
        last_time = int(frame["time_id"].iloc[-1])
        last_mask = frame["time_id"].to_numpy(copy=False) == last_time
        complete = frame.loc[~last_mask]
        carry = frame.loc[last_mask].copy()
        if not complete.empty:
            yield complete.reset_index(drop=True)
    if carry is not None and not carry.empty:
        yield carry.reset_index(drop=True)


def transform_selected(
    frame: pd.DataFrame,
    feature_columns: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(values, lower, upper, out=values)
    values -= center
    values /= scale
    np.clip(values, -10.0, 10.0, out=values)
    return values


def cross_sectional_deviation(values: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    means = np.add.reduceat(values, starts, axis=0) / counts[:, None]
    return values - np.repeat(means, counts, axis=0)


def build_history_design(
    paths: list[Path],
    artifact: dict[str, object],
    history_positions: np.ndarray,
    sample_modulo: int,
    history: AssetHistory | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, AssetHistory]:
    feature_columns = list(artifact["selected_features"])
    lower = np.asarray(artifact["lower"], dtype=np.float32)
    upper = np.asarray(artifact["upper"], dtype=np.float32)
    center = np.asarray(artifact["center"], dtype=np.float32)
    scale = np.asarray(artifact["scale"], dtype=np.float32)
    history = history or AssetHistory(feature_count=len(history_positions), window_size=5)
    columns = ["time_id", "asset_id", "weight", *feature_columns, "target"]

    design_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    asset_parts: list[np.ndarray] = []
    for path in paths:
        kept = 0
        for frame in iter_complete_time_batches(path, columns):
            time_ids = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            asset_ids = frame["asset_id"].to_numpy(dtype=np.int8, copy=False)
            raw = transform_selected(frame, feature_columns, lower, upper, center, scale)
            cross_sectional = cross_sectional_deviation(raw, time_ids)
            historical = raw[:, history_positions]
            previous, difference, rolling_mean, rolling_deviation = history.transform(historical, asset_ids)
            mask = time_ids % sample_modulo == 0
            if mask.any():
                design_parts.append(
                    np.column_stack(
                        [
                            raw[mask],
                            cross_sectional[mask],
                            previous[mask],
                            difference[mask],
                            rolling_mean[mask],
                            rolling_deviation[mask],
                        ]
                    ).astype(np.float32, copy=False)
                )
                target_parts.append(frame.loc[mask, "target"].to_numpy(dtype=np.float32, copy=True))
                weight_parts.append(frame.loc[mask, "weight"].to_numpy(dtype=np.float32, copy=True))
                asset_parts.append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int8, copy=True))
                kept += int(mask.sum())
        print(f"history design {path.name}: {kept:,} sampled rows", flush=True)

    return (
        np.concatenate(design_parts),
        np.concatenate(target_parts),
        np.concatenate(weight_parts),
        np.concatenate(asset_parts),
        history,
    )
