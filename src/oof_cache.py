"""Validated access to production-equivalent OOF component caches.

The experiment directory historically opened ``.npz`` files ad hoc.  This module
makes cache identity, row alignment and provenance explicit so structural-signal
experiments cannot silently compare predictions produced by different fold grids.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

CORE_COLUMNS = ("target", "weight", "time_id", "asset_id", "fold", "prediction_raw")
COMPONENT_COLUMNS = ("market_ridge", "market_lgbm", "market", "e_lgbm", "e_target", "xs_lgbm")


@dataclass(frozen=True)
class OOFBundle:
    path: Path
    arrays: Mapping[str, np.ndarray]
    report: Mapping[str, object]
    sha256: str

    @property
    def valid_mask(self) -> np.ndarray:
        return self.arrays["fold"] >= 0

    def valid(self, names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        selected = tuple(names) if names is not None else tuple(self.arrays)
        mask = self.valid_mask
        return {name: self.arrays[name][mask] for name in selected}


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(cache_path: Path, report_dir: Path | None) -> Path:
    directory = report_dir or cache_path.parent.parent / "experiments"
    return directory / f"{cache_path.stem}.json"


def validate_oof_arrays(arrays: Mapping[str, np.ndarray], *, require_components: bool = True) -> None:
    required = set(CORE_COLUMNS)
    if require_components:
        required.update(COMPONENT_COLUMNS)
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"OOF cache is missing required arrays: {missing}")

    lengths = {name: len(value) for name, value in arrays.items() if np.ndim(value) >= 1}
    expected = lengths["target"]
    bad = {name: size for name, size in lengths.items() if size != expected}
    if bad:
        raise ValueError(f"OOF arrays have inconsistent lengths; expected {expected}, got {bad}")
    if expected == 0:
        raise ValueError("OOF cache is empty")

    time_id = np.asarray(arrays["time_id"])
    asset_id = np.asarray(arrays["asset_id"])
    fold = np.asarray(arrays["fold"])
    if np.any(np.diff(time_id) < 0):
        raise ValueError("OOF rows are not sorted by time_id")
    if np.any(asset_id < 0):
        raise ValueError("OOF cache contains negative asset_id")
    if not np.any(fold >= 0):
        raise ValueError("OOF cache contains no validation rows")

    for name in ("target", "weight"):
        values = np.asarray(arrays[name])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"OOF array {name!r} contains non-finite values")
    valid_mask = fold >= 0
    finite_components = ("prediction_raw", "market_ridge", "market_lgbm", "market",
                         "e_lgbm", "xs_lgbm")
    for name in finite_components:
        if name in arrays and not np.all(np.isfinite(np.asarray(arrays[name])[valid_mask])):
            raise ValueError(f"OOF array {name!r} contains non-finite validation values")
    if np.any(np.asarray(arrays["weight"]) < 0):
        raise ValueError("OOF cache contains negative weights")

    # A time_id must belong wholly to one fold. Splitting a cross section across
    # train/validation would invalidate all rank and set-level features.
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    stops = np.r_[starts[1:], len(time_id)]
    for start, stop in zip(starts, stops):
        if np.ptp(fold[start:stop]) != 0:
            raise ValueError(f"time_id {int(time_id[start])} spans multiple folds")


def load_oof_bundle(
    path: str | Path,
    *,
    report_dir: str | Path | None = None,
    require_components: bool = True,
) -> OOFBundle:
    cache_path = Path(path).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    validate_oof_arrays(arrays, require_components=require_components)

    report_path = _report_path(cache_path, None if report_dir is None else Path(report_dir))
    report: Mapping[str, object] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reported_cache = report.get("cache")
        if reported_cache and Path(str(reported_cache)).name != cache_path.name:
            raise ValueError(
                f"OOF report/cache identity mismatch: {reported_cache!r} vs {cache_path.name!r}"
            )
    return OOFBundle(cache_path, arrays, report, file_sha256(cache_path))


def assert_row_alignment(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> None:
    for name in ("time_id", "asset_id", "target", "weight", "fold"):
        if name not in reference or name not in candidate:
            raise ValueError(f"alignment key {name!r} is missing")
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"OOF row alignment failed for {name!r}")
