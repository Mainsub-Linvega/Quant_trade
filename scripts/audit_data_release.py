"""Create or compare an auditable fingerprint of a competition data release.

Use before retraining after an organizer refresh. The snapshot records manifest content, file hashes,
Parquet row counts/time ranges/schema, and aggregate train/test counts. Comparison is read-only and exits
with status 2 when releases differ, making it suitable as an explicit gate before cache invalidation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot or compare a competition data release.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "data_audits" /
                                                "data_release_current.json"))
    parser.add_argument("--baseline", default=None,
                        help="Existing snapshot to compare against. Omit to create a snapshot only.")
    parser.add_argument("--no-file-hash", action="store_true",
                        help="Skip SHA256 for a fast metadata-only audit (not suitable for the final gate).")
    parser.add_argument("--fail-if-unchanged", action="store_true",
                        help="Exit 3 if baseline and current release are identical.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest_paths(data_root: Path, split: str) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = manifest.get("files", {}).get(split, [])
        if names:
            return [data_root / str(name) for name in names]
    return sorted((data_root / split).glob("*.parquet"))


def column_min_max(path: Path, column: str) -> tuple[int | None, int | None]:
    parquet = pq.ParquetFile(path)
    minimum: int | None = None
    maximum: int | None = None
    column_index = parquet.schema_arrow.get_field_index(column)
    if column_index < 0:
        return None, None
    metadata_complete = True
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(column_index).statistics
        if stats is None or not stats.has_min_max:
            metadata_complete = False
            break
        local_min, local_max = int(stats.min), int(stats.max)
        minimum = local_min if minimum is None else min(minimum, local_min)
        maximum = local_max if maximum is None else max(maximum, local_max)
    if metadata_complete:
        return minimum, maximum
    for batch in parquet.iter_batches(batch_size=250_000, columns=[column]):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        if not len(values):
            continue
        local_min, local_max = int(values.min()), int(values.max())
        minimum = local_min if minimum is None else min(minimum, local_min)
        maximum = local_max if maximum is None else max(maximum, local_max)
    return minimum, maximum


def inspect_parquet(path: Path, data_root: Path, include_hash: bool) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    time_min, time_max = column_min_max(path, "time_id")
    row_min, row_max = column_min_max(path, "row_id")
    return {
        "path": str(path.relative_to(data_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path) if include_hash else None,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": schema.names,
        "schema": str(schema),
        "time_id_min": time_min,
        "time_id_max": time_max,
        "row_id_min": row_min,
        "row_id_max": row_max,
    }


def snapshot(data_root: Path, include_hash: bool) -> dict[str, Any]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        splits[split] = [inspect_parquet(path, data_root, include_hash)
                         for path in manifest_paths(data_root, split)]
    sample = data_root / "sample_submission.csv"
    payload = {
        "data_root": str(data_root.resolve()),
        "hashes_included": include_hash,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "sample_submission": {
            "path": str(sample.relative_to(data_root)), "bytes": sample.stat().st_size,
            "sha256": sha256_file(sample) if include_hash else None,
        } if sample.exists() else None,
        "splits": splits,
        "summary": {
            split: {"files": len(files), "rows": sum(item["rows"] for item in files),
                    "bytes": sum(item["bytes"] for item in files),
                    "time_id_min": min((item["time_id_min"] for item in files
                                        if item["time_id_min"] is not None), default=None),
                    "time_id_max": max((item["time_id_max"] for item in files
                                        if item["time_id_max"] is not None), default=None)}
            for split, files in splits.items()
        },
    }
    return payload


def file_identity(item: dict[str, Any]) -> dict[str, Any]:
    keys = ["bytes", "rows", "row_groups", "columns", "schema", "time_id_min", "time_id_max",
            "row_id_min", "row_id_max"]
    if item.get("sha256") is not None:
        keys.append("sha256")
    return {key: item.get(key) for key in keys}


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"changed": False, "manifest_changed": False,
                              "sample_submission_changed": False, "splits": {}}
    result["manifest_changed"] = baseline.get("manifest") != current.get("manifest")
    base_sample = baseline.get("sample_submission")
    cur_sample = current.get("sample_submission")
    result["sample_submission_changed"] = base_sample != cur_sample
    result["changed"] = result["manifest_changed"] or result["sample_submission_changed"]
    for split in ("train", "test"):
        before = {item["path"]: item for item in baseline.get("splits", {}).get(split, [])}
        after = {item["path"]: item for item in current.get("splits", {}).get(split, [])}
        added = sorted(after.keys() - before.keys())
        removed = sorted(before.keys() - after.keys())
        modified = sorted(path for path in before.keys() & after.keys()
                          if file_identity(before[path]) != file_identity(after[path]))
        row_delta = (sum(item["rows"] for item in after.values())
                     - sum(item["rows"] for item in before.values()))
        result["splits"][split] = {"added": added, "removed": removed, "modified": modified,
                                    "row_delta": row_delta}
        result["changed"] = result["changed"] or bool(added or removed or modified)
    result["cache_action"] = (
        "invalidate outputs/cache and rebuild every data-derived artifact"
        if result["changed"] else "keep caches; no release change detected"
    )
    return result


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    current = snapshot(data_root, include_hash=not args.no_file_hash)
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current["comparison"] = compare(baseline, current)
        current["comparison"]["baseline"] = str(baseline_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": current["summary"],
                      "comparison": current.get("comparison")}, ensure_ascii=False, indent=2))
    if args.baseline and current["comparison"]["changed"]:
        raise SystemExit(2)
    if args.baseline and args.fail_if_unchanged and not current["comparison"]["changed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
