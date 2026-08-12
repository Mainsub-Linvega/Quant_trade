"""Audit a private-submission zip without extracting it into the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REQUIRED = {"main.py", "model/baseline_model.json", "model/hybrid_meta.json"}
FORBIDDEN_NAMES = {"train.py"}
FORBIDDEN_PREFIXES = ("src/", "data/", "outputs/", ".git/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit submission zip contents and model metadata.")
    parser.add_argument("zip_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-scale", type=float, default=None)
    parser.add_argument("--expected-iterations", type=int, default=None)
    parser.add_argument("--expected-seeds", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def audit(path: Path, expected_scale: float | None = None,
          expected_iterations: int | None = None, expected_seeds: int | None = None) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        duplicates = sorted({name for name in names if names.count(name) > 1})
        missing = sorted(REQUIRED - set(names))
        forbidden = sorted(name for name in names
                           if Path(name).name in FORBIDDEN_NAMES
                           or name.startswith(FORBIDDEN_PREFIXES)
                           or "__pycache__" in Path(name).parts or name.endswith(".pyc"))
        meta = json.loads(archive.read("model/hybrid_meta.json")) if not missing else {}
        model_files = list(meta.get("lgbm_model_files") or [])
        absent_models = sorted(name for name in model_files if f"model/{name}" not in names)
        checks = {
            "required_files_present": not missing,
            "no_forbidden_files": not forbidden,
            "no_duplicate_entries": not duplicates,
            "all_declared_models_present": not absent_models,
            "prediction_scale_matches": (expected_scale is None or
                                           abs(float(meta.get("prediction_scale", float("nan")))
                                               - expected_scale) < 1e-12),
            "iterations_match": (expected_iterations is None or
                                  meta.get("num_iteration") == expected_iterations),
            "seed_count_matches": (expected_seeds is None or len(model_files) == expected_seeds),
        }
        return {
            "zip": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "files": names, "missing": missing, "forbidden": forbidden,
            "duplicates": duplicates, "absent_declared_models": absent_models,
            "meta_summary": {"prediction_scale": meta.get("prediction_scale"),
                             "num_iteration": meta.get("num_iteration"),
                             "history_window": meta.get("history_window"),
                             "history_positions": len(meta.get("history_positions") or []),
                             "lgbm_model_files": model_files},
            "checks": checks, "passed": all(checks.values()),
        }


def main() -> None:
    args = parse_args()
    result = audit(Path(args.zip_path), args.expected_scale, args.expected_iterations,
                   args.expected_seeds)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
