"""Gate and orchestrate fixed-structure retraining after an organizer data refresh.

The command refuses to train unless ``audit_data_release.py`` proves that the training split changed.
Default mode is a dry-run command plan. ``--execute`` runs the two existing production trainers into a new
candidate directory and never overwrites ``strategies/*/model``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain fixed v3 structure after a verified data refresh.")
    parser.add_argument("--audit", required=True, help="Updated audit JSON containing comparison to baseline.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--candidate-dir", default=str(ROOT / "outputs" / "candidates" /
                                                        "v3_hybrid_extended_fixed"))
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--ridge-feature-count", type=int, default=200)
    parser.add_argument("--lgbm-feature-count", type=int, default=200)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--num-iteration", type=int, default=480)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--prediction-scale", type=float, default=1.16)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def command_plan(args: argparse.Namespace) -> list[list[str]]:
    python = str(ROOT / ".venv" / "bin" / "python")
    candidate = str(Path(args.candidate_dir))
    return [
        [python, str(ROOT / "strategies" / "v1_ridge" / "train.py"),
         "--data-root", args.data_root, "--model-dir", candidate,
         "--train-partitions", "999", "--sample-modulo", str(args.sample_modulo),
         "--validation-sample-modulo", "10", "--sampling", "phase_balanced",
         "--feature-count", str(args.ridge_feature_count), "--ridge-alpha", str(args.ridge_alpha),
         "--prediction-scale", str(args.prediction_scale), "--skip-validation"],
        [python, str(ROOT / "strategies" / "v3_hybrid" / "train.py"),
         "--data-root", args.data_root, "--model-dir", candidate,
         "--sample-modulo", str(args.sample_modulo), "--sampling", "phase_balanced",
         "--feature-count", str(args.lgbm_feature_count), "--history-count", "40",
         "--history-window", "5", "--num-iteration", str(args.num_iteration),
         "--n-seeds", str(args.n_seeds), "--prediction-scale", str(args.prediction_scale)],
    ]


def validate_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison") or {}
    train = comparison.get("splits", {}).get("train", {})
    train_changed = bool(train.get("added") or train.get("removed") or train.get("modified")
                         or train.get("row_delta"))
    if not comparison.get("changed") or not train_changed:
        raise SystemExit("audit does not prove a changed training split; fixed retraining is blocked")
    return comparison


def main() -> None:
    args = parse_args()
    comparison = validate_audit(Path(args.audit))
    candidate = Path(args.candidate_dir)
    plan = command_plan(args)
    print(json.dumps({"audit": args.audit, "comparison": comparison,
                      "candidate_dir": str(candidate), "commands": plan,
                      "execute": args.execute}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if candidate.exists():
        if not args.force:
            raise SystemExit(f"candidate exists: {candidate}; pass --force")
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)
    for command in plan:
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"fixed-structure extended-data candidate: {candidate}")


if __name__ == "__main__":
    main()
