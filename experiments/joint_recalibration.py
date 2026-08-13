"""Pre-registered joint recalibration matrix for the post-refresh window.

This script does not perform selection today. It emits/validates the finite matrix that must be run only
after data-release audit and metric ruler verification. Correlated knobs are represented jointly so the
operator cannot accidentally interpret a one-dimensional sweep as an independent gain.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RIDGE_ALPHA = (1_000_000.0, 2_000_000.0, 4_000_000.0)
RIDGE_FEATURES = (200, 323)
RIDGE_MODULO = (5, 3)
LGBM_CAPACITY = {
    "stronger": {"num_leaves": 31, "min_data_fraction": 0.006,
                 "learning_rate": 0.02, "feature_fraction": 0.5, "lambda_l2": 10.0},
    "current": {"num_leaves": 63, "min_data_fraction": 12000 / 3_500_000,
                "learning_rate": 0.03, "feature_fraction": 0.7, "lambda_l2": 1.0},
    "looser": {"num_leaves": 95, "min_data_fraction": 0.002,
               "learning_rate": 0.025, "feature_fraction": 0.8, "lambda_l2": 0.5},
}
ROUNDS = (320, 480, 640)


def matrix() -> dict:
    ridge = [{"name": f"r_a{int(alpha)}_f{features}_m{modulo}",
              "ridge_alpha": alpha, "feature_count": features, "sample_modulo": modulo}
             for alpha, features, modulo in itertools.product(RIDGE_ALPHA, RIDGE_FEATURES, RIDGE_MODULO)]
    lgbm = [{"name": f"l_{capacity}_r{rounds}", "capacity": capacity,
             "num_iteration": rounds, **settings}
            for capacity, settings in LGBM_CAPACITY.items() for rounds in ROUNDS]
    return {
        "classification": "post-refresh joint recalibration; no one-dimensional conclusions",
        "selection_metric": "paired absolute peak against the fixed strong baseline",
        "ridge": ridge, "lgbm": lgbm,
        "gates": ["data audit proves train changed", "public-period metric ruler matches known CSV scores",
                  "all preprocessing fitted inside each training fold", "report A/B/peak and runtime",
                  "select once; do not extend the matrix after reading outcomes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the fixed post-refresh recalibration matrix.")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "experiments" /
                                                "joint_recalibration_plan.json"))
    args = parser.parse_args()
    payload = matrix()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "ridge_configs": len(payload["ridge"]),
                      "lgbm_configs": len(payload["lgbm"])}, indent=2))


if __name__ == "__main__":
    main()
