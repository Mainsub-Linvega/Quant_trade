"""顺序验收两个 Ridge 模型，不生成提交 CSV。

按官方 Time-Series API 的逐 time_id 顺序分别调用正式模型和候选模型，只累计
预测差异、合法性与耗时统计。模型预测不会写盘。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STRATEGY_DIR = _REPO_ROOT / "strategies" / "v1_ridge"
for _path in (str(_REPO_ROOT), str(_STRATEGY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from main import Model
from src.artifact import sha256_file
from timeseries_api.runner import coerce_prediction, iter_test_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Ridge candidate without writing predictions.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument(
        "--baseline-model",
        default=str(_STRATEGY_DIR / "model" / "baseline_model.json"),
    )
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_start = time.perf_counter()
    baseline = Model(model_path=args.baseline_model)
    baseline_init = time.perf_counter() - init_start
    init_start = time.perf_counter()
    candidate = Model(model_path=args.candidate_model)
    candidate_init = time.perf_counter() - init_start

    rows = calls = invalid_baseline = invalid_candidate = 0
    clipped_baseline = clipped_candidate = 0
    baseline_seconds = candidate_seconds = 0.0
    baseline_max_seconds = candidate_max_seconds = 0.0
    max_abs_diff = 0.0
    sum_diff2 = sum_baseline = sum_candidate = 0.0
    sum_baseline2 = sum_candidate2 = sum_cross = 0.0
    baseline_clip = float(baseline.prediction_clip)
    candidate_clip = float(candidate.prediction_clip)
    run_start = time.perf_counter()

    for _, test in iter_test_slices(args.data_root, split=args.split):
        started = time.perf_counter()
        raw_baseline = baseline.predict(test.copy())
        elapsed = time.perf_counter() - started
        baseline_seconds += elapsed
        baseline_max_seconds = max(baseline_max_seconds, elapsed)

        started = time.perf_counter()
        raw_candidate = candidate.predict(test.copy())
        elapsed = time.perf_counter() - started
        candidate_seconds += elapsed
        candidate_max_seconds = max(candidate_max_seconds, elapsed)

        baseline_pred, invalid = coerce_prediction(raw_baseline, len(test))
        invalid_baseline += invalid
        candidate_pred, invalid = coerce_prediction(raw_candidate, len(test))
        invalid_candidate += invalid

        diff = candidate_pred - baseline_pred
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(diff))))
        sum_diff2 += float(np.dot(diff, diff))
        sum_baseline += float(baseline_pred.sum())
        sum_candidate += float(candidate_pred.sum())
        sum_baseline2 += float(np.dot(baseline_pred, baseline_pred))
        sum_candidate2 += float(np.dot(candidate_pred, candidate_pred))
        sum_cross += float(np.dot(baseline_pred, candidate_pred))
        clipped_baseline += int((np.abs(baseline_pred) >= baseline_clip - 1e-12).sum())
        clipped_candidate += int((np.abs(candidate_pred) >= candidate_clip - 1e-12).sum())
        rows += len(test)
        calls += 1

    mean_baseline = sum_baseline / rows
    mean_candidate = sum_candidate / rows
    var_baseline = max(0.0, sum_baseline2 / rows - mean_baseline**2)
    var_candidate = max(0.0, sum_candidate2 / rows - mean_candidate**2)
    covariance = sum_cross / rows - mean_baseline * mean_candidate
    correlation = covariance / math.sqrt(var_baseline * var_candidate) if var_baseline > 0 and var_candidate > 0 else 0.0

    payload = {
        "baseline_model": str(Path(args.baseline_model)),
        "baseline_model_sha256": sha256_file(args.baseline_model),
        "candidate_model": str(Path(args.candidate_model)),
        "candidate_model_sha256": sha256_file(args.candidate_model),
        "split": args.split,
        "rows": rows,
        "predict_calls": calls,
        "prediction_comparison": {
            "max_abs_diff": max_abs_diff,
            "rmse_diff": math.sqrt(sum_diff2 / rows),
            "correlation": correlation,
            "baseline_mean": mean_baseline,
            "candidate_mean": mean_candidate,
            "baseline_std": math.sqrt(var_baseline),
            "candidate_std": math.sqrt(var_candidate),
            "baseline_clipped_rows": clipped_baseline,
            "candidate_clipped_rows": clipped_candidate,
            "baseline_invalid_rows": invalid_baseline,
            "candidate_invalid_rows": invalid_candidate,
        },
        "timing": {
            "baseline_init_seconds": baseline_init,
            "candidate_init_seconds": candidate_init,
            "baseline_predict_total_seconds": baseline_seconds,
            "candidate_predict_total_seconds": candidate_seconds,
            "baseline_mean_predict_seconds": baseline_seconds / calls,
            "candidate_mean_predict_seconds": candidate_seconds / calls,
            "baseline_max_predict_seconds": baseline_max_seconds,
            "candidate_max_predict_seconds": candidate_max_seconds,
            "wall_seconds": time.perf_counter() - run_start,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
