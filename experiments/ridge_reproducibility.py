"""检查 Ridge 在不同 BLAS 线程数下是否得到同一模型。

父进程分别启动 1/4 线程子进程；每个子进程加载同一份确定性样本、拟合同一模型，
并在后一分区的固定样本上预测。所有中间文件都放在临时目录，不触碰正式模型。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import load_time_sample, train_files
from src.metric import weighted_zero_mean_r2
from train import fit_model, predict_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ridge models across BLAS thread counts.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--train-partitions", type=int, default=2)
    parser.add_argument("--sample-modulo", type=int, default=20)
    parser.add_argument("--valid-sample-modulo", type=int, default=50)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--ridge-tol", type=float, default=1e-8)
    parser.add_argument("--ridge-max-iter", type=int, default=2000)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output", default=None)
    parser.add_argument("--child-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-threads", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def child_run(args: argparse.Namespace) -> None:
    files = train_files(Path(args.data_root))
    train_paths = files[-(args.train_partitions + 1):-1]
    valid_path = files[-1]
    x_train, y_train, w_train, t_train = load_time_sample(train_paths, args.sample_modulo)
    artifact, selected = fit_model(
        x_train, y_train, w_train, t_train,
        args.feature_count, args.ridge_alpha,
        ridge_tol=args.ridge_tol,
        ridge_max_iter=args.ridge_max_iter,
    )
    del x_train, y_train, w_train, t_train
    x_valid, y_valid, w_valid, t_valid = load_time_sample(
        [valid_path], args.valid_sample_modulo
    )
    prediction = predict_array(artifact, x_valid, t_valid, selected, 1.0, 1e9)
    np.savez(
        args.child_output,
        selected=selected,
        coef=np.asarray(artifact["coef"], dtype=np.float64),
        intercept=np.asarray([artifact["intercept"]], dtype=np.float64),
        prediction=prediction.astype(np.float64),
        target=y_valid.astype(np.float64),
        weight=w_valid.astype(np.float64),
        n_iter=np.asarray([artifact["ridge_n_iter"]], dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    if args.child_output:
        child_run(args)
        return

    with tempfile.TemporaryDirectory(prefix="ridge-repro-") as temp_dir:
        outputs: dict[int, dict[str, np.ndarray]] = {}
        for threads in args.threads:
            path = Path(temp_dir) / f"threads_{threads}.npz"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--data-root", args.data_root,
                "--train-partitions", str(args.train_partitions),
                "--sample-modulo", str(args.sample_modulo),
                "--valid-sample-modulo", str(args.valid_sample_modulo),
                "--feature-count", str(args.feature_count),
                "--ridge-alpha", str(args.ridge_alpha),
                "--ridge-tol", str(args.ridge_tol),
                "--ridge-max-iter", str(args.ridge_max_iter),
                "--child-output", str(path),
                "--child-threads", str(threads),
            ]
            env = os.environ.copy()
            env.update({
                "OPENBLAS_NUM_THREADS": str(threads),
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
            })
            subprocess.run(command, check=True, env=env)
            with np.load(path) as handle:
                outputs[threads] = {name: handle[name].copy() for name in handle.files}

    reference_threads = args.threads[0]
    reference = outputs[reference_threads]
    comparisons = {}
    for threads in args.threads[1:]:
        current = outputs[threads]
        same_selected = bool(np.array_equal(reference["selected"], current["selected"]))
        coef_diff = float(np.max(np.abs(reference["coef"] - current["coef"])))
        pred_diff = float(np.max(np.abs(reference["prediction"] - current["prediction"])))
        reference_score = weighted_zero_mean_r2(
            reference["target"], reference["prediction"], reference["weight"]
        )
        current_score = weighted_zero_mean_r2(
            current["target"], current["prediction"], current["weight"]
        )
        comparisons[f"{reference_threads}_vs_{threads}"] = {
            "same_selected": same_selected,
            "max_abs_coef_diff": coef_diff,
            "max_abs_prediction_diff": pred_diff,
            "abs_score_diff": abs(reference_score - current_score),
            "score_reference": reference_score,
            "score_current": current_score,
            "n_iter_reference": int(reference["n_iter"][0]),
            "n_iter_current": int(current["n_iter"][0]),
        }
    payload = {
        "configuration": {
            "threads": args.threads,
            "sample_modulo": args.sample_modulo,
            "ridge_tol": args.ridge_tol,
            "ridge_max_iter": args.ridge_max_iter,
        },
        "comparisons": comparisons,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
