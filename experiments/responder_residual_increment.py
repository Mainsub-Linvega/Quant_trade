"""Stage C: test whether Stage-B responder predictions add target residual signal.

This is the inexpensive, strict gate before a full history-LightGBM confirmation. For every outer fold it
fits the target baseline and responder models on the same transformed Ridge design, calibrates a linear
increment only on an inner holdout, and evaluates the frozen combination on the outer validation window.
The script consumes only Stage-B-passed families and refuses to run if Stage B did not pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.metric import scale_invariant_score
from src.validation import rolling_time_folds
from responder_predictability import fit_multi_ridge, group_metrics, predict_multi
from responder_targets import RESPONDER_COLUMNS, load_rows_with_responders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict responder residual increment gate.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--stage-b", default=str(ROOT / "outputs" / "experiments" /
                                                 "responder_predictability.json"))
    parser.add_argument("--label", default="responder_residual_increment")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=39_480)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--sampling", default="periodic", choices=["periodic", "phase_balanced"])
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--inner-fraction", type=float, default=0.25)
    parser.add_argument("--max-families", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def optimal_weights(residual: np.ndarray, auxiliary: np.ndarray, weight: np.ndarray,
                    ridge: float = 1e-8) -> np.ndarray:
    """Weighted least-squares calibration on an inner holdout only."""
    w = np.maximum(weight.astype(np.float64), 0.0)
    x = auxiliary.astype(np.float64)
    y = residual.astype(np.float64)
    gram = x.T @ (w[:, None] * x)
    rhs = x.T @ (w * y)
    penalty = ridge * max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    return np.linalg.solve(gram + penalty * np.eye(len(gram)), rhs)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.label}.json"; md_path = out / f"{args.label}.md"
    if (json_path.exists() or md_path.exists()) and not args.force:
        raise SystemExit(f"{json_path} or {md_path} exists; pass --force")
    stage_b_path = Path(args.stage_b)
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    if not stage_b.get("verdict", {}).get("enter_stage_c"):
        raise SystemExit("Stage B did not pass; Stage C must not run")
    passed = [row for row in stage_b["summary"]["clusters"] if row["pass"]]
    passed.sort(key=lambda row: row["mean_peak"], reverse=True)
    passed = passed[:args.max_families]
    if not passed:
        raise SystemExit("Stage B contains no passed families")

    # Freeze one representative per family before this experiment. Selection was already made inside
    # each Stage-B outer fold; choose the modal representative with lexical tie-break, never target delta.
    selected_names: list[str] = []
    for row in passed:
        counts: dict[str, int] = {}
        for name in row["selected_representatives"]:
            counts[name] = counts.get(name, 0) + 1
        selected_names.append(sorted(counts, key=lambda name: (-counts[name], name))[0])
    responder_indices = [RESPONDER_COLUMNS.index(name) for name in selected_names]

    print(f"loading {len(selected_names)} frozen Stage-B representatives: {selected_names}", flush=True)
    data = load_rows_with_responders(Path(args.data_root), args.sample_modulo, args.sampling)
    complete = np.all(np.isfinite(data["responders"][:, responder_indices]), axis=1)
    for key in data:
        data[key] = data[key][complete]
    all_ids = data["time_id"]
    folds = rolling_time_folds(np.unique(all_ids), args.n_folds, args.train_window, args.embargo)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        tr = np.isin(all_ids, train_ids); va = np.isin(all_ids, valid_ids)
        cut = int(len(train_ids) * (1.0 - args.inner_fraction))
        inner_train_ids = train_ids[:max(1, cut - args.embargo)]
        inner_valid_ids = train_ids[cut:]
        itr = tr & np.isin(all_ids, inner_train_ids)
        iva = tr & np.isin(all_ids, inner_valid_ids)
        target_matrix_inner = np.column_stack([
            data["target"][itr], data["responders"][itr][:, responder_indices]
        ])
        inner_model = fit_multi_ridge(
            data["features"][itr], all_ids[itr], data["target"][itr], target_matrix_inner,
            np.maximum(data["weight"][itr], 0.0), args.feature_count, args.ridge_alpha,
        )
        pred_inner = predict_multi(inner_model, data["features"][iva], all_ids[iva])
        target_inner_std = ((data["target"][iva] - inner_model["target_means"][0]) /
                            max(inner_model["target_stds"][0], 1e-30))
        aux_inner = pred_inner[:, 1:]
        coefficients = optimal_weights(target_inner_std - pred_inner[:, 0], aux_inner,
                                       data["weight"][iva])

        target_matrix_outer = np.column_stack([
            data["target"][tr], data["responders"][tr][:, responder_indices]
        ])
        outer_model = fit_multi_ridge(
            data["features"][tr], all_ids[tr], data["target"][tr], target_matrix_outer,
            np.maximum(data["weight"][tr], 0.0), args.feature_count, args.ridge_alpha,
        )
        pred_outer = predict_multi(outer_model, data["features"][va], all_ids[va])
        target_outer_std = ((data["target"][va] - outer_model["target_means"][0]) /
                            max(outer_model["target_stds"][0], 1e-30))
        baseline = pred_outer[:, 0]
        candidate = baseline + pred_outer[:, 1:] @ coefficients
        w_va = data["weight"][va]
        base_metric = scale_invariant_score(target_outer_std, baseline, w_va)
        candidate_metric = scale_invariant_score(target_outer_std, candidate, w_va)
        residual_corr = []
        residual = target_outer_std - baseline
        for j in range(len(selected_names)):
            if np.std(pred_outer[:, j + 1]) > 0 and np.std(residual) > 0:
                residual_corr.append(float(np.corrcoef(pred_outer[:, j + 1], residual)[0, 1]))
            else:
                residual_corr.append(None)
        fold_rows.append({
            "fold": fold_index,
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "coefficients": dict(zip(selected_names, coefficients.tolist())),
            "responder_prediction_target_residual_correlation": dict(zip(selected_names, residual_corr)),
            "baseline": base_metric, "candidate": candidate_metric,
            "delta_peak": float(candidate_metric["peak"] - base_metric["peak"]),
            "relative_gain": float(candidate_metric["peak"] / base_metric["peak"] - 1.0),
            "candidate_components": group_metrics(target_outer_std, candidate, w_va, all_ids[va],
                                                   data["asset_id"][va]),
            "elapsed_seconds": time.perf_counter() - fold_started,
        })
        print(f"fold {fold_index}: peak {base_metric['peak']:.8f} -> {candidate_metric['peak']:.8f} "
              f"({fold_rows[-1]['relative_gain']*100:+.2f}%)", flush=True)
        del inner_model, outer_model, pred_inner, pred_outer
        gc.collect()

    base = np.array([row["baseline"]["peak"] for row in fold_rows])
    cand = np.array([row["candidate"]["peak"] for row in fold_rows])
    delta = cand - base
    drop_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    checks = {
        "mean_delta_positive": float(delta.mean()) > 0,
        "positive_4of5_or_80pct": int((delta > 0).sum()) >= math.ceil(0.8 * len(delta)),
        "survives_drop_best": float(drop_best.mean()) > 0,
        "relative_gain_at_least_1pct": float(delta.mean() / base.mean()) >= 0.01,
    }
    passed_gate = all(checks.values())
    payload = {
        "question": "Do strict OOF predictions of Stage-B-passed responder families add target residual signal?",
        "configuration": vars(args),
        "data_fingerprint": {"stage_b": str(stage_b_path), "stage_b_sha256": sha256_file(stage_b_path)},
        "frozen_families": [{"cluster": row["cluster"], "members": row["members"],
                             "representative": name} for row, name in zip(passed, selected_names)],
        "data_quality": {"sampled_rows": int(len(complete)), "complete_rows": int(complete.sum()),
                         "excluded_rows": int((~complete).sum())},
        "folds": fold_rows,
        "summary": {"baseline_peak_mean": float(base.mean()), "candidate_peak_mean": float(cand.mean()),
                    "mean_delta": float(delta.mean()), "relative_gain": float(delta.mean() / base.mean()),
                    "positive_folds": int((delta > 0).sum()), "mean_delta_drop_best": float(drop_best.mean())},
        "verdict": {"checks": checks, "pass": passed_gate,
                    "next": "run full history-LightGBM confirmation" if passed_gate else "stop responder NN line"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"# Responder 残差增量（`{args.label}`）", "",
             f"冻结代表：{', '.join(selected_names)}", "",
             "| fold | baseline peak | candidate peak | relative gain |", "|---:|---:|---:|---:|"]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | {row['baseline']['peak']:.8f} | "
                     f"{row['candidate']['peak']:.8f} | {row['relative_gain']*100:+.2f}% |")
    lines += ["", f"平均增益：**{payload['summary']['relative_gain']*100:+.2f}%**；"
              f"正向折 {payload['summary']['positive_folds']}/{len(fold_rows)}；"
              f"去最好一折 Δ={payload['summary']['mean_delta_drop_best']:.6g}。", "",
              f"**{'PASS' if passed_gate else 'STOP'}** — {payload['verdict']['next']}", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}; pass={passed_gate}", flush=True)


if __name__ == "__main__":
    main()
