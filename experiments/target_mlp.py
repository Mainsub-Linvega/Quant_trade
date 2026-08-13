"""Target-only two-head MLP screen against cached v3 OOF predictions.

The model intentionally avoids responder labels. A small market head predicts the time-level target mean;
a cross-sectional head predicts the asset residual from current deviation, the validated history40 block,
and asset one-hot identity. Both heads are trained only on each outer training window. Acceptance is based
on the raw 50/50 ensemble with the strong cached v3 baseline; oracle weights are not used.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge",
             ROOT / "strategies" / "v4_mlp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.metric import scale_invariant_score
from src.validation import rolling_time_folds
from features import apply_robust_transform, cross_sectional_deviation
from history_peak import build_lag_cache, history_blocks, transform_with
from lgbm_xs import load_rows
from mlp_numpy import NumpyMLP
from mt_predictability import group_starts
from train import robust_transform_fit, select_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-only two-head MLP OOF screen.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="target_mlp_screen")
    parser.add_argument("--baseline-cache", default=str(ROOT / "outputs" / "cache" /
                                                         "temporal_v3_oof_1seed160.npz"))
    parser.add_argument("--candidate-meta", default=str(ROOT / "outputs" / "candidates" /
                                                         "v3_hybrid_r480_phasebal_hist" /
                                                         "hybrid_meta.json"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=39_480)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--sampling", default="phase_balanced", choices=["periodic", "phase_balanced"])
    parser.add_argument("--current-feature-count", type=int, default=100)
    parser.add_argument("--market-hidden", type=int, nargs="+", default=[32])
    parser.add_argument("--cross-hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts, axis=0) / counts[:, None]


def standardize_target(values: np.ndarray, weight: np.ndarray):
    weight = np.maximum(weight.astype(np.float64), 0.0)
    total = float(weight.sum())
    mean = float(np.dot(weight, values) / total)
    variance = float(np.dot(weight, (values - mean) ** 2) / total)
    std = max(float(np.sqrt(max(variance, 0.0))), 1e-8)
    return ((values - mean) / std).astype(np.float64), mean, std


def fit_mlp(design: np.ndarray, target: np.ndarray, weight: np.ndarray,
            hidden: tuple[int, ...], args: argparse.Namespace, seed: int):
    standardized, mean, std = standardize_target(target, weight)
    estimator = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=args.alpha,
        batch_size=args.batch_size, learning_rate_init=args.learning_rate,
        max_iter=args.max_iter, shuffle=True, random_state=seed, tol=0.0,
        early_stopping=False, n_iter_no_change=args.max_iter + 1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(design, standardized, sample_weight=np.maximum(weight, 0.0))
    numpy_model = NumpyMLP.from_sklearn(estimator)
    parity_probe = design[:min(4096, len(design))]
    parity = float(np.max(np.abs(estimator.predict(parity_probe)
                                 - numpy_model.predict(parity_probe))))
    if parity > 1e-5:
        raise RuntimeError(f"sklearn/NumPy MLP mismatch: {parity}")
    return estimator, numpy_model, mean, std, parity


def build_baseline_map(cache_path: Path, sampled_rows: int):
    with np.load(cache_path, allow_pickle=False) as cache:
        index = cache["row_index"].astype(np.int64)
        prediction = cache["prediction_baseline"].astype(np.float64)
    result = np.full(sampled_rows, np.nan, dtype=np.float64)
    result[index] = prediction
    return result


def fold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = np.array([row["baseline"]["peak"] for row in rows])
    mlp = np.array([row["mlp"]["peak"] for row in rows])
    blend = np.array([row["equal_blend"]["peak"] for row in rows])
    delta = blend - base
    drop = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    checks = {
        "mlp_at_least_70pct_of_baseline": float(mlp.mean() / base.mean()) >= 0.70,
        "blend_gain_at_least_5pct": float(delta.mean() / base.mean()) >= 0.05,
        "blend_positive_at_least_4of5_or_80pct": int((delta > 0).sum()) >= int(np.ceil(0.8 * len(delta))),
        "blend_survives_drop_best": float(drop.mean()) > 0,
    }
    return {
        "baseline_peak_mean": float(base.mean()), "mlp_peak_mean": float(mlp.mean()),
        "equal_blend_peak_mean": float(blend.mean()),
        "mlp_relative_to_baseline": float(mlp.mean() / base.mean()),
        "blend_relative_gain": float(delta.mean() / base.mean()),
        "blend_positive_folds": int((delta > 0).sum()),
        "blend_mean_delta_drop_best": float(drop.mean()),
        "prediction_correlation_mean": float(np.mean([row["prediction_correlation"] for row in rows])),
        "checks": checks, "pass": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.label}.json"; md_path = out / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} or {md_path} exists; pass --force")
    started = time.perf_counter()
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    tid = data["time_id"]; aid = data["asset_id"]
    folds = rolling_time_folds(np.unique(tid), args.n_folds, args.train_window, args.embargo)
    baseline_all = build_baseline_map(Path(args.baseline_cache), len(tid))

    meta = json.loads(Path(args.candidate_meta).read_text(encoding="utf-8"))
    lgbm_global = np.array([int(name.split("_")[-1]) for name in meta["lgbm_features"]], dtype=np.int64)
    history_global = lgbm_global[np.asarray(meta["history_positions"], dtype=np.int64)]
    lag_cache = build_lag_cache(
        [Path(path) for path in __import__("src.io", fromlist=["train_files"]).train_files(Path(args.data_root))],
        history_global, args.sample_modulo, 5, sampling=args.sampling,
    )
    if not (np.array_equal(lag_cache["time_id"], tid)
            and np.array_equal(lag_cache["asset_id"], aid)):
        raise SystemExit("history cache is not row-aligned")

    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        tr = np.isin(tid, train_ids); va = np.isin(tid, valid_ids)
        if not np.all(np.isfinite(baseline_all[va])):
            raise SystemExit("baseline cache does not cover this fold; regenerate with matching configuration")
        t_train, stats = robust_transform_fit(data["features"][tr].copy())
        t_valid = transform_with(data["features"][va], stats)
        y_train = data["target"][tr].astype(np.float64)
        y_valid = data["target"][va].astype(np.float64)
        w_train = np.maximum(data["weight"][tr].astype(np.float64), 0.0)
        w_valid = np.maximum(data["weight"][va].astype(np.float64), 0.0)
        tid_train, tid_valid = tid[tr], tid[va]
        aid_train, aid_valid = aid[tr], aid[va]
        starts_train = group_starts(tid_train); counts_train = np.diff(np.r_[starts_train, len(tid_train)])
        starts_valid = group_starts(tid_valid); counts_valid = np.diff(np.r_[starts_valid, len(tid_valid)])
        market_y_train = np.add.reduceat(y_train, starts_train) / counts_train
        market_y_valid = np.add.reduceat(y_valid, starts_valid) / counts_valid
        market_weight_train = np.add.reduceat(w_train, starts_train)
        e_train = y_train - np.repeat(market_y_train, counts_train)

        market_selected = select_features(t_train, y_train, w_train, args.current_feature_count)
        cross_selected = select_features(t_train, e_train, np.ones_like(e_train), args.current_feature_count)
        market_train = group_mean(t_train[:, market_selected], starts_train, counts_train)
        market_valid = group_mean(t_valid[:, market_selected], starts_valid, counts_valid)
        cross_train = cross_sectional_deviation(t_train[:, cross_selected].copy(), tid_train)
        cross_valid = cross_sectional_deviation(t_valid[:, cross_selected].copy(), tid_valid)
        lo, hi, ce, sc = (stats[key][history_global] for key in ("lower", "upper", "center", "scale"))
        hist_train = history_blocks(lag_cache["lags"][tr], lag_cache["count"][tr],
                                    t_train[:, history_global], lo, hi, ce, sc)
        hist_valid = history_blocks(lag_cache["lags"][va], lag_cache["count"][va],
                                    t_valid[:, history_global], lo, hi, ce, sc)
        asset_train = np.eye(15, dtype=np.float32)[aid_train]
        asset_valid = np.eye(15, dtype=np.float32)[aid_valid]
        cross_design_train = np.ascontiguousarray(np.column_stack([cross_train, *hist_train, asset_train]))
        cross_design_valid = np.ascontiguousarray(np.column_stack([cross_valid, *hist_valid, asset_valid]))

        market_est, _, market_mean, market_std, market_parity = fit_mlp(
            market_train, market_y_train, market_weight_train, tuple(args.market_hidden),
            args, args.seed + fold_index)
        cross_est, _, cross_mean, cross_std, cross_parity = fit_mlp(
            cross_design_train, e_train, np.ones_like(e_train), tuple(args.cross_hidden),
            args, args.seed + 100 + fold_index)
        market_prediction = market_est.predict(market_valid) * market_std + market_mean
        e_prediction = cross_est.predict(cross_design_valid) * cross_std + cross_mean
        e_prediction -= np.repeat(np.add.reduceat(e_prediction, starts_valid) / counts_valid, counts_valid)
        mlp_prediction = np.repeat(market_prediction, counts_valid) + e_prediction
        baseline = baseline_all[va]
        equal_blend = 0.5 * baseline + 0.5 * mlp_prediction
        correlation = float(np.corrcoef(baseline, mlp_prediction)[0, 1])
        row = {
            "fold": fold_index, "train_rows": int(tr.sum()), "valid_rows": int(va.sum()),
            "market_rows": int(len(market_train)),
            "design": {"market_columns": int(market_train.shape[1]),
                       "cross_columns": int(cross_design_train.shape[1])},
            "iterations": {"market": int(market_est.n_iter_), "cross": int(cross_est.n_iter_)},
            "numpy_parity": {"market_max_abs": market_parity, "cross_max_abs": cross_parity},
            "baseline": scale_invariant_score(y_valid, baseline, w_valid),
            "mlp": scale_invariant_score(y_valid, mlp_prediction, w_valid),
            "equal_blend": scale_invariant_score(y_valid, equal_blend, w_valid),
            "prediction_correlation": correlation,
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        fold_rows.append(row)
        print(f"fold {fold_index}: base={row['baseline']['peak']:.8f} "
              f"mlp={row['mlp']['peak']:.8f} blend={row['equal_blend']['peak']:.8f} "
              f"corr={correlation:.4f} ({row['elapsed_seconds']:.1f}s)", flush=True)
        del t_train, t_valid, cross_design_train, cross_design_valid, hist_train, hist_valid
        del market_est, cross_est
        gc.collect()

    summary = fold_summary(fold_rows)
    payload = {
        "question": "Can a target-only two-head MLP provide a useful low-correlation ensemble component?",
        "configuration": vars(args),
        "architecture": {"market": "time-level mean features -> MLP",
                         "cross": "current cross-sectional deviation + validated history40 + asset one-hot -> MLP",
                         "target": "market + demean(cross)", "responders_used": False},
        "folds": fold_rows, "summary": summary,
        "verdict": {"pass": summary["pass"],
                    "next": "export/deployment confirmation" if summary["pass"] else "stop MLP search"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    lines = [f"# Target-only MLP 筛选（`{args.label}`）", "",
             "| fold | baseline | MLP | equal blend | corr |", "|---:|---:|---:|---:|---:|"]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | {row['baseline']['peak']:.8f} | {row['mlp']['peak']:.8f} | "
                     f"{row['equal_blend']['peak']:.8f} | {row['prediction_correlation']:.4f} |")
    lines += ["", f"MLP/base={summary['mlp_relative_to_baseline']:.1%}，"
              f"等权集成增益={summary['blend_relative_gain']:+.2%}，"
              f"正折={summary['blend_positive_folds']}/{len(fold_rows)}，"
              f"平均相关={summary['prediction_correlation_mean']:.4f}。", "",
              f"**{payload['verdict']['next']}**", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
