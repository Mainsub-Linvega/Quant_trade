"""Frozen residual adapters trained on the earliest OOF fold and evaluated later.

The residual atlas exposes two plausible, deployable failure modes:

1. the two market estimates disagree and their fixed 0.5/0.5 blend may be regime dependent;
2. cross-sectional quality is strongly asset dependent.

This script does an honest gate: fit every second-stage parameter on fold 0 only, freeze it,
and evaluate folds 1..4. Inputs are restricted to signals available at inference time. No target-
derived regime (realized volatility, realized market return, residual magnitude) is allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.metric import scale_invariant_score, weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(_REPO_ROOT / "outputs" / "cache" /
                                         "v3_production_oof_phasebal_prodwindow_exact.npz"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_residual_adapters")
    p.add_argument("--meta-fold", type=int, default=0)
    p.add_argument("--market-ridge-alpha", type=float, default=10.0)
    p.add_argument("--asset-shrink", type=float, default=50_000.0,
                   help="pseudo weighted energy pulling each asset slope toward 1")
    p.add_argument("--market-lambda", type=float, default=0.5)
    p.add_argument("--blend-weight", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def starts_counts(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)]).astype(np.int64)
    return starts, counts


def group_mean(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.add.reduceat(values, starts) / counts, counts)


def compose_hybrid_prediction(market_ridge: np.ndarray, market_lgbm: np.ndarray,
                              e_ridge: np.ndarray, e_lgbm: np.ndarray,
                              market_lambda: float,
                              blend_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    market = (1.0 - market_lambda) * market_ridge + market_lambda * market_lgbm
    cross = (1.0 - blend_weight) * e_ridge + blend_weight * e_lgbm
    return market, cross, market + cross


def market_features(time_id: np.ndarray, asset_id: np.ndarray, e_pred: np.ndarray,
                    market_ridge: np.ndarray, market_lgbm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, counts = starts_counts(time_id)
    mr = market_ridge[starts]
    ml = market_lgbm[starts]
    cross_rms = np.sqrt(np.add.reduceat(e_pred * e_pred, starts) / counts)
    phase = time_id[starts] % 10
    phase_onehot = np.eye(10, dtype=np.float64)[phase]
    coverage = counts.astype(np.float64)
    # All columns are available online from current predictions/time_id/asset coverage.
    design = np.column_stack([
        mr, ml, ml - mr, np.abs(ml - mr), np.abs(mr), np.abs(ml),
        cross_rms, coverage, phase_onehot,
    ])
    return design, starts, counts


def fit_market_experts(time_id: np.ndarray, asset_id: np.ndarray, target: np.ndarray,
                       weight: np.ndarray, e_pred: np.ndarray, market_ridge: np.ndarray,
                       market_lgbm: np.ndarray, alpha: float):
    design, starts, counts = market_features(time_id, asset_id, e_pred, market_ridge, market_lgbm)
    target_market = group_mean(target, starts, counts)[starts]
    group_weight = np.add.reduceat(weight, starts)
    linear = Ridge(alpha=alpha, fit_intercept=True)
    linear.fit(design, target_market, sample_weight=group_weight)
    hgb = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=250, l2_regularization=20.0, random_state=2026,
    )
    hgb.fit(design, target_market, sample_weight=group_weight)
    return linear, hgb


def predict_market(model, time_id: np.ndarray, asset_id: np.ndarray, e_pred: np.ndarray,
                   market_ridge: np.ndarray, market_lgbm: np.ndarray) -> np.ndarray:
    design, _, counts = market_features(time_id, asset_id, e_pred, market_ridge, market_lgbm)
    return np.repeat(model.predict(design), counts)


def fit_asset_slopes(time_id: np.ndarray, target: np.ndarray, weight: np.ndarray,
                     asset_id: np.ndarray, e_pred: np.ndarray, shrink: float,
                     fixed_cross: np.ndarray | None = None,
                     variable_weight: float = 1.0) -> np.ndarray:
    starts, counts = starts_counts(time_id)
    target_cross = target - group_mean(target, starts, counts)
    if fixed_cross is not None:
        if variable_weight == 0.0:
            raise ValueError("variable_weight must be non-zero")
        target_cross = (target_cross - fixed_cross) / variable_weight
    n_assets = int(asset_id.max()) + 1
    slopes = np.ones(n_assets, dtype=np.float64)
    for asset in range(n_assets):
        mask = asset_id == asset
        numerator = float(np.dot(weight[mask], e_pred[mask] * target_cross[mask]))
        denominator = float(np.dot(weight[mask], e_pred[mask] * e_pred[mask]))
        slopes[asset] = (numerator + shrink) / (denominator + shrink)
    return slopes


def apply_asset_slopes(time_id: np.ndarray, asset_id: np.ndarray, e_pred: np.ndarray,
                       slopes: np.ndarray) -> np.ndarray:
    starts, counts = starts_counts(time_id)
    adjusted = e_pred * slopes[asset_id]
    return adjusted - group_mean(adjusted, starts, counts)


def paired_summary(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    delta = candidate - baseline
    without_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    base = float(baseline.mean())
    return {
        "baseline_mean": base, "candidate_mean": float(candidate.mean()),
        "mean_delta": float(delta.mean()),
        "relative_gain": float(delta.mean() / base) if base > 0 else float("nan"),
        "positive_folds": int((delta > 0).sum()), "n_folds": int(len(delta)),
        "mean_delta_drop_best": float(without_best.mean()),
        "relative_gain_drop_best": float(without_best.mean() / base) if base > 0 else float("nan"),
        "per_fold_delta": [float(v) for v in delta],
        "pass": bool(delta.mean() > 0 and without_best.mean() > 0 and (delta > 0).sum() >= 3),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    with np.load(args.oof, allow_pickle=False) as d:
        fold = d["fold"].astype(np.int16)
        valid = fold >= 0
        arrays = {name: d[name][valid] for name in (
            "target", "weight", "time_id", "asset_id", "fold",
            "market_ridge", "market_lgbm", "e_ridge", "e_lgbm")}
    target = arrays["target"].astype(np.float64)
    weight = np.maximum(arrays["weight"].astype(np.float64), 0.0)
    time_id = arrays["time_id"].astype(np.int64)
    asset_id = arrays["asset_id"].astype(np.int64)
    fold = arrays["fold"].astype(np.int16)
    market_ridge = arrays["market_ridge"].astype(np.float64)
    market_lgbm = arrays["market_lgbm"].astype(np.float64)
    e_ridge = arrays["e_ridge"].astype(np.float64)
    e_pred = arrays["e_lgbm"].astype(np.float64)
    market, base_cross, baseline = compose_hybrid_prediction(
        market_ridge, market_lgbm, e_ridge, e_pred,
        args.market_lambda, args.blend_weight)

    meta = fold == args.meta_fold
    if not meta.any():
        raise SystemExit(f"meta fold {args.meta_fold} is absent")
    linear, hgb = fit_market_experts(
        time_id[meta], asset_id[meta], target[meta], weight[meta], e_pred[meta],
        market_ridge[meta], market_lgbm[meta], args.market_ridge_alpha)
    fixed_cross = (1.0 - args.blend_weight) * e_ridge
    slopes = fit_asset_slopes(
        time_id[meta], target[meta], weight[meta], asset_id[meta],
        e_pred[meta], args.asset_shrink, fixed_cross=fixed_cross[meta],
        variable_weight=args.blend_weight)

    # A single release scale is learned on meta fold and frozen for all later folds.
    arms: dict[str, np.ndarray] = {
        "baseline": baseline.copy(),
        "market_linear": baseline.copy(),
        "market_hgb": baseline.copy(),
        "cross_asset": baseline.copy(),
        "linear_plus_asset": baseline.copy(),
        "hgb_plus_asset": baseline.copy(),
    }
    for current_fold in np.unique(fold):
        mask = fold == current_fold
        market_linear = predict_market(linear, time_id[mask], asset_id[mask], e_pred[mask],
                                       market_ridge[mask], market_lgbm[mask])
        market_hgb = predict_market(hgb, time_id[mask], asset_id[mask], e_pred[mask],
                                    market_ridge[mask], market_lgbm[mask])
        adjusted_lgbm = apply_asset_slopes(
            time_id[mask], asset_id[mask], e_pred[mask], slopes)
        cross_asset = ((1.0 - args.blend_weight) * e_ridge[mask]
                       + args.blend_weight * adjusted_lgbm)
        arms["market_linear"][mask] = market_linear + base_cross[mask]
        arms["market_hgb"][mask] = market_hgb + base_cross[mask]
        arms["cross_asset"][mask] = market[mask] + cross_asset
        arms["linear_plus_asset"][mask] = market_linear + cross_asset
        arms["hgb_plus_asset"][mask] = market_hgb + cross_asset

    meta_scales = {}
    for name, pred in arms.items():
        meta_scales[name] = float(scale_invariant_score(target[meta], pred[meta], weight[meta])["optimal_scale"])

    eval_folds = [int(v) for v in np.unique(fold) if int(v) != args.meta_fold]
    fold_rows: list[dict[str, Any]] = []
    for current_fold in eval_folds:
        mask = fold == current_fold
        row: dict[str, Any] = {"fold": current_fold, "arms": {}}
        for name, pred in arms.items():
            peak = scale_invariant_score(target[mask], pred[mask], weight[mask])
            frozen_score = weighted_zero_mean_r2(
                target[mask], pred[mask] * meta_scales[name], weight[mask])
            row["arms"][name] = {
                "peak": float(peak["peak"]), "optimal_scale": float(peak["optimal_scale"]),
                "frozen_scale": meta_scales[name], "frozen_scale_score": float(frozen_score),
            }
        fold_rows.append(row)

    baseline_peak = np.array([row["arms"]["baseline"]["peak"] for row in fold_rows])
    baseline_fixed = np.array([row["arms"]["baseline"]["frozen_scale_score"] for row in fold_rows])
    summary: dict[str, Any] = {}
    for name in arms:
        if name == "baseline":
            continue
        peaks = np.array([row["arms"][name]["peak"] for row in fold_rows])
        fixed = np.array([row["arms"][name]["frozen_scale_score"] for row in fold_rows])
        summary[name] = {
            "peak": paired_summary(peaks, baseline_peak),
            "frozen_scale_score": paired_summary(fixed, baseline_fixed),
        }

    payload = {
        "experiment": "v3_residual_adapters", "oof": str(args.oof),
        "meta_fold": args.meta_fold, "eval_folds": eval_folds,
        "config": {"market_ridge_alpha": args.market_ridge_alpha,
                   "asset_shrink": args.asset_shrink,
                   "market_lambda": args.market_lambda,
                   "blend_weight": args.blend_weight},
        "asset_slopes": [float(v) for v in slopes],
        "meta_scales": meta_scales,
        "folds": fold_rows, "summary": summary,
        "gate": "mean delta > 0, drop-best delta > 0, positive in >=3/4 later folds",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen v3 residual adapters", "", f"Meta-train fold: `{args.meta_fold}`; evaluation folds: `{eval_folds}`", "",
             "## Asset slopes", "", "`" + ", ".join(f"{v:.3f}" for v in slopes) + "`", "",
             "## Honest later-fold results", "", "| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen-scale gain | Fixed gate |",
             "|---|---:|---:|---:|:---:|---:|:---:|"]
    for name, result in summary.items():
        peak, fixed = result["peak"], result["frozen_scale_score"]
        lines.append(f"| `{name}` | {peak['relative_gain']*100:+.2f}% | {peak['positive_folds']}/{peak['n_folds']} | "
                     f"{peak['relative_gain_drop_best']*100:+.2f}% | {'PASS' if peak['pass'] else 'FAIL'} | "
                     f"{fixed['mean_delta']:+.8f} | {'PASS' if fixed['pass'] else 'FAIL'} |")
    lines += ["", "## Per-fold peak", "", "| Fold | " + " | ".join(f"`{name}`" for name in arms) + " |",
              "|---:|" + "---:|" * len(arms)]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | " + " | ".join(f"{row['arms'][name]['peak']:.8f}" for name in arms) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)
    print("\n".join(lines[:20]), flush=True)


if __name__ == "__main__":
    main()
