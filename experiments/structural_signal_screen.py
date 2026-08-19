"""Strict OOF screening for genuinely new structural signal representations.

Arms
----
market_set
    Predict the time-level market target from cross-sectional distribution summaries,
    replacing the existing row-model market component while keeping XS fixed.
market_asset_panel
    Preserve anonymous asset identity in a fixed asset x feature panel, including the
    previous panel and first difference, so the market model can learn cross-asset interactions.
xs_rank
    Fit a correction to earlier strict-OOF production residuals from percentile ranks,
    robust z-scores and tail positions; the first fold is an explicit no-op.
xs_residual_select
    Re-select raw features against earlier strict-OOF production residuals inside each
    train window, then fit a small zero-mean XS correction.  This is not the already-rejected 323-wide
    arm, whose columns were selected against the original target.

No submission file or production artifact is written.  Defaults are the cheap 1-seed /
160-round screening tier; passing arms must later be confirmed at 3 seeds x 480 rounds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.lgbm_xs import FEATURE_COLUMNS, load_rows
from src.metric import scale_invariant_score
from src.oof_cache import load_oof_bundle

DEFAULT_CACHE = ROOT / "outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"
ARMS = ("market_set", "market_asset_panel", "xs_rank", "xs_residual_select")


def group_bounds(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    unique = time_id[starts]
    return starts, counts, unique


def group_mean(values: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    starts, counts, _ = group_bounds(time_id)
    if values.ndim == 1:
        means = np.add.reduceat(values, starts) / counts
    else:
        means = np.add.reduceat(values, starts, axis=0) / counts[:, None]
    return np.repeat(means, counts, axis=0)


def group_values(values: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    """One value per time_id (unweighted mean), preserving time order."""
    starts, counts, _ = group_bounds(time_id)
    return np.add.reduceat(values, starts, axis=0) / (counts[:, None] if values.ndim == 2 else counts)


def stable_top_correlated(x: np.ndarray, y: np.ndarray, top_k: int,
                          sample_rows: int = 500_000) -> np.ndarray:
    """Deterministic train-only absolute-correlation screen with finite-value handling."""
    if len(x) != len(y):
        raise ValueError("x/y length mismatch")
    if top_k <= 0:
        return np.empty(0, dtype=np.int64)
    if len(y) > sample_rows:
        index = np.linspace(0, len(y) - 1, sample_rows, dtype=np.int64)
        x, y = x[index], y[index]
    y = np.asarray(y, np.float64)
    y = y - np.nanmean(y)
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        col = np.asarray(x[:, j], np.float64)
        finite = np.isfinite(col) & np.isfinite(y)
        if finite.sum() < 3:
            continue
        c = col[finite] - col[finite].mean()
        yy = y[finite]
        den = np.sqrt(np.dot(c, c) * np.dot(yy, yy))
        scores[j] = abs(float(np.dot(c, yy) / den)) if den > 0 else 0.0
    # mergesort makes ties deterministic by original feature index.
    return np.argsort(-scores, kind="stable")[: min(top_k, x.shape[1])]


def _group_cube(x: np.ndarray, time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-size cross sections to a dense cube with NaN sentinel rows."""
    starts, counts, _ = group_bounds(time_id)
    width = int(counts.max())
    cube = np.full((len(starts), width, x.shape[1]), np.nan, dtype=np.float32)
    if np.all(counts == counts[0]):
        cube[:] = x.reshape(len(starts), int(counts[0]), x.shape[1])
    else:
        for group, (start, count) in enumerate(zip(starts, counts)):
            cube[group, :count] = x[start:start + count]
    return cube, counts


def _flatten_groups(cube: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if np.all(counts == counts[0]):
        return cube[:, :int(counts[0])].reshape(-1, cube.shape[-1])
    return np.concatenate([cube[i, :count] for i, count in enumerate(counts)], axis=0)


def cross_sectional_rank_features(x: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    """Per-row percentile rank, median/MAD z-score, and two-sided tail distance."""
    cube, counts = _group_cube(np.asarray(x, np.float32), time_id)
    finite = np.isfinite(cube)
    filled = np.where(finite, cube, np.nan)
    median = np.nanmedian(filled, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(filled - median), axis=1, keepdims=True)
    mad = np.where(np.isfinite(mad) & (mad > 1e-6), mad, 1.0)
    robust_z = np.clip((filled - median) / mad, -12.0, 12.0)

    # Cross sections are only ~15 rows, so stable double argsort is cheap and exact.
    sortable = np.where(finite, cube, np.inf)
    order = np.argsort(sortable, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable").astype(np.float32)
    denom = np.maximum(counts - 1, 1).astype(np.float32)[:, None, None]
    percentile = ranks / denom - np.float32(0.5)
    tail = np.maximum(np.abs(percentile) - np.float32(0.3), 0.0)
    blocks = [np.nan_to_num(percentile), np.nan_to_num(robust_z), np.nan_to_num(tail)]
    return _flatten_groups(np.concatenate(blocks, axis=2), counts).astype(np.float32, copy=False)


def asset_panel_features(x: np.ndarray, time_id: np.ndarray, asset_id: np.ndarray,
                         *, n_assets: int | None = None, dynamics: bool = True) -> np.ndarray:
    """Fixed asset-major panel per time_id, with presence flags and causal panel changes."""
    x = np.asarray(x, dtype=np.float32)
    asset_id = np.asarray(asset_id, dtype=np.int64)
    starts, counts, _ = group_bounds(time_id)
    assets = int(asset_id.max()) + 1 if n_assets is None else int(n_assets)
    panel = np.zeros((len(starts), assets, x.shape[1]), dtype=np.float32)
    present = np.zeros((len(starts), assets), dtype=np.float32)
    for group, (start, count) in enumerate(zip(starts, counts)):
        ids = asset_id[start:start + count]
        if np.any(ids < 0) or np.any(ids >= assets) or len(np.unique(ids)) != len(ids):
            raise ValueError("asset panel requires unique in-range asset_id per time_id")
        panel[group, ids] = np.nan_to_num(x[start:start + count])
        present[group, ids] = 1.0
    current = panel.reshape(len(starts), -1)
    if not dynamics:
        return np.column_stack([current, present])
    lag = np.zeros_like(current); lag[1:] = current[:-1]
    delta = current - lag
    return np.column_stack([current, lag, delta, present]).astype(np.float32, copy=False)


def market_summary_features(x: np.ndarray, time_id: np.ndarray, *, dynamics: bool = True) -> np.ndarray:
    """Time-level distribution summaries; all current-time statistics are deployment-visible."""
    cube, _counts = _group_cube(np.asarray(x, np.float32), time_id)
    q = np.nanquantile(cube, (0.10, 0.25, 0.50, 0.75, 0.90), axis=1)
    mean = np.nanmean(cube, axis=1)
    std = np.nanstd(cube, axis=1)
    positive = np.nanmean(cube > 0, axis=1)
    current = np.concatenate([mean, std, q[1], q[2], q[3], q[4] - q[0], positive], axis=1)
    current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not dynamics:
        return current
    lag = np.zeros_like(current)
    lag[1:] = current[:-1]
    delta = current - lag
    gap = np.zeros((len(current), 1), dtype=np.float32)
    _, _, unique = group_bounds(time_id)
    gap[1:, 0] = np.minimum(np.diff(unique), 1000).astype(np.float32)
    return np.concatenate([current, lag, delta, np.abs(delta), gap], axis=1)


def _sanitize_fit(train: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.nanquantile(train, 0.005, axis=0)
    upper = np.nanquantile(train, 0.995, axis=0)
    lower = np.nan_to_num(lower, nan=-1.0); upper = np.nan_to_num(upper, nan=1.0)
    bad = upper <= lower
    upper[bad] = lower[bad] + 1.0
    return (np.nan_to_num(np.clip(train, lower, upper)).astype(np.float32),
            np.nan_to_num(np.clip(valid, lower, upper)).astype(np.float32))


def fit_lgbm(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray,
             *, weight: np.ndarray | None, iterations: int, seed: int,
             num_threads: int, market: bool) -> np.ndarray:
    import lightgbm as lgb
    params: dict[str, Any] = {
        "objective": "regression", "metric": "None", "verbosity": -1,
        "learning_rate": 0.03 if not market else 0.02,
        "num_leaves": 31 if not market else 15,
        "min_data_in_leaf": 20_000 if not market else 1_000,
        "feature_fraction": 0.65 if not market else 0.55,
        "bagging_fraction": 0.7, "bagging_freq": 1,
        "lambda_l2": 10.0 if not market else 30.0,
        "seed": seed, "feature_fraction_seed": seed + 17, "bagging_seed": seed + 31,
        "num_threads": num_threads, "force_col_wise": True,
    }
    dataset = lgb.Dataset(train_x, label=train_y, weight=weight, free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=iterations)
    return booster.predict(valid_x, num_iteration=iterations).astype(np.float64)


def metrics(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    result = scale_invariant_score(y, pred, weight)
    return {"A": float(result["A"]), "B": float(result["B"]),
            "peak": float(result["peak"]), "optimal_scale": float(result["optimal_scale"])}


def paired_summary(base: list[dict[str, float]], arm: list[dict[str, float]]) -> dict[str, Any]:
    bp = np.array([x["peak"] for x in base]); ap = np.array([x["peak"] for x in arm])
    delta = ap - bp
    best = int(np.argmax(delta))
    keep = np.arange(len(delta)) != best
    base_A = np.mean([x["A"] for x in base]); arm_A = np.mean([x["A"] for x in arm])
    base_B = np.mean([x["B"] for x in base]); arm_B = np.mean([x["B"] for x in arm])
    rel_A = (arm_A - base_A) / abs(base_A) if base_A else 0.0
    rel_B = (arm_B - base_B) / abs(base_B) if base_B else 0.0
    relative = float(delta.mean() / bp.mean())
    checks = {
        "mean_delta_positive": bool(delta.mean() > 0),
        "at_least_4_of_5_positive": bool(np.sum(delta > 0) >= min(4, len(delta))),
        "drop_best_positive": bool(delta[keep].mean() > 0) if keep.any() else False,
        "relative_gain_at_least_2pct": bool(relative >= 0.02),
        "two_delta_A_exceeds_delta_B": bool(2 * rel_A > rel_B),
    }
    return {"mean_delta": float(delta.mean()), "relative": relative,
            "positive_folds": int(np.sum(delta > 0)), "n_folds": len(delta),
            "drop_best_delta": float(delta[keep].mean()) if keep.any() else float("nan"),
            "relative_delta_A": float(rel_A), "relative_delta_B": float(rel_B),
            "per_fold_delta": delta.tolist(), "checks": checks, "pass_screen": all(checks.values())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--output-dir", default=str(ROOT / "outputs/experiments"))
    p.add_argument("--label", default="structural_signal_screen_1s160")
    p.add_argument("--arms", nargs="*", choices=ARMS, default=list(ARMS))
    p.add_argument("--folds", nargs="*", type=int, default=None)
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", choices=("periodic", "phase_balanced"), default="phase_balanced")
    p.add_argument("--market-top-k", type=int, default=20)
    p.add_argument("--xs-top-k", type=int, default=32)
    p.add_argument("--iterations", type=int, default=160)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.label}.json"; md_path = out / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force")

    started = time.perf_counter()
    bundle = load_oof_bundle(args.cache)
    cache = bundle.arrays
    print("loading sampled raw features...", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    for name in ("time_id", "asset_id", "target", "weight"):
        if not np.array_equal(data[name], cache[name]):
            raise SystemExit(f"raw data and OOF cache are not aligned on {name}")
    x = data["features"]
    y = data["target"].astype(np.float64, copy=False)
    w = np.maximum(data["weight"].astype(np.float64, copy=False), 0.0)
    tid = data["time_id"]; aid = data["asset_id"]

    report_folds = {int(row["fold"]): row for row in bundle.report.get("folds", [])}
    folds = sorted(int(v) for v in np.unique(cache["fold"]) if v >= 0)
    if args.folds is not None:
        folds = [f for f in folds if f in set(args.folds)]
    if not folds:
        raise SystemExit("no selected validation folds")

    per_arm: dict[str, list[dict[str, float]]] = {arm: [] for arm in args.arms}
    base_metrics: list[dict[str, float]] = []
    fold_details: list[dict[str, Any]] = []
    for fold in folds:
        if fold not in report_folds:
            raise SystemExit(f"cache report is missing fold {fold} metadata")
        row = report_folds[fold]
        tr_lo, tr_hi = row["train_time_range"]
        va_mask = cache["fold"] == fold
        tr = slice(int(np.searchsorted(tid, tr_lo, side="left")),
                   int(np.searchsorted(tid, tr_hi, side="right")))
        va_idx = np.flatnonzero(va_mask)
        va = slice(int(va_idx[0]), int(va_idx[-1]) + 1)
        if not np.all(va_mask[va]):
            raise AssertionError("validation fold rows are not contiguous")

        y_tr, y_va = y[tr], y[va]; w_tr, w_va = w[tr], w[va]
        tid_tr, tid_va = tid[tr], tid[va]
        aid_tr, aid_va = aid[tr], aid[va]
        x_tr, x_va = x[tr], x[va]
        base_raw = cache["prediction_raw"][va]
        base = metrics(y_va, base_raw, w_va); base_metrics.append(base)
        detail: dict[str, Any] = {"fold": fold, "baseline": base, "arms": {}}
        print(f"fold {fold}: train={len(y_tr):,} valid={len(y_va):,}", flush=True)

        target_market_tr = group_values(y_tr, tid_tr)
        target_market_va = group_values(y_va, tid_va)
        e_tr = y_tr - group_mean(y_tr, tid_tr)

        for arm in args.arms:
            arm_started = time.perf_counter()
            if arm in ("market_set", "market_asset_panel"):
                means_tr = group_values(x_tr, tid_tr)
                chosen = stable_top_correlated(means_tr, target_market_tr, args.market_top_k,
                                               sample_rows=len(target_market_tr))
                if arm == "market_set":
                    train_design = market_summary_features(x_tr[:, chosen], tid_tr)
                    valid_design = market_summary_features(x_va[:, chosen], tid_va)
                else:
                    n_assets = int(max(aid.max(), aid_tr.max(), aid_va.max())) + 1
                    train_design = asset_panel_features(x_tr[:, chosen], tid_tr, aid_tr, n_assets=n_assets)
                    valid_design = asset_panel_features(x_va[:, chosen], tid_va, aid_va, n_assets=n_assets)
                train_design, valid_design = _sanitize_fit(train_design, valid_design)
                pred_group = fit_lgbm(train_design, target_market_tr, valid_design,
                                      weight=None, iterations=args.iterations, seed=args.seed,
                                      num_threads=args.num_threads, market=True)
                _, counts_va, _ = group_bounds(tid_va)
                pred_market = np.repeat(pred_group, counts_va)
                candidate = base_raw - cache["market"][va] + pred_market
            elif arm in ("xs_rank", "xs_residual_select"):
                # A true residual correction needs baseline predictions on its training
                # rows. Only earlier strict-OOF folds qualify; fold 0 therefore becomes
                # an explicit no-op rather than using in-sample or future-fitted values.
                prior = cache["fold"][tr] >= 0
                if not prior.any():
                    chosen = np.empty(0, dtype=np.int64)
                    candidate = base_raw.copy()
                    detail.setdefault("notes", {})[arm] = "no earlier OOF rows; strict no-op"
                else:
                    prior_x = x_tr[prior]
                    prior_tid = tid_tr[prior]
                    prior_w = w_tr[prior]
                    residual = y_tr[prior] - cache["prediction_raw"][tr][prior]
                    chosen = stable_top_correlated(prior_x, residual, args.xs_top_k)
                    if arm == "xs_rank":
                        train_design = cross_sectional_rank_features(prior_x[:, chosen], prior_tid)
                        valid_design = cross_sectional_rank_features(x_va[:, chosen], tid_va)
                    else:
                        train_design = prior_x[:, chosen] - group_mean(prior_x[:, chosen], prior_tid)
                        valid_design = x_va[:, chosen] - group_mean(x_va[:, chosen], tid_va)
                    train_design, valid_design = _sanitize_fit(train_design, valid_design)
                    correction = fit_lgbm(train_design, residual, valid_design, weight=prior_w,
                                          iterations=args.iterations, seed=args.seed,
                                          num_threads=args.num_threads, market=False)
                    correction -= group_mean(correction, tid_va)
                    candidate = base_raw + correction
            else:  # pragma: no cover
                raise AssertionError(arm)
            result = metrics(y_va, candidate, w_va)
            per_arm[arm].append(result)
            detail["arms"][arm] = {"metric": result,
                                    "selected_features": [FEATURE_COLUMNS[int(i)] for i in chosen],
                                    "elapsed_seconds": time.perf_counter() - arm_started}
            print(f"  {arm}: peak={result['peak']:.8f} "
                  f"delta={result['peak']-base['peak']:+.3e}", flush=True)
        fold_details.append(detail)

    summaries = {arm: paired_summary(base_metrics, values) for arm, values in per_arm.items()}
    payload = {
        "experiment": "structural_signal_screen",
        "question": "Do set-level market state, cross-sectional order statistics, or residual-directed selection add a new production-orthogonal signal?",
        "cache": {"path": str(bundle.path), "sha256": bundle.sha256},
        "config": {"arms": args.arms, "folds": folds, "iterations": args.iterations,
                   "seed": args.seed, "market_top_k": args.market_top_k,
                   "xs_top_k": args.xs_top_k, "sample_modulo": args.sample_modulo,
                   "sampling": args.sampling},
        "screening_rule": "1 seed / 160 rounds; mean positive, >=4/5 positive, drop-best positive, >=2%, 2dA>dB. Passing arms require 3x480 confirmation and paired bootstrap before promotion.",
        "summaries": summaries, "folds": fold_details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Structural signal screen", "", f"OOF cache: `{bundle.path.name}` (`{bundle.sha256[:12]}`)", "",
             "| Arm | Mean delta | Relative | Positive folds | Drop-best | dA | dB | Screen |",
             "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for arm, s in summaries.items():
        lines.append(f"| `{arm}` | {s['mean_delta']:+.3e} | {s['relative']*100:+.2f}% | "
                     f"{s['positive_folds']}/{s['n_folds']} | {s['drop_best_delta']:+.3e} | "
                     f"{s['relative_delta_A']*100:+.2f}% | {s['relative_delta_B']*100:+.2f}% | "
                     f"{'PASS' if s['pass_screen'] else 'FAIL'} |")
    lines += ["", "> A screening PASS is not a production decision. Run 3-seed x 480-round confirmation, paired block bootstrap, full-resolution online parity, and delivery-runtime gates.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
