"""Stage B: strict rolling out-of-sample feature-to-responder predictability.

Clusters come from Stage A, but the representative of each cluster is chosen only on an inner split of
each outer training window. The outer validation responder values are never used for selection or fitting.
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
from responder_targets import (RESPONDER_COLUMNS, build_design, gram_and_rhs,
                               load_rows_with_responders, solve_ridge, weighted_center,
                               weighted_moments, standardize_target)
from train import robust_transform_fit, select_features
from features import apply_robust_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict OOF responder predictability.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--analysis", default=str(ROOT / "outputs" / "experiments" / "responder_analysis.json"))
    parser.add_argument("--label", default="responder_predictability")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=39_480)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--sampling", default="periodic", choices=["periodic", "phase_balanced"])
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--inner-fraction", type=float, default=0.25)
    parser.add_argument("--max-clusters", type=int, default=None,
                        help="Smoke/debug only: keep clusters with strongest Stage-A representatives.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fit_multi_ridge(features: np.ndarray, time_ids: np.ndarray, target_for_selection: np.ndarray,
                    targets: np.ndarray, weight: np.ndarray, feature_count: int,
                    alpha: float) -> dict[str, Any]:
    transformed, stats = robust_transform_fit(features.copy())
    selected = select_features(transformed, target_for_selection, weight, feature_count)
    design = build_design(transformed, time_ids, selected)
    del transformed
    means = np.zeros(targets.shape[1], dtype=np.float64)
    stds = np.zeros(targets.shape[1], dtype=np.float64)
    standardized = np.empty(targets.shape, dtype=np.float64)
    for j in range(targets.shape[1]):
        means[j], stds[j] = weighted_moments(targets[:, j], weight)
        standardized[:, j] = standardize_target(targets[:, j], weight, means[j], stds[j])
    mean_x, _ = weighted_center(design, weight)
    gram, rhs = gram_and_rhs(design, weight, mean_x, standardized)
    beta = solve_ridge(gram, rhs, alpha)
    return {"stats": stats, "selected": selected, "mean_x": mean_x, "beta": beta,
            "target_means": means, "target_stds": stds}


def predict_multi(model: dict[str, Any], features: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
    transformed = features.copy()
    stats = model["stats"]
    apply_robust_transform(transformed, stats["lower"], stats["upper"], stats["center"], stats["scale"])
    design = build_design(transformed, time_ids, model["selected"])
    prediction = (design.astype(np.float64) - model["mean_x"]) @ model["beta"]
    return prediction


def group_metrics(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray,
                  time_ids: np.ndarray, asset_ids: np.ndarray | None = None) -> dict[str, Any]:
    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite].astype(np.float64)
    prediction = prediction[finite].astype(np.float64)
    weight = np.maximum(weight[finite].astype(np.float64), 0.0)
    time_ids = time_ids[finite]
    asset_ids = asset_ids[finite] if asset_ids is not None else None
    overall = scale_invariant_score(target, prediction, weight)
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    total_w = np.add.reduceat(weight, starts)
    target_market = np.add.reduceat(weight * target, starts) / total_w
    pred_market = np.add.reduceat(weight * prediction, starts) / total_w
    target_xs = target - np.repeat(target_market, counts)
    pred_xs = prediction - np.repeat(pred_market, counts)
    market = scale_invariant_score(target_market, pred_market, total_w)
    cross_section = scale_invariant_score(target_xs, pred_xs, weight)
    ics: list[float] = []
    for start, count in zip(starts, counts):
        stop = int(start + count)
        y = target[start:stop]; p = prediction[start:stop]
        if len(y) < 3 or np.std(y) <= 0 or np.std(p) <= 0:
            continue
        corr = float(np.corrcoef(y, p)[0, 1])
        if np.isfinite(corr):
            ics.append(corr)
    ic = np.asarray(ics, dtype=np.float64)
    asset_peaks: dict[str, float] = {}
    if asset_ids is not None:
        for asset in np.unique(asset_ids):
            mask = asset_ids == asset
            if mask.sum() >= 3:
                asset_peaks[str(int(asset))] = scale_invariant_score(
                    target[mask], prediction[mask], weight[mask]
                )["peak"]
    return {
        "overall": overall, "market": market, "cross_section": cross_section,
        "pearson": float(np.corrcoef(target, prediction)[0, 1]) if np.std(prediction) > 0 else None,
        "cross_section_ic_mean": float(ic.mean()) if len(ic) else None,
        "cross_section_icir": float(ic.mean() / ic.std()) if len(ic) and ic.std() > 0 else None,
        "cross_section_positive_ic_rate": float((ic > 0).mean()) if len(ic) else None,
        "cross_section_count": int(len(ic)), "valid_rows": int(len(target)),
        "asset_peaks": asset_peaks,
        "positive_assets": int(sum(value > 0 for value in asset_peaks.values())),
        "asset_count": int(len(asset_peaks)),
    }


def load_clusters(path: Path, max_clusters: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clusters = payload["clustering"]["clusters"]
    score = {row["name"]: abs(row.get("target_pearson") or 0.0) for row in payload["columns"]}
    clusters = sorted(clusters, key=lambda c: score.get(c["representative"], 0.0), reverse=True)
    return clusters if max_clusters is None else clusters[:max_clusters]


def aggregate_cluster(cluster: dict[str, Any], folds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [fold["clusters"][str(cluster["cluster"])] for fold in folds]
    peaks = np.array([row["metrics"]["overall"]["peak"] for row in rows])
    market = np.array([row["metrics"]["market"]["peak"] for row in rows])
    xs = np.array([row["metrics"]["cross_section"]["peak"] for row in rows])
    required_positive = math.ceil(0.8 * len(rows))
    drop_best = float(np.delete(peaks, int(np.argmax(peaks))).mean()) if len(peaks) > 1 else float(peaks[0])
    multi_member = len(cluster["members"]) >= 2
    inner_consistency = float(np.mean([row["inner_positive_member_rate"] for row in rows]))
    positive_assets = np.array([row["metrics"]["positive_assets"] for row in rows])
    asset_counts = np.array([row["metrics"]["asset_count"] for row in rows])
    asset_rate = float(np.divide(positive_assets.sum(), max(asset_counts.sum(), 1)))
    checks = {
        "mean_peak_positive": float(peaks.mean()) > 0,
        "positive_folds": int((peaks > 0).sum()) >= required_positive,
        "survives_drop_best": drop_best > 0,
        "cross_section_mean_positive": float(xs.mean()) > 0,
        "multi_member_family": multi_member,
        "inner_family_direction_consistent": inner_consistency >= 0.5,
        "not_single_asset": asset_rate >= 2.0 / 3.0,
    }
    return {
        "cluster": int(cluster["cluster"]), "members": cluster["members"],
        "selected_representatives": [row["selected"] for row in rows],
        "mean_peak": float(peaks.mean()), "positive_folds": int((peaks > 0).sum()),
        "required_positive_folds": required_positive, "mean_peak_drop_best": drop_best,
        "mean_market_peak": float(market.mean()), "mean_cross_section_peak": float(xs.mean()),
        "inner_positive_member_rate_mean": inner_consistency,
        "positive_asset_rate": asset_rate,
        "checks": checks, "pass": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"; md_path = output_dir / f"{args.label}.md"
    if (json_path.exists() or md_path.exists()) and not args.force:
        raise SystemExit(f"{json_path} or {md_path} exists; pass --force")
    analysis_path = Path(args.analysis)
    clusters = load_clusters(analysis_path, args.max_clusters)
    candidate_names = sorted({name for cluster in clusters for name in cluster["members"]})
    candidate_indices = np.array([RESPONDER_COLUMNS.index(name) for name in candidate_names])
    print(f"loading sampled data for {len(clusters)} clusters / {len(candidate_names)} responders", flush=True)
    data = load_rows_with_responders(Path(args.data_root), args.sample_modulo, args.sampling)
    complete = np.all(np.isfinite(data["responders"][:, candidate_indices]), axis=1)
    excluded = int((~complete).sum())
    for key in data:
        data[key] = data[key][complete]
    all_ids = data["time_id"]
    unique_ids = np.unique(all_ids)
    folds = rolling_time_folds(unique_ids, args.n_folds, args.train_window, args.embargo)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        tr = np.isin(all_ids, train_ids); va = np.isin(all_ids, valid_ids)
        cut = int(len(train_ids) * (1.0 - args.inner_fraction))
        inner_train_ids = train_ids[:max(1, cut - args.embargo)]
        inner_valid_ids = train_ids[cut:]
        itr = tr & np.isin(all_ids, inner_train_ids)
        iva = tr & np.isin(all_ids, inner_valid_ids)
        y_inner = data["responders"][itr][:, candidate_indices]
        inner_model = fit_multi_ridge(data["features"][itr], all_ids[itr], data["target"][itr],
                                      y_inner, np.maximum(data["weight"][itr], 0.0),
                                      args.feature_count, args.ridge_alpha)
        inner_pred = predict_multi(inner_model, data["features"][iva], all_ids[iva])
        inner_metrics: dict[str, dict[str, Any]] = {}
        for j, name in enumerate(candidate_names):
            raw = data["responders"][iva, RESPONDER_COLUMNS.index(name)]
            standardized = (raw - inner_model["target_means"][j]) / max(
                inner_model["target_stds"][j], 1e-30
            )
            inner_metrics[name] = group_metrics(standardized, inner_pred[:, j], data["weight"][iva],
                                                all_ids[iva], data["asset_id"][iva])
        selected: dict[str, str] = {}
        for cluster in clusters:
            selected[str(cluster["cluster"])] = max(
                cluster["members"], key=lambda name: inner_metrics[name]["overall"]["peak"]
            )
        selected_names = [selected[str(cluster["cluster"])] for cluster in clusters]
        outer_targets = np.column_stack([
            data["target"][tr],
            *[data["responders"][tr, RESPONDER_COLUMNS.index(name)] for name in selected_names],
        ])
        outer_model = fit_multi_ridge(data["features"][tr], all_ids[tr], data["target"][tr],
                                      outer_targets, np.maximum(data["weight"][tr], 0.0),
                                      args.feature_count, args.ridge_alpha)
        outer_pred = predict_multi(outer_model, data["features"][va], all_ids[va])
        cluster_rows: dict[str, Any] = {}
        for j, cluster in enumerate(clusters, start=1):
            name = selected[str(cluster["cluster"])]
            member_peaks = [inner_metrics[m]["overall"]["peak"] for m in cluster["members"]]
            cluster_rows[str(cluster["cluster"])] = {
                "selected": name,
                "inner_member_peaks": dict(zip(cluster["members"], member_peaks)),
                "inner_positive_member_rate": float(np.mean(np.asarray(member_peaks) > 0)),
                "metrics": group_metrics(
                    (data["responders"][va, RESPONDER_COLUMNS.index(name)]
                     - outer_model["target_means"][j]) / max(outer_model["target_stds"][j], 1e-30),
                    outer_pred[:, j], data["weight"][va], all_ids[va], data["asset_id"][va]),
            }
        fold_rows.append({
            "fold": fold_index,
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "inner_train_time_range": [int(inner_train_ids[0]), int(inner_train_ids[-1])],
            "inner_valid_time_range": [int(inner_valid_ids[0]), int(inner_valid_ids[-1])],
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "train_rows": int(tr.sum()), "valid_rows": int(va.sum()),
            "baseline_target_metrics": group_metrics(
                (data["target"][va] - outer_model["target_means"][0])
                / max(outer_model["target_stds"][0], 1e-30),
                outer_pred[:, 0], data["weight"][va], all_ids[va], data["asset_id"][va]),
            "clusters": cluster_rows,
            "elapsed_seconds": time.perf_counter() - fold_started,
        })
        print(f"fold {fold_index}: {len(selected_names)} representatives, "
              f"{fold_rows[-1]['elapsed_seconds']:.1f}s", flush=True)
        del inner_model, inner_pred, outer_model, outer_pred, outer_targets
        gc.collect()

    summaries = [aggregate_cluster(cluster, fold_rows) for cluster in clusters]
    passed = [row for row in summaries if row["pass"]]
    missing_rates = [
        (column.get("nonfinite_count") or 0) / max(column.get("finite_count", 0)
                                                   + (column.get("nonfinite_count") or 0), 1)
        for column in json.loads(analysis_path.read_text(encoding="utf-8"))["columns"]
        if column["name"] in candidate_names
    ]
    if any(rate > 0.005 for rate in missing_rates):
        raise SystemExit("A tested responder exceeds the 0.5% missing-rate threshold; "
                         "implement its per-target Gram before continuing")
    payload = {
        "question": "Can currently visible features predict any responder family out of sample?",
        "configuration": vars(args),
        "data_fingerprint": {"analysis_path": str(analysis_path), "analysis_sha256": file_sha256(analysis_path)},
        "data_quality": {"sampled_rows_before_complete_case": int(len(complete)),
                         "complete_case_rows": int(complete.sum()), "excluded_rows": excluded,
                         "excluded_rate": float(excluded / len(complete)),
                         "policy": "shared Gram uses rows finite for every tested responder"},
        "folds": fold_rows,
        "summary": {"clusters_tested": len(clusters), "clusters_passed": len(passed),
                    "passed_clusters": [row["cluster"] for row in passed], "clusters": summaries},
        "verdict": {"status": "pass" if passed else "stop", "enter_stage_c": bool(passed),
                    "reason": "At least one family passed every preregistered gate" if passed else
                              "No responder family passed every preregistered gate"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"# Responder 样本外可预测性（`{args.label}`）", "",
             f"- 测试族群：{len(clusters)}", f"- 通过族群：{len(passed)}",
             f"- complete-case 排除：{excluded:,} / {len(complete):,} ({excluded/len(complete):.4%})", "",
             "| cluster | members | mean peak | +folds | drop best | market | cross-section | pass |",
             "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in summaries:
        lines.append(f"| {row['cluster']} | {len(row['members'])} | {row['mean_peak']:.6g} | "
                     f"{row['positive_folds']}/{len(fold_rows)} | {row['mean_peak_drop_best']:.6g} | "
                     f"{row['mean_market_peak']:.6g} | {row['mean_cross_section_peak']:.6g} | "
                     f"{'✅' if row['pass'] else '❌'} |")
    lines += ["", "## 裁决", "", f"`{payload['verdict']['status']}` — {payload['verdict']['reason']}", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}; verdict={payload['verdict']['status']}", flush=True)


if __name__ == "__main__":
    main()
