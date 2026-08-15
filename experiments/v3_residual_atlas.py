"""从严格 v3 OOF 生成 market/cross residual atlas。

这个脚本只读 OOF cache，不训练模型、不生成提交文件。它把当前组合的失效拆成：

* market residual：每个 time_id 的真实截面均值 − 预测截面均值；
* cross residual：去掉上述 market 分量后的逐资产残差；
* fold / phase / asset / market-volatility / prediction-magnitude / model-disagreement。

输出 JSON/Markdown，给后续 conditional market expert 或 cross residual adapter 提供
候选门禁。所有候选必须先在 OOF bucket 上跨 fold 稳定，不能只看 pooled 平均。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(_REPO_ROOT / "outputs" / "cache" /
                                         "v3_production_oof_phasebal_prodwindow_exact.npz"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_residual_atlas")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_starts(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ids)]).astype(np.int64)
    return starts, counts


def broadcast_group_mean(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    starts, counts = group_starts(ids)
    return np.repeat(np.add.reduceat(values, starts) / counts, counts)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_rows(label: np.ndarray, target: np.ndarray, prediction: np.ndarray,
                   weight: np.ndarray) -> dict[str, Any]:
    """按 label 汇总，返回可跨 fold 比较的 SSE/energy/score。"""
    result: dict[str, Any] = {}
    for key in np.unique(label):
        mask = label == key
        w = np.maximum(weight[mask], 0.0)
        y = target[mask]
        p = prediction[mask]
        energy = float(np.dot(w, y * y))
        sse = float(np.dot(w, (y - p) ** 2))
        result[str(key)] = {
            "rows": int(mask.sum()),
            "weight": float(w.sum()),
            "target_energy": energy,
            "sse": sse,
            "score": float(1.0 - sse / energy) if energy > 0 else 0.0,
            "prediction_rms": float(np.sqrt(np.mean(p * p))),
            "residual_bias": float(np.average(y - p, weights=w)) if w.sum() > 0 else float("nan"),
        }
    return result


def component_summary(name: str, target: np.ndarray, prediction: np.ndarray,
                      weight: np.ndarray) -> dict[str, Any]:
    w = np.maximum(weight, 0.0)
    energy = float(np.dot(w, target * target))
    sse = float(np.dot(w, (target - prediction) ** 2))
    return {
        "name": name, "rows": int(len(target)), "score": float(1.0 - sse / energy),
        "sse": sse, "target_energy": energy,
        "rmse": float(np.sqrt(np.average((target - prediction) ** 2, weights=w))),
        "bias": float(np.average(target - prediction, weights=w)),
        "corr": safe_corr(target, prediction),
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
        target_all = d["target"].astype(np.float64)
        weight_all = d["weight"].astype(np.float64)
        time_all = d["time_id"].astype(np.int64)
        asset_all = d["asset_id"].astype(np.int64)
        fold_all = d["fold"].astype(np.int16)
        prediction_all = d["prediction"].astype(np.float64)
        raw_prediction_all = d["prediction_raw"].astype(np.float64)
        market_all = d["market"].astype(np.float64)
        market_ridge_all = d["market_ridge"].astype(np.float64)
        market_lgbm_all = d["market_lgbm"].astype(np.float64)
        e_lgbm_all = d["e_lgbm"].astype(np.float64)

    valid = (fold_all >= 0) & np.isfinite(prediction_all)
    if not valid.any():
        raise SystemExit("OOF cache contains no valid predictions")
    target = target_all[valid]
    weight = np.maximum(weight_all[valid], 0.0)
    time_id = time_all[valid]
    asset_id = asset_all[valid]
    fold = fold_all[valid]
    prediction = prediction_all[valid]
    raw_prediction = raw_prediction_all[valid]
    market = market_all[valid]
    market_ridge = market_ridge_all[valid]
    market_lgbm = market_lgbm_all[valid]
    e_lgbm = e_lgbm_all[valid]

    actual_market = broadcast_group_mean(target, time_id)
    actual_cross = target - actual_market
    # Decompose the actual emitted prediction, including any scale/clip applied by OOF.
    predicted_market = broadcast_group_mean(prediction, time_id)
    predicted_cross = prediction - predicted_market
    market_residual = actual_market - predicted_market
    cross_residual = actual_cross - predicted_cross
    raw_predicted_market = broadcast_group_mean(raw_prediction, time_id)
    raw_predicted_cross = raw_prediction - raw_predicted_market
    raw_market_residual = actual_market - raw_predicted_market
    raw_cross_residual = actual_cross - raw_predicted_cross

    starts, counts = group_starts(time_id)
    n_groups = len(starts)
    group_time = time_id[starts]
    group_fold = fold[starts]
    group_actual_market = actual_market[starts]
    group_market_residual = market_residual[starts]
    group_cross_rms = np.sqrt(np.add.reduceat(cross_residual * cross_residual, starts) / counts)
    group_market_gap = np.abs((market_lgbm - market_ridge)[starts])
    group_target_rms = np.sqrt(np.add.reduceat(target * target, starts) / counts)
    group_prediction_rms = np.sqrt(np.add.reduceat(prediction * prediction, starts) / counts)
    group_actual_vol = np.sqrt(np.add.reduceat(actual_cross * actual_cross, starts) / counts)

    def qbucket(values: np.ndarray, q: int = 4) -> np.ndarray:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, q + 1)))
        if len(edges) <= 2:
            return np.zeros(len(values), dtype=np.int64)
        return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)

    group_phase = group_time % 10
    group_vol_bucket = qbucket(group_actual_vol)
    group_market_bucket = qbucket(np.abs(group_actual_market))
    group_pred_bucket = qbucket(group_prediction_rms)
    group_gap_bucket = qbucket(group_market_gap)

    # Expand group labels back to rows for weighted score summaries.
    def expand(values: np.ndarray) -> np.ndarray:
        return np.repeat(values, counts)

    labels = {
        "fold": fold.astype(str),
        "phase": (time_id % 10).astype(str),
        "asset": asset_id.astype(str),
        "market_vol_quartile": expand(group_vol_bucket).astype(str),
        "market_abs_quartile": expand(group_market_bucket).astype(str),
        "prediction_rms_quartile": expand(group_pred_bucket).astype(str),
        "ridge_lgbm_gap_quartile": expand(group_gap_bucket).astype(str),
    }

    component_metrics = {
        "emitted_prediction": component_summary("emitted_prediction", target, prediction, weight),
        "raw_prediction": component_summary("raw_prediction", target, raw_prediction, weight),
        "market_only": component_summary("market_only", actual_market, predicted_market, weight),
        "cross_only": component_summary("cross_only", actual_cross, predicted_cross, weight),
        "market_ridge": component_summary("market_ridge", actual_market, market_ridge, weight),
        "market_lgbm": component_summary("market_lgbm", actual_market, market_lgbm, weight),
        "e_lgbm": component_summary("e_lgbm", actual_cross, e_lgbm, weight),
    }
    buckets = {
        name: {
            "prediction": aggregate_rows(label, target, prediction, weight),
            "raw_prediction": aggregate_rows(label, target, raw_prediction, weight),
        }
        for name, label in labels.items()
    }

    # Fold stability: each bucket is considered visible only if its delta is positive in
    # the same direction in at least 3/5 folds, not merely because one large bucket dominates.
    stability: dict[str, Any] = {}
    for name in ("phase", "asset", "market_vol_quartile", "market_abs_quartile",
                 "prediction_rms_quartile", "ridge_lgbm_gap_quartile"):
        by_bucket: dict[str, Any] = {}
        for bucket in np.unique(labels[name]):
            fold_values = []
            for f in np.unique(fold):
                mask = (labels[name] == bucket) & (fold == f)
                if mask.sum() < 20:
                    continue
                w = weight[mask]
                denom = float(np.dot(w, target[mask] * target[mask]))
                sse_model = float(np.dot(w, (target[mask] - prediction[mask]) ** 2))
                sse_market = float(np.dot(w, (target[mask] - predicted_market[mask]) ** 2))
                if denom > 0:
                    fold_values.append({"fold": int(f), "model_score": 1 - sse_model / denom,
                                        "market_score": 1 - sse_market / denom,
                                        "delta_model_vs_market": (sse_market - sse_model) / denom,
                                        "rows": int(mask.sum())})
            deltas = np.array([v["delta_model_vs_market"] for v in fold_values], dtype=float)
            by_bucket[str(bucket)] = {
                "folds": fold_values,
                "positive_folds": int((deltas > 0).sum()),
                "mean_delta_model_vs_market": float(deltas.mean()) if len(deltas) else float("nan"),
                "stable_positive_3of5": bool(len(deltas) >= 3 and (deltas > 0).sum() >= 3),
            }
        stability[name] = by_bucket

    worst_time_order = np.argsort(-(group_market_residual ** 2 + group_cross_rms ** 2))[:50]
    worst_times = [{
        "time_id": int(group_time[i]), "fold": int(group_fold[i]),
        "actual_market": float(group_actual_market[i]),
        "market_residual": float(group_market_residual[i]),
        "cross_rms": float(group_cross_rms[i]), "target_rms": float(group_target_rms[i]),
        "prediction_rms": float(group_prediction_rms[i]), "ridge_lgbm_gap": float(group_market_gap[i]),
    } for i in worst_time_order]

    payload = {
        "experiment": "v3_residual_atlas",
        "oof": str(args.oof),
        "rows": int(len(target)), "time_ids": int(n_groups),
        "components": component_metrics,
        "bucket_reports": buckets,
        "stability": stability,
        "worst_time_ids": worst_times,
        "diagnostic_definition": {
            "market_residual": "mean_i(y_it) - mean_i(pred_it)",
            "cross_residual": "(y_it - mean_i(y_it)) - (pred_it - mean_i(pred_it))",
            "stable_positive_3of5": "bucket's model-vs-market SSE delta is positive in >=3 observed folds",
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
                         encoding="utf-8")

    lines = ["# v3 residual atlas", "", f"OOF rows: `{len(target):,}`; time_ids: `{n_groups:,}`", "",
             "## Component metrics", "", "| Component | Score | RMSE | Bias | Corr |", "|---|---:|---:|---:|---:|"]
    for name, row in component_metrics.items():
        lines.append(f"| `{name}` | {row['score']:.8f} | {row['rmse']:.8f} | {row['bias']:+.3e} | "
                     f"{row['corr']:.4f} |")
    lines += ["", "## Stable bucket candidates", "", "Only buckets with model-vs-market positive delta in at least 3 observed folds are listed.", ""]
    for name, rows in stability.items():
        candidates = [(bucket, row) for bucket, row in rows.items() if row["stable_positive_3of5"]]
        lines.append(f"### {name}")
        if not candidates:
            lines.append("- none")
        else:
            for bucket, row in candidates:
                lines.append(f"- `{bucket}`: mean delta={row['mean_delta_model_vs_market']:+.4e}, "
                             f"positive folds={row['positive_folds']}")
        lines.append("")
    lines += ["## Worst time_id diagnostics", "", "| time_id | fold | market residual | cross RMS | target RMS | ridge/LGBM gap |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in worst_times[:20]:
        lines.append(f"| {row['time_id']} | {row['fold']} | {row['market_residual']:+.3e} | "
                     f"{row['cross_rms']:.3e} | {row['target_rms']:.3e} | {row['ridge_lgbm_gap']:.3e} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
