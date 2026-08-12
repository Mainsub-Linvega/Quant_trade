"""Stage A: reproducible, streaming structural analysis of responder labels.

This script intentionally makes no claim that a responder can improve the deployed target model.
It describes contemporaneous label structure only; deployable predictability is tested separately by
``responder_predictability.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import train_files

RESPONDER_COLUMNS = [f"responder_{i:02d}" for i in range(47)]
BASE_COLUMNS = ["row_id", "time_id", "asset_id", "weight", "target"]
QUANTILES = np.array([0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream responder structure statistics.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="responder_analysis")
    parser.add_argument("--batch-size", type=int, default=120_000)
    parser.add_argument("--spearman-modulo", type=int, default=20)
    parser.add_argument("--cluster-threshold", type=float, default=0.15)
    parser.add_argument("--smoke-partitions", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(sw: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray,
              syy: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sy / sw
        vx = sxx - sx * sx / sw
        vy = syy - sy * sy / sw
        out = cov / np.sqrt(np.maximum(vx * vy, 0.0))
    return np.where((sw > 0) & (vx > 0) & (vy > 0), out, np.nan)


@dataclass
class VectorMoments:
    n: int

    def __post_init__(self) -> None:
        self.sw = np.zeros(self.n, dtype=np.float64)
        self.sx = np.zeros(self.n, dtype=np.float64)
        self.sxx = np.zeros(self.n, dtype=np.float64)
        self.sy = np.zeros(self.n, dtype=np.float64)
        self.syy = np.zeros(self.n, dtype=np.float64)
        self.sxy = np.zeros(self.n, dtype=np.float64)

    def add(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
        finite = np.isfinite(x) & np.isfinite(y)[:, None]
        w = np.where(finite, np.maximum(weight, 0.0)[:, None], 0.0)
        xv = np.where(finite, x, 0.0).astype(np.float64, copy=False)
        yv = np.where(np.isfinite(y), y, 0.0).astype(np.float64, copy=False)
        self.sw += w.sum(axis=0)
        self.sx += (w * xv).sum(axis=0)
        self.sxx += (w * xv * xv).sum(axis=0)
        self.sy += (w * yv[:, None]).sum(axis=0)
        self.syy += (w * (yv * yv)[:, None]).sum(axis=0)
        self.sxy += (w * xv * yv[:, None]).sum(axis=0)

    def corr(self) -> np.ndarray:
        return safe_corr(self.sw, self.sx, self.sy, self.sxx, self.syy, self.sxy)


class PairMoments:
    def __init__(self, n: int) -> None:
        shape = (n, n)
        self.sw = np.zeros(shape, dtype=np.float64)
        self.sx = np.zeros(shape, dtype=np.float64)
        self.sxx = np.zeros(shape, dtype=np.float64)
        self.sxy = np.zeros(shape, dtype=np.float64)

    def add(self, values: np.ndarray, weight: np.ndarray) -> None:
        finite = np.isfinite(values)
        mask = finite.astype(np.float64)
        x = np.where(finite, values, 0.0).astype(np.float64, copy=False)
        wmask = np.maximum(weight, 0.0)[:, None] * mask
        self.sw += mask.T @ wmask
        self.sx += x.T @ wmask
        self.sxx += (x * x).T @ wmask
        self.sxy += x.T @ (np.maximum(weight, 0.0)[:, None] * x)

    def corr(self) -> np.ndarray:
        return safe_corr(self.sw, self.sx, self.sx.T, self.sxx, self.sxx.T, self.sxy)


def grouped_slices(time_ids: np.ndarray) -> list[tuple[int, int]]:
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(time_ids)]
    return list(zip(starts.tolist(), stops.tolist()))


def weighted_group_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    w = np.where(finite, np.maximum(weights, 0.0)[:, None], 0.0)
    total = w.sum(axis=0)
    return np.divide((w * np.where(finite, values, 0.0)).sum(axis=0), total,
                     out=np.full(values.shape[1], np.nan), where=total > 0)


def pearson_rows(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    for j in range(values.shape[1]):
        finite = np.isfinite(values[:, j]) & np.isfinite(target)
        if finite.sum() < 3:
            continue
        x = values[finite, j].astype(np.float64)
        y = target[finite].astype(np.float64)
        x -= x.mean(); y -= y.mean()
        denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
        if denom > 0:
            out[j] = float(np.dot(x, y) / denom)
    return out


def process_time_groups(time_ids: np.ndarray, asset_ids: np.ndarray, responders: np.ndarray,
                        target: np.ndarray, weight: np.ndarray, market: VectorMoments,
                        cross_section: VectorMoments, ic_sum: np.ndarray, ic_sq: np.ndarray,
                        ic_pos: np.ndarray, ic_n: np.ndarray, lag: VectorMoments,
                        last_by_asset: dict[int, np.ndarray]) -> int:
    groups = 0
    for start, stop in grouped_slices(time_ids):
        r = responders[start:stop]
        y = target[start:stop]
        w = np.maximum(weight[start:stop], 0.0)
        assets = asset_ids[start:stop]
        finite_y = np.isfinite(y)
        if not finite_y.any() or w[finite_y].sum() <= 0:
            continue
        y_mean = float(np.dot(w[finite_y], y[finite_y]) / w[finite_y].sum())
        r_mean = weighted_group_mean(r, w)
        group_weight = np.array([float(w.sum())])
        market.add(r_mean[None, :], np.array([y_mean]), group_weight)
        cross_section.add(r - r_mean[None, :], y - y_mean, w)
        corr = pearson_rows(r, y)
        finite_corr = np.isfinite(corr)
        ic_sum[finite_corr] += corr[finite_corr]
        ic_sq[finite_corr] += corr[finite_corr] ** 2
        ic_pos[finite_corr] += corr[finite_corr] > 0
        ic_n[finite_corr] += 1

        previous = np.full_like(r, np.nan, dtype=np.float64)
        for row, asset in enumerate(assets):
            key = int(asset)
            if key in last_by_asset:
                previous[row] = last_by_asset[key]
            last_by_asset[key] = r[row].astype(np.float64, copy=True)
        lag.add(previous, y, w)
        groups += 1
    return groups


def main() -> None:
    args = parse_args()
    if args.spearman_modulo <= 0:
        raise SystemExit("--spearman-modulo must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if (json_path.exists() or md_path.exists()) and not args.force:
        raise SystemExit(f"{json_path} or {md_path} exists; pass --force to overwrite")

    files = train_files(Path(args.data_root))
    if args.smoke_partitions is not None:
        files = files[:args.smoke_partitions]
    if not files:
        raise SystemExit("no training parquet files")

    started = time.perf_counter()
    n_resp = len(RESPONDER_COLUMNS)
    total_rows = 0
    finite_counts = np.zeros(n_resp, dtype=np.int64)
    mins = np.full(n_resp, np.inf)
    maxs = np.full(n_resp, -np.inf)
    responder_target = VectorMoments(n_resp)
    responder_pairs = PairMoments(n_resp)
    partition_corr: list[list[float | None]] = []
    asset_stats = {asset: VectorMoments(n_resp) for asset in range(15)}
    market = VectorMoments(n_resp)
    cross_section = VectorMoments(n_resp)
    lag = VectorMoments(n_resp)
    ic_sum = np.zeros(n_resp); ic_sq = np.zeros(n_resp)
    ic_pos = np.zeros(n_resp, dtype=np.int64); ic_n = np.zeros(n_resp, dtype=np.int64)
    last_by_asset: dict[int, np.ndarray] = {}
    sampled_r: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []
    sampled_w: list[np.ndarray] = []
    sampled_row: list[np.ndarray] = []
    group_count = 0

    carry: dict[str, np.ndarray] | None = None
    columns = [*BASE_COLUMNS, *RESPONDER_COLUMNS]
    for path in files:
        part = VectorMoments(n_resp)
        part_rows = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size, columns=columns):
            frame = batch.to_pandas()
            row_id = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            weight = np.maximum(frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0)
            target = frame["target"].to_numpy(dtype=np.float64, copy=False)
            resp = frame[RESPONDER_COLUMNS].to_numpy(dtype=np.float64, copy=True)
            n = len(frame); total_rows += n; part_rows += n
            finite = np.isfinite(resp)
            finite_counts += finite.sum(axis=0)
            mins = np.minimum(mins, np.min(np.where(finite, resp, np.inf), axis=0))
            maxs = np.maximum(maxs, np.max(np.where(finite, resp, -np.inf), axis=0))
            responder_target.add(resp, target, weight)
            responder_pairs.add(resp, weight)
            part.add(resp, target, weight)
            for asset, stats in asset_stats.items():
                mask = aid == asset
                if mask.any():
                    stats.add(resp[mask], target[mask], weight[mask])
            sample = row_id % args.spearman_modulo == 0
            if sample.any():
                sampled_r.append(resp[sample].astype(np.float32))
                sampled_y.append(target[sample].astype(np.float32))
                sampled_w.append(weight[sample].astype(np.float32))
                sampled_row.append(row_id[sample].copy())

            current = {"time": tid, "asset": aid, "resp": resp, "target": target, "weight": weight}
            if carry is not None:
                for key in current:
                    current[key] = np.concatenate([carry[key], current[key]])
                carry = None
            last_start = int(np.flatnonzero(current["time"] == current["time"][-1])[0])
            if last_start > 0:
                group_count += process_time_groups(
                    current["time"][:last_start], current["asset"][:last_start],
                    current["resp"][:last_start], current["target"][:last_start],
                    current["weight"][:last_start], market, cross_section, ic_sum, ic_sq,
                    ic_pos, ic_n, lag, last_by_asset,
                )
            carry = {key: value[last_start:].copy() for key, value in current.items()}
        partition_corr.append([None if not np.isfinite(x) else float(x) for x in part.corr()])
        print(f"{path.name}: {part_rows:,} rows ({time.perf_counter()-started:.1f}s)", flush=True)

    if carry is not None:
        group_count += process_time_groups(
            carry["time"], carry["asset"], carry["resp"], carry["target"], carry["weight"],
            market, cross_section, ic_sum, ic_sq, ic_pos, ic_n, lag, last_by_asset,
        )

    sample_r = np.concatenate(sampled_r)
    sample_y = np.concatenate(sampled_y)
    sample_w = np.concatenate(sampled_w)
    sample_rows = np.concatenate(sampled_row)
    sample_quantiles = np.nanquantile(sample_r, QUANTILES, axis=0).T
    spearman = np.full(n_resp, np.nan)
    for j in range(n_resp):
        finite = np.isfinite(sample_r[:, j]) & np.isfinite(sample_y)
        if finite.sum() >= 3:
            spearman[j] = np.corrcoef(rankdata(sample_r[finite, j]), rankdata(sample_y[finite]))[0, 1]

    corr_matrix = responder_pairs.corr()
    corr_for_pca = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_for_pca, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_for_pca)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / max(float(eigenvalues.sum()), 1e-30)
    distance = np.clip(1.0 - np.abs(corr_for_pca), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, t=args.cluster_threshold, criterion="distance")

    target_corr = responder_target.corr()
    part_arr = np.array([[np.nan if x is None else x for x in row] for row in partition_corr])
    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == cluster_id)
        representative = int(members[np.nanargmax(np.abs(target_corr[members]))])
        clusters.append({
            "cluster": int(cluster_id),
            "members": [RESPONDER_COLUMNS[i] for i in members],
            "representative": RESPONDER_COLUMNS[representative],
        })

    def json_float(x: float) -> float | None:
        return None if not np.isfinite(x) else float(x)

    columns_payload = []
    market_corr = market.corr(); xs_corr = cross_section.corr(); lag_corr = lag.corr()
    for j, name in enumerate(RESPONDER_COLUMNS):
        ic_mean = ic_sum[j] / ic_n[j] if ic_n[j] else np.nan
        ic_var = ic_sq[j] / ic_n[j] - ic_mean * ic_mean if ic_n[j] else np.nan
        columns_payload.append({
            "name": name,
            "finite_count": int(finite_counts[j]),
            "nonfinite_count": int(total_rows - finite_counts[j]),
            "nonfinite_rate": float((total_rows - finite_counts[j]) / total_rows),
            "min": json_float(mins[j]), "max": json_float(maxs[j]),
            "quantiles": {str(q): json_float(sample_quantiles[j, i]) for i, q in enumerate(QUANTILES)},
            "weighted_mean": json_float(responder_target.sx[j] / responder_target.sw[j]),
            "weighted_std": json_float(math.sqrt(max(responder_target.sxx[j] / responder_target.sw[j] -
                                                       (responder_target.sx[j] / responder_target.sw[j]) ** 2, 0.0))),
            "target_pearson": json_float(target_corr[j]),
            "sample_spearman": json_float(spearman[j]),
            "partition_pearson": [json_float(x) for x in part_arr[:, j]],
            "partition_same_sign": int(np.sum(np.sign(part_arr[:, j]) == np.sign(target_corr[j]))),
            "partition_std": json_float(np.nanstd(part_arr[:, j])),
            "asset_pearson": [json_float(asset_stats[a].corr()[j]) for a in sorted(asset_stats)],
            "market_component_pearson": json_float(market_corr[j]),
            "cross_section_pearson": json_float(xs_corr[j]),
            "cross_section_ic_mean": json_float(ic_mean),
            "cross_section_icir": json_float(ic_mean / math.sqrt(max(ic_var, 0.0))) if ic_var > 0 else None,
            "cross_section_positive_ic_rate": json_float(ic_pos[j] / ic_n[j]) if ic_n[j] else None,
            "cross_section_time_count": int(ic_n[j]),
            "lag1_responder_to_current_target_pearson": json_float(lag_corr[j]),
            "cluster": int(labels[j]),
        })

    elapsed = time.perf_counter() - started
    payload = {
        "question": "What stable contemporaneous structure and redundancy exist among the 47 responder labels?",
        "scope_warning": "This report does not test deployable feature-to-responder predictability and must not claim score improvement.",
        "configuration": vars(args),
        "data_fingerprint": [{"path": str(p.relative_to(Path(args.data_root))), "size": p.stat().st_size,
                              "sha256": sha256_file(p)} for p in files],
        "summary": {
            "rows": int(total_rows), "time_groups": int(group_count),
            "spearman_sample_rows": int(len(sample_r)), "cluster_count": int(len(clusters)),
            "components_for_90pct": int(np.searchsorted(np.cumsum(explained), 0.9) + 1),
        },
        "columns": columns_payload,
        "responder_correlation": corr_matrix.tolist(),
        "pca": {"eigenvalues": eigenvalues.tolist(), "explained_variance": explained.tolist(),
                "cumulative_explained_variance": np.cumsum(explained).tolist(),
                "loadings": eigenvectors.tolist()},
        "clustering": {"distance": "1-abs(weighted Pearson)", "threshold": args.cluster_threshold,
                       "linkage": tree.tolist(), "clusters": clusters},
        "sample": {"rule": f"row_id % {args.spearman_modulo} == 0",
                   "first_row_id": int(sample_rows.min()), "last_row_id": int(sample_rows.max())},
        "verdict": {"status": "descriptive_only", "may_claim_score_gain": False},
        "elapsed_seconds": elapsed,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    ranked = sorted(columns_payload, key=lambda row: abs(row["target_pearson"] or 0.0), reverse=True)[:12]
    lines = [
        f"# Responder 结构分析（`{args.label}`)", "",
        "> 这是同期标签结构报告，不是可部署提分证据。", "",
        f"- 行数：{total_rows:,}", f"- time_id 截面数：{group_count:,}",
        f"- 确定性 Spearman 样本：{len(sample_r):,} 行", f"- 聚类数：{len(clusters)}",
        f"- PCA 达到 90% 累计解释方差：{payload['summary']['components_for_90pct']} 个成分", "",
        "## 与 target 同期关系最强的列", "",
        "| responder | Pearson | Spearman | market | cross-section | IC | cluster |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.4f}"
        lines.append(f"| {row['name']} | {fmt(row['target_pearson'])} | {fmt(row['sample_spearman'])} | "
                     f"{fmt(row['market_component_pearson'])} | {fmt(row['cross_section_pearson'])} | "
                     f"{fmt(row['cross_section_ic_mean'])} | {row['cluster']} |")
    lines += ["", "## 稳定族群", "", "| cluster | representative | members |", "|---:|---|---|"]
    for cluster in clusters:
        lines.append(f"| {cluster['cluster']} | {cluster['representative']} | {', '.join(cluster['members'])} |")
    lines += ["", "## 裁决", "", "阶段 A 只冻结族群与候选代表；是否可预测、是否能补 target 残差由阶段 B/C 决定。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path} and {md_path} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
