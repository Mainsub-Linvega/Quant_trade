"""同预算比较周期采样与全相位平衡采样的 Ridge 泛化。

训练仍只保留约 1/sample_modulo 的完整 time_id 截面；验证流式读取整个未来分区，
覆盖全部相位。输出固定 scale 分数以及不受 scale 影响的 A/B/峰值，不写正式模型。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, load_time_sample, train_files
from src.metric import weighted_zero_mean_r2_from_sums
from train import fit_model, predict_array

ARMS = ("periodic", "phase_balanced")
VALID_COLUMNS = ["time_id", "weight", *FEATURE_COLUMNS, "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare periodic and phase-balanced Ridge sampling.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="ridge_phase_sampling")
    parser.add_argument("--validation-partitions", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--train-partitions", type=int, default=4)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--phase-period", type=int, default=10)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--ridge-tol", type=float, default=1e-8)
    parser.add_argument("--ridge-max-iter", type=int, default=2000)
    parser.add_argument("--prediction-scale", type=float, default=1.13)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=30_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_complete_time_batches(path: Path, batch_size: int):
    carry: pd.DataFrame | None = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=VALID_COLUMNS):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last_time = int(frame["time_id"].iloc[-1])
        last_mask = frame["time_id"].to_numpy(copy=False) == last_time
        complete = frame.loc[~last_mask]
        carry = frame.loc[last_mask].copy()
        if not complete.empty:
            yield complete
    if carry is not None and not carry.empty:
        yield carry


def stream_evaluate(
    path: Path,
    artifact: dict[str, object],
    selected: np.ndarray,
    scale: float,
    clip: float,
    phase_period: int,
    batch_size: int,
) -> dict[str, object]:
    denominator = cross = prediction_energy = fixed_sse = 0.0
    rows = 0
    phase_den = np.zeros(phase_period, dtype=np.float64)
    phase_sse = np.zeros(phase_period, dtype=np.float64)
    phase_rows = np.zeros(phase_period, dtype=np.int64)

    for frame in iter_complete_time_batches(path, batch_size):
        features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        time_ids = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
        target = frame["target"].to_numpy(dtype=np.float64, copy=False)
        weight = np.maximum(frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0)
        raw_prediction = predict_array(artifact, features, time_ids, selected, 1.0, 1e9).astype(np.float64)
        prediction = np.clip(raw_prediction * scale, -clip, clip)

        denominator += float(np.dot(weight, target * target))
        cross += float(np.dot(weight, target * raw_prediction))
        prediction_energy += float(np.dot(weight, raw_prediction * raw_prediction))
        fixed_sse += float(np.dot(weight, (target - prediction) ** 2))
        rows += len(frame)

        phases = time_ids % phase_period
        for phase in np.unique(phases):
            mask = phases == phase
            phase_den[phase] += float(np.dot(weight[mask], target[mask] * target[mask]))
            phase_sse[phase] += float(np.dot(weight[mask], (target[mask] - prediction[mask]) ** 2))
            phase_rows[phase] += int(mask.sum())

    if denominator <= 0.0:
        return {
            "rows": rows,
            "score_denominator": denominator,
            "fixed_scale_score": 0.0,
            "A": 0.0,
            "B": 0.0,
            "optimal_scale_unclipped": 0.0,
            "peak_score_unclipped": 0.0,
            "phase_scores": {},
            "phase_rows": {str(phase): int(phase_rows[phase]) for phase in range(phase_period)},
        }
    a_value = cross / denominator
    b_value = prediction_energy / denominator
    optimal_scale = a_value / b_value if b_value > 0 else 0.0
    return {
        "rows": rows,
        "score_denominator": denominator,
        "fixed_scale_score": weighted_zero_mean_r2_from_sums(fixed_sse, denominator),
        "A": a_value,
        "B": b_value,
        "optimal_scale_unclipped": optimal_scale,
        "peak_score_unclipped": a_value * a_value / b_value if b_value > 0 else 0.0,
        "phase_scores": {
            str(phase): weighted_zero_mean_r2_from_sums(phase_sse[phase], phase_den[phase])
            for phase in range(phase_period) if phase_den[phase] > 0
        },
        "phase_rows": {str(phase): int(phase_rows[phase]) for phase in range(phase_period)},
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{args.label} 已存在；换 --label 或加 --force")

    files = train_files(Path(args.data_root))
    results: list[dict[str, object]] = []
    for valid_index in args.validation_partitions:
        train_indices = list(range(valid_index - args.train_partitions, valid_index))
        if train_indices[0] < 0 or valid_index >= len(files):
            raise ValueError("fold configuration does not fit available partitions")
        fold: dict[str, object] = {
            "valid_partition": valid_index,
            "train_partitions": train_indices,
            "arms": {},
        }
        selected_by_arm: dict[str, set[int]] = {}
        for arm in ARMS:
            started = time.perf_counter()
            x_train, y_train, w_train, t_train = load_time_sample(
                [files[index] for index in train_indices],
                args.sample_modulo,
                sampling=arm,
                phase_period=args.phase_period,
            )
            train_summary = {
                "rows": int(len(y_train)),
                "time_ids": int(len(np.unique(t_train))),
                "nonnegative_weight_sum": float(np.maximum(w_train.astype(np.float64), 0.0).sum()),
                "phase_time_ids": {
                    str(phase): int(np.unique(t_train[t_train % args.phase_period == phase]).size)
                    for phase in range(args.phase_period)
                },
            }
            artifact, selected = fit_model(
                x_train, y_train, w_train, t_train,
                args.feature_count, args.ridge_alpha,
                ridge_tol=args.ridge_tol,
                ridge_max_iter=args.ridge_max_iter,
            )
            del x_train, y_train, w_train, t_train
            gc.collect()
            evaluation = stream_evaluate(
                files[valid_index], artifact, selected,
                args.prediction_scale, args.prediction_clip,
                args.phase_period, args.batch_size,
            )
            selected_by_arm[arm] = set(map(int, selected))
            fold["arms"][arm] = {
                "train": train_summary,
                "ridge_n_iter": int(artifact["ridge_n_iter"]),
                "selected_features": [int(value) for value in selected],
                "evaluation": evaluation,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            print(
                f"p{valid_index:03d} {arm}: fixed={evaluation['fixed_scale_score']:.8f} "
                f"peak={evaluation['peak_score_unclipped']:.8f} "
                f"a*={evaluation['optimal_scale_unclipped']:.4f}",
                flush=True,
            )
            del artifact, selected
            gc.collect()
        overlap = selected_by_arm["periodic"] & selected_by_arm["phase_balanced"]
        union = selected_by_arm["periodic"] | selected_by_arm["phase_balanced"]
        fold["selected_jaccard"] = len(overlap) / len(union)
        results.append(fold)

    def values(arm: str, key: str) -> np.ndarray:
        return np.asarray([fold["arms"][arm]["evaluation"][key] for fold in results], dtype=np.float64)

    summary: dict[str, object] = {"arms": {}}
    for arm in ARMS:
        fixed = values(arm, "fixed_scale_score")
        peak = values(arm, "peak_score_unclipped")
        denominators = values(arm, "score_denominator")
        summary["arms"][arm] = {
            "fixed_macro_mean": float(fixed.mean()),
            "fixed_pooled": float(np.average(fixed, weights=denominators)),
            "peak_macro_mean": float(peak.mean()),
            "positive_folds": int((fixed > 0).sum()),
        }
    fixed_delta = values("phase_balanced", "fixed_scale_score") - values("periodic", "fixed_scale_score")
    peak_delta = values("phase_balanced", "peak_score_unclipped") - values("periodic", "peak_score_unclipped")
    periodic_denominators = values("periodic", "score_denominator")
    balanced_denominators = values("phase_balanced", "score_denominator")
    if not np.array_equal(periodic_denominators, balanced_denominators):
        raise AssertionError("both arms must use identical validation denominators")
    pooled_fixed_delta = float(np.average(fixed_delta, weights=periodic_denominators))
    summary["paired_delta"] = {
        "fixed_by_fold": fixed_delta.tolist(),
        "fixed_mean": float(fixed_delta.mean()),
        "fixed_pooled": pooled_fixed_delta,
        "fixed_positive_folds": int((fixed_delta > 0).sum()),
        "peak_by_fold": peak_delta.tolist(),
        "peak_mean": float(peak_delta.mean()),
        "peak_positive_folds": int((peak_delta > 0).sum()),
        "selected_jaccard_mean": float(np.mean([fold["selected_jaccard"] for fold in results])),
    }

    payload = {
        "hypothesis": "same-budget phase-balanced training improves full-phase future validation",
        "configuration": vars(args),
        "sampling": {
            "periodic": "time_id % sample_modulo == 0",
            "phase_balanced": "((time_id // phase_period) + time_id % phase_period) % sample_modulo == 0",
        },
        "summary": summary,
        "folds": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Ridge 同预算全相位采样实验", "",
        "训练保留完整 time_id 截面；验证读取整个未来分区（全部相位）。", "",
        "| Fold | periodic fixed | balanced fixed | Δ fixed | periodic peak | balanced peak | Δ peak | selected Jaccard |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in results:
        p = fold["arms"]["periodic"]["evaluation"]
        b = fold["arms"]["phase_balanced"]["evaluation"]
        lines.append(
            f"| p{fold['valid_partition']:03d} | {p['fixed_scale_score']:.8f} | {b['fixed_scale_score']:.8f} | "
            f"{b['fixed_scale_score']-p['fixed_scale_score']:+.3e} | {p['peak_score_unclipped']:.8f} | "
            f"{b['peak_score_unclipped']:.8f} | {b['peak_score_unclipped']-p['peak_score_unclipped']:+.3e} | "
            f"{fold['selected_jaccard']:.3f} |"
        )
    delta = summary["paired_delta"]
    lines += [
        "",
        f"固定 scale pooled(Δ)={delta['fixed_pooled']:+.3e}；macro mean(Δ)={delta['fixed_mean']:+.3e}，"
        f"正折 {delta['fixed_positive_folds']}/{len(results)}。",
        f"逐折 oracle 峰值仅作幅度诊断：mean(Δ)={delta['peak_mean']:+.3e}，"
        f"正折 {delta['peak_positive_folds']}/{len(results)}。",
        f"选中特征 Jaccard 均值={delta['selected_jaccard_mean']:.3f}。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
