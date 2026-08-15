"""Responder 重新审计：严格 OOF 预测 → 固定过去窗口上的非线性二层门禁。

旧的 responder 结论主要来自线性换目标/线性残差叠加。本脚本回答一个更窄、但此前没有回答的
问题：**若先用可见 feature 严格样本外地预测 target 和全部 responder，非线性二层模型能否从
responder 预测中获得稳定的 target 增量？**

为防止二层模型逐折重训后把 regime 漂移误当成 responder 收益，正式裁决使用固定的历史校准窗：
只在最早的 OOF fold 上训练二层模型，冻结后评估后续四折。每种容量同时训练 target-only 对照和
[target + 47 responder] 候选，二者的差才是 responder 增量。

默认口径是生产等效的 modulo 5 / phase_balanced / 78,960 sampled time_ids。缓存阶段选用全部
323 个 feature，避免旧 Stage-B 的 target-only top-200 选列成为先验过滤器。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    .venv/bin/python experiments/responder_nonlinear_reaudit.py --stage all --force

输出：
    outputs/cache/responder_nonlinear_reaudit_oof.npz
    outputs/experiments/responder_nonlinear_reaudit.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "experiments", ROOT / "strategies" / "v1_ridge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from responder_predictability import fit_multi_ridge, predict_multi
from responder_targets import RESPONDER_COLUMNS, load_rows_with_responders
from src.metric import scale_invariant_score
from src.validation import rolling_time_folds

SPECS: dict[str, dict[str, float | int]] = {
    "strong": {
        "max_leaf_nodes": 7,
        "min_samples_leaf": 1000,
        "l2_regularization": 30.0,
        "learning_rate": 0.04,
        "max_iter": 120,
    },
    "current": {
        "max_leaf_nodes": 15,
        "min_samples_leaf": 500,
        "l2_regularization": 10.0,
        "learning_rate": 0.05,
        "max_iter": 150,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict nonlinear responder re-audit.")
    parser.add_argument("--stage", choices=["cache", "audit", "all"], default="all")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--cache", default=str(ROOT / "outputs" / "cache" /
                                                "responder_nonlinear_reaudit_oof.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--label", default="responder_nonlinear_reaudit")
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--sampling", choices=["periodic", "phase_balanced"],
                        default="phase_balanced")
    parser.add_argument("--train-window", type=int, default=78_960)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--feature-count", type=int, default=323)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_cache(args: argparse.Namespace, cache_path: Path) -> dict[str, Any]:
    if cache_path.exists() and not args.force:
        raise SystemExit(f"cache exists: {cache_path}; pass --force")
    started = time.perf_counter()
    data = load_rows_with_responders(Path(args.data_root), args.sample_modulo, args.sampling)
    complete = np.all(np.isfinite(data["responders"]), axis=1)
    original_index = np.flatnonzero(complete).astype(np.int64)
    for key in data:
        data[key] = data[key][complete]

    time_ids = data["time_id"]
    folds = rolling_time_folds(np.unique(time_ids), args.n_folds, args.train_window, args.embargo)
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("row_index", "target", "weight", "time_id", "asset_id",
                            "prediction_target", "prediction_responders", "fold")
    }
    for fold_index, (train_ids, valid_ids) in enumerate(folds):
        fold_started = time.perf_counter()
        train = np.isin(time_ids, train_ids)
        valid = np.isin(time_ids, valid_ids)
        targets = np.column_stack([data["target"][train], data["responders"][train]])
        model = fit_multi_ridge(
            data["features"][train], time_ids[train], data["target"][train], targets,
            np.maximum(data["weight"][train], 0.0), args.feature_count, args.ridge_alpha,
        )
        prediction = predict_multi(model, data["features"][valid], time_ids[valid]).astype(np.float32)
        valid_index = np.flatnonzero(valid)
        parts["row_index"].append(original_index[valid_index])
        parts["target"].append(data["target"][valid].astype(np.float32))
        parts["weight"].append(data["weight"][valid].astype(np.float32))
        parts["time_id"].append(time_ids[valid].astype(np.int64))
        parts["asset_id"].append(data["asset_id"][valid].astype(np.int16))
        parts["prediction_target"].append(prediction[:, 0])
        parts["prediction_responders"].append(prediction[:, 1:])
        parts["fold"].append(np.full(int(valid.sum()), fold_index, dtype=np.int8))
        print(f"fold {fold_index}: {int(valid.sum()):,} rows, "
              f"{time.perf_counter()-fold_started:.1f}s", flush=True)
        del model, prediction, targets
        gc.collect()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{key: np.concatenate(value) for key, value in parts.items()},
        responder_names=np.asarray(RESPONDER_COLUMNS),
    )
    return {
        "complete_case_rows": int(complete.sum()),
        "excluded_rows": int((~complete).sum()),
        "oof_rows": int(sum(len(value) for value in parts["target"])),
        "cache_bytes": int(cache_path.stat().st_size),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def estimator(spec: dict[str, float | int], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        early_stopping=False,
        random_state=seed,
    )


def paired_summary(target_only: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = candidate - target_only
    drop_best = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    checks = {
        "mean_increment_positive": float(delta.mean()) > 0.0,
        "positive_at_least_3of4": int((delta > 0).sum()) >= 3,
        "survives_drop_best": float(drop_best.mean()) > 0.0,
        "relative_increment_at_least_3pct": float(delta.mean() / target_only.mean()) >= 0.03,
    }
    return {
        "target_only_peak_mean": float(target_only.mean()),
        "with_responders_peak_mean": float(candidate.mean()),
        "responder_relative_gain": float(delta.mean() / target_only.mean()),
        "positive_folds": int((delta > 0).sum()),
        "drop_best_relative": float(drop_best.mean() / target_only.mean()),
        "per_fold_delta": delta.tolist(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_cache(args: argparse.Namespace, cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as cache:
        fold = cache["fold"].astype(np.int64)
        target = cache["target"].astype(np.float64)
        weight = cache["weight"].astype(np.float64)
        target_prediction = cache["prediction_target"].astype(np.float32)
        responder_prediction = cache["prediction_responders"].astype(np.float32)
        asset_id = cache["asset_id"].astype(np.int64)

    unique_folds = sorted(set(fold.tolist()))
    if len(unique_folds) < 2:
        raise SystemExit("cache must contain at least two OOF folds")
    calibration_fold = unique_folds[0]
    calibration = fold == calibration_fold
    asset_one_hot = np.eye(15, dtype=np.float32)[asset_id]
    target_design = np.column_stack([target_prediction, asset_one_hot])
    responder_design = np.column_stack([target_prediction, responder_prediction, asset_one_hot])

    models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
    for index, (name, spec) in enumerate(SPECS.items()):
        models[name] = {
            "target_only": estimator(spec, 2026 + index).fit(
                target_design[calibration], target[calibration],
                sample_weight=np.maximum(weight[calibration], 0.0),
            ),
            "with_responders": estimator(spec, 3026 + index).fit(
                responder_design[calibration], target[calibration],
                sample_weight=np.maximum(weight[calibration], 0.0),
            ),
        }

    rows: list[dict[str, Any]] = []
    for fold_index in unique_folds[1:]:
        valid = fold == fold_index
        row: dict[str, Any] = {"fold": int(fold_index), "arms": {}}
        for name, pair in models.items():
            target_only = pair["target_only"].predict(target_design[valid])
            with_responders = pair["with_responders"].predict(responder_design[valid])
            row["arms"][name] = {
                "target_only": scale_invariant_score(target[valid], target_only, weight[valid]),
                "with_responders": scale_invariant_score(target[valid], with_responders, weight[valid]),
            }
        rows.append(row)
        print(f"audit fold {fold_index}: " + "  ".join(
            f"{name} {entry['target_only']['peak']:.8f}->{entry['with_responders']['peak']:.8f}"
            for name, entry in row["arms"].items()), flush=True)

    summary = {}
    for name in SPECS:
        target_only = np.asarray([row["arms"][name]["target_only"]["peak"] for row in rows])
        candidate = np.asarray([row["arms"][name]["with_responders"]["peak"] for row in rows])
        summary[name] = paired_summary(target_only, candidate)
    return {
        "method": (
            "all 323 features -> strict multi-target Ridge OOF predictions; nonlinear level-2 models "
            "fit once on the earliest OOF fold and frozen for all later folds"
        ),
        "calibration_fold": int(calibration_fold),
        "evaluation_folds": [int(value) for value in unique_folds[1:]],
        "specs": SPECS,
        "folds": rows,
        "summary": summary,
        "verdict": {
            "pass": any(value["pass"] for value in summary.values()),
            "next": "confirm against strong v3 OOF" if any(value["pass"] for value in summary.values())
                    else "do not open responder multi-task training",
        },
    }


def render(payload: dict[str, Any]) -> str:
    cfg = payload["configuration"]
    lines = [
        "# Responder 非线性重新审计",
        "",
        "**问题**：严格 OOF 的 responder 预测，能否在同容量 target-only 非线性对照之上提供稳定增量？",
        "",
        f"- 采样：modulo {cfg['sample_modulo']} / {cfg['sampling']}",
        f"- 训练窗口：{cfg['train_window']:,} sampled time_ids；embargo {cfg['embargo']}",
        f"- 一级模型：全部 {cfg['feature_count']} 个特征，target + 47 responder 共享 Ridge 设计",
        f"- 二级模型：只在 OOF fold {payload['audit']['calibration_fold']} 训练，冻结后评估 "
        f"fold {payload['audit']['evaluation_folds']}",
        "",
        "| capacity | target-only peak | +responder peak | relative | +folds | drop best | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, row in payload["audit"]["summary"].items():
        lines.append(
            f"| `{name}` | {row['target_only_peak_mean']:.8f} | "
            f"{row['with_responders_peak_mean']:.8f} | {row['responder_relative_gain']*100:+.2f}% | "
            f"{row['positive_folds']}/4 | {row['drop_best_relative']*100:+.2f}% | "
            f"{'✅' if row['pass'] else '❌'} |"
        )
    lines += [
        "",
        f"**{'PASS' if payload['audit']['verdict']['pass'] else 'STOP'}** — "
        f"{payload['audit']['verdict']['next']}",
        "",
        "限制：本门禁的 target-only 对照仍是 Ridge OOF 的二层校准，不等同于当前生产 v3。"
        "只有与强 v3 OOF 配对后仍增益，才允许进入 GPU 多任务模型。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache)
    cache_info = None
    if args.stage in {"cache", "all"}:
        cache_info = build_cache(args, cache_path)
        print(json.dumps(cache_info, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "cache":
        return
    if not cache_path.exists():
        raise SystemExit(f"cache not found: {cache_path}")
    audit = audit_cache(args, cache_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "Do nonlinear combinations of strict OOF responder predictions add target signal?",
        "configuration": vars(args),
        "cache": {"path": str(cache_path), "bytes": int(cache_path.stat().st_size),
                  "build": cache_info},
        "audit": audit,
    }
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path} / {md_path}; pass --force")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(payload), encoding="utf-8")
    print(render(payload), flush=True)


if __name__ == "__main__":
    main()
