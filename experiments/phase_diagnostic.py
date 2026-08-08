"""诊断：sample_modulo 从 10 换到 5 后分数腰斩，是评估集变难还是训练集变差？

背景：同样的 fold 时间范围、同样的 per-row alpha，训练行数翻倍，
      modulo 10 得 0.00133767，modulo 5 只有 0.00071851。

modulo 10 只取 time_id % 10 == 0（记作相位 0），modulo 5 取相位 0 和 5。
把两个变量拆开：

  A. 训练相位 0，     评估相位 0        —— 已知 0.00133767
  B. 训练相位 0+5，   评估相位 0+5      —— 已知 0.00071851
  C. 训练相位 0+5，   评估相位 0 / 相位 5 分开   ←—— 本脚本

若 C 的相位 0 分数 ≈ A：说明训练混相位无害，是相位 5 这批行本身难预测。
若 C 的相位 0 分数 ≈ B：说明训练里混进相位 5 把模型带坏了，与评估集无关。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/phase_diagnostic.py
输出：outputs/experiments/phase_diagnostic.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_time_folds
from train import fit_model, predict_array
from walk_forward_rolling import PROD_SAMPLED_WINDOW, load_all_sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-split diagnostic for sample_modulo.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--phase-period", type=int, default=10,
                        help="按 time_id %% phase_period 给验证行分组报分")
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=0.5)
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading all partitions...", flush=True)
    data = load_all_sampled(train_files(Path(args.data_root)), args.sample_modulo)
    all_time_ids = data["time_id"]
    unique_time_ids = np.unique(all_time_ids)
    train_window = int(len(unique_time_ids) * 4 / 9)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, train_window, args.embargo)
    fold_alpha = args.ridge_alpha * train_window / PROD_SAMPLED_WINDOW
    phases = sorted({int(t) % args.phase_period for t in unique_time_ids[:1000]})
    print(f"{len(all_time_ids):,} rows, {len(unique_time_ids):,} time_ids, "
          f"train_window={train_window:,}, fold_alpha={fold_alpha:,.0f}, 相位={phases}", flush=True)

    results = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        train_set = np.isin(all_time_ids, train_ids)
        valid_set = np.isin(all_time_ids, valid_ids)

        artifact, selected = fit_model(
            data["features"][train_set], data["target"][train_set],
            data["weight"][train_set], data["time_id"][train_set],
            args.feature_count, fold_alpha,
        )
        v_time = data["time_id"][valid_set]
        prediction = predict_array(
            artifact, data["features"][valid_set], v_time, selected,
            args.prediction_scale, args.prediction_clip,
        )
        v_target = data["target"][valid_set]
        v_weight = data["weight"][valid_set]

        row = {"fold": index, "all": float(weighted_zero_mean_r2(v_target, prediction, v_weight))}
        for phase in phases:
            mask = (v_time % args.phase_period) == phase
            row[f"phase{phase}"] = float(
                weighted_zero_mean_r2(v_target[mask], prediction[mask], v_weight[mask])
            )
            row[f"phase{phase}_rows"] = int(mask.sum())
        row["elapsed_seconds"] = float(time.perf_counter() - started)
        results.append(row)
        print(
            f"fold {index:2d}: 全部={row['all']:.8f}  "
            + "  ".join(f"相位{p}={row[f'phase{p}']:.8f}" for p in phases)
            + f"  ({row['elapsed_seconds']:.0f}s)",
            flush=True,
        )
        del artifact, selected, prediction, train_set, valid_set, v_time, v_target, v_weight
        gc.collect()

    summary = {"all": float(np.mean([r["all"] for r in results]))}
    for phase in phases:
        summary[f"phase{phase}"] = float(np.mean([r[f"phase{phase}"] for r in results]))

    payload = {
        "question": "sample_modulo 10→5 分数腰斩：评估集变难 还是 训练集变差？",
        "configuration": {
            "n_folds": args.n_folds, "train_window": train_window, "embargo": args.embargo,
            "sample_modulo": args.sample_modulo, "phase_period": args.phase_period,
            "feature_count": args.feature_count, "ridge_alpha": fold_alpha,
            "prediction_scale": args.prediction_scale,
        },
        "reference": {
            "modulo10_train_phase0_eval_phase0": 0.00133767,
            "modulo5_train_both_eval_both": 0.00071851,
        },
        "summary": summary,
        "folds": results,
    }
    (output_dir / "phase_diagnostic.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 相位诊断：sample_modulo 10→5 分数腰斩的原因",
        "",
        f"训练用 modulo {args.sample_modulo}（含相位 {phases}），验证段按 time_id % "
        f"{args.phase_period} 拆开报分。",
        "",
        "| Fold | 全部 | " + " | ".join(f"相位 {p}" for p in phases) + " |",
        "|---:|---:|" + "---:|" * len(phases),
    ]
    for r in results:
        lines.append(
            f"| {r['fold']} | {r['all']:.8f} | "
            + " | ".join(f"{r[f'phase{p}']:.8f}" for p in phases) + " |"
        )
    lines.append(
        f"| **mean** | **{summary['all']:.8f}** | "
        + " | ".join(f"**{summary[f'phase{p}']:.8f}**" for p in phases) + " |"
    )
    lines += [
        "",
        "## 对照",
        "",
        "| 场景 | 分数 |",
        "|---|---:|",
        "| A 训练相位 0，评估相位 0（modulo 10） | 0.00133767 |",
        "| B 训练相位 0+5，评估相位 0+5（modulo 5） | 0.00071851 |",
        f"| C 训练相位 0+5，评估**只看相位 0** | {summary.get('phase0', float('nan')):.8f} |",
        "",
        "读法：C ≈ A → 训练混相位无害，是相位 5 那批行本身难预测；",
        "C ≈ B → 训练里混进相位 5 把模型带坏了，与评估集无关。",
    ]
    (output_dir / "phase_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
