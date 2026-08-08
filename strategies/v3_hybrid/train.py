"""v3_hybrid 训练：只训 LightGBM 的截面分量，岭回归部分冻结复用生产模型。

## 为什么岭回归不重训

`model/baseline_model.json` 是 `strategies/v1_ridge/` 的生产模型原样拷贝
（sha `c23a8cfb`，公榜实测 0.00187232）。冻结它有两个好处：

1. **唯一的变量是截面分量** —— 与公榜 0.00187232 对比时，差异纯粹来自 LightGBM
2. 那份模型的可复现性已经验收过（严格求解器，7 道门禁）

所以本脚本只产出 LightGBM 部分 + `hybrid_meta.json`。

## 超参从哪来（全部预注册，不在这里搜）

来自 `lgbm_blend_unweighted`（5 折、3 种子、无泄漏口径、判据由 `verdict_of()` 机器判）：

- 候选 `xs_loose`：`num_leaves 63 / lr 0.03 / feature_fraction 0.7 / lambda_l2 1.0`，
  `min_data_in_leaf` 取训练行数的 12000/3.5e6 ≈ 0.343%
- 轮数取**逐折 best_iteration 的折均 160**（逐折范围 44~302）——
  照增补包的做法，不在全量上再搜一次
- 混合 `blend50`：折后 +11.1%，四个臂里最高（`replace` 原始分更高但 ΔB 涨 17.9%，折后只剩 +9.8%）

## 口径

目标 `e = y − 无权截面均值(y)`。**无权**是因为推理端拿不到 weight
（test 无该列，且 runner 的 `forbidden` 会剥掉），详见 `main.py` 的模块注释。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python strategies/v3_hybrid/train.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments"),
              str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train import robust_transform_fit, select_features        # v1_ridge 的生产实现
from features import apply_robust_transform, cross_sectional_deviation
from lgbm_xs import load_rows                                   # 流式加载（带 asset_id）

# 全部来自 lgbm_blend_unweighted，预注册、不在这里搜
SPEC = {"num_leaves": 63, "learning_rate": 0.03,
        "feature_fraction": 0.7, "lambda_l2": 1.0}
MIN_DATA_FRAC = 12000 / 3_500_000        # lgbm_blend 里 min_data_in_leaf 与行数的比例
NUM_ITERATION = 160                       # 逐折 best_iteration 的折均（范围 44~302）
BLEND_WEIGHT = 0.5                        # blend50：ê 里 LGBM 占一半，不拟合权重


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LightGBM cross-sectional part of v3_hybrid.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", default="phase_balanced")
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--num-iteration", type=int, default=NUM_ITERATION)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=16)
    # scale 是纯后处理旋钮，最终值由公榜两点法定；这里先写本地最优做占位
    p.add_argument("--prediction-scale", type=float, default=0.856)
    p.add_argument("--prediction-clip", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    assert (model_dir / "baseline_model.json").exists(), \
        "model/baseline_model.json 缺失 —— 它是 v1_ridge 生产模型的冻结拷贝"

    print(f"加载训练数据（modulo {args.sample_modulo} / {args.sampling}）…", flush=True)
    d = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features, y, w, tid, aid = d["features"], d["target"], d["weight"], d["time_id"], d["asset_id"]
    del d, w                                   # 权重只有训练端有，这里刻意不用（见模块注释）
    assert np.all(np.diff(tid) >= 0), "行未按 time_id 排序，截面聚合会算错"

    # 目标：无权截面残差
    starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(tid)])
    e = y - np.repeat(np.add.reduceat(y, starts) / counts, counts)
    resid = float(np.abs(np.add.reduceat(e, starts)).max())
    assert resid < 1e-8, f"逐 time_id 的 e 之和不为 0（{resid:.2e}）"
    print(f"{len(tid):,} 行 / {len(starts):,} 个 time_id；截面残差校验 {resid:.2e} ✅", flush=True)

    # 预处理与选列（全量拟合 —— 这是最终模型，没有留出段）
    scratch, stats = robust_transform_fit(features.copy())
    selected = select_features(scratch, e, np.ones_like(e), args.feature_count)
    del scratch
    lgbm_features = [f"feature_{i:03d}" for i in selected]

    raw = features[:, selected].copy()
    del features
    apply_robust_transform(raw, stats["lower"][selected], stats["upper"][selected],
                           stats["center"][selected], stats["scale"][selected])
    dev = cross_sectional_deviation(raw, tid)
    del raw
    design = np.ascontiguousarray(np.column_stack([dev, aid.astype(np.float32)]))
    del dev
    cat = design.shape[1] - 1
    min_data = max(20, int(round(MIN_DATA_FRAC * len(design))))
    print(f"设计矩阵 {design.shape[0]:,} × {design.shape[1]}，"
          f"min_data_in_leaf={min_data:,}，{args.n_seeds} 种子 × {args.num_iteration} 轮", flush=True)

    model_files = []
    for s in range(args.n_seeds):
        params = {
            **SPEC, "objective": "regression", "metric": "l2", "verbosity": -1,
            "num_threads": args.num_threads, "min_data_in_leaf": min_data,
            "bagging_fraction": 0.7, "bagging_freq": 1,
            "deterministic": True, "force_row_wise": True, "feature_pre_filter": False,
            "seed": args.seed + s, "bagging_seed": args.seed + 1000 + s,
            "feature_fraction_seed": args.seed + 2000 + s,
        }
        t0 = time.perf_counter()
        ds = lgb.Dataset(design, label=e, params=params,
                         categorical_feature=[cat], free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=args.num_iteration)
        name = f"lgbm_seed{args.seed + s}.txt"
        booster.save_model(str(model_dir / name), num_iteration=args.num_iteration)
        model_files.append(name)
        print(f"  种子 {args.seed + s}: {time.perf_counter()-t0:.0f}s → {name}", flush=True)
        del booster, ds

    meta = {
        "strategy": "v3_hybrid_ridge_plus_lgbm_cross_section",
        "blend_weight": BLEND_WEIGHT,
        "blend_note": "最终 ê = (1−w)·ê_ridge + w·ê_lgbm，w=0.5 是先验、不拟合（ROADMAP §5）",
        "num_iteration": args.num_iteration,
        "num_iteration_source": "lgbm_blend_unweighted 逐折 best_iteration 的折均（范围 44~302）",
        "lgbm_model_files": model_files,
        "lgbm_features": lgbm_features,
        "lower": stats["lower"][selected].tolist(),
        "upper": stats["upper"][selected].tolist(),
        "center": stats["center"][selected].tolist(),
        "scale": stats["scale"][selected].tolist(),
        "prediction_scale": args.prediction_scale,
        "prediction_clip": args.prediction_clip,
        "scale_note": "scale 是纯后处理旋钮；此处为本地最优占位，最终值由公榜两点法定",
        "cross_sectional_mean": "unweighted",
        "cross_sectional_mean_note":
            "推理端拿不到 weight（test 无该列且 runner 的 forbidden 会剥掉），"
            "所以拆解、投影、训练目标一律无权。已在 lgbm_blend_unweighted 上重测，PASS 成立。",
        "lgbm_params": {**SPEC, "min_data_in_leaf": min_data,
                        "bagging_fraction": 0.7, "bagging_freq": 1},
        "train_rows": int(len(design)),
        "sample_modulo": args.sample_modulo,
        "sampling": args.sampling,
        "ridge_model_sha_note": "baseline_model.json 是 v1_ridge 生产模型的冻结拷贝，不重训",
    }
    (model_dir / "hybrid_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {model_dir/'hybrid_meta.json'}")
    print(f"模型文件：{model_files}")


if __name__ == "__main__":
    main()
