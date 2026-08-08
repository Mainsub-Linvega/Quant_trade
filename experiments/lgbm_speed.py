"""LightGBM 训练提速基准 —— 在**真实设计矩阵**上量，不靠猜。

## 为什么要这个脚本

合成随机数据上测过一轮（20 万行 × 200 列 / 50 轮）：

| 配置 | 耗时 |
|---|---:|
| CPU（`deterministic` + `force_row_wise`） | 3.1s |
| CPU 去掉 `deterministic` | 2.2s |
| GPU（OpenCL）+ `deterministic` | 2.2s |
| GPU 裸跑 | 2.0s |

但**合成数据没有真实结构、规模也只有实际的 1/5**，分箱边界与分裂行为都不一样。
这里在真实的截面块设计矩阵上重量一次。

## 判据：只快不准没有意义

每个配置除了报耗时，**必须报预测与基线的差异**。
`max|Δpred| / std(pred) < 1e-6` 才算「数值等价、可以换」；
否则这个配置只能用来探路，**不能用于出结论的实验**
（否则同一条研究线里混进两套数值口径，之前的结论就没法对读了）。

## 一次只变一个因子

基线 = 现在 `lgbm_blend.py` 的实际设置：
`device=cpu, deterministic=True, force_row_wise=True, num_threads=16, max_bin=255`，
每个种子重建一次 Dataset。然后逐个变量单独改。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/lgbm_speed.py
输出：outputs/experiments/lgbm_speed.{json,md}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.validation import rolling_time_folds
from train import robust_transform_fit, select_features
from features import apply_robust_transform, cross_sectional_deviation
from lgbm_xs import load_rows, cross_sectional_target

# 与 lgbm_blend 的 xs_shrunk 同一组超参（min_data_in_leaf 按行数比例定）
SPEC = {"num_leaves": 15, "min_data_frac": 100000 / 3_500_000,
        "learning_rate": 0.02, "feature_fraction": 0.4, "lambda_l2": 30.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LightGBM training on the real design matrix.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--report", default="lgbm_speed")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", default="phase_balanced")
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--inner-frac", type=float, default=0.10)
    p.add_argument("--rounds", type=int, default=300,
                   help="固定轮数（不早停）—— 各配置必须做等量的功才可比")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.report}.json", out_dir / f"{args.report}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"报告已存在：{json_path}。要覆盖请显式加 --force")

    print(f"加载真实数据（modulo {args.sample_modulo} / {args.sampling}）…", flush=True)
    d = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features, y, w, tid, aid = d["features"], d["target"], d["weight"], d["time_id"], d["asset_id"]
    del d
    e, _ = cross_sectional_target(y, w, tid)
    uniq = tid[np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]]

    # 取第一折，构造与 lgbm_blend 完全相同的设计矩阵
    train_ids, _valid_ids = rolling_time_folds(
        uniq, args.n_folds, int(len(uniq) * 4 / 9), args.embargo)[0]
    n_tr = len(train_ids)
    n_inner = max(1, int(n_tr * args.inner_frac))
    it_ids = train_ids[: n_tr - n_inner - args.embargo]
    iv_ids = train_ids[n_tr - n_inner:]

    def rows_of(ids):
        return np.arange(int(np.searchsorted(tid, ids[0], "left")),
                         int(np.searchsorted(tid, ids[-1], "right")))

    it_rows, iv_rows = rows_of(it_ids), rows_of(iv_ids)
    scratch, stats = robust_transform_fit(features[it_rows].copy())
    selected = select_features(scratch, y[it_rows], w[it_rows], args.feature_count)
    del scratch
    raw = features[:, selected].copy()
    apply_robust_transform(raw, stats["lower"][selected], stats["upper"][selected],
                           stats["center"][selected], stats["scale"][selected])
    dev = cross_sectional_deviation(raw, tid)
    del raw, features
    x_it = np.ascontiguousarray(np.column_stack([dev[it_rows], aid[it_rows].astype(np.float32)]))
    x_iv = np.ascontiguousarray(np.column_stack([dev[iv_rows], aid[iv_rows].astype(np.float32)]))
    del dev
    y_it, w_it = e[it_rows], w[it_rows]
    cat = x_it.shape[1] - 1
    min_data = max(20, int(round(SPEC["min_data_frac"] * len(it_rows))))
    print(f"训练矩阵 {x_it.shape[0]:,} 行 × {x_it.shape[1]} 列，"
          f"min_data_in_leaf={min_data:,}，固定 {args.rounds} 轮 × {args.n_seeds} 种子\n", flush=True)

    base_params = {k: v for k, v in SPEC.items() if k != "min_data_frac"}
    base_params.update({"objective": "regression", "metric": "l2", "verbosity": -1,
                        "bagging_fraction": 0.7, "bagging_freq": 1,
                        "feature_pre_filter": False, "min_data_in_leaf": min_data})

    def run(tag: str, *, device="cpu", threads=16, max_bin=255,
            deterministic=True, reuse_dataset=False) -> dict[str, Any]:
        """跑一个配置，返回耗时与 3 种子平均预测。"""
        p = {**base_params, "num_threads": threads, "device_type": device,
             "max_bin": max_bin, "deterministic": deterministic}
        if deterministic:
            p["force_row_wise"] = True
        acc = np.zeros(len(x_iv))
        t0 = time.perf_counter()
        shared = (lgb.Dataset(x_it, label=y_it, weight=w_it, params=p,
                              categorical_feature=[cat], free_raw_data=False)
                  if reuse_dataset else None)
        for s in range(args.n_seeds):
            pp = {**p, "seed": args.seed + s, "bagging_seed": args.seed + 1000 + s,
                  "feature_fraction_seed": args.seed + 2000 + s}
            ds = shared if reuse_dataset else lgb.Dataset(
                x_it, label=y_it, weight=w_it, params=pp,
                categorical_feature=[cat], free_raw_data=False)
            b = lgb.train(pp, ds, num_boost_round=args.rounds)
            acc += b.predict(x_iv)
            del b
        secs = time.perf_counter() - t0
        del shared
        return {"tag": tag, "seconds": float(secs), "pred": acc / float(args.n_seeds),
                "device": device, "threads": threads, "max_bin": max_bin,
                "deterministic": deterministic, "reuse_dataset": reuse_dataset}

    # 一次只变一个因子，最后一个是组合
    plan = [
        ("baseline (cpu/16线程/max_bin255/每种子重建)", {}),
        ("Dataset 每候选建一次",                        {"reuse_dataset": True}),
        ("GPU (OpenCL)",                                {"device": "gpu"}),
        ("8 线程",                                      {"threads": 8}),
        ("32 线程",                                     {"threads": 32}),
        ("max_bin=63 ⚠️会改变模型",                     {"max_bin": 63}),
        ("组合：GPU + Dataset 复用",                    {"device": "gpu", "reuse_dataset": True}),
    ]

    rows: list[dict[str, Any]] = []
    base_pred = base_secs = None
    for tag, kw in plan:
        try:
            r = run(tag, **kw)
        except Exception as ex:                       # GPU 不可用等
            print(f"  {tag:38s} ❌ {str(ex).splitlines()[0][:70]}", flush=True)
            rows.append({"tag": tag, "error": str(ex).splitlines()[0][:200], **kw})
            continue
        if base_pred is None:
            base_pred, base_secs = r["pred"], r["seconds"]
        std = float(np.std(base_pred))
        rel = float(np.abs(r["pred"] - base_pred).max() / max(std, 1e-30))
        rows.append({k: v for k, v in r.items() if k != "pred"} | {
            "speedup": float(base_secs / r["seconds"]),
            "max_abs_pred_diff_rel_std": rel,
            "numerically_equivalent": bool(rel < 1e-6),
        })
        print(f"  {tag:38s} {r['seconds']:7.1f}s  ×{base_secs/r['seconds']:.2f}  "
              f"Δpred/std={rel:.2e} {'✅等价' if rel < 1e-6 else '⚠️不等价'}", flush=True)

    payload = {
        "question": "LightGBM 训练能提速多少？换了之后数值还等价吗？",
        "criterion": "max|Δpred|/std(pred) < 1e-6 才算数值等价、可用于出结论的实验",
        "baseline": "device=cpu, deterministic=True, force_row_wise=True, "
                    "num_threads=16, max_bin=255, 每个种子重建 Dataset",
        "configuration": {
            "rows": int(x_it.shape[0]), "columns": int(x_it.shape[1]),
            "min_data_in_leaf": min_data, "rounds": args.rounds, "n_seeds": args.n_seeds,
            "sample_modulo": args.sample_modulo, "sampling": args.sampling,
            "note": "固定轮数、不早停 —— 各配置必须做等量的功才可比",
        },
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# LightGBM 训练提速基准（真实设计矩阵）", "",
             f"{x_it.shape[0]:,} 行 × {x_it.shape[1]} 列，固定 {args.rounds} 轮 × "
             f"{args.n_seeds} 种子（不早停，各配置做等量的功）。", "",
             "**判据**：`max|Δpred|/std(pred) < 1e-6` 才算数值等价。"
             "只快不准没有意义 —— 不等价的配置不能用于出结论的实验，"
             "否则同一条研究线里混进两套数值口径。", "",
             "| 配置 | 耗时 | 提速 | Δpred/std | 数值等价 |", "|---|---:|---:|---:|:--:|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['tag']} | ❌ | — | — | {r['error'][:40]} |")
        else:
            lines.append(f"| {r['tag']} | {r['seconds']:.1f}s | ×{r['speedup']:.2f} | "
                         f"{r['max_abs_pred_diff_rel_std']:.1e} | "
                         f"{'✅' if r['numerically_equivalent'] else '⚠️'} |")
    lines += ["", "## 决策规则（预先写死）", "",
              "- 总提速 **< 1.5 倍** → 只保留 Dataset 复用（零风险零成本），不折腾 GPU",
              "- 总提速 **≥ 2 倍且数值等价** → 后续大规模跑默认开；"
              "已出结论的实验**不重跑**，避免混入两套数值口径",
              "- **数值不等价** → 一律不用于出结论的实验，最多探路", "",
              "⚠️ `max_bin=63` **会改变模型**，这里只测速度与预测差异，不改默认值。", "",
              "⚠️ 评测环境是 4 核 / 12GB / **无 GPU**，GPU 只用于本地训练，"
              "推理端不依赖它（LightGBM 模型是纯文本，推理只是遍历树）。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n产物：{json_path}\n     {md_path}")


if __name__ == "__main__":
    main()
