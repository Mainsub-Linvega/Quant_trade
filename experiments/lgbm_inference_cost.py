"""🚨 阻塞项：LightGBM 在评测环境里推理跑得完吗？

## 为什么这是阻塞项

评测环境：**4 核 / 12GB / 无 GPU / 无网络**，且 **某个 time_id 超时就把那一刻的预测全部置 0**。

- 岭回归实测：0.42 ms/次 × 214,538 次 = **87 秒**，余量很大
- LightGBM：**3 个种子 × 数百到近千棵树**，`lgbm_blend` 里 `xs_shrunk` 有几折早停到 891 轮
  —— **从没测过**

这条不过，`lgbm_mt` / `lgbm_xs` / `lgbm_blend` 三轮实验的价值全是零。

## 量什么

官方 runner 每次 `predict` 恰好喂**一个 time_id 的全部行**（约 15 行）。
15 行是极小批 —— **LightGBM 的每次调用固定开销很可能盖过树遍历本身**，
所以必须按「15 行一批、调用 21.4 万次」这个真实形态量，不能拿大批量吞吐去换算。

同时量**轮数 → 耗时**的曲线：知道成本怎么随树的数量涨，才能在
「精度」和「跑得完」之间做取舍（`lgbm_blend` 里各折早停从 62 到 891 轮不等）。

环境用 `OMP_NUM_THREADS=4` 限制，对齐评测端的 4 核。

用法：
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/lgbm_inference_cost.py
输出：outputs/experiments/lgbm_inference_cost.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
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

# 与 lgbm_blend 胜出的那组一致（xs_loose：原始与折后都赢）
SPEC = {"num_leaves": 63, "min_data_frac": 12000 / 3_500_000,
        "learning_rate": 0.03, "feature_fraction": 0.7, "lambda_l2": 1.0}

TEST_PREDICT_CALLS = 214_538      # 测试集真实的 predict 调用次数
RIDGE_TOTAL_SECONDS = 87.0        # 岭回归实测总耗时（ridge_strict_acceptance）
TREE_COUNTS = (100, 200, 300, 450, 600, 900)


def stage(label: str) -> float:
    """打印当前 RSS。首跑 exit 137 时因为管道接了 tail、一行日志都没看到，
    根因至今不明 —— 根因不明的时候，可观测性比任何优化都重要。"""
    with open("/proc/self/statm") as fh:
        rss_pages = int(fh.read().split()[1])
    gb = rss_pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    print(f"    [RSS {gb:6.2f} GB] {label}", flush=True)
    return gb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Can LightGBM finish inference in the eval box?")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--report", default="lgbm_inference_cost")
    p.add_argument("--force", action="store_true")
    # 推理成本只取决于「模型形状」（树数 × 叶子数，由参数控制）与「调用形态」
    # （每批约 15 行 × 201 列），**不取决于训练集大小** —— 所以这里用小样本
    # 训出同样形状的模型即可，内存从约 7 GB 降到约 1.5 GB。
    # 唯一的保留是树的实际深度可能略有不同（num_leaves 是上界不是等式），
    # 所以报告里会打印实测平均树深与总叶子数，让这个迁移假设可核验。
    p.add_argument("--sample-modulo", type=int, default=40)
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--max-rounds", type=int, default=max(TREE_COUNTS))
    p.add_argument("--n-calls", type=int, default=500, help="计时用的 predict 调用次数")
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

    # 线程只靠 OMP_NUM_THREADS 环境变量限制（这本来就是对齐评测端 4 核的正确做法）。
    # 不再每次 predict 传 num_threads —— LightGBM 4.x 会把 predict 的 kwargs 当预测参数
    # 每次重新解析、可能重建内部 predictor，在 3.6 万次调用的循环里既慢又可疑。
    omp = os.environ.get("OMP_NUM_THREADS")
    print(f"OMP_NUM_THREADS={omp}（对齐评测端 4 核）"
          + ("" if omp == "4" else "  ⚠️ 不是 4，计时不代表评测环境！"), flush=True)
    print(f"加载数据（modulo {args.sample_modulo}）…", flush=True)
    d = load_rows(Path(args.data_root), args.sample_modulo, "phase_balanced")
    features, y, w, tid, aid = d["features"], d["target"], d["weight"], d["time_id"], d["asset_id"]
    del d
    stage("加载完成")
    e, _ = cross_sectional_target(y, w, tid)
    uniq = tid[np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]]

    train_ids, valid_ids = rolling_time_folds(uniq, 5, int(len(uniq) * 4 / 9), 6)[-1]

    def rows_of(ids):
        return np.arange(int(np.searchsorted(tid, ids[0], "left")),
                         int(np.searchsorted(tid, ids[-1], "right")))

    tr_rows, va_rows = rows_of(train_ids), rows_of(valid_ids)
    scratch, stats = robust_transform_fit(features[tr_rows].copy())
    selected = select_features(scratch, e[tr_rows], w[tr_rows], args.feature_count)
    del scratch
    stage("预处理与选列完成")
    raw = features[:, selected].copy()
    apply_robust_transform(raw, stats["lower"][selected], stats["upper"][selected],
                           stats["center"][selected], stats["scale"][selected])
    dev = cross_sectional_deviation(raw, tid)
    del raw, features
    x = np.column_stack([dev, aid.astype(np.float32)])
    del dev
    stage("设计矩阵完成")
    cat = x.shape[1] - 1
    min_data = max(20, int(round(SPEC["min_data_frac"] * len(tr_rows))))

    print(f"训练 {len(tr_rows):,} 行 × {x.shape[1]} 列，{args.n_seeds} 种子 × "
          f"{args.max_rounds} 轮…", flush=True)
    boosters, model_bytes = [], 0
    for s in range(args.n_seeds):
        p = {k: v for k, v in SPEC.items() if k != "min_data_frac"}
        p.update({"objective": "regression", "verbosity": -1, "num_threads": 16,
                  "min_data_in_leaf": min_data, "bagging_fraction": 0.7, "bagging_freq": 1,
                  "deterministic": True, "force_row_wise": True, "feature_pre_filter": False,
                  "seed": args.seed + s, "bagging_seed": args.seed + 1000 + s,
                  "feature_fraction_seed": args.seed + 2000 + s})
        ds = lgb.Dataset(x[tr_rows], label=e[tr_rows], weight=w[tr_rows], params=p,
                         categorical_feature=[cat], free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=args.max_rounds)
        boosters.append(b)
        model_bytes += len(b.model_to_string().encode())
        del ds
        stage(f"种子 {s} 训练完成")

    # 树的实际形状 —— 小样本训练能否代表全量，就靠这两个数核验
    dumps = [b.dump_model() for b in boosters]
    def _depth(node, d=1):
        if "leaf_value" in node:
            return d
        return max(_depth(node["left_child"], d + 1), _depth(node["right_child"], d + 1))
    depths = [_depth(t["tree_structure"]) for dm in dumps for t in dm["tree_info"]]
    leaves = [t["num_leaves"] for dm in dumps for t in dm["tree_info"]]
    del dumps
    print(f"训练完成。{args.n_seeds} 个模型纯文本共 {model_bytes/2**20:.1f} MiB；"
          f"平均树深 {np.mean(depths):.1f}（最大 {max(depths)}）、"
          f"平均叶子数 {np.mean(leaves):.1f}", flush=True)
    stage("训练全部完成")

    # ---- 按真实形态计时：一个 time_id 一批（约 15 行），调用 n_calls 次
    tid_va = tid[va_rows]
    starts = np.r_[0, np.flatnonzero(tid_va[1:] != tid_va[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(va_rows)])
    x_va = x[va_rows]                       # ← 提到推导外面。原来在推导里每轮重算一次，
    del x                                   #    2000 轮累计 220 GB 的分配/释放
    batches = [np.ascontiguousarray(x_va[s:s + c])
               for s, c in zip(starts[:args.n_calls], counts[:args.n_calls])]
    del x_va
    mean_rows = float(np.mean([len(b) for b in batches]))
    stage("计时批次准备完成")
    print(f"计时：{len(batches):,} 次调用，每批平均 {mean_rows:.1f} 行\n", flush=True)

    results = []
    for n_trees in TREE_COUNTS:
        if n_trees > args.max_rounds:
            continue
        for b in boosters:                       # 预热，避开首次调用的初始化开销
            b.predict(batches[0], num_iteration=n_trees)
        t0 = time.perf_counter()
        for batch in batches:
            acc = None
            for b in boosters:
                p = b.predict(batch, num_iteration=n_trees)
                acc = p if acc is None else acc + p
            acc /= len(boosters)
        elapsed = time.perf_counter() - t0
        per_call_ms = elapsed / len(batches) * 1000.0
        total = per_call_ms / 1000.0 * TEST_PREDICT_CALLS
        results.append({
            "n_trees": n_trees, "n_seeds": args.n_seeds,
            "per_call_ms": per_call_ms,
            "extrapolated_total_seconds": total,
            "extrapolated_total_minutes": total / 60.0,
            "vs_ridge_x": total / RIDGE_TOTAL_SECONDS,
        })
        print(f"  {n_trees:4d} 棵 × {args.n_seeds} 种子  {per_call_ms:7.3f} ms/次  → "
              f"21.4 万次共 {total/60:6.1f} 分钟（岭回归的 {total/RIDGE_TOTAL_SECONDS:.1f} 倍）",
              flush=True)
    stage("计时全部完成")

    payload = {
        "question": "LightGBM 在评测环境（4 核 / 12GB / 无 GPU）里推理跑得完吗？",
        "why_blocking": "超时该 time_id 置 0。这条不过，前面三轮实验价值为零。",
        "measurement_shape": (f"按真实形态：一个 time_id 一批（平均 {mean_rows:.1f} 行）、"
                              f"调用 {len(batches):,} 次外推到 {TEST_PREDICT_CALLS:,} 次。"
                              "15 行是极小批，LightGBM 的每次调用固定开销可能盖过树遍历本身，"
                              "所以不能拿大批量吞吐换算。"),
        "reference": {"ridge_total_seconds": RIDGE_TOTAL_SECONDS,
                      "ridge_per_call_ms": RIDGE_TOTAL_SECONDS / TEST_PREDICT_CALLS * 1000},
        "configuration": {
            "spec": SPEC, "min_data_in_leaf": min_data, "n_seeds": args.n_seeds,
            "omp_num_threads": omp,
            "sample_modulo": args.sample_modulo,
            "train_rows": int(len(tr_rows)), "columns": int(cat + 1),
            "model_text_bytes": model_bytes,
            # 小样本训练能否代表全量，靠这两个数核验（num_leaves 是上界不是等式）
            "tree_shape": {"mean_depth": float(np.mean(depths)), "max_depth": int(max(depths)),
                           "mean_leaves": float(np.mean(leaves)), "n_trees_total": len(depths)},
            "why_small_sample_is_ok":
                "推理成本只取决于模型形状（树数 × 叶子数，由参数控制）与调用形态"
                "（每批约 15 行 × 201 列），不取决于训练集大小。tree_shape 用来核验这个假设。",
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# LightGBM 推理成本 —— 评测环境跑得完吗", "",
             f"评测环境 **4 核 / 12GB / 无 GPU**，超时该 time_id 置 0。",
             f"按真实形态计时：一个 time_id 一批（平均 {mean_rows:.1f} 行）、"
             f"{len(batches):,} 次调用外推到 {TEST_PREDICT_CALLS:,} 次。",
             "",
             "> 15 行是极小批 —— **LightGBM 的每次调用固定开销很可能盖过树遍历本身**，"
             "所以不能拿大批量吞吐去换算，必须按这个形态量。",
             "",
             f"对照：岭回归实测 **{RIDGE_TOTAL_SECONDS:.0f} 秒**"
             f"（{RIDGE_TOTAL_SECONDS/TEST_PREDICT_CALLS*1000:.3f} ms/次）。",
             "",
             f"{args.n_seeds} 个种子的模型纯文本共 **{model_bytes/2**20:.1f} MiB**（私榜 zip 要装得下）。",
             "",
             f"⚠️ 模型是在 **modulo {args.sample_modulo}（{len(tr_rows):,} 行）** 上训的，不是全量 —— "
             "推理成本只取决于模型形状与调用形态，不取决于训练集大小。"
             f"实测树形状：**平均深度 {np.mean(depths):.1f}（最大 {max(depths)}）、"
             f"平均叶子数 {np.mean(leaves):.1f}**（`num_leaves` 上限 {SPEC['num_leaves']}），"
             "这两个数就是用来核验这个迁移假设的。",
             "",
             f"| 树的数量（×{args.n_seeds} 种子） | ms/次 | 21.4 万次总耗时 | 相对岭回归 |",
             "|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['n_trees']} | {r['per_call_ms']:.3f} | "
                     f"{r['extrapolated_total_minutes']:.1f} 分钟 | "
                     f"{r['vs_ridge_x']:.1f}× |")
    lines += ["", "## 怎么读", "",
              "- **主办方没公布超时阈值**，所以这里只给绝对耗时，不自行判定「通过/不通过」。",
              "  参照系：岭回归 87 秒跑完且从未超时，说明预算至少不止 87 秒。",
              "- 成本随树的数量近似线性 → 可以用轮数换时间。`lgbm_blend` 里各折早停"
              "从 62 到 891 轮不等，最终模型该取多少轮是个**精度 vs 跑得完**的取舍。",
              "- 种子数是线性因子：3 种子降到 1 种子直接省 3 倍，代价是失去种子平均的降方差。",
              "- 若耗时不可接受，还有两条路：**减树**（降 `num_leaves` / 早停更严）、"
              "或**纯 numpy 树遍历**（`model_to_string` 是纯文本，也顺带解决评测端有没有"
              "lightgbm 的问题）。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n产物：{json_path}\n     {md_path}")


if __name__ == "__main__":
    main()
