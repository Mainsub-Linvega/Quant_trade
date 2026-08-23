"""peer 对轴收口：把 oracle 的 `peer_e_lag1` 换成**可部署**量之后，那 +3.29% 还剩多少？

## 这个格子为什么空着

`xs_peer_pair_confirm_3s480`（8/23）在 3s480 上拿到 pooled **+3.29% / 5-of-5 折 /
去最好折 +2.93% / `2ΔA>ΔB` / bootstrap CI 下界 +2.30%**，六道门禁过五道，只差检出下限
—— 与长窗 w512 当年 confirm 档同一个桶。**但特征是 oracle**：`peer_e_lag1` 由真实 target
反推（`e = y − 截面均值`），而 `main.py` 的 `forbidden` 在 `predict()` 之前就剥掉 target。

ROADMAP §5 因此留了一条重新开放条件：「换成模型自身对搭档的历史预测值」。
那句话至今**没有数字**。本脚本给它一个。

## 为什么是缓存探针，不是把列换掉重跑树

`e_lgbm` 只在 OOF cache 的 `fold>=0` 行有值。按 `xs_peer_pair_probe` 自己的 fold 版图，
**训练段**覆盖率是：

    fold0 0.0%   fold1 25%   fold2 50%   fold3 75%   fold4 100%

fold 0 的 peer 列会恒为零（两臂等价），fold 1–3 的覆盖率**与时间强相关** ⟹ 树可以学到
「peer 列非零 ⟹ 处在较晚时期」这个伪时间信号。**直接换列不是有效实验。**
要让它有效，得在训练段也生成 `ê`，那等于重跑一遍扩展 fold 版图的 OOF（小时级）。

⟹ 改为只在**验证段**评估（`ê` 覆盖 100%，无时间混淆），零训练，复用
`horizon_auxiliary_cache_probe.evaluate_arm`（2026-08-22 抽出，抽取前后逐字段验证过）。

⚠️ **本探针的局限先写在这里，不放到结论里找补**：缓存探针是**线性**配比，而树是非线性
用这个特征的；逐有向对拆 6 列只覆盖到 `asset_id × peer` 的**一阶**交互。所以本探针的
阴性结果**弱于**树的阴性结果 —— 这正是 `oracle_lag1` 阳性对照存在的理由：它定量回答
「这把线性尺子能不能看见树看见的那个东西」。看不见就判 `INCONCLUSIVE`，不判「没效果」。

## 四个臂

| 臂 | peer 量 | 作用 |
|---|---|---|
| `oracle_lag1`     | 真实 `e_j(t−1)`            | 阳性对照 |
| `deployable_lag1` | `ê_j(t−1)`                 | 主问题 |
| `deployable_now`  | `ê_j(t)`                   | 两阶段变体（同一次 predict 内可得）|
| `shuffled_lag1`   | **非搭档**资产的 `ê(t−1)`  | 阴性对照，防「多一列就变好」|

每个臂拆成 **6 列**（第 k 列只在有向对 k 的目标资产行上非零）传给 `evaluate_arm`，
它对 `[base, *auxes]` 一起解系数 ⟹ 自然得到逐有向对的系数。既有函数一行不改。

判据先于结果落在 `outputs/experiments/xs_peer_deployable_plan.json`，其 sha256 记进结果。

用法：
    .venv/bin/python experiments/xs_peer_deployable_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from function_class_probe import CACHE_PATH  # noqa: E402
from horizon_auxiliary_cache_probe import (evaluate_arm, group_index,  # noqa: E402
                                           zero_mean_per_time)
from src.oof_cache import assert_reproducible_cache  # noqa: E402
from xs_peer_pair_probe import PAIRS  # noqa: E402

OUTPUT_DIR = _REPO_ROOT / "outputs" / "experiments"
LABEL = "xs_peer_deployable_probe"
PLAN_PATH = OUTPUT_DIR / "xs_peer_deployable_plan.json"

# 阴性对照：**目标**资产集合与 PAIRS 完全相同（于是非零行一模一样），只把搭档换成
# 9 个不参与任何真对子的资产。这样「多一列」的效应被抵消掉，剩下的只有「是不是这个搭档」。
NEG_PAIRS = {0: 3, 6: 4, 2: 5, 14: 7, 1: 8, 13: 9}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(CACHE_PATH))
    p.add_argument("--plan", default=str(PLAN_PATH))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--label", default=LABEL)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cross_sectional_residual(target: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    """`e = y − 逐 time_id 无权截面均值`，与 `xs_peer_pair_probe.build_peer_feature` 同口径。

    行必须已按 time_id 升序（调用方保证）。
    """
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(np.float64)
    return target - np.repeat(np.add.reduceat(target, starts) / counts, counts.astype(int))


def peer_columns(source: np.ndarray, time_id: np.ndarray, asset_id: np.ndarray,
                 eval_mask: np.ndarray, pairs: dict[int, int], lag: int) -> list[np.ndarray]:
    """把一个搭档量摊成**逐有向对**的 6 列，只在各自目标资产的行上非零。

    ⚠️ pivot 建在**全部**采样 time_id 上（含 `fold=-1`）再 shift，这样「上一个采样 time_id」
    与 `xs_peer_pair_probe.build_peer_feature` 是同一个语义；切到评估行是**之后**才做的。
    若改成只在评估行上 shift，折边界处的「上一期」会跨过整段训练区，口径就变了。

    NaN 一律填 0 —— 与原探针同一约定（`np.nan_to_num(vals, nan=0.0)`）。
    `ê` 在 `fold=-1` 行是 NaN，因此只影响「上一期落在训练区」的那几行边界。
    """
    frame = pd.DataFrame({"time_id": time_id, "asset_id": asset_id, "v": source})
    pivot = frame.pivot_table(index="time_id", columns="asset_id", values="v").sort_index()
    shifted = pivot.shift(lag) if lag else pivot

    t_eval, a_eval = time_id[eval_mask], asset_id[eval_mask]
    out: list[np.ndarray] = []
    for asset, partner in pairs.items():
        col = np.zeros(len(t_eval), dtype=np.float64)
        rows = a_eval == asset
        vals = shifted[partner].reindex(t_eval[rows]).to_numpy()
        col[rows] = np.nan_to_num(vals, nan=0.0)
        out.append(col)
    return out


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and json_path.exists():
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    plan_path = Path(args.plan)
    if not plan_path.is_file():
        raise SystemExit(f"预注册文件不存在：{plan_path} —— 判据必须先于结果落盘")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if {int(k): int(v) for k, v in plan["pairs"]["expected"].items()} != PAIRS:
        raise SystemExit("预注册的 PAIRS 与 xs_peer_pair_probe.PAIRS 不一致 ⟹ 对子被动过")
    if set(NEG_PAIRS) != set(PAIRS):
        raise SystemExit("阴性对照的**目标**资产集合必须与 PAIRS 相同")
    if any(NEG_PAIRS[a] == PAIRS[a] for a in PAIRS) or set(NEG_PAIRS.values()) & set(PAIRS):
        raise SystemExit("阴性对照的搭档与真对子有交集 ⟹ 不是干净对照")

    cache_path = Path(args.cache)
    assert_reproducible_cache(cache_path)
    with np.load(cache_path, allow_pickle=False) as d:
        time_id = d["time_id"].astype(np.int64)
        asset_id = d["asset_id"].astype(np.int64)
        target = d["target"].astype(np.float64)
        weight = np.maximum(d["weight"].astype(np.float64), 0.0)
        fold = d["fold"].astype(np.int64)
        e_lgbm = d["e_lgbm"].astype(np.float64)

    order = np.argsort(time_id, kind="stable")
    time_id, asset_id, target = time_id[order], asset_id[order], target[order]
    weight, fold, e_lgbm = weight[order], fold[order], e_lgbm[order]

    e_true = cross_sectional_residual(target, time_id)
    eval_mask = fold >= 0

    # 自检 1：评估段的 ê 必须 100% 有值。不足就退出 —— 覆盖不全会让「零填充」与时间相关，
    # 正是本探针要避开的那个混淆。
    coverage = float(np.isfinite(e_lgbm[eval_mask]).mean())
    if coverage < 1.0:
        raise SystemExit(f"评估段 ê 覆盖率只有 {coverage:.4%}，不是 100% ⟹ 存在时间混淆")

    y = target[eval_mask]
    w = weight[eval_mask]
    base = e_lgbm[eval_mask]
    starts, gidx, n_groups = group_index(time_id[eval_mask])
    group_fold = fold[eval_mask][starts]
    fold_list = sorted(set(int(f) for f in group_fold))
    if len(fold_list) < 3:
        raise SystemExit(f"只有 {len(fold_list)} 折，评估折不足")

    # 自检 2：基准本身应当已经是逐 time_id 零均值的截面分量（生产 ê 出厂就投影过）。
    base_mean_abs = float(np.max(np.abs(
        base - zero_mean_per_time(base, starts, len(base)))))
    print(f"缓存 {cache_path.name}\n"
          f"评估行 {int(eval_mask.sum()):,} / 全量 {len(time_id):,}；"
          f"time_id 组 {n_groups:,}；折 {fold_list}\n"
          f"ê 覆盖 {coverage:.2%}；基准偏离零均值 max|Δ| {base_mean_abs:.3e}", flush=True)

    arms = {
        "oracle_lag1":     (e_true, PAIRS, 1),
        "deployable_lag1": (e_lgbm, PAIRS, 1),
        "deployable_now":  (e_lgbm, PAIRS, 0),
        "shuffled_lag1":   (e_lgbm, NEG_PAIRS, 1),
    }
    results: dict[str, Any] = {}
    for name, (source, pairs, lag) in arms.items():
        auxes = peer_columns(source, time_id, asset_id, eval_mask, pairs, lag)
        nonzero = float(np.mean(np.count_nonzero(np.column_stack(auxes), axis=1) > 0))
        # 逐臂一条独立的 bootstrap 流 ⟹ 结果与臂顺序无关（evaluate_arm docstring 的建议）
        rng = np.random.default_rng(args.boot_seed)
        res = evaluate_arm(name, auxes, base, "pure_e", y=y, w=w, starts=starts, gidx=gidx,
                           n_groups=n_groups, group_fold=group_fold, fold_list=fold_list,
                           boot_rng=rng, block_size=args.block_size, n_boot=args.n_boot)
        res["nonzero_row_fraction"] = nonzero
        res["source"] = "e_true" if source is e_true else "e_lgbm"
        res["lag"] = lag
        res["pairs"] = {str(k): int(v) for k, v in pairs.items()}
        results[name] = res

    # ⚠️ 判据按**预注册原文**：`oracle 过`（七道门禁全过），不是「oracle 为正」。
    # 2026-08-23 初版实现误写成后者，而实测恰好落在两者之间（oracle 为正但不过门禁）——
    # 那正是最容易被结果牵着走的地方。以严格的那条为准，宁可判 INCONCLUSIVE。
    oracle_pass = results["oracle_lag1"]["pass"]
    oracle_positive = results["oracle_lag1"]["mean_delta"] > 0
    deployable_pass = [n for n in ("deployable_lag1", "deployable_now") if results[n]["pass"]]
    negative_clean = not results["shuffled_lag1"]["pass"]

    # 事后分析（**不在预注册里**，只作旁证，不参与裁决）：相对阴性对照的差。
    # 「多加 6 列」本身有代价，对零比大小会把这个代价算到臂头上；对阴性对照比才干净。
    shuffled_rel = results["shuffled_lag1"]["relative"]
    for name, r in results.items():
        r["posthoc_contrast_vs_negative_control_pp"] = (r["relative"] - shuffled_rel) * 100.0

    if not negative_clean:
        verdict = "VOID_NEGATIVE_CONTROL_PASSED"
        reading = ("阴性对照也过了门禁 ⟹ 整份作废。先查是不是「多一列就变好」，"
                   "不得把任何一个臂当成发现。")
    elif not oracle_pass:
        verdict = "INCONCLUSIVE_NO_DETECTION_POWER"
        reading = ("按预注册：阳性对照 `oracle_lag1` **没有过门禁** ⟹ **这把线性尺子对该机制"
                   "检出力不足**，因此两个可部署臂的阴性结果不能升级为「没效果」。"
                   + ("⭐ oracle 仍为正（"
                      f"{results['oracle_lag1']['relative']*100:+.2f}%，"
                      f"{results['oracle_lag1']['positive_folds']}/"
                      f"{results['oracle_lag1']['n_folds']} 折，bootstrap CI 下界为正），"
                      "说明尺子并非全无分辨力，只是分辨不到 3% 门槛。" if oracle_positive else "")
                   + " 实操结论仍是不推进：见事后旁证与 2026-08-23 的前置测量。")
    elif deployable_pass:
        verdict = "DEPLOYABLE_ARM_PASSED"
        reading = (f"可部署臂 {deployable_pass} 过门禁且阴性对照干净 ⟹ 才谈树版；"
                   "而树版需先解决训练段 ê 覆盖（扩展 fold 版图的 OOF，小时级）。")
    else:
        verdict = "REJECTED"
        reading = ("两个可部署臂都不过，阳性对照过门禁、阴性对照干净 ⟹ "
                   "**peer 轴由证据关闭**：结构是真的，但推理端拿不到承载它的量。")

    payload = {
        "experiment": LABEL,
        "plan_sha256": sha256_file(plan_path),
        "cache": {"path": str(cache_path), "sha256": sha256_file(cache_path),
                  "rows_total": int(len(time_id)), "rows_eval": int(eval_mask.sum()),
                  "groups": int(n_groups), "folds": fold_list,
                  "ehat_coverage_eval": coverage},
        "baseline": {"name": "e_lgbm", "evaluate_arm_bname": "pure_e",
                     "max_abs_deviation_from_zero_mean": base_mean_abs},
        "pairs": {str(k): int(v) for k, v in PAIRS.items()},
        "negative_control_pairs": {str(k): int(v) for k, v in NEG_PAIRS.items()},
        "arms": results,
        "verdict": verdict,
        "reading": reading,
        "oracle_positive_but_not_passing": bool(oracle_positive and not oracle_pass),
        "posthoc_note": ("`posthoc_contrast_vs_negative_control_pp` 不在预注册里，只作旁证。"
                         "读法：oracle 高出阴性对照才说明尺子有分辨力；可部署臂若落在对照"
                         "同侧或更低，则它带来的不是 peer 信息。"),
        "limitation": ("缓存探针是线性配比、逐有向对 6 列只覆盖一阶交互；"
                       "树是非线性用这个特征的 ⟹ 本探针的阴性结果弱于树的阴性结果。"
                       "oracle_lag1 就是用来定量这条局限的。"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = [f"# peer 搭档量：oracle vs 可部署（`{LABEL}`）", "",
             f"> 预注册判据 sha256 `{payload['plan_sha256']}`，先于结果落盘。",
             f"> 缓存 `{cache_path.name}`，评估 {int(eval_mask.sum()):,} 行 / "
             f"{n_groups:,} 个 time_id / {len(fold_list)} 折（fold 0 被系数拟合消耗）。",
             f"> ê 覆盖 **{coverage:.2%}**（不足 100% 即报错退出）。", "",
             "| 臂 | 搭档量 | 滞后 | Δpeak 折均 | 相对 | 正折 | 去最好折 | CI 下界 | 减阴性对照 | 判定 |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|"]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {r['source']} | {r['lag']} | {r['mean_delta']:+.3e} | "
            f"**{r['relative']*100:+.2f}%** | {r['positive_folds']}/{r['n_folds']} | "
            f"{r['mean_delta_drop_best']:+.3e} | {r['paired_bootstrap']['p2.5']:+.3e} | "
            f"{r['posthoc_contrast_vs_negative_control_pp']:+.2f}pp | "
            f"{'✅' if r['pass'] else '❌'} |")
    lines += ["", f"> 「减阴性对照」列是**事后**旁证，不在预注册里：{payload['posthoc_note']}"]
    lines += ["", "### 逐门槛", ""]
    for name, r in results.items():
        lines.append(f"**`{name}`**")
        lines += [f"- {'✅' if v else '❌'} `{k}`" for k, v in r["checks"].items()]
        lines.append("")
    lines += [f"## 裁决：{verdict}", "", f"> {reading}", "",
              f"⚠️ {payload['limitation']}", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n裁决：{verdict}\n{reading}\nwrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
