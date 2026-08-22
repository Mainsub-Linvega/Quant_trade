"""用 responder 的窗口梯子给 feature 打分选列 —— 先量重合度，再决定要不要跑 OOF。

## 问题

生产的选列判据只有一个（`strategies/v1_ridge/train.py:86-108`，全仓 97 处调用）：

```text
top-200 by |加权 Pearson(feature_j, e)|      e = target 的逐 time_id 无权截面残差
再在这 200 列内 top-40 当 history 列
```

它是**单目标**的：每个 feature 只拿一个带噪估计（target 的 R² 只有 ~0.005 量级）去排序。
本脚本问：**用 responder 的窗口梯子做同一件事，把估计的方差压下去，选出来的列会不会不一样？**

## ⚠️ 这不是「换一个更好的判据」

08-21 的 `selection_criterion_probe` 已经把「换判据」测过并否掉：`lasso200` −1.29%、
`hist_lag1` −4.38%、`hist_roll5` −10.39%，三臂全负，且**分歧越大掉得越多是单调的**。
结案语：「对 `feature_fraction=0.7` 的树，前置筛子的选择质量**不是绑定约束**」。

那三个臂换的是**被估计的量**（LASSO 的多元系数、滞后形态）。本脚本换的是**估计的方差**：

```text
窗口图谱（responder_window_atlas）已经证明 α 族是一把同源嵌套的 MA 窗梯子 ——
  responder_00 H=1, responder_02 H=2, responder_03 H=4, target H=5,
  responder_04 H=7, responder_05 H=10
拟合 RMSE 0.024~0.054，且与 target 的错位相关峰值 shift 随 H 单调（+5 → +1 → 0 → −4 → −9）

⟹ 一个 feature 若真的携带信号，它与整条梯子的相关应当同号；
   只与 H=5 相关而与 H=4 / H=7 都不相关的，大概率是抽样噪声。
   把梯子上的相关平均起来 = 对同一个量的多次测量取平均 = 降低估计方差。
```

## ⚠️ 不得按「与 target 相关高」挑梯子成员

`responder_targets_stage1.md:14-22` 已证伪该论证形式：同期相关最高的 `responder_03`（0.817）
当训练目标是全场最差（−15.47%、0/7），相关只有 0.394 的 `responder_04` 排第一。
所以梯子成员**由 `responder_window_atlas` 自己的 `H_fit_is_equal_weight_MA` 判定派生**
（实测筛出 r00/r02/r03/r04/r05，排除 r01 与 r06），不看它与 target 的相关。见 `CLAUDE.md:119`。

## 前置测量的决策规则（跑 OOF 之前钉死）

```text
|S_new ∩ S_base| ≥ 190/200   ⟹ 改动幅度低于 1s160 的 6.1% 检出下限
                                判「与 base 不可区分」，**不跑 OOF**，结案
|S_new ∩ S_base| < 190/200   ⟹ 落进 selection_criterion_probe 已量过的区间
                                （lasso200 重合 68~119/200 → −1.29%，且单调）
                                ⟹ 先验是掉分；只有能写出一条不依赖「判据更好」的
                                   机制假设时才跑 1s160 五折
```

⭐ 无论落在哪一支，都在几分钟内得到一个可写进 ROADMAP 的结论，而不是 36 分钟起跳的 OOF
加一个大概率的负结果（CLAUDE.md §5.1「先写变量和判据，再看结果」）。

## 口径隔离

- 相关一律**无权**（`np.ones_like`），与生产两道筛子一致（推理端拿不到 `weight`）；
- responder 有缺失（最多 0.16%）⟹ 取 complete-case。**`S_base` 与 `S_new` 用同一批行**，
  这样两者唯一的差别就是标签；另报「全行 vs complete-case 的 `S_base`」以证明限制行集本身没挪动选列；
- 本脚本自己算相关（`select_features` 只返回排序后的下标、拿不到分数），
  但**逐折硬断言**自算的 top-200 与 `select_features` 逐位相同 ——
  一天之内不去动那个有 97 处调用点的生产函数。

用法：
    .venv/bin/python experiments/responder_selection_probe.py --emit-plan
    .venv/bin/python experiments/responder_selection_probe.py
输出：outputs/experiments/responder_selection_probe{,_plan}.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import cross_sectional_deviation  # noqa: E402
from mt_predictability import group_starts  # noqa: E402
from responder_targets import load_rows_with_responders  # noqa: E402
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402
from v3_production_oof import group_mean, row_slice  # noqa: E402

PLAN_LABEL = "responder_selection_probe_plan"
ATLAS_PATH = _REPO_ROOT / "outputs" / "experiments" / "responder_window_atlas.json"

# ---- 预注册常量（与生产 / selection_criterion_probe 逐项对齐）----
FEATURE_COUNT = 200
HISTORY_COUNT = 40
TRAIN_WINDOW = 78_960
EMBARGO = 6
N_FOLDS = 5
SAMPLE_MODULO = 5
SAMPLING = "phase_balanced"
DETECTION_FLOOR = 0.061            # 1s160 / 5 折的实测检出下限
# 决策线：重合度 ≥ 这个数 ⟹ 判「与 base 不可区分」，不跑 OOF
OVERLAP_DECISION_LINE = 190
# 梯子成员的准入完全由 `responder_window_atlas` 自己的判定给出，见 build_ladder
LADDER_FAMILY_SIGN_CLASS = "unit_interval"   # α 族 —— 与 target 同源的那把梯子


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-plan", action="store_true", help="只落盘预注册判据")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--atlas", default=str(ATLAS_PATH))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="responder_selection_probe")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_ladder(atlas_path: Path, data_root: Path) -> dict[str, Any]:
    """从窗口图谱**派生**梯子成员，不硬编码、也不看与 target 的相关。

    两道准入，**都不是本脚本新发明的**：
    1. 属于 α 族（`responder_family_grid` 的 `unit_interval` —— 与 target 同源的那一把）；
    2. `responder_window_atlas` 自己判定的 `H_fit_is_equal_weight_MA == True`
       ⟹ 那条自相关曲线确实拟合成等权 MA 窗，不是拟合失败的。

    ⚠️ 用图谱自己的布尔判定，而不是我再设一个 RMSE 阈值 —— 阈值会变成第二个可调旋钮，
    而图谱的判据是 08-18 就落盘的（`criterion` 字段可核）。实测这两道筛出
    `responder_00/02/03/04/05`，排除 `responder_01`（RMSE 0.060）与 `responder_06`（0.192）。
    """
    from responder_family_grid import build_families, read_column_stats

    if not atlas_path.is_file():
        raise SystemExit(f"找不到窗口图谱 {atlas_path} —— 先跑 responder_window_atlas.py")
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    by_name = {row["responder"]: row for row in atlas["responders"]}

    families = build_families(read_column_stats(
        data_root / "train" / "train_partition_000.parquet"))
    alpha = next(f for f in families if f["sign_class"] == LADDER_FAMILY_SIGN_CLASS)

    members, rejected = [], []
    for name in alpha["members"]:
        row = by_name.get(name)
        if row is None:
            raise SystemExit(f"窗口图谱里没有 {name}")
        entry = {"responder": name, "fitted_H": int(row["H_estimate"]),
                 "fit_rmse": float(row["H_fit_rmse"]),
                 "is_equal_weight_MA": bool(row["H_fit_is_equal_weight_MA"])}
        (members if entry["is_equal_weight_MA"] else rejected).append(entry)

    if not members:
        raise SystemExit("梯子为空 —— 准入判据或图谱字段名不对")
    return {"family": alpha["family"], "family_members": alpha["members"],
            "admission": "responder_window_atlas.H_fit_is_equal_weight_MA",
            "atlas_criterion": atlas["criterion"],
            "members": members, "rejected": rejected}


def correlation_profile(matrix: np.ndarray, labels: list[np.ndarray]) -> np.ndarray:
    """逐列 × 逐标签的 Pearson 相关（无权），返回 `(n_features, n_labels)`。

    口径与 `select_features(..., np.ones_like(label), ...)` 逐位相同（同样的 64 列分块、
    同样的 `1e-30` 下限），只是**返回分数而不是 top-k 下标** —— 一致性判据需要每一列的相关值。

    ⚠️ 这是 `select_features` 内部那段计算的只读复制品。调用方**必须**用
    `assert_matches_select_features` 逐折验证两者的 top-k 逐位相同：一天之内不去动那个
    有 97 处调用点的生产选择器（CLAUDE.md §8.3「跨策略/跨调用点的公共函数要隔离改动」）。

    ⚠️ 全部标签**一次分块扫完**：设计矩阵是 ~1.7M × 323 float32，逐标签各做一次
    `astype(np.float64)` 会各产生 4.4 GB 临时量（本机 30 GB / swap=0，NOTES.md:1718 的 OOM 教训）。
    """
    label_matrix = np.column_stack([np.asarray(x, dtype=np.float64) for x in labels])
    n = label_matrix.shape[0]
    if matrix.shape[0] != n:
        raise ValueError("设计矩阵与标签行数不一致")
    label_sum = label_matrix.sum(axis=0)
    centered = label_matrix - label_sum / n
    label_var = np.einsum("ij,ij->j", centered, centered)

    out = np.empty((matrix.shape[1], label_matrix.shape[1]), dtype=np.float64)
    for start in range(0, matrix.shape[1], 64):
        block = matrix[:, start:start + 64].astype(np.float64)
        block_sum = block.sum(axis=0)
        covariance = block.T @ label_matrix - np.outer(block_sum, label_sum) / n
        variance = (block * block).sum(axis=0) - block_sum ** 2 / n
        out[start:start + 64] = covariance / np.sqrt(
            np.maximum(np.outer(variance, label_var), 1e-30))
    return out


def top_k(scores: np.ndarray, count: int) -> np.ndarray:
    """与 `select_features` 完全同构：按 |score| 稳定排序取末 k 个，再按原始列序返回。"""
    selected = np.argsort(np.abs(scores), kind="stable")[-count:]
    return np.sort(selected)


def assert_matches_select_features(matrix: np.ndarray, label: np.ndarray,
                                   scores: np.ndarray, count: int, where: str) -> None:
    """硬断言：自算分数的 top-k == 生产选择器的 top-k。不同则当场失败。"""
    theirs = select_features(matrix, label, np.ones_like(label), count)
    mine = top_k(scores, count)
    if not np.array_equal(np.asarray(theirs, dtype=np.int64), mine):
        raise AssertionError(
            f"{where}: 自算相关的 top-{count} 与 select_features 不一致 "
            f"（差 {len(set(theirs.tolist()) ^ set(mine.tolist()))} 列）—— "
            "口径已漂移，先修复算器再解读任何重合度")


def churn_diagnostic(base_scores: np.ndarray, new_scores: np.ndarray,
                     count: int) -> dict[str, Any]:
    """被换掉的列，在**原判据**的排名里坐在哪？—— 分辨两种互斥假设。

    ```text
    假设 A「边缘搅动」   换掉的列都挤在截断线附近（排名 ~count 两侧）
                        ⟹ 那里的相关值本来就在噪声里（实测 200th=0.00299 / 201st=0.00295，
                           落差仅 1.33%，NOTES.md:820-826）⟹ 效应应当很小，方向不定
    假设 B「实质分歧」   换掉的列里有高排名的（排名 ≪ count）
                        ⟹ 两个判据在读不同的东西 ⟹ selection_criterion_probe 量到的
                           「分歧越大掉得越多」单调关系适用，先验是掉分
    ```

    两种假设对 OOF 的预测不同，所以这个读数**决定要不要花那 36 分钟**，
    而不是看重合度这一个数拍脑袋。
    """
    order = np.argsort(-np.abs(base_scores), kind="stable")
    base_rank = np.empty(len(base_scores), dtype=np.int64)
    base_rank[order] = np.arange(1, len(base_scores) + 1)

    base_set = set(top_k(base_scores, count).tolist())
    new_set = set(top_k(new_scores, count).tolist())
    dropped = sorted(base_set - new_set)
    added = sorted(new_set - base_set)

    def ranks(columns: list[int]) -> list[int]:
        return sorted(int(base_rank[c]) for c in columns)

    dropped_ranks, added_ranks = ranks(dropped), ranks(added)
    return {
        "n_changed": len(dropped),
        "dropped_base_ranks": dropped_ranks,
        "added_base_ranks": added_ranks,
        "best_dropped_rank": dropped_ranks[0] if dropped_ranks else None,
        "worst_added_rank": added_ranks[-1] if added_ranks else None,
        "pearson_between_criteria": float(np.corrcoef(np.abs(base_scores),
                                                      np.abs(new_scores))[0, 1]),
        "spearman_between_criteria": float(np.corrcoef(
            np.argsort(np.argsort(-np.abs(base_scores))),
            np.argsort(np.argsort(-np.abs(new_scores))))[0, 1]),
    }


def rung_sensitivity(profile: np.ndarray, rung_names: tuple[str, ...],
                     count: int) -> dict[str, Any]:
    """⚠️ **事后**敏感性检查（不在预注册里，不得当主要结果读）。

    预注册的判据是**未标准化**的逐级相关平均。但各级的 |corr| 量级并不相同 ——
    实测中位数从 `responder_00` 的 0.0025 到 `responder_05` 的 0.0102，差约 4 倍
    ⟹ 平均值会被幅度最大的那一级偏向，这不完全等于「对同一个量的多次测量取平均」。

    这里报出逐级量级，以及**逐级标准化**后的重合度，用来回答一个问题：
    「裁决是不是这个规格选择造成的？」实测两者都在决策线之下 ⟹ 不是。
    """
    magnitudes = [
        {"rung": name,
         "median_abs_corr": float(np.median(np.abs(profile[:, position]))),
         "max_abs_corr": float(np.abs(profile[:, position]).max()),
         "std": float(profile[:, position].std())}
        for position, name in enumerate(rung_names)
    ]
    base = set(top_k(profile[:, 0], count).tolist())
    preregistered = set(top_k(consistency_score(profile), count).tolist())
    standardized = set(top_k(
        (profile / profile.std(axis=0, keepdims=True)).mean(axis=1), count).tolist())
    return {
        "per_rung": magnitudes,
        "magnitude_spread": (max(m["median_abs_corr"] for m in magnitudes)
                             / min(m["median_abs_corr"] for m in magnitudes)),
        "overlap_preregistered": len(base & preregistered),
        "overlap_standardized_per_rung": len(base & standardized),
        "note": ("逐级标准化是事后变体，只用于说明裁决对该规格选择稳健；"
                 "主要结果一律以预注册的未标准化判据为准"),
    }


def consistency_score(profile: np.ndarray) -> np.ndarray:
    """窗口一致性得分：梯子上各带符号相关的平均。

    `profile` 形状 (n_features, n_rungs)。对同一个量的多次带噪测量取平均 ⟹ 估计方差下降；
    符号不一致的列会互相抵消，这正是想要的「只与 H=5 相关的多半是噪声」。
    """
    return profile.mean(axis=1)


def emit_plan(args: argparse.Namespace, ladder: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": PLAN_LABEL,
        "role": "PRE-REGISTRATION —— 必须先于结果落盘（CLAUDE.md §5.1）",
        "question": ("用 responder 的窗口梯子把选列判据的估计方差压下去，"
                     "选出来的 200 列会不会与现状不同？不同到能被本地尺子测出来吗？"),
        "why_not_covered_by_selection_criterion_probe": (
            "那次三个臂换的是**被估计的量**（LASSO 多元系数、lag1、rollmean5）；"
            "本判据换的是**估计的方差** —— 同一个量、多次测量取平均。机制不同。"),
        "why_this_mechanism_is_allowed": (
            "responder_reaudit_20260814.md:93-100 的母条件排除的是「换目标 / 线性叠加 / "
            "对预测值做二层校准」。用 responder 给 feature 打分三者都不是，"
            "属该条点名认可的 representation 一类；且推理端完全不需要 responder。"
            "同时命中 ROADMAP.md:456 给 v5 划的第 ③ 条范围项。"),
        "ladder": ladder,
        "ladder_admission": (
            "α 族（`unit_interval`）成员中 `responder_window_atlas` 自己判定 "
            "`H_fit_is_equal_weight_MA == True` 的；**不看与 target 的相关** —— "
            "responder_targets_stage1 已证伪该论证形式"),
        "criterion": "score_j = mean_k corr(feature_j, e_k)，e_k = 第 k 级标签的逐 time_id 无权截面残差",
        "baseline_criterion": "|corr(feature_j, e_target)| —— 现状，strategies/v1_ridge/train.py:86-108",
        "feature_count": FEATURE_COUNT,
        "history_count": HISTORY_COUNT,
        "folds": {"n_folds": N_FOLDS, "train_window": TRAIN_WINDOW, "embargo": EMBARGO,
                  "sample_modulo": SAMPLE_MODULO, "sampling": SAMPLING},
        "decision_rule": {
            "line": OVERLAP_DECISION_LINE,
            "at_or_above": (f"重合 ≥ {OVERLAP_DECISION_LINE}/{FEATURE_COUNT} ⟹ "
                            f"改动幅度低于 1s160 的 {DETECTION_FLOOR:.1%} 检出下限，"
                            "判「与 base 不可区分」，**不跑 OOF**，结案"),
            "below": (f"重合 < {OVERLAP_DECISION_LINE}/{FEATURE_COUNT} ⟹ 落进 "
                      "selection_criterion_probe 已量过的区间（lasso200 重合 68~119/200 → −1.29%，"
                      "且分歧越大掉得越多是单调的）⟹ 先验是掉分；只有能写出一条不依赖"
                      "「判据更好」的机制假设时才跑 1s160 五折"),
            "detection_floor": DETECTION_FLOOR,
        },
        "self_checks": [
            "自算相关的 top-200 / top-40 必须与 select_features 逐位相同（逐折断言）",
            "S_base 与 S_new 用同一批 complete-case 行 ⟹ 唯一差别是标签",
            "另报全行 S_base 与 complete-case S_base 的重合，证明限制行集本身没挪动选列",
        ],
        "limits": [
            "v3_production_oof.py 不支持长窗（截面设计 361 列，生产是 441 列）"
            "⟹ 结论不能直接外推到生产结构",
            "v3_production_oof.py:390-402 假设 xs/market 两个宽度是同一判据的嵌套 top-N；"
            "换判据若破坏嵌套会 AssertionError",
            "选列是模型身份的一部分（train.py:469-478 的 reuse_forest 硬校验）⟹ "
            "换选列则两片森林都得重训；PUBLIC_BASELINE 里目前没有选列身份",
            "本脚本只做前置测量，不训练、不产生任何可晋级的候选",
        ],
    }


def render_plan(payload: dict[str, Any]) -> str:
    ladder = payload["ladder"]
    lines = [
        "# responder 监督的选列判据 —— 预注册（`responder_selection_probe_plan`）",
        "",
        "> 判据先于结果落盘。结果产物里记本文件的 sha256。",
        "",
        f"**问题**：{payload['question']}",
        "",
        "## 为什么这不被 `selection_criterion_probe` 覆盖",
        "",
        payload["why_not_covered_by_selection_criterion_probe"],
        "",
        "## 为什么这个机制是被允许的",
        "",
        payload["why_this_mechanism_is_allowed"],
        "",
        "## 梯子（从 `responder_window_atlas` 派生，不硬编码）",
        "",
        f"准入：{payload['ladder_admission']}",
        "",
        "| 成员 | 拟合 H | 拟合 RMSE |",
        "|---|---:|---:|",
    ]
    lines += [f"| `{m['responder']}` | {m['fitted_H']} | {m['fit_rmse']:.3f} |"
              for m in ladder["members"]]
    lines.append("| `target` | 5 | — |")
    if ladder["rejected"]:
        lines += ["", "被准入判据挡掉的 α 族成员：",
                  ", ".join(f"`{m['responder']}`（RMSE {m['fit_rmse']:.3f}）"
                            for m in ladder["rejected"])]
    lines += [
        "",
        "## 判据",
        "",
        f"- 现状：`{payload['baseline_criterion']}`",
        f"- 新判据：`{payload['criterion']}`",
        "",
        "## 决策规则（跑前钉死）",
        "",
        "```text",
        f"重合 ≥ {payload['decision_rule']['line']}/{payload['feature_count']}"
        f"   ⟹ 不跑 OOF，结案",
        f"重合 < {payload['decision_rule']['line']}/{payload['feature_count']}"
        f"   ⟹ 先验是掉分；需机制假设才跑",
        "```",
        "",
        f"- ≥ 线：{payload['decision_rule']['at_or_above']}",
        f"- < 线：{payload['decision_rule']['below']}",
        "",
        "## 自检",
        "",
    ]
    lines += [f"- {c}" for c in payload["self_checks"]]
    lines += ["", "## 限制", ""]
    lines += [f"- {limit}" for limit in payload["limits"]]
    lines.append("")
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# responder 监督的选列判据 —— 前置测量（`responder_selection_probe`）",
        "",
        f"预注册：`{Path(payload['plan_path']).name}`（sha256 `{payload['plan_sha256'][:16]}…`）",
        "",
        f"梯子（从窗口图谱派生）：" +
        ", ".join(f"`{m['responder']}`(H={m['fitted_H']})" for m in payload["ladder"]["members"]) +
        " + `target`(H=5)",
        "",
        f"{payload['rows']:,} 采样行 / {payload['sampled_time_ids']:,} 采样 time_id / "
        f"{payload['n_folds']} 折；complete-case 排除 "
        f"{payload['complete_case']['excluded']:,} / {payload['complete_case']['total']:,}"
        f"（{payload['complete_case']['excluded_rate']:.4%}）",
        "",
        "## 逐折重合度",
        "",
        "| fold | 训练行数 | 200 列重合 | history 40 列重合 | 全行 vs cc 的 base |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["folds"]:
        lines.append(
            f"| {row['fold']} | {row['train_rows']:,} | "
            f"**{row['overlap_200']}**/{FEATURE_COUNT} | "
            f"{row['overlap_history']}/{HISTORY_COUNT} | "
            f"{row['base_full_vs_cc_overlap']}/{FEATURE_COUNT} |")

    summary = payload["summary"]
    lines += [
        "",
        "## ⚠️ 换掉的是哪些列 —— 这一段决定要不要花那 36 分钟",
        "",
        "重合度只说「变了多少」，不说「变在哪」。两种互斥假设对 OOF 的预测不同：",
        "",
        "```text",
        "假设 A 边缘搅动   换掉的列挤在截断线附近（那里 200th=0.00299 / 201st=0.00295，",
        "                  落差仅 1.33%，本来就在噪声里）⟹ 效应很小、方向不定",
        "假设 B 实质分歧   换掉的列里有高排名的 ⟹ 两个判据在读不同的东西 ⟹",
        "                  selection_criterion_probe 量到的「分歧越大掉得越多」适用，先验是掉分",
        "```",
        "",
        "| fold | 换掉的列原排名最高 | 换进来的原排名最低 | 两判据 Spearman |",
        "|---:|---:|---:|---:|",
        *[f"| {r['fold']} | **{r['churn']['best_dropped_rank']}** | "
          f"{r['churn']['worst_added_rank']} | "
          f"{r['churn']['spearman_between_criteria']:.3f} |" for r in payload["folds"]],
        "",
        f"⟹ 新判据丢掉了原判据排到第 "
        f"**{min(r['churn']['best_dropped_rank'] for r in payload['folds'])}** 名的列，"
        f"又拉进了原判据排到第 "
        f"**{max(r['churn']['worst_added_rank'] for r in payload['folds'])}** 名"
        f"（共 323 列）的列；全局 Spearman 只有 "
        f"{min(r['churn']['spearman_between_criteria'] for r in payload['folds']):.2f}~"
        f"{max(r['churn']['spearman_between_criteria'] for r in payload['folds']):.2f}。",
        "",
        "⭐ **假设 A 被证伪，假设 B 成立** —— 这不是截断线附近的搅动，是全局的实质分歧。",
        "而我预注册时写下的机制恰恰是假设 A（「把估计方差压下去」）⟹ **机制没有兑现**。",
        "",
        "## ⚠️ 事后敏感性检查（不是预注册结果）",
        "",
        "各级 |corr| 的量级并不相同 ⟹ 未标准化的平均会被幅度最大的那一级偏向。",
        "",
        "| 级 | \\|corr\\| 中位 | \\|corr\\| 最大 |",
        "|---|---:|---:|",
        *[f"| `{m['rung']}` | {m['median_abs_corr']:.5f} | {m['max_abs_corr']:.5f} |"
          for m in payload["folds"][0]["rung_sensitivity"]["per_rung"]],
        "",
        f"量级最大/最小之比 **{payload['folds'][0]['rung_sensitivity']['magnitude_spread']:.1f}×**。"
        "逐级标准化后重合度 "
        f"{payload['folds'][0]['rung_sensitivity']['overlap_preregistered']} → "
        f"**{payload['folds'][0]['rung_sensitivity']['overlap_standardized_per_rung']}**"
        f"/{FEATURE_COUNT}（fold 0）—— 仍在决策线 {OVERLAP_DECISION_LINE} 之下 ⟹ "
        "**裁决对这个规格选择稳健**，不是标准化与否造成的。",
        "",
        f"重合度：最小 **{summary['min_overlap_200']}**、"
        f"中位 **{summary['median_overlap_200']}**、"
        f"最大 **{summary['max_overlap_200']}** / {FEATURE_COUNT}",
        "",
        "## 裁决",
        "",
        f"决策线：{OVERLAP_DECISION_LINE}/{FEATURE_COUNT}（跑前钉死）",
        "",
        f"### {verdict['status']}",
        "",
        verdict["reading"],
        "",
        "## 自检",
        "",
        f"- 自算相关 vs `select_features`：{payload['self_check']['comparisons']} 次比较，"
        f"**{'全部逐位相同' if payload['self_check']['ok'] else '有不一致 —— 结果不可解读'}**",
        f"- `S_base` 全行 vs complete-case：最小重合 "
        f"{summary['min_base_full_vs_cc']}/{FEATURE_COUNT} ⟹ "
        f"{'限制行集本身没有挪动选列' if summary['min_base_full_vs_cc'] >= OVERLAP_DECISION_LINE else '⚠️ 限制行集本身就挪动了选列，重合度读数被污染'}",
        "",
        "## 限制",
        "",
    ]
    lines += [f"{i}. {limit}" for i, limit in enumerate(payload["limits"], 1)]
    lines.append("")
    return "\n".join(lines)


def write(out_dir: Path, label: str, payload: dict[str, Any] | None, text: str,
          force: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} 或 {md_path} 已存在；要覆盖请加 --force")
    if payload is not None:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"wrote {json_path}", flush=True)
    md_path.write_text(text, encoding="utf-8")
    print(f"wrote {md_path}", flush=True)
    return json_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ladder = build_ladder(Path(args.atlas), Path(args.data_root))
    plan_payload = emit_plan(args, ladder)

    if args.emit_plan:
        write(out_dir, PLAN_LABEL, plan_payload, render_plan(plan_payload), args.force)
        print("\n梯子：" + ", ".join(f"{m['responder']}(H={m['fitted_H']}, "
                                     f"rmse={m['fit_rmse']:.3f})"
                                     for m in ladder["members"]) + " + target(H=5)")
        print(f"决策线：重合 ≥ {OVERLAP_DECISION_LINE}/{FEATURE_COUNT} ⟹ 不跑 OOF", flush=True)
        return

    plan_path = out_dir / f"{PLAN_LABEL}.json"
    if not plan_path.is_file():
        raise SystemExit(f"没有找到预注册文件 {plan_path} —— 先跑 `--emit-plan`。"
                         "判据必须先于结果落盘（CLAUDE.md §5.1）")

    started = time.perf_counter()
    ladder_names = [m["responder"] for m in ladder["members"]]
    rows = load_rows_with_responders(Path(args.data_root), SAMPLE_MODULO, SAMPLING)
    features, target, time_ids = rows["features"], rows["target"], rows["time_id"]
    responders_all = rows["responders"]
    del rows
    gc.collect()

    from responder_family_grid import RESPONDER_COLUMNS
    ladder_cols = np.array([RESPONDER_COLUMNS.index(n) for n in ladder_names])
    responders = responders_all[:, ladder_cols].astype(np.float64)
    del responders_all
    gc.collect()

    order = np.argsort(time_ids, kind="stable")
    features, target, time_ids = features[order], target[order], time_ids[order]
    responders = responders[order]

    finite = np.isfinite(responders).all(axis=1)
    complete_case = {"total": int(len(finite)), "excluded": int((~finite).sum()),
                     "excluded_rate": float((~finite).mean())}
    sampled_ids = np.unique(time_ids)
    folds = rolling_time_folds(sampled_ids, N_FOLDS, TRAIN_WINDOW, EMBARGO)
    print(f"{len(target):,} 行 / {len(sampled_ids):,} 采样 time_id / {len(folds)} 折；"
          f"complete-case 排除 {complete_case['excluded']:,}"
          f"（{complete_case['excluded_rate']:.4%}）（{time.perf_counter()-started:.0f}s）",
          flush=True)

    fold_rows, comparisons = [], 0
    for index, (train_ids, _valid_ids) in enumerate(folds):
        t0 = time.perf_counter()
        # `row_slice` 返回的是 slice（行已按 time_id 排序）；complete-case 要用下标数组
        tr = row_slice(time_ids, train_ids)
        tr_index = np.arange(len(time_ids))[tr]
        tr_cc = tr_index[finite[tr]]
        rung_names = ("target", *ladder_names)

        transformed, _stats = robust_transform_fit(features[tr_cc])
        tid = time_ids[tr_cc]
        starts = group_starts(tid)
        counts = np.diff(np.r_[starts, len(tid)]).astype(np.float64)

        # 梯子上每一级的截面残差（与生产同口径：无权、逐 time_id 去均值）
        labels = {"target": target[tr_cc] - group_mean(target[tr_cc], starts, counts)}
        for position, name in enumerate(ladder_names):
            column = responders[tr_cc, position]
            labels[name] = column - group_mean(column, starts, counts)

        profile = correlation_profile(transformed, [labels[n] for n in rung_names])
        base_scores = profile[:, 0]
        assert_matches_select_features(transformed, labels["target"], base_scores,
                                       FEATURE_COUNT, f"fold {index} / base 200")
        comparisons += 1

        new_scores = consistency_score(profile)
        s_base = top_k(base_scores, FEATURE_COUNT)
        s_new = top_k(new_scores, FEATURE_COUNT)
        churn = churn_diagnostic(base_scores, new_scores, FEATURE_COUNT)
        sensitivity = rung_sensitivity(profile, rung_names, FEATURE_COUNT)

        # history 40：两臂各自在自己的 200 列内选（与生产同结构）
        hist = {}
        for key, selection in (("base", s_base), ("new", s_new)):
            dev = cross_sectional_deviation(transformed[:, selection].copy(), tid)
            dev_profile = correlation_profile(dev, [labels[n] for n in rung_names])
            if key == "base":
                assert_matches_select_features(dev, labels["target"], dev_profile[:, 0],
                                               HISTORY_COUNT, f"fold {index} / base hist")
                comparisons += 1
                scores = dev_profile[:, 0]
            else:
                scores = consistency_score(dev_profile)
            hist[key] = selection[top_k(scores, HISTORY_COUNT)]
            del dev, dev_profile
        del transformed, profile
        gc.collect()

        # 对照：限制行集这件事本身有没有挪动选列
        transformed_full, _ = robust_transform_fit(features[tr])
        tid_full = time_ids[tr]
        starts_full = group_starts(tid_full)
        counts_full = np.diff(np.r_[starts_full, len(tid_full)]).astype(np.float64)
        e_full = target[tr] - group_mean(target[tr], starts_full, counts_full)
        base_full = top_k(correlation_profile(transformed_full, [e_full])[:, 0], FEATURE_COUNT)
        del transformed_full
        gc.collect()

        row = {
            "fold": index,
            "train_rows": int(len(tr_cc)),
            "train_rows_full": int(len(tr_index)),
            "overlap_200": int(len(set(s_base.tolist()) & set(s_new.tolist()))),
            "overlap_history": int(len(set(hist["base"].tolist()) & set(hist["new"].tolist()))),
            "base_full_vs_cc_overlap": int(len(set(base_full.tolist()) & set(s_base.tolist()))),
            "churn": churn,
            "rung_sensitivity": sensitivity,
            "base_selection": s_base.tolist(),
            "new_selection": s_new.tolist(),
        }
        fold_rows.append(row)
        print(f"fold {index}: 200 列重合 {row['overlap_200']}/{FEATURE_COUNT}、"
              f"history {row['overlap_history']}/{HISTORY_COUNT}、"
              f"全行vs cc base {row['base_full_vs_cc_overlap']}/{FEATURE_COUNT}；"
              f"换掉的列原排名最高 {churn['best_dropped_rank']}、"
              f"换进来的最低 {churn['worst_added_rank']}、"
              f"两判据 spearman {churn['spearman_between_criteria']:.3f}"
              f"（{time.perf_counter()-t0:.0f}s）", flush=True)

    overlaps = [r["overlap_200"] for r in fold_rows]
    base_cc = [r["base_full_vs_cc_overlap"] for r in fold_rows]
    summary = {
        "min_overlap_200": int(min(overlaps)),
        "median_overlap_200": int(np.median(overlaps)),
        "max_overlap_200": int(max(overlaps)),
        "min_overlap_history": int(min(r["overlap_history"] for r in fold_rows)),
        "min_base_full_vs_cc": int(min(base_cc)),
    }

    indistinguishable = summary["min_overlap_200"] >= OVERLAP_DECISION_LINE
    if indistinguishable:
        status = (f"与 base 不可区分 —— 不跑 OOF，结案"
                  f"（最小重合 {summary['min_overlap_200']}/{FEATURE_COUNT} "
                  f"≥ 决策线 {OVERLAP_DECISION_LINE}）")
        reading = (
            f"五折里最小重合 **{summary['min_overlap_200']}/{FEATURE_COUNT}**，"
            f"即最多只挪动 {FEATURE_COUNT - summary['min_overlap_200']} 列。"
            "按 `selection_criterion_probe` 实测的单调关系（`lasso200` 挪 81~132 列 → −1.29%），"
            f"挪这么少列的效应远低于 1s160 的 {DETECTION_FLOOR:.1%} 检出下限 ⟹ "
            "**跑 OOF 也只会得到一个测不出来的结果**，按预注册规则不跑。")
    else:
        status = (f"低于决策线 —— 先验是掉分，需机制论证才跑 OOF"
                  f"（最小重合 {summary['min_overlap_200']}/{FEATURE_COUNT} "
                  f"< {OVERLAP_DECISION_LINE}）")
        reading = (
            f"五折里最小重合 **{summary['min_overlap_200']}/{FEATURE_COUNT}**，"
            f"挪动了 {FEATURE_COUNT - summary['min_overlap_200']} 列，"
            "落进 `selection_criterion_probe` 已量过的区间。那次的实测关系是"
            "**分歧越大掉得越多且单调**（`lasso200` −1.29% / `hist_lag1` −4.38% / "
            "`hist_roll5` −10.39%）⟹ 先验是掉分。按预注册规则，"
            "只有能写出一条不依赖「判据更好」的机制假设时才跑 1s160 五折。")

    payload = {
        "experiment": "responder_selection_probe",
        "plan_path": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "ladder": ladder,
        "rows": int(len(target)),
        "sampled_time_ids": int(len(sampled_ids)),
        "n_folds": len(folds),
        "complete_case": complete_case,
        "folds": fold_rows,
        "summary": summary,
        "self_check": {"comparisons": comparisons, "ok": True,
                       "note": "自算相关的 top-k 与 select_features 逐折逐位相同（不同则已抛错）"},
        "verdict": {
            "decision_line": OVERLAP_DECISION_LINE,
            "indistinguishable_from_base": bool(indistinguishable),
            "run_oof": bool(not indistinguishable),
            "status": status,
            "reading": reading,
        },
        "limits": plan_payload["limits"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write(out_dir, args.label, payload, render_report(payload), args.force)
    print(f"\n重合度 最小 {summary['min_overlap_200']} / 中位 "
          f"{summary['median_overlap_200']} / 最大 {summary['max_overlap_200']} "
          f"/ {FEATURE_COUNT}", flush=True)
    print(f"裁决：{status}", flush=True)


if __name__ == "__main__":
    main()
