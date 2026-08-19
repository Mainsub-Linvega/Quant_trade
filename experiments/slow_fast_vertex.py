"""slow/fast 直线上的抛物线顶点标定 —— ①类后处理，公榜是正确的尺子。

## 机制（这是代数，不是猜测）

把预测沿两个已交过的点连成的直线参数化：

    c(t) = (1−t)·(1.16, 1.16) + t·(0.4496, 1.2530)
    pred(t) = c_s(t)·slow + c_f(t)·fast = pred(0) + t·[pred(1) − pred(0)]

`pred(t)` 对 t 是**逐行线性**的，而比赛指标

    Score = [2⟨y,pred⟩_w − ⟨pred,pred⟩_w] / ⟨y,y⟩_w

对 pred 是二次的 ⟹ **`Score(t)` 是 t 的精确二次式**（只要不触 clip）：

    Score(t) = S0 + b·t + a·t²        a = −⟨d,d⟩_w / D，  d = pred(1) − pred(0)

⭐ `a` **恒为负**（内积的负数），这是构造决定的，不是待检验的假设。所以：

- 三个系数、两个已知点 ⟹ 定不下来；**再取一点即闭式解出顶点**；
- 若解出来 `a ≥ 0`，唯一的解释是**某一点触了限，或哪次提交的模型身份和记录不符**
  （08-13 那类事故的形状）⟹ 停下来查，不要用 t*。

## 增益只取决于顶点位置

由 `S1 − S0 = a(1 − 2t*)` 可得（推导见 `_gain_from_vertex`）：

    gain(t*) = Score(t*) − S1 = (S1 − S0) · (t*−1)² / (2t*−1)

⟹ 花公榜名额之前就能把「S2 落在哪 ⟹ 能拿多少」整张表写死。这就是本脚本
`--emit-plan` 的产物，它**必须先于第三点提交落盘**（CLAUDE.md §5.1：先写变量和判据）。

## 预注册的采纳门槛（跑之前钉死，不因为看到 S2 而改）

1. **完整性**：`S2 < 2·S1 − S0`（等价于 a<0）。不满足 ⟹ 退出，查触限/查模型身份。
2. **私榜半步收缩**：`c_used = c1 + 0.5·(c* − c1)`，即 `t_used = (1+t*)/2`。
   恰好拿到理论增益的 **75%**（`(t*−1)² − ((t*−1)/2)² = 0.75(t*−1)²`）。
   理由不是「0.51× 迁移率」（那是本地→公榜，不是公榜→私榜），而是：曲率小时顶点估计
   本身病态敏感，向已验证点收缩是标准对冲；且抛物线在顶点附近平坦 ⟹ 收一半只损失 25%。
3. **采纳线**：沿用仓库已有的预注册判据（2026-08-17 asset adapter：「|Δ| < 1e-5 视为不可辨别」）。
   **收缩后预测增益 < 1e-5 ⟹ 私榜维持 t=1 不动。**
4. **clip 边界**：`max|pred(t)| < 0.5` 必须逐点核过（本脚本从盘上两份 CSV 实测，不靠估算）。

⚠️ 本脚本**不生成任何提交 CSV**（CLAUDE.md §1.4）。它只产出预注册表和顶点解；
CSV 由用户用 `experiments/slow_fast_csv.py --scale-slow/--scale-fast` 生成并提交。

用法：
    # 第一步：落盘预注册（必须在提交第三点之前）
    .venv/bin/python experiments/slow_fast_vertex.py --emit-plan
    # 第二步：用户提交 t=2 拿到 S2 之后，解顶点
    .venv/bin/python experiments/slow_fast_vertex.py --s2 0.00412345
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---- 两个已交过的点。真值来源：experiments/ledger.csv ----
# t=0：2026-08-13 `v3_hybrid_mkt_shrunk`，单一 scale 1.16
S0_PUBLIC = 0.0039977510
# t=1：2026-08-17 CSV 后处理版 / 2026-08-18 官方 runner 版，两次逐位同分
S1_PUBLIC = 0.0041150085

C0 = (1.16, 1.16)            # t=0 的 (slow, fast) scale —— 即单一 scale
C1 = (0.4496, 1.2530)        # t=1，当前生产（= 1.16 × relative 0.387610 / 1.080181）
PUBLIC_SCALE = 1.16          # meta 里的 prediction_scale；relative = c / PUBLIC_SCALE
PREDICTION_CLIP = 0.5

# ---- 预注册常量：跑之前钉死，不搜索 ----
SHRINK = 0.5                 # 私榜半步收缩 c_used = c1 + SHRINK·(c* − c1)
ADOPT_MIN_DELTA = 1e-5       # 沿用 2026-08-17 asset adapter 的「不可辨别」阈值

PLAN_LABEL = "slow_fast_line_geometry"
SOLVE_LABEL = "slow_fast_vertex_solution"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-plan", action="store_true",
                   help="产出预注册表（默认动作；给了 --s2 就改成解顶点）")
    p.add_argument("--s2", type=float, default=None,
                   help="第三个点（t=--t3）的公榜分数；给了就解顶点")
    p.add_argument("--t3", type=float, default=2.0, help="第三个点的 t（默认 2.0）")
    p.add_argument("--s0", type=float, default=S0_PUBLIC)
    p.add_argument("--s1", type=float, default=S1_PUBLIC)
    p.add_argument("--csv-t0", default=str(_REPO_ROOT / "outputs" / "submission_mkt_shrunk.csv"))
    p.add_argument("--csv-t1",
                   default=str(_REPO_ROOT / "outputs" / "submission_mkt_shrunk_slowfast.csv"))
    p.add_argument("--test-glob", default=str(_REPO_ROOT / "data" / "test" / "*.parquet"))
    p.add_argument("--k-real-steps", type=int, default=2000,
                   help="slow/fast 的真实步窗口，用于锚点交叉验证（生产值 2000）")
    p.add_argument("--skip-anchor-check", action="store_true",
                   help="跳过锚点交叉验证（只在没有 test parquet 时用）")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def coefficients(t: float) -> tuple[float, float]:
    """直线上参数 t 处的 (slow, fast) 绝对 scale。"""
    return (C0[0] + t * (C1[0] - C0[0]), C0[1] + t * (C1[1] - C0[1]))


def relatives(c: tuple[float, float]) -> tuple[float, float]:
    """meta 口径：`slow_fast_*_relative` = 绝对 scale / prediction_scale。"""
    return (c[0] / PUBLIC_SCALE, c[1] / PUBLIC_SCALE)


def _gain_from_vertex(s0: float, s1: float, t_star: float) -> float:
    """gain(t*) = (S1−S0)·(t*−1)²/(2t*−1)。

    推导：Score(t)=S0+bt+at²，t*=−b/(2a) ⟹ b=−2at*；代入 t=1 得 S1−S0=a(1−2t*)。
    又 Score(t*)−Score(1) = −(b+2a)²/(4a) = |a|(t*−1)²，把 |a|=(S1−S0)/(2t*−1) 代回即可。
    """
    return (s1 - s0) * (t_star - 1.0) ** 2 / (2.0 * t_star - 1.0)


def load_line_predictions(csv_t0: Path, csv_t1: Path
                          ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """按 row_id 对齐读出 t=0 与 t=1 的逐行预测，并带回 row_id 供锚点校验用。

    两份 CSV 都是 8 位小数 ⟹ 逐行误差 ~1e-8，对 0.5 这道限幅判定完全够用。
    """
    a = pd.read_csv(csv_t0)
    b = pd.read_csv(csv_t1)
    for frame, path in ((a, csv_t0), (b, csv_t1)):
        if list(frame.columns) != ["row_id", "target"]:
            raise SystemExit(f"{path} 的列不是 [row_id, target]：{list(frame.columns)}")
    merged = a.merge(b, on="row_id", how="inner", suffixes=("_t0", "_t1"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise SystemExit(f"row_id 对不齐：t0={len(a):,} t1={len(b):,} 交集={len(merged):,}")
    return (merged["target_t0"].to_numpy(np.float64),
            merged["target_t1"].to_numpy(np.float64), merged)


def anchor_cross_check(merged: pd.DataFrame, test_glob: str, k_real_steps: int) -> dict:
    """从 t=0 的 CSV 按记录的变换重算 t=1，与盘上那份逐行比对。

    ⚠️ 为什么必须做：`outputs/experiments/public_replay_inventory.json` 里
    `submission_mkt_shrunk.csv` 的归属是 **inferred（按模型名推断）**，不是 sha256 硬校验 ——
    而整条抛物线都架在「它就是 t=0 那一枪」上。若这个锚点错了，解出来的 t* 全是错的，
    而且**不会有任何别的东西提示你**（08-13 那类事故的形状）。
    两份 CSV 相互印证到 8 位小数的舍入地板，就等于两次独立提交的记录自洽。
    """
    import glob as _glob

    import pyarrow.parquet as pq
    from slow_fast_csv import (OOF_C_FAST, OOF_C_SLOW, OOF_GLOBAL_SCALE,
                               causal_trailing_mean)

    files = sorted(_glob.glob(test_glob))
    if not files:
        raise SystemExit(f"no test parquet matched {test_glob}")
    keys = pd.concat([pq.ParquetFile(f).read(columns=["row_id", "time_id", "asset_id"]).to_pandas()
                      for f in files], ignore_index=True)
    frame = merged.merge(keys, on="row_id", how="left", validate="one_to_one")
    if frame["time_id"].isna().any():
        raise SystemExit("有 row_id 在 test parquet 里找不到 —— 锚点校验无法进行")
    frame = frame.sort_values(["time_id", "asset_id"], kind="stable")

    raw = frame["target_t0"].to_numpy(np.float64) / PUBLIC_SCALE
    slow = causal_trailing_mean(raw, frame["time_id"].to_numpy(np.int64),
                                frame["asset_id"].to_numpy(np.int64), k_real_steps)
    rebuilt = np.clip(PUBLIC_SCALE * OOF_C_SLOW / OOF_GLOBAL_SCALE * slow
                      + PUBLIC_SCALE * OOF_C_FAST / OOF_GLOBAL_SCALE * (raw - slow),
                      -PREDICTION_CLIP, PREDICTION_CLIP)
    difference = np.abs(rebuilt - frame["target_t1"].to_numpy(np.float64))
    return {
        "max_abs_difference": float(difference.max()),
        "median_abs_difference": float(np.median(difference)),
        "rows_over_1e_7": int((difference > 1e-7).sum()),
        "rows": int(len(difference)),
        "k_real_steps": int(k_real_steps),
        "passes": bool(difference.max() < 1e-7),
        "note": ("从 t=0 的 CSV 按 ledger 记录的 slow/fast 变换重算 t=1；"
                 "CSV 是 8 位小数 ⟹ 期望 ~1e-8 的舍入地板"),
    }


def clip_geometry(p0: np.ndarray, p1: np.ndarray, probes: tuple[float, ...]) -> dict:
    """沿直线扫 max|pred| 与触限行数，并二分出不触限的最大 t。"""
    d = p1 - p0
    rows = []
    for t in probes:
        values = np.abs(p0 + t * d)
        rows.append({"t": float(t), "max_abs_pred": float(values.max()),
                     "clipped_rows": int((values >= PREDICTION_CLIP).sum()),
                     "coefficients": list(coefficients(t))})
    low, high = 1.0, 20.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if np.abs(p0 + mid * d).max() < PREDICTION_CLIP:
            low = mid
        else:
            high = mid
    return {"probes": rows, "max_clip_free_t": float(low), "rows": int(len(p0))}


def solve_vertex(s0: float, s1: float, s2: float, t3: float) -> dict:
    """由三点解二次式。t3 一般是 2.0，但保持通用（三点必须互异）。"""
    matrix = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [t3 * t3, t3, 1.0]], dtype=np.float64)
    a, b, c = np.linalg.solve(matrix, np.array([s0, s1, s2], dtype=np.float64))
    result = {"a": float(a), "b": float(b), "c": float(c), "t3": float(t3)}
    if a >= 0.0:
        result["concave"] = False
        return result
    t_star = float(-b / (2.0 * a))
    result.update({
        "concave": True,
        "t_star": t_star,
        "score_at_vertex": float(s0 - b * b / (4.0 * a)),
        "gain_full": float(_gain_from_vertex(s0, s1, t_star)),
    })
    return result


def _anchor_note(anchor: dict | None) -> str:
    if anchor is None:
        return "⚠️ 本次跳过了锚点交叉验证（`--skip-anchor-check`）—— S0 的归属未经独立印证。"
    return "\n".join([
        "⚠️ `public_replay_inventory` 里 `submission_mkt_shrunk.csv` 的归属是 "
        "**inferred（按模型名推断）**，不是 sha256 硬校验 —— 而整条抛物线都架在它是 t=0 上。",
        "",
        "从它按 ledger 记录的 slow/fast 变换重算 t=1，与盘上那份逐行比对：",
        "",
        "```text",
        f"逐行比对        {anchor['rows']:,} 行（K_real = {anchor['k_real_steps']}）",
        f"max|Δ|          {anchor['max_abs_difference']:.3e}     ← CSV 是 8 位小数，"
        "舍入地板就在 ~1e-8",
        f"中位|Δ|         {anchor['median_abs_difference']:.3e}",
        f"超过 1e-7 的行  {anchor['rows_over_1e_7']:,}",
        "```",
        "",
        "⟹ **两份 CSV 相互印证，S0 锚点可信。**"
        if anchor["passes"] else "⟹ ❌ **对不上 —— A 线不能开工。**",
    ])


def render_plan(payload: dict) -> str:
    geometry = payload["clip_geometry"]
    lines = [
        "# slow/fast 直线几何与顶点标定预注册（`slow_fast_line_geometry`）",
        "",
        "> ⚠️ **这份文件必须先于第三点提交落盘。** 它写死了「S2 落在哪 ⟹ 拿到多少 ⟹ 改不改交付」，",
        "> 是 CLAUDE.md §5.1「先写变量和判据，再看结果」的执行件。看到 S2 之后不得修改本文件。",
        "",
        "## 1. 直线与两个已知点",
        "",
        "```text",
        f"c(t) = (1−t)·{C0} + t·{C1}",
        f"t=0  c={tuple(round(v, 4) for v in coefficients(0))}  S0 = {payload['s0']:.10f}"
        "   （单一 scale 1.16，ledger 2026-08-13 mkt_shrunk）",
        f"t=1  c={tuple(round(v, 4) for v in coefficients(1))}  S1 = {payload['s1']:.10f}"
        "   （当前生产 slow/fast，ledger 2026-08-17/18）",
        f"S1 − S0 = {payload['s1'] - payload['s0']:.6e}",
        "```",
        "",
        "`Score(t)` 是 t 的**精确**二次式，二次项系数 `a = −⟨d,d⟩_w/D` **恒为负**（构造决定）。",
        "",
        "## 2. 限幅几何（从盘上两份 CSV 实测，不是估算）",
        "",
        f"逐行对齐 **{geometry['rows']:,}** 行；`pred(t) = pred(0) + t·[pred(1) − pred(0)]`。",
        "",
        "| t | c(slow, fast) | max\\|pred\\| | 触限行数 |",
        "|---:|---|---:|---:|",
    ]
    for row in geometry["probes"]:
        c = row["coefficients"]
        lines.append(f"| {row['t']:.2f} | ({c[0]:+.4f}, {c[1]:.4f}) | "
                     f"{row['max_abs_pred']:.6f} | {row['clipped_rows']:,} |")
    lines += [
        "",
        f"**不触 clip 的最大 t ≈ {geometry['max_clip_free_t']:.4f}**"
        f"（clip = ±{PREDICTION_CLIP}）。超过它二次式失效，与现有公榜分不可直接比大小。",
        "",
        f"⟹ 第三点取 **t = {payload['t3']}**，"
        f"c = ({coefficients(payload['t3'])[0]:+.4f}, {coefficients(payload['t3'])[1]:.4f})，"
        f"meta relative = ({relatives(coefficients(payload['t3']))[0]:.6f}, "
        f"{relatives(coefficients(payload['t3']))[1]:.6f})。",
        "⚠️ `slow_relative` 为负是**正常**的 —— 那是个高通滤波，与「预测整体过度平滑」的诊断同向。",
        "",
        "## 2b. 锚点交叉验证（S0 那一枪的归属）",
        "",
        payload["anchor_note"],
        "",
        "## 3. 完整性检查（先于一切解读）",
        "",
        "```text",
        f"S2 必须 < 2·S1 − S0 = {payload['integrity_upper_bound']:.10f}",
        "否则 a ≥ 0 ⟹ 曲线不是凹的 ⟹ 某点触了限，或哪次提交的模型身份与记录不符",
        "（08-13 那类事故的形状）⟹ **停下来查，不要用 t\\***",
        "```",
        "",
        f"顶点方向只看一个比较：**S2 > S0 = {payload['s0']:.7f} ⟹ t\\* > 1**（还该往前走）；",
        "**S2 < S0 ⟹ t\\* < 1**（已经越过顶点）。",
        "",
        "## 4. S2 → t\\* → 增益（**本表在提交前写死**）",
        "",
        "```text",
        "gain(t*) = (S1 − S0) · (t*−1)² / (2t*−1)        只取决于顶点位置",
        "```",
        "",
        "| S2 | t\\* | Score(t\\*) | 相对当前 | 半步收缩后 | 可达？ |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for row in payload["s2_table"]:
        reach = "OK" if row["reachable"] else f"❌ t\\*>{geometry['max_clip_free_t']:.2f}"
        lines.append(f"| {row['s2']:.7f} | {row['t_star']:.3f} | {row['score_at_vertex']:.8f} | "
                     f"{row['relative_gain']:+.2%} | {row['relative_gain_shrunk']:+.2%} | {reach} |")
    lines += [
        "",
        "## 5. 预注册的采纳门槛（钉死，看到 S2 后不得修改）",
        "",
        f"1. **私榜半步收缩**：`c_used = c1 + {SHRINK}·(c* − c1)`，即 `t_used = (1+t*)/2`。",
        f"   恰好拿到理论增益的 **{1 - (1 - SHRINK) ** 2:.0%}**"
        f"（`(t*−1)² − ((1−{SHRINK})(t*−1))² = {1 - (1 - SHRINK) ** 2:.2f}(t*−1)²`）。",
        f"2. **采纳线**：收缩后预测增益 `< {ADOPT_MIN_DELTA:g}` ⟹ **私榜维持 t=1 不动**。",
        f"   这条沿用仓库已有的预注册判据（2026-08-17 asset adapter：|Δ| < 1e-5 视为不可辨别），",
        f"   不是为本实验新造的。换算过来是 **t\\* < {payload['t_star_adopt_threshold']:.3f}**"
        f"（约 S2 < {payload['s2_adopt_threshold']:.7f}）就不改交付。",
        "3. 不因为看到 S2 而改门槛、不搜第二个收缩系数、不换第三点位置重取。",
        "4. 采纳时**不是改 CSV 交私榜** —— 系数必须走",
        "   `scripts/promote_v3_candidate.py --slow-fast-slow-relative/--slow-fast-fast-relative`",
        "   写进候选 meta，再过完整套转正门禁（CLAUDE.md §6）。",
        "",
        "## 6. 期望值的诚实说明",
        "",
        "上表显示典型情形（S2 落在 S0 与 S1 之间或略高）只有 **+0.0%~+0.9%**；",
        "+2% 以上只在 S2 贴近 a→0 边界时出现，而那时 t\\* 已越过 clip 边界、拿不到全部。",
        "⟹ 这次公榜名额值得花（8/23 后作废、不用即归零；三点同源、配对测量极精确），",
        "但**不要指望它翻盘**。",
    ]
    return "\n".join(lines) + "\n"


def render_solution(payload: dict) -> str:
    solve = payload["solve"]
    lines = [
        "# slow/fast 顶点解（`slow_fast_vertex_solution`）",
        "",
        f"预注册文件：`{payload['plan_path']}`（sha256 `{payload['plan_sha256'][:16]}…`）",
        "⟹ 判据先于本结果落盘，可核验。",
        "",
        "## 输入三点",
        "",
        "```text",
        f"t=0            S0 = {payload['s0']:.10f}",
        f"t=1            S1 = {payload['s1']:.10f}",
        f"t={payload['t3']:<12g} S2 = {payload['s2']:.10f}",
        "```",
        "",
        "## 完整性检查",
        "",
        "```text",
        f"S2 < 2·S1 − S0 = {payload['integrity_upper_bound']:.10f} ?   "
        f"{'PASS' if solve['concave'] else '**FAIL**'}",
        f"a = {solve['a']:.6e}   （必须 < 0）",
        "```",
    ]
    if not solve["concave"]:
        lines += [
            "",
            "## ❌ 停止",
            "",
            "`a ≥ 0` ⟹ 曲线不是凹的。这在数学上不可能由「三点都是同一组预测的线性变换、",
            "且都没触限」产生 ⟹ **某个点触了限，或哪次提交的模型身份与记录不符**。",
            "先查，不要用 t\\*。",
        ]
        return "\n".join(lines) + "\n"

    c_star = payload["c_star"]
    c_used = payload["c_used"]
    lines += [
        "",
        "## 顶点",
        "",
        "```text",
        f"t*             = {solve['t_star']:.6f}",
        f"Score(t*)      = {solve['score_at_vertex']:.10f}",
        f"相对当前 S1    = {solve['gain_full'] / payload['s1']:+.4%}   "
        f"（绝对 {solve['gain_full']:+.6e}）",
        f"c*             = ({c_star[0]:+.6f}, {c_star[1]:.6f})",
        f"clip 可达      = {'是' if payload['reachable'] else '否（t* 已越过 clip 边界）'}",
        "```",
        "",
        f"## 预注册的私榜点（半步收缩 {SHRINK}）",
        "",
        "```text",
        f"t_used         = {payload['t_used']:.6f}",
        f"c_used         = ({c_used[0]:+.6f}, {c_used[1]:.6f})",
        f"meta relative  = ({payload['relative_used'][0]:.9f}, {payload['relative_used'][1]:.9f})",
        f"预测增益       = {payload['gain_shrunk']:+.6e}"
        f"  （相对 S1 {payload['gain_shrunk'] / payload['s1']:+.4%}）",
        f"采纳阈值       = {ADOPT_MIN_DELTA:g}",
        "```",
        "",
        f"## 判定：**{payload['verdict']}**",
        "",
        payload["verdict_note"],
    ]
    if payload["adopt"]:
        lines += [
            "",
            "采纳路径（**不是**改 CSV，必须进模型身份）：",
            "",
            "```bash",
            ".venv/bin/python scripts/promote_v3_candidate.py \\",
            "    --candidate outputs/candidates/v3_hybrid_slowfast \\",
            f"    --slow-fast-slow-relative {payload['relative_used'][0]:.9f} \\",
            f"    --slow-fast-fast-relative {payload['relative_used'][1]:.9f} \\",
            "    --off-baseline          # 有意偏离公榜标定，必须显式按下",
            "```",
            "",
            "然后过完整套转正门禁（CLAUDE.md §6），再由用户打包。",
        ]
    return "\n".join(lines) + "\n"


def write_artifacts(out_dir: Path, label: str, payload: dict, text: str, force: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"{json_path} 或 {md_path} 已存在；要覆盖请加 --force")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(text, encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)
    return json_path


def emit_plan(args: argparse.Namespace) -> None:
    p0, p1, merged = load_line_predictions(Path(args.csv_t0), Path(args.csv_t1))
    geometry = clip_geometry(p0, p1, (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    anchor = (None if args.skip_anchor_check
              else anchor_cross_check(merged, args.test_glob, args.k_real_steps))
    if anchor is not None and not anchor["passes"]:
        raise SystemExit(
            f"锚点交叉验证失败：max|Δ| = {anchor['max_abs_difference']:.3e} ≥ 1e-7 ⟹ "
            "t=0 的那份 CSV 不是 slow/fast 变换的输入，S0 锚点存疑，A 线不能开工")
    s0, s1 = args.s0, args.s1
    bound = 2.0 * s1 - s0

    table = []
    for s2 in (0.00380, 0.00390, s0, 0.00405, 0.00410, 0.00412, 0.004150, 0.004180, 0.004200):
        solve = solve_vertex(s0, s1, float(s2), args.t3)
        if not solve["concave"]:
            continue
        gain = solve["gain_full"]
        table.append({
            "s2": float(s2), "t_star": solve["t_star"],
            "score_at_vertex": solve["score_at_vertex"],
            "relative_gain": gain / s1,
            "relative_gain_shrunk": gain * (1.0 - (1.0 - SHRINK) ** 2) / s1,
            "reachable": solve["t_star"] <= geometry["max_clip_free_t"],
        })

    # 采纳阈值换算：0.75·gain_full ≥ ADOPT_MIN_DELTA ⟹ 解 (t*−1)²/(2t*−1) 的阈值
    keep = 1.0 - (1.0 - SHRINK) ** 2
    grid = np.linspace(1.0, 6.0, 500_001)
    ok = keep * _gain_from_vertex(s0, s1, grid) >= ADOPT_MIN_DELTA
    t_threshold = float(grid[ok][0]) if ok.any() else float("nan")
    # 反解对应的 S2：Score(t3) = S0 + b·t3 + a·t3²，其中 a=(S1−S0)/(1−2t*)、b=−2a·t*
    a_thr = (s1 - s0) / (1.0 - 2.0 * t_threshold)
    s2_threshold = float(s0 - 2.0 * a_thr * t_threshold * args.t3 + a_thr * args.t3 ** 2)

    payload = {
        "experiment": "slow_fast_line_geometry",
        "role": "PRE-REGISTRATION —— 必须先于第三点提交落盘",
        "s0": s0, "s1": s1, "t3": args.t3,
        "c0": list(C0), "c1": list(C1),
        "c_t3": list(coefficients(args.t3)),
        "relative_t3": list(relatives(coefficients(args.t3))),
        "prediction_scale": PUBLIC_SCALE, "prediction_clip": PREDICTION_CLIP,
        "integrity_upper_bound": bound,
        "clip_geometry": geometry,
        "anchor_cross_check": anchor,
        "s2_table": table,
        "shrink": SHRINK,
        "shrink_keeps_fraction_of_gain": keep,
        "adopt_min_delta": ADOPT_MIN_DELTA,
        "adopt_min_delta_source": "2026-08-17 asset adapter 预注册判据「|Δ| < 1e-5 视为不可辨别」",
        "t_star_adopt_threshold": t_threshold,
        "s2_adopt_threshold": s2_threshold,
        "csv_t0": str(args.csv_t0), "csv_t1": str(args.csv_t1),
    }
    payload["anchor_note"] = _anchor_note(anchor)
    write_artifacts(Path(args.output_dir), args.label or PLAN_LABEL,
                    payload, render_plan(payload), args.force)
    print(f"\n第三点：t={args.t3}  c=({payload['c_t3'][0]:+.4f}, {payload['c_t3'][1]:.4f})  "
          f"relative=({payload['relative_t3'][0]:.6f}, {payload['relative_t3'][1]:.6f})", flush=True)
    print(f"完整性上界 S2 < {bound:.10f}；采纳阈值 t* ≥ {t_threshold:.3f}"
          f"（约 S2 ≥ {s2_threshold:.7f}）", flush=True)


def solve(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    plan_path = out_dir / f"{PLAN_LABEL}.json"
    if not plan_path.is_file():
        raise SystemExit(
            f"没有找到预注册文件 {plan_path} —— 先跑 `--emit-plan`。"
            "判据必须先于结果落盘（CLAUDE.md §5.1）")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    s0, s1, s2 = plan["s0"], plan["s1"], float(args.s2)
    t3 = plan["t3"]
    if abs(t3 - args.t3) > 1e-12:
        raise SystemExit(f"--t3 {args.t3} 与预注册的 {t3} 不一致 —— 不得事后改第三点位置")
    result = solve_vertex(s0, s1, s2, t3)
    max_t = plan["clip_geometry"]["max_clip_free_t"]

    payload = {
        "experiment": "slow_fast_vertex_solution",
        "plan_path": str(plan_path), "plan_sha256": plan_sha,
        "s0": s0, "s1": s1, "s2": s2, "t3": t3,
        "integrity_upper_bound": plan["integrity_upper_bound"],
        "solve": result,
    }
    if not result["concave"]:
        payload.update({"verdict": "STOP —— a ≥ 0，曲线不是凹的", "adopt": False,
                        "verdict_note": "见 md"})
        write_artifacts(out_dir, args.label or SOLVE_LABEL, payload,
                        render_solution(payload), args.force)
        raise SystemExit("a ≥ 0 ⟹ 停止。查触限 / 查模型身份，不要用 t*。")

    t_star = result["t_star"]
    t_used = 1.0 + SHRINK * (t_star - 1.0)
    gain_shrunk = float(_gain_from_vertex(s0, s1, t_star)
                        - _gain_from_vertex(s0, s1, t_star) * (1.0 - SHRINK) ** 2)
    adopt = bool(gain_shrunk >= ADOPT_MIN_DELTA)
    reachable = bool(t_star <= max_t and t_used <= max_t)
    payload.update({
        "c_star": list(coefficients(t_star)),
        "t_used": t_used, "c_used": list(coefficients(t_used)),
        "relative_used": list(relatives(coefficients(t_used))),
        "gain_shrunk": gain_shrunk,
        "reachable": reachable,
        "max_clip_free_t": max_t,
        "adopt": adopt and reachable,
        "verdict": ("采纳（走 promote，不是改 CSV）" if adopt and reachable else
                    "不改交付，私榜维持 t=1"),
        "verdict_note": (
            f"收缩后预测增益 {gain_shrunk:.3e} ≥ 阈值 {ADOPT_MIN_DELTA:g} ⟹ 采纳。"
            if adopt and reachable else
            f"收缩后预测增益 {gain_shrunk:.3e} < 阈值 {ADOPT_MIN_DELTA:g}"
            "（预注册的「不可辨别」线）⟹ **不改交付**。"
            if not adopt else
            f"t_used={t_used:.4f} 越过 clip 边界 {max_t:.4f} ⟹ 不改交付。"),
    })
    write_artifacts(out_dir, args.label or SOLVE_LABEL, payload,
                    render_solution(payload), args.force)
    print(f"\nt* = {t_star:.6f}   Score(t*) = {result['score_at_vertex']:.10f}"
          f"   相对当前 {result['gain_full'] / s1:+.4%}", flush=True)
    print(f"半步收缩 t_used = {t_used:.6f}  relative = "
          f"({payload['relative_used'][0]:.9f}, {payload['relative_used'][1]:.9f})", flush=True)
    print(f"判定：{payload['verdict']}", flush=True)


def main() -> None:
    args = parse_args()
    if args.s2 is not None:
        solve(args)
    else:
        emit_plan(args)


if __name__ == "__main__":
    main()
