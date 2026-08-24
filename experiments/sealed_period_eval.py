"""密封期尺子：把公榜期尾段封存成本地测试集，让 8/23 之后的决定可测。

## 它要解决什么

8/23 回补的**就是公榜期的标签**（`docs/data_description.md:172`「公榜截止后会发布标签回补
数据，该部分数据将作为扩展训练数据使用」）。实测：`data/test/*.parquet` 326 列、**无
weight / target / responder**；`data/train/*.parquet` 375 列、带 weight/target/47 个 responder。

那段数据只能用一次 —— **当训练数据，或当干净测试集，用来训了就不能再当测试**：

```text
公榜期   time_id 888,480–1,105,919   217,440 个 real time_id   3,217,458 行
现有 OOF 5 折验证行数合计            约 150 万行（每折约 30 万）
⟹ 公榜期评估行数约是现有 OOF 的 2.1×，而且是最近的一段
```

现有 OOF 的检出下限是基准 peak 的 6.1%（1s160）/ 8.7%（3s480），正是它把 `mkt323` (+1.09%)、
`phase_id` (+1.1%)、`responder_00` (+1.38%)、`lag3+lag10` (+0.38%)、扩展窗 (+1.08%) 全判成
「测不出来」。⟹ 若把回补数据全部拿去训练，8/23–8/31 最可能的结局是「重训了，但测不出有没有
用，于是按 RUNBOOK D6 维持现状」。本脚本就是为了避免那个结局。

## ⭐ 为什么不需要反解 raw

官方 runner 输出的是**已乘 scale、已限幅**的值。但比较模型用的是 `peak = A²/B`，而
`f → c·f` 时 `A → cA`、`B → c²B` ⟹ **peak 对全局缩放严格不变**。限幅是唯一的非线性步骤，
所以只要**触限 0 行**，直接拿 runner 输出算 peak 就与拿 raw 算逐位等价。

⚠️ 早期设计里写的是 `raw = pred / prediction_scale` —— 那在 **slow/fast 下是错的**：最终值是
`clip(s_slow·slow + s_fast·fast)`，两个分量各有 scale，除以单一 `prediction_scale` 还原不出
raw。本脚本改为「断言触限 0 行 + 直接算 peak」，少一步除法也少一个错。
`optimal_scale` 仍然会报，但它**不是**尺度不变量，只在「相对该候选自己提交时的 scale」这个
意义上可读。

## ⚠️ 不写提交格式 CSV

官方 runner 必须给一个 `output_path`，本脚本把它指向 `tempfile.TemporaryDirectory` 内，读完
即随临时目录销毁；预测落 `outputs/cache/sealed_pred_<label>.npz`。两个理由：
(a) CLAUDE.md §1.4「不生成公榜 CSV」；(b) 盘上不留看起来像提交文件、8/31 可能被误传的东西
（ROADMAP P0 已经因为盘上三个 zip 写过一次警告）。

用法：
    # ① 判据先落盘（跑之前钉死）
    .venv/bin/python experiments/sealed_period_eval.py --emit-plan

    # ② 出一个候选在 test 期的预测（约 6 分钟，走官方 runner）
    .venv/bin/python experiments/sealed_period_eval.py \
        --candidate strategies/v3_hybrid --label seal_production

    # ③ 裁决（8/23 有标签之后）
    .venv/bin/python experiments/sealed_period_eval.py \
        --baseline seal_production --arms mkt323=seal_mkt323 \
        --labels <回补数据目录> --label sealed_tier2_mkt323
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent),
              str(_REPO_ROOT / "timeseries_api")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.metric import scale_invariant_score  # noqa: E402

# ============================ 预注册常量（跑之前钉死，改动必须是有意的） ============================

SEAL_TIME_IDS = 60_000
"""密封测试集大小，单位 real time_id。2026-08-20 由用户选定（约 88.8 万行、占回补 27.6%）。"""

N_BLOCKS = 4
"""密封期切成几块。⚠️ 这是唯一必须在标签到手前定死的旋钮 —— 看过答案再选分块 = 看结果选参。"""

EMBARGO_REAL_TIME_IDS = 30
"""训练段与密封段之间的 embargo。= OOF 协议的 6 采样步 × sample_modulo 5（src/validation.py:40）。"""

MIN_POSITIVE_BLOCKS = 3
"""≥3/4 块 —— RUNBOOK D2「≥4/5 折」在 4 块下的映射。"""

MIN_RELATIVE_GAIN = 0.03
"""RUNBOOK D2 的「相对 ≥ 3%」，读的是块均相对增益。"""

BOOTSTRAP_CHUNKS_PER_BLOCK = 25
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026
CI_ALPHA = 0.05
"""配对 block bootstrap：每块切 25 个 chunk（600 real time_id）⟹ 全期 100 个 chunk，
有放回重抽 2000 次取 95% 分位。peak 的三个分量都是**和**，所以按 chunk 的充分统计量
累加即可精确重算，不必每次重扫行。"""

PLAN_LABEL = "sealed_period_plan"
CACHE_PREFIX = "sealed_pred_"

TIER1: tuple[tuple[str, str, float, str], ...] = (
    ("production_slowfast", "strategies/v3_hybrid", 0.0041150085, "基准，必须先打"),
    ("mkt_shrunk", "outputs/candidates/v3_hybrid_mkt_shrunk", 0.0039977510,
     "密封期能不能重现 +2.93% 这个已知差"),
    ("mktwe", "outputs/candidates/v3_hybrid_r480_pb_hist_mktwe", 0.0039673997, "第三个标定点"),
    ("asset_adapter", "outputs/candidates/v3_asset_cross_3s480_shrink500", 0.0039908352,
     "OOF 说 +1.99%、公榜说 −0.17% —— 密封期站哪边"),
    ("r960", "outputs/candidates/v3_hybrid_r960_pb_hist_mktwe", 0.0037609312, "负控制（−5.20%）"),
    ("xs_shrunk", "outputs/candidates/v3_hybrid_xs_shrunk", 0.0035771492, "负控制（−9.84%）"),
)
"""零重训成本、盘上现成。这一层的产出不是找收益，是**标定这把尺子的检出下限**。"""

TIER2: tuple[tuple[str, str], ...] = (
    ("mkt323", "市场块选列 323（ROADMAP 已写明「回补数据后按原规格复验一次」，+1.09%/3-of-5）"),
    ("v4r_regime", "V4-R 压缩 market regime（ROADMAP §3.7 唯一保留的原规格复验项）"),
    ("phase_id", "phase_id 作特征（+1.1%、3/5 折）"),
    ("lag3_lag10", "lag3+lag10（+0.38%、3/5 折、drop-best 为负）"),
    ("responder_00", "pure_e/responder_00 辅助（+1.38%、3/4 折、drop-best 为负）"),
)
"""每个一次 3s480 重训。⚠️ 只有 Tier 1 标定出的检出下限低于各自点估计时才开跑。"""


# ============================ 几何：密封段与分块 ============================


def seal_geometry(test_time_id_min: int, test_time_id_max: int,
                  seal_time_ids: int = SEAL_TIME_IDS,
                  n_blocks: int = N_BLOCKS,
                  embargo: int = EMBARGO_REAL_TIME_IDS) -> dict[str, Any]:
    """由 test 期的 time_id 范围推出密封段、embargo 和决策期训练段的边界。

    全部是闭区间。分块必须整除 —— 预注册的东西不允许「差不多」。
    """
    span = test_time_id_max - test_time_id_min + 1
    if seal_time_ids <= 0 or seal_time_ids > span:
        raise SystemExit(f"密封段 {seal_time_ids} 超出 test 期跨度 {span}")
    if seal_time_ids % n_blocks != 0:
        raise SystemExit(f"密封段 {seal_time_ids} 不能被 {n_blocks} 块整除 —— "
                         "预注册的分块必须精确，不接受余数落在最后一块")
    seal_start = test_time_id_max - seal_time_ids + 1
    per_block = seal_time_ids // n_blocks
    blocks = [{"block": i,
               "time_id_min": seal_start + i * per_block,
               "time_id_max": seal_start + (i + 1) * per_block - 1}
              for i in range(n_blocks)]
    return {
        "test_time_id_min": int(test_time_id_min),
        "test_time_id_max": int(test_time_id_max),
        "test_span_time_ids": int(span),
        "seal_time_ids": int(seal_time_ids),
        "seal_time_id_min": int(seal_start),
        "seal_time_id_max": int(test_time_id_max),
        "embargo_real_time_ids": int(embargo),
        "decision_train_time_id_max": int(seal_start - embargo - 1),
        "n_blocks": int(n_blocks),
        "block_time_ids": int(per_block),
        "blocks": blocks,
    }


# ============================ 门禁前置：限幅与标签 ============================


def clip_hits(prediction: np.ndarray, clip: float) -> int:
    """触限行数 —— 只数，不判。"""
    return int(np.sum(np.abs(prediction) >= clip - 1e-12))


def assert_no_clip_hits(prediction: np.ndarray, clip: float, *, where: str = "") -> int:
    """限幅是唯一的非线性步骤；触限 > 0 行时 peak 不再与 raw 上的 peak 等价。

    ⚠️⚠️ 2026-08-24 修**作用域**：本断言此前作用在**全量 test 期**（3,217,458 行）的预测上，
    而 peak 只由**密封段**那 856,319 行算出来（`arm_view` 紧接着的每一步都是 `pred[seal]`）
    ⟹ 一行落在评估窗口**之外**的触限就能毙掉整个臂。

    实测代价：`r960`（两个负控制之一）在全窗触限**恰好 1 行 / 3,217,458**，
    而**密封段内 0 行**、段内 `max|pred| = 0.4620392` —— 按旧作用域它会被
    `SystemExit` 踢出标定，而它对 peak 的有效性其实毫无影响。
    更糟的是脚本用 `set -e` 串跑时，它还会连带中断排在后面的臂。

    ⟹ 判据只看**进入 peak 计算的那些行**；全窗的计数仍然记进 npz 元数据（`clip_hits_full`），
    因为「榜上那份 CSV 触没触限」是交付时要知道的事，只是不该由它否决段内比较。
    """
    touched = clip_hits(prediction, clip)
    if touched:
        raise SystemExit(
            f"{touched:,} 行触到限幅 {clip}{where} ⟹ peak 与 raw 上的 peak 不再等价，"
            "本脚本拒绝在这种情况下出 peak 比较（CLAUDE.md §5.5）。")
    return touched


def load_backfill_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读回补标签，返回 (row_id, target, weight)。

    ⚠️ 缺 `weight` 直接拒绝。公榜口径是 Σw(y−ŷ)²/Σw·y²，静默退化成无权会得到一个
    **看起来正常但错的**分数 —— 与 `public_replay.py:158` 同一道守卫。
    """
    import pyarrow.parquet as pq

    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"{path} 下没有 parquet")
    want = ["row_id", "target", "weight"]
    parts = []
    for p in files:
        names = set(pq.ParquetFile(p).schema_arrow.names)
        missing = [c for c in want if c not in names]
        if missing:
            raise SystemExit(
                f"{p.name} 缺列 {missing}。公榜口径是 Σw(y−ŷ)²/Σw·y²，"
                "**没有 weight 就不能算**，本脚本拒绝静默退化成无权。")
        parts.append(pq.ParquetFile(p).read(columns=want).to_pandas())
    import pandas as pd
    frame = pd.concat(parts, ignore_index=True)
    return (frame["row_id"].to_numpy(np.int64),
            frame["target"].to_numpy(np.float64),
            np.maximum(frame["weight"].to_numpy(np.float64), 0.0))


def align_labels(pred_row_id: np.ndarray, label_row_id: np.ndarray) -> np.ndarray:
    """把标签按预测的 row_id 顺序对齐，返回索引。**拒绝部分 join。**

    回补数据若换了 row_id 口径，部分 join 会算出一个看起来正常但错的分数 —— 8/23 拿它做
    采纳决策就完了。与 `public_replay.py:205` 同一道守卫，但这里升级为硬失败：那边是
    复算历史提交（少几份还能看），这里是**裁决**。
    """
    order = np.argsort(label_row_id, kind="stable")
    sorted_ids = label_row_id[order]
    pos = np.searchsorted(sorted_ids, pred_row_id)
    inside = pos < len(sorted_ids)
    hit = np.zeros(len(pred_row_id), dtype=bool)
    hit[inside] = sorted_ids[pos[inside]] == pred_row_id[inside]
    if not hit.all():
        raise SystemExit(
            f"标签只覆盖 {hit.mean():.4%} 的预测行 ⟹ 不能用于裁决。"
            "若回补数据的 row_id 与 test 不同口径，改按 (time_id, asset_id) 连接。")
    return order[pos]


# ============================ 分块打分与配对 bootstrap ============================


def _chunk_sums(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray,
                chunk_id: np.ndarray, n_chunks: int) -> np.ndarray:
    """每个 chunk 的三个和：Σw·y·p、Σw·p²、Σw·y²。peak 的分量全是和 ⟹ 可累加重算。"""
    out = np.zeros((n_chunks, 3), dtype=np.float64)
    out[:, 0] = np.bincount(chunk_id, weights=weight * target * prediction, minlength=n_chunks)
    out[:, 1] = np.bincount(chunk_id, weights=weight * prediction * prediction, minlength=n_chunks)
    out[:, 2] = np.bincount(chunk_id, weights=weight * target * target, minlength=n_chunks)
    return out


def _peak_from_sums(sums: np.ndarray) -> float:
    """sums = [Σw·y·p, Σw·p², Σw·y²] ⟹ peak = A²/B，其中 A、B 共用分母 Σw·y²。"""
    numerator, energy, denominator = sums
    if denominator <= 0.0 or energy <= 0.0:
        return 0.0
    a = numerator / denominator
    b = energy / denominator
    return float(a * a / b)


def block_metrics(time_id: np.ndarray, target: np.ndarray, prediction: np.ndarray,
                  weight: np.ndarray, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐块出 `scale_invariant_score`（A/B/peak/optimal_scale）。"""
    out = []
    for block in blocks:
        mask = (time_id >= block["time_id_min"]) & (time_id <= block["time_id_max"])
        rows = int(mask.sum())
        if rows == 0:
            raise SystemExit(f"block {block['block']} 没有行 —— 密封段边界与数据对不上")
        score = scale_invariant_score(target[mask], prediction[mask], weight[mask])
        out.append({"block": block["block"], "rows": rows,
                    "time_id_range": [block["time_id_min"], block["time_id_max"]],
                    "A": score["A"], "B": score["B"], "peak": score["peak"],
                    "optimal_scale": score["optimal_scale"]})
    return out


def paired_block_bootstrap(base_sums: np.ndarray, arm_sums: np.ndarray,
                           samples: int = BOOTSTRAP_SAMPLES,
                           seed: int = BOOTSTRAP_SEED,
                           alpha: float = CI_ALPHA) -> dict[str, Any]:
    """配对 block bootstrap：对 chunk 有放回重抽，重算 pooled peak 之差的相对量。

    配对体现在**两个臂用同一批重抽 chunk** —— 否则测的是两个独立样本之差，方差会大得多。
    """
    n_chunks = base_sums.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_chunks, size=(samples, n_chunks))
    relative = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = draws[i]
        base_peak = _peak_from_sums(base_sums[idx].sum(axis=0))
        arm_peak = _peak_from_sums(arm_sums[idx].sum(axis=0))
        relative[i] = (arm_peak / base_peak - 1.0) if base_peak > 0.0 else np.nan
    finite = relative[np.isfinite(relative)]
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"samples": int(samples), "chunks": int(n_chunks),
            "ci_low": float(lo), "ci_high": float(hi),
            "median": float(np.median(finite))}


# ============================ 判据 ============================


def judge(baseline: list[dict[str, Any]], arm: list[dict[str, Any]],
          pooled_baseline: dict[str, float], pooled_arm: dict[str, float],
          bootstrap: dict[str, Any] | None = None,
          detection_floor: float | None = None) -> dict[str, Any]:
    """RUNBOOK D2 六道门禁在 4 块密封期上的映射。

    ⚠️ 第七道「超过检出下限」在 Tier 1 标定出来之前是**未知**，不是自动通过。
    传 `detection_floor=None` 时该项判 `None`，整体判定同时标 `pending_calibration`。
    """
    if len(baseline) != len(arm):
        raise SystemExit("两臂块数不同 —— 不是配对比较")
    per_block = []
    for b, a in zip(baseline, arm):
        if b["block"] != a["block"] or b["rows"] != a["rows"]:
            raise SystemExit(f"block {b['block']} 的行数在两臂之间不同 —— 不是配对比较")
        relative = (a["peak"] / b["peak"] - 1.0) if b["peak"] > 0.0 else float("nan")
        per_block.append({"block": b["block"], "rows": b["rows"],
                          "baseline_peak": b["peak"], "arm_peak": a["peak"],
                          "relative": relative})
    relatives = np.array([r["relative"] for r in per_block], dtype=np.float64)
    block_mean = float(np.mean(relatives))
    positive = int(np.sum(relatives > 0.0))
    drop_best = float(np.mean(np.sort(relatives)[:-1]))

    delta_a = (pooled_arm["A"] / pooled_baseline["A"] - 1.0) if pooled_baseline["A"] else float("nan")
    delta_b = (pooled_arm["B"] / pooled_baseline["B"] - 1.0) if pooled_baseline["B"] else float("nan")
    pooled_relative = ((pooled_arm["peak"] / pooled_baseline["peak"] - 1.0)
                       if pooled_baseline["peak"] > 0.0 else float("nan"))

    gates: dict[str, Any] = {
        "1_block_mean_positive": bool(block_mean > 0.0),
        "2_positive_blocks_at_least": bool(positive >= MIN_POSITIVE_BLOCKS),
        "3_survives_drop_best_block": bool(drop_best > 0.0),
        "4_relative_gain_at_least": bool(block_mean >= MIN_RELATIVE_GAIN),
        "5_two_delta_a_gt_delta_b": bool(2.0 * delta_a > delta_b),
        "6_paired_ci_lower_positive": (bool(bootstrap["ci_low"] > 0.0) if bootstrap else None),
        "7_above_detection_floor": (bool(block_mean >= detection_floor)
                                    if detection_floor is not None else None),
    }
    pending = [k for k, v in gates.items() if v is None]
    decided = [v for v in gates.values() if v is not None]
    return {
        "per_block": per_block,
        "block_mean_relative": block_mean,
        "pooled_relative": pooled_relative,
        "positive_blocks": positive,
        "n_blocks": len(per_block),
        "drop_best_block_relative": drop_best,
        "delta_A": float(delta_a),
        "delta_B": float(delta_b),
        "bootstrap": bootstrap,
        "detection_floor": detection_floor,
        "gates": gates,
        "pending_gates": pending,
        "passes": bool(all(decided) and not pending),
        "verdict": ("PASS" if (all(decided) and not pending)
                    else "PENDING_CALIBRATION" if (all(decided) and pending) else "FAIL"),
    }


# ============================ 预注册产物 ============================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_payload(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "sealed_period_plan",
        "registered": "2026-08-20",
        "question": "把公榜期尾段封存成密封测试集，标定它的检出下限，再决定哪些「测不出来」的候选值得复验",
        "geometry": geometry,
        "gates": {
            "1_block_mean_positive": "块均相对增益 > 0",
            "2_positive_blocks_at_least": f"≥{MIN_POSITIVE_BLOCKS}/{N_BLOCKS} 块为正（D2「≥4/5 折」的映射）",
            "3_survives_drop_best_block": "去掉最好一块后仍 > 0",
            "4_relative_gain_at_least": f"块均相对增益 ≥ {MIN_RELATIVE_GAIN:.0%}",
            "5_two_delta_a_gt_delta_b": "2ΔA > ΔB（pooled）",
            "6_paired_ci_lower_positive": "配对 block bootstrap 的 CI 下界 > 0",
            "7_above_detection_floor": "超过检出下限 —— ⚠️ 该值由 Tier 1 标定，标定前判 None 而非自动通过",
        },
        "bootstrap": {"chunks_per_block": BOOTSTRAP_CHUNKS_PER_BLOCK,
                      "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                      "ci": f"{1 - CI_ALPHA:.0%} percentile"},
        "reading": ("比较用 peak = A²/B（尺度不变），不用单点分（CLAUDE.md §5.5）。"
                    "官方 runner 输出已乘 scale、已限幅；触限 0 行时 peak 与 raw 上的 peak 逐位等价，"
                    "所以不反解 raw —— slow/fast 下两个分量各有 scale，除以单一 prediction_scale 是错的。"),
        "stage_order": [
            "D0 审计 + D0.3 修尺子（不变）",
            "D0.4 用 Tier 1 盘上现成候选标定密封期尺子的检出下限",
            "D1/D2 重训 + 裁决 —— 训练止于 decision_train_time_id_max，评分在密封期",
            "D3+ Tier 2 复验（排在 recency 之后，RUNBOOK 顺序不变）",
            "D4.5 决定拍完后，最终交付件用 100% 数据重训，只过 D4 转正门禁（机械正确性）",
            "D5/D6 不变",
        ],
        "d4_5_risk": ("最终交付件训练在没有任何评估覆盖过的数据上。缓解：与刚被密封期验过的是"
                      "同一结构，且 D4 覆盖机械正确性。回退：D4.5 任一门禁不过就交决策期那份"
                      "（它被密封期评过），再不行交当前生产 —— 三层都有落盘产物。"),
        "tier1": [{"name": n, "model_dir": d, "published_public_score": s, "question": q}
                  for n, d, s, q in TIER1],
        "tier2": [{"name": n, "note": q} for n, q in TIER2],
        "not_doing": [
            "不生成提交格式 CSV，不打包 zip，不 commit（CLAUDE.md §1.2 / §1.4）",
            "不动生产目录和 hybrid_meta.json",
            "8/23 之前不打公榜枪",
            "不用密封期反复调参：候选清单预注册，每个候选只打一次分",
            "不因为看到密封期结果改块数、改门槛、或往清单里加项",
            "不把 Tier 2 重训排到 recency 前面",
        ],
    }


def render_plan(payload: dict[str, Any]) -> str:
    geometry = payload["geometry"]
    lines = [
        "# 密封期尺子 —— 预注册", "",
        f"注册日期：**{payload['registered']}**（标签到手**之前**）", "",
        f"**问题**：{payload['question']}", "",
        "## 切分", "", "```text",
        f"test 期        {geometry['test_time_id_min']:,} – {geometry['test_time_id_max']:,}"
        f"   {geometry['test_span_time_ids']:,} 个 real time_id",
        f"密封测试集     {geometry['seal_time_id_min']:,} – {geometry['seal_time_id_max']:,}"
        f"   {geometry['seal_time_ids']:,} 个",
    ]
    for block in geometry["blocks"]:
        lines.append(f"  block {block['block']}      {block['time_id_min']:,} – "
                     f"{block['time_id_max']:,}   {geometry['block_time_ids']:,} 个")
    lines += [
        f"embargo        {geometry['embargo_real_time_ids']} real time_id"
        f"（= OOF 的 6 采样步 × sample_modulo 5）",
        f"决策期训练     ≤ {geometry['decision_train_time_id_max']:,}",
        "```", "",
        "## 门禁", "", "| # | 判据 |", "|---|---|",
    ]
    for key, text in payload["gates"].items():
        lines.append(f"| {key.split('_')[0]} | {text} |")
    boot = payload["bootstrap"]
    lines += [
        "", f"配对 block bootstrap：每块 {boot['chunks_per_block']} 个 chunk、"
        f"重抽 {boot['samples']} 次、seed {boot['seed']}、{boot['ci']}。", "",
        "## 读数口径", "", payload["reading"], "",
        "## 阶段顺序", "",
    ]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(payload["stage_order"])]
    lines += ["", f"⚠️ D4.5 的风险：{payload['d4_5_risk']}", "",
              "## Tier 1 — 零重训成本，盘上现成", "",
              "| 候选 | 模型目录 | 公榜真值 | 这一枪问什么 |", "|---|---|---:|---|"]
    for item in payload["tier1"]:
        lines.append(f"| `{item['name']}` | `{item['model_dir']}` | "
                     f"{item['published_public_score']:.10f} | {item['question']} |")
    lines += ["", "⟹ 六个已知公榜真值 + 块级方差 = **这把尺子的检出下限**。它决定 Tier 2 值不值得花重训。",
              "", "## Tier 2 — 每个一次 3s480 重训", ""]
    for item in payload["tier2"]:
        lines.append(f"- `{item['name']}` —— {item['note']}")
    lines += ["", "⚠️ 只有 Tier 1 标定出的检出下限低于各自点估计时才开跑。", "",
              "## 明确不做", ""]
    lines += [f"- {s}" for s in payload["not_doing"]]
    return "\n".join(lines) + "\n"


# ============================ 候选推理 ============================


def test_time_id_bounds(data_root: Path) -> tuple[int, int]:
    """从 test 分区读出 time_id 的真实范围 —— 不硬编码，几何由数据决定。"""
    import pyarrow.parquet as pq
    lo, hi = None, None
    for path in sorted((data_root / "test").glob("*.parquet")):
        column = pq.ParquetFile(path).read(columns=["time_id"]).to_pandas()["time_id"]
        lo = int(column.min()) if lo is None else min(lo, int(column.min()))
        hi = int(column.max()) if hi is None else max(hi, int(column.max()))
    if lo is None:
        raise SystemExit(f"{data_root / 'test'} 下没有 parquet")
    return lo, hi


# ⚠️ 2026-08-24 补 `long_window` 与 history 两项。此前这张表硬编码 9 个键，
# 而 `long_window` 是 08-21 转正的那块结构（441 列 vs 361 列）——报告号称打印「模型身份」
# 却分不出这两种模型，与 08-18/19/21/23/24 同型。
IDENTITY_KEYS = ("blend_weight", "num_iteration", "prediction_scale", "prediction_clip",
                 "market_lambda", "cross_section_weighted", "long_window", "history_window",
                 "slow_fast_window", "slow_fast_slow_relative", "slow_fast_fast_relative")


def baseline_overrides(model_dir: Path, *, slow_fast: bool = False) -> dict[str, float]:
    """把候选目录的**占位** meta 拨回公榜口径 —— 否则量的不是有公榜真值的那个模型。

    ⚠️⚠️ 2026-08-24 事故（首轮 Tier 1 标定作废，已重跑）：本函数之前不存在，
    调用处传的是 `stage({}, ...)`，即**原样使用候选目录的 meta**。而
    `promote_v3_candidate.PUBLIC_BASELINE` 的注释早就写明：

    > 所有公榜好成绩都是 `experiments/variant_submission.py --blend-weight 1.0 --scale 1.16`
    > 在**临时副本**上覆写出来的，生产 meta 从来没被同步过。

    候选目录落的是 `train.py` 的本地占位 `blend_weight=0.5 / prediction_scale=0.856`
    ⟹ 首轮六个臂里有**四个**跑的根本不是那个有公榜真值的模型。
    实测证据（密封期 `max|pred|` 对公榜 CSV 的 `max|pred|`）：

    ```text
    production_slowfast  0.402099 / 0.4020988  比值 1.0000  ✅（生产 meta 本来就是 1.0/1.16）
    asset_adapter        0.414722 / 0.4147218  比值 1.0000  ✅
    mkt_shrunk           0.243468 / 0.4046632  比值 0.6017  ❌
    mktwe                0.301355 / 0.4489862  比值 0.6712  ❌
    r960                 0.325353 / 0.5000000  比值 0.6507  ❌
    xs_shrunk            0.291319 / 0.4217869  比值 0.6907  ❌
    ```

    ⭐ 四个比值**互不相同**是关键证据：若只差 `prediction_scale`，比值应恒为
    0.856/1.16 = 0.73793。比值散开说明 `blend_weight` 0.5→1.0 也在里面，
    而那**不是缩放、是另一个模型**（`ê = (1−w)·ê_ridge + w·ê_lgbm`）。
    peak = A²/B 对缩放不变救不了这一项。

    `blend_weight` 与 `prediction_scale` 都从 `PUBLIC_BASELINE` 取，不在这里写常量
    （CLAUDE.md §7）。对生产目录是无操作（它的 meta 已经是这两个值）。
    """
    import sys as _sys
    _sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from promote_v3_candidate import PUBLIC_BASELINE
    finally:
        _sys.path.remove(str(_REPO_ROOT / "scripts"))
    meta = json.loads((resolve_model_dir(model_dir) / "hybrid_meta.json")
                      .read_text(encoding="utf-8"))
    keys = ["blend_weight", "prediction_scale"]
    # ⚠️ 2026-08-24：slow/fast 三键**默认不补**，必须显式要（`--slow-fast`）。
    # 两种臂的「按交付口径」是**不同的**：
    #   · Tier 1 那五个历史臂的公榜真值是在 slow/fast 转正**之前**打的 ⟹ 补上就不再是那个模型
    #     （实测佐证：不补时它们的 max|pred| 与留档公榜 CSV 逐位对上，比值 1.0000）；
    #   · 8/23 之后的重训候选必须**补上** —— `train.py` 的 CLI 里没有 slow/fast 概念，
    #     任何重训候选都必定缺这三个键，而 `main.py:222` 是
    #     `PredictionTrail(...) if window else None` ⟹ **缺键静默关掉 slow/fast**。
    #     不补就等于拿「扩展数据 + 丢了 slow/fast」去比「当前数据 + 有 slow/fast」，
    #     两个变量混在一起（CLAUDE.md §5.2），而 slow/fast 公榜实测值 +2.93%。
    # 沿用 RUNBOOK D1 坑 1 的 (b) 路：**沿用当前标定**，不借机重标定 ——
    # 这样候选与基准的唯一差别就是训练数据。
    if slow_fast:
        keys += ["slow_fast_window", "slow_fast_slow_relative", "slow_fast_fast_relative"]
    overrides: dict[str, float] = {}
    for key in keys:
        want = PUBLIC_BASELINE[key]          # ⚠️ 取**原值**：`slow_fast_window` 是 int，
        try:                                  # 转成 2000.0 会让 staged meta 与生产 meta 分型
            actual = float(meta[key])
        except (KeyError, TypeError, ValueError):
            actual = float("nan")
        # ⚠️ 不能写成 `if abs(actual - want) >= 1e-12`：`actual` 是 NaN（缺键/非数）时
        # 该式恒为 **False** ⟹ 缺键被静默判成「不用覆写」。缺键必须算偏离。
        if not (abs(actual - float(want)) < 1e-12):
            overrides[key] = want
    return overrides


def resolve_model_dir(path: Path) -> Path:
    """两种布局都接受：生产的 meta 在 `model/` 下，候选的 meta 在本层。

    `variant_submission.stage()` 要的是**模型产物目录本身**（它整个 copytree 成 `package/model`）。
    给错一层只会在 stage 里抛一个裸 `FileNotFoundError`，看不出是口径问题 —— 所以在这里判死。
    """
    path = Path(path)
    if (path / "hybrid_meta.json").is_file():
        return path
    if (path / "model" / "hybrid_meta.json").is_file():
        return path / "model"
    raise SystemExit(f"{path} 下找不到 hybrid_meta.json（本层和 model/ 都找过）")


def sealed_rows(row_id: np.ndarray, data_root: Path) -> np.ndarray:
    """给 test 期的 row_id 序列返回「落在密封段内」的布尔掩码。

    几何从 `seal_geometry` 派生（唯一真值源）；time_id 从 test 分区按 row_id 对齐读入。
    用 searchsorted 对齐而不是逐行查字典 —— 320 万行上后者要好几秒。
    """
    import pandas as pd
    import pyarrow.parquet as pq

    index = pd.concat([pq.read_table(path, columns=["row_id", "time_id"]).to_pandas()
                       for path in sorted((data_root / "test").glob("*.parquet"))],
                      ignore_index=True)
    idx_row = index["row_id"].to_numpy(np.int64)
    idx_time = index["time_id"].to_numpy(np.int64)
    order = np.argsort(idx_row, kind="stable")
    idx_row, idx_time = idx_row[order], idx_time[order]

    pos = np.searchsorted(idx_row, row_id)
    if pos.max() >= len(idx_row) or not np.array_equal(idx_row[np.minimum(pos, len(idx_row) - 1)],
                                                       row_id):
        raise SystemExit("预测的 row_id 与 test 索引对不齐 —— 口径对不上，拒绝出掩码")

    seal_min = seal_geometry(int(idx_time.min()), int(idx_time.max()))["seal_time_id_min"]
    return idx_time[pos] >= seal_min


def run_candidate(model_dir: Path, data_root: Path, cache_path: Path,
                  *, slow_fast: bool = False) -> dict[str, Any]:
    """走官方 runner 在 test 期出预测，落 npz。**不留提交格式 CSV。**"""
    model_dir = resolve_model_dir(model_dir)
    from runner import run_strategy  # 官方 runner，只读引用
    from variant_submission import stage

    import pandas as pd

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sealed_period_") as workspace:
        workspace = Path(workspace)
        package = stage(baseline_overrides(model_dir, slow_fast=slow_fast),
                        workspace, model_dir)
        staged_meta = json.loads(
            (package / "model" / "hybrid_meta.json").read_text(encoding="utf-8"))
        identity = {k: staged_meta.get(k) for k in IDENTITY_KEYS}
        identity["n_history_positions"] = len(staged_meta.get("history_positions") or [])
        identity["n_lgbm_files"] = len(staged_meta.get("lgbm_model_files", []))
        identity["n_market_files"] = len(staged_meta.get("market_model_files", []))
        print("  入包模型身份：" + ", ".join(f"{k}={v}" for k, v in identity.items()), flush=True)

        # ⚠️ output_path 指向临时目录 —— 读完随目录销毁，盘上不留提交格式 CSV。
        raw_path = workspace / "runner_output.csv"
        print("  跑官方 runner（全量 test）…", flush=True)
        result = run_strategy(
            data_root=str(data_root), strategy_dir=str(package), output_path=str(raw_path),
            split="test", model_init_timeout_seconds=None, per_step_timeout_seconds=None,
            total_timeout_seconds=None, timeout_policy="zero_step",
        )
        print(f"  status={result.status} rows={result.rows:,} "
              f"predict_total={result.timing.predict_total_seconds / 60:.2f} 分钟 "
              f"timeouts={result.timing.predict_timeout_count}", flush=True)
        if result.status != "ok" or result.timing.predict_timeout_count:
            raise SystemExit(f"runner 未干净完成：status={result.status}、"
                             f"timeouts={result.timing.predict_timeout_count}")
        frame = pd.read_csv(raw_path)

    row_id = frame["row_id"].to_numpy(np.int64)
    prediction = frame["target"].to_numpy(np.float64)
    if not np.all(np.isfinite(prediction)):
        raise SystemExit("预测里有非有限值")
    clip = float(staged_meta.get("prediction_clip", 0.5))
    # ⚠️ 2026-08-24：判据只看**密封段**（peak 只由那 856,319 行算出来），全窗计数仅作记录。
    # 旧作用域会因为一行落在评估窗口之外的触限毙掉整个臂 —— `r960` 实测就是这样
    # （全窗 1 行 / 3,217,458，段内 0 行）。详见 `assert_no_clip_hits` 的说明。
    full_hits = clip_hits(prediction, clip)
    seal_mask = sealed_rows(row_id, data_root)
    touched = assert_no_clip_hits(prediction[seal_mask], clip, where="（密封段内）")
    if full_hits:
        print(f"  ⚠️ 全窗触限 {full_hits:,} 行（密封段内 0 行）—— 记录但不否决段内比较",
              flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model_dir": str(model_dir), "identity": identity, "clip": clip,
            "clip_hits": touched, "clip_hits_full": full_hits,
            "sealed_rows": int(seal_mask.sum()), "rows": int(len(row_id)),
            "max_abs_prediction": float(np.max(np.abs(prediction))),
            "runner_status": result.status,
            "predict_total_seconds": float(result.timing.predict_total_seconds),
            "elapsed_seconds": float(time.perf_counter() - started)}
    np.savez_compressed(cache_path, row_id=row_id, prediction=prediction,
                        meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"  → {cache_path}  max|pred|={meta['max_abs_prediction']:.6f} 触限 {touched} 行",
          flush=True)
    return meta


def load_prediction(cache_dir: Path, label: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = cache_dir / f"{CACHE_PREFIX}{label}.npz"
    if not path.exists():
        raise SystemExit(f"找不到 {path} —— 先用 --candidate 出这一臂的预测")
    with np.load(path, allow_pickle=False) as bundle:
        return (bundle["row_id"], bundle["prediction"],
                json.loads(str(bundle["meta_json"])))


def synthetic_labels(row_id: np.ndarray, seed: int = 20260820) -> tuple[np.ndarray, np.ndarray]:
    """干跑用的合成标签。**只用于验证链路**，产物强制 adjudication_valid=false。"""
    rng = np.random.default_rng(seed)
    weight = rng.uniform(0.5, 1.5, len(row_id))
    target = rng.normal(0.0, 0.02, len(row_id))
    return target, weight


# ============================ main ============================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="密封期尺子：预注册 / 出预测 / 裁决")
    p.add_argument("--emit-plan", action="store_true", help="落盘预注册判据（默认动作）")
    p.add_argument("--candidate", default=None, help="候选模型目录（走官方 runner 出预测）")
    p.add_argument("--baseline", default=None, help="基准臂的 cache label")
    p.add_argument("--arms", nargs="*", default=[], metavar="NAME=LABEL")
    p.add_argument("--labels", default=None, help="回补数据目录或 parquet")
    p.add_argument("--synthetic-labels", action="store_true",
                   help="干跑：用合成标签走通链路，产物强制 adjudication_valid=false")
    p.add_argument("--detection-floor", type=float, default=None,
                   help="Tier 1 标定出来的检出下限（相对量）；不给则第 7 道判 None")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--cache-dir", default=str(_REPO_ROOT / "outputs" / "cache"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--slow-fast", action="store_true",
                   help="给候选补上 PUBLIC_BASELINE 的 slow/fast 三键 —— "
                        "重训候选必须加（train.py 不产出它们，缺键会静默关掉）；"
                        "Tier 1 那些 slow/fast 转正前的历史臂**不要**加")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _write(out_dir: Path, label: str, payload: dict[str, Any], markdown: str,
           force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{label}.json", out_dir / f"{label}.md"
    if not force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"\n→ {json_path}\n→ {md_path}")


def main() -> None:
    args = parse_args()
    data_root, cache_dir = Path(args.data_root), Path(args.cache_dir)
    out_dir = Path(args.output_dir)

    lo, hi = test_time_id_bounds(data_root)
    geometry = seal_geometry(lo, hi)

    # ---- 出预测
    if args.candidate:
        label = args.label or Path(args.candidate).name
        print(f"候选 {args.candidate} → {CACHE_PREFIX}{label}.npz")
        run_candidate(Path(args.candidate), data_root,
                      cache_dir / f"{CACHE_PREFIX}{label}.npz",
                      slow_fast=args.slow_fast)
        return

    # ---- 裁决
    if args.baseline:
        plan_path = out_dir / f"{PLAN_LABEL}.json"
        if not plan_path.exists():
            raise SystemExit(
                f"没有预注册文件 {plan_path} —— 先跑 --emit-plan。"
                "判据必须在看到结果之前落盘（CLAUDE.md §5.1）。")
        registered = json.loads(plan_path.read_text(encoding="utf-8"))
        if registered["geometry"] != geometry:
            raise SystemExit("当前数据推出的密封段几何与预注册文件不同 —— "
                             "数据变了就必须重新预注册，不能沿用旧判据。")
        if not args.labels and not args.synthetic_labels:
            raise SystemExit("裁决需要 --labels，或 --synthetic-labels 做干跑")

        base_row_id, base_pred, base_meta = load_prediction(cache_dir, args.baseline)
        if args.synthetic_labels:
            target, weight = synthetic_labels(base_row_id)
            label_row_id = base_row_id
        else:
            label_row_id, target, weight = load_backfill_labels(Path(args.labels))
        take = align_labels(base_row_id, label_row_id)
        target, weight = target[take], weight[take]

        # 密封段的 time_id：由 test 分区索引给出
        import pyarrow.parquet as pq
        import pandas as pd
        index = pd.concat(
            [pq.ParquetFile(p).read(columns=["row_id", "time_id"]).to_pandas()
             for p in sorted((data_root / "test").glob("*.parquet"))], ignore_index=True)
        idx_order = np.argsort(index["row_id"].to_numpy(np.int64), kind="stable")
        idx_row = index["row_id"].to_numpy(np.int64)[idx_order]
        idx_time = index["time_id"].to_numpy(np.int64)[idx_order]
        pos = np.searchsorted(idx_row, base_row_id)
        # 不校验的 searchsorted 会把找不到的 row_id 静默映到邻居的 time_id ⟹ 分块全错但数看起来正常
        if pos.max() >= len(idx_row) or not np.array_equal(idx_row[np.minimum(pos, len(idx_row) - 1)],
                                                           base_row_id):
            raise SystemExit("预测的 row_id 在 test 分区索引里对不上 —— 无法确定它们落在哪个 time_id")
        time_id = idx_time[pos]

        seal = (time_id >= geometry["seal_time_id_min"]) & (time_id <= geometry["seal_time_id_max"])
        print(f"密封段：{int(seal.sum()):,} 行 / {geometry['seal_time_ids']:,} 个 time_id")

        # chunk 必须整除进块，否则最后一个 chunk 跨块、bootstrap 的配对结构就破了
        if geometry["block_time_ids"] % BOOTSTRAP_CHUNKS_PER_BLOCK:
            raise SystemExit(f"每块 {geometry['block_time_ids']} 个 time_id 不能被 "
                             f"{BOOTSTRAP_CHUNKS_PER_BLOCK} 个 chunk 整除")
        chunk_span = geometry["block_time_ids"] // BOOTSTRAP_CHUNKS_PER_BLOCK
        chunk_id = ((time_id[seal] - geometry["seal_time_id_min"]) // chunk_span).astype(np.int64)
        n_chunks = N_BLOCKS * BOOTSTRAP_CHUNKS_PER_BLOCK
        if chunk_id.min() < 0 or chunk_id.max() >= n_chunks:
            raise SystemExit("chunk 编号越界 —— 密封段掩码与几何不一致")

        def arm_view(label: str) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, dict]:
            row_id, pred, meta = load_prediction(cache_dir, label)
            if not np.array_equal(row_id, base_row_id):
                raise SystemExit(f"{label} 的 row_id 与基准不同 —— 不是配对比较")
            # ⚠️ 只看密封段 —— 下面每一行都是 pred[seal]（见 assert_no_clip_hits 的说明）
            assert_no_clip_hits(pred[seal], meta["clip"], where="（密封段内）")
            blocks = block_metrics(time_id[seal], target[seal], pred[seal], weight[seal],
                                   geometry["blocks"])
            pooled = scale_invariant_score(target[seal], pred[seal], weight[seal])
            sums = _chunk_sums(target[seal], pred[seal], weight[seal], chunk_id, n_chunks)
            return dict(pooled), blocks, sums, meta

        base_pooled, base_blocks, base_sums, _ = arm_view(args.baseline)
        results = []
        for spec in args.arms:
            if "=" not in spec:
                raise SystemExit(f"--arms 要 NAME=LABEL，收到 {spec}")
            name, arm_label = spec.split("=", 1)
            arm_pooled, arm_blocks, arm_sums, arm_meta = arm_view(arm_label)
            boot = paired_block_bootstrap(base_sums, arm_sums)
            verdict = judge(base_blocks, arm_blocks, base_pooled, arm_pooled,
                            boot, args.detection_floor)
            verdict.update({"name": name, "cache_label": arm_label,
                            "model_dir": arm_meta.get("model_dir")})
            results.append(verdict)
            print(f"  {name:<20} 块均 {verdict['block_mean_relative']:+.2%} "
                  f"正块 {verdict['positive_blocks']}/{verdict['n_blocks']} "
                  f"去最好块 {verdict['drop_best_block_relative']:+.2%} → {verdict['verdict']}")

        label = args.label or "sealed_period_eval"
        payload = {
            "experiment": "sealed_period_eval",
            "adjudication_valid": not args.synthetic_labels,
            "synthetic_labels": bool(args.synthetic_labels),
            "plan_sha256": sha256_file(plan_path),
            "geometry": geometry,
            "baseline": {"cache_label": args.baseline, "pooled": base_pooled,
                         "blocks": base_blocks},
            "arms": results,
            "sealed_rows": int(seal.sum()),
        }
        if args.synthetic_labels:
            payload["warning"] = ("合成标签 ⟹ **任何门禁判定都不作数**，本次只验证链路"
                                  "（join / 分块 / bootstrap / 判据）能跑通。")
            for item in payload["arms"]:
                item["verdict"] = "DRY_RUN"
        _write(out_dir, label, payload, render_summary(payload), args.force)
        return

    # ---- 预注册（默认）
    payload = plan_payload(geometry)
    _write(out_dir, args.label or PLAN_LABEL, payload, render_plan(payload), args.force)


def render_summary(payload: dict[str, Any]) -> str:
    geometry = payload["geometry"]
    lines = ["# 密封期裁决", ""]
    if payload["synthetic_labels"]:
        lines += [f"> ⚠️ {payload['warning']} `adjudication_valid=false`", ""]
    lines += [
        f"密封段 `{geometry['seal_time_id_min']:,}–{geometry['seal_time_id_max']:,}`"
        f"（{payload['sealed_rows']:,} 行 / {geometry['n_blocks']} 块），"
        f"预注册 sha256 `{payload['plan_sha256'][:16]}…`", "",
        f"基准 `{payload['baseline']['cache_label']}` pooled peak "
        f"**{payload['baseline']['pooled']['peak']:.8f}**", "",
        "| 臂 | 块均 | 正块 | 去最好块 | pooled | ΔA | ΔB | CI 下界 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for arm in payload["arms"]:
        ci = arm["bootstrap"]["ci_low"] if arm.get("bootstrap") else float("nan")
        lines.append(
            f"| `{arm['name']}` | {arm['block_mean_relative']:+.2%} | "
            f"{arm['positive_blocks']}/{arm['n_blocks']} | "
            f"{arm['drop_best_block_relative']:+.2%} | {arm['pooled_relative']:+.2%} | "
            f"{arm['delta_A']:+.2%} | {arm['delta_B']:+.2%} | {ci:+.2%} | {arm['verdict']} |")
    pending = {p for arm in payload["arms"] for p in arm.get("pending_gates", [])}
    if pending:
        lines += ["", f"⚠️ 未判定的门禁：{', '.join(sorted(pending))} —— "
                      "检出下限要由 Tier 1 标定出来才有值，标定前不自动通过。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
