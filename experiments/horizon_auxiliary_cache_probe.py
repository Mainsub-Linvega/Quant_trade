"""horizon auxiliary 的准入筛：responder_00 / responder_02 能不能补强 v3 的 target 残差？

## 为什么要问（这个格子确实空着）

`responder_window_atlas` 测出 responder 是一把窗口梯子，`responder_00`(H=1) 与
`responder_02`(H=2) 是仅有的两个「比 target(H=5) 更短」的候选。但 2026-08-14 的
responder 重新审计（`responder_reaudit_20260814.md`）已经在**强 v3、生产等效口径、
固定过去校准**下否掉了同一个机制：线性叠加 −18.81%（1/5 折）、非线性二层
−11.70%/−17.03%、受控增量 −7.76%。

⚠️ 但那一轮**没有测过 00 和 02**。查 `responder_predictability_reaudit_phasebal_prodwindow.json`：

    cluster 24 = {responder_00}  mean_peak 0.02945  5/5 折  drop-best 0.02857  pass=false
    cluster 22 = {responder_02}  mean_peak 0.00141  5/5 折  drop-best 0.00130  pass=false

七项 check 里**只有 `multi_member_family` 一项为 false**（它们是单成员族），其余六项全过。
也就是说这两个 responder 被挡在 Stage C 门外**不是因为证据，是因为一条稳健性启发式**
（要求族里 ≥2 个成员）。Stage C 只吃了 8 个通过族（27/24/37/15/08/40/18/11）。

⟹ 「这两个短窗口 responder 能否补 target 残差」是真空白格。本脚本用**已有缓存**把它填掉，
不训练任何新模型。

## 先验（写在前面，免得看到结果再编故事）

- 被测过的那 8 个族可预测性 peak 是 0.235~0.654，比 responder_00 高 **8~22×**、
  比 responder_02 高 **170~460×** —— 它们**更**可预测，结果仍是 −18.81%。
- responder_00 的可预测性 **7.7× 集中在 market 侧**（market 0.0711 / cross 0.0092），
  而 market 侧六条路 2026-08-17 已全关。
- 机制上：auxiliary 预测 `â=f(X)` 与主模型同用一套 X、同一折训练段，能贡献的只有归纳偏置。

## 设计

- 两份现有缓存按 `(time_id<<8)|asset_id` 复合键连接（实测 responder 缓存 1,460,912 行
  100% 落在强 v3 OOF 的 fold>=0 行内）：
  - auxiliary ← `responder_oof_phasebal_prodwindow_f323.npz` 的 `prediction_responders`
    （严格 OOF、phase_balanced/modulo5/train_window 78,960/5 折/embargo 6/全 323 特征）
  - baseline  ← `v3_production_oof_confirm_3s480_phasebal_prodwindow.npz`
- 组合系数按**扩展窗口只用 fold 0..k−1 拟合**，冻结后在 fold k 上评估（fold 0 被拟合消耗
  ⟹ 4 个评估折）。与 `asset_blend_check.py` 同一套 moments/solve/score/bootstrap。
- 两套基准分别报告：`full` = `prediction_raw`（未乘 scale、未 clip，符合
  `scale_invariant_score` 的契约）；`pure_e` = `e_lgbm`，auxiliary 先投影成逐 time_id
  无权零均值（与 `main.py` 的 `e_lgbm -= e_lgbm.mean()` 同口径）。

## ⚠️ 限制（负结果的边界在哪）

1. 缓存里的 responder OOF 是 **Ridge 强度**。这与 08-14 对那 8 个族用的是同一把尺子
   ⟹ 可比；但严格说负结果只证否「Ridge 强度的 auxiliary 无增量」，不证否 LGBM 强度。
   本脚本因此只是**准入筛**，不是终审。
2. 基准取的是它在评估折上**重新解出的最优 scale**（peak），候选却必须用**冻结系数**
   ⟹ 对候选不利。`null_frozen_scale` 臂（不加任何 auxiliary、只冻结一个 scale）就是用来
   量化这个让步有多大的；每个臂同时报告 `delta_vs_frozen_baseline` 作为剥掉让步后的诊断。
3. 基准不含 slow/fast 后处理（生产已是 slow 0.4496 / fast 1.2530）⟹ 本轮增量与 slow/fast
   的交互未验证。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/horizon_auxiliary_cache_probe.py
输出：outputs/experiments/<label>.{json,md}
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
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from market_model import sign_test_p  # noqa: E402

RESP_CACHE = _REPO_ROOT / "outputs" / "cache" / "responder_oof_phasebal_prodwindow_f323.npz"
V3_CACHE = (_REPO_ROOT / "outputs" / "cache" /
            "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz")
EXPECTED_ROWS = 1_460_912          # 实测交集行数；对不上说明缓存被换过，直接失败
ASSET_BITS = 8                     # asset_id ∈ [0,14] ⟹ 8 bit 足够
MIN_RELATIVE_GAIN = 0.03           # 08-14 重新开放条件的原值（不是 NEXT_STEPS 里的 1%）
MIN_POSITIVE_FOLDS = 3             # 4 个评估折里至少 3 个
BASELINES = {"full": "prediction_raw", "pure_e": "e_lgbm"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resp-cache", default=str(RESP_CACHE))
    p.add_argument("--v3-cache", default=str(V3_CACHE))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="horizon_auxiliary_cache_probe")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--shuffle-seed", type=int, default=20260818)
    p.add_argument("--limit-groups", type=int, default=None,
                   help="只保留前 N 个 time_id（烟测用，会跳过行数断言）")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def group_index(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """行已按 time_id 排序 ⟹ 返回 (组起点, 逐行组号, 组数)。"""
    if np.any(np.diff(time_id) < 0):
        raise AssertionError("行未按 time_id 升序 ⟹ 组内统计会串组")
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(int)
    return starts, np.repeat(np.arange(len(starts)), counts), len(starts)


def zero_mean_per_time(values: np.ndarray, starts: np.ndarray, n_rows: int) -> np.ndarray:
    counts = np.diff(np.r_[starts, n_rows]).astype(np.float64)
    return values - np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))


def load_aligned(args: argparse.Namespace) -> dict[str, Any]:
    """两份缓存按复合键连接；行序沿用 responder 缓存（已按 time_id 升序）。"""
    with np.load(args.resp_cache, allow_pickle=False) as d:
        resp = {k: d[k] for k in ("target", "weight", "time_id", "asset_id", "fold",
                                  "prediction_responders", "prediction_target")}
        names = [str(x) for x in d["responder_names"]]
    with np.load(args.v3_cache, allow_pickle=False) as d:
        keep = d["fold"] >= 0
        v3 = {k: d[k][keep] for k in ("target", "weight", "time_id", "asset_id", "fold",
                                      "prediction_raw", "e_lgbm")}

    def key(t: np.ndarray, a: np.ndarray) -> np.ndarray:
        if a.min() < 0 or a.max() >= (1 << ASSET_BITS):
            raise AssertionError("asset_id 超出复合键位宽")
        return (t.astype(np.int64) << ASSET_BITS) | a.astype(np.int64)

    kr, kv = key(resp["time_id"], resp["asset_id"]), key(v3["time_id"], v3["asset_id"])
    order = np.argsort(kv, kind="stable")
    pos = np.searchsorted(kv[order], kr)
    if np.any(pos >= len(kv)) or np.any(kv[order][np.minimum(pos, len(kv) - 1)] != kr):
        raise AssertionError("responder 缓存有行不在 v3 OOF 验证行内 ⟹ 口径不一致")
    take = order[pos]

    # 口径漂移的硬检查：两份缓存在匹配行上必须描述同一批样本
    for field, tol in (("target", 1e-6), ("weight", 1e-6)):
        diff = float(np.max(np.abs(v3[field][take].astype(np.float64)
                                   - resp[field].astype(np.float64))))
        if diff > tol:
            raise AssertionError(f"两缓存 {field} 不一致，max|Δ|={diff:.3e}")
    # 两份缓存的 fold 边界错开约 90 个 time_id（各自的 rolling_time_folds 建在不同的
    # unique time_id 列表上）。两种标注在因果上都成立 —— 缓存里存的都只是各自模型的 OOF
    # 验证行 —— 但折号必须唯一，否则「只用过去折」的语义会含混。⟹ 丢掉不一致的行。
    agree = v3["fold"][take].astype(np.int64) == resp["fold"].astype(np.int64)
    n_dropped = int((~agree).sum())
    if n_dropped > 0.001 * len(agree):
        raise AssertionError(f"fold 标签不一致的行占 {n_dropped/len(agree):.3%}，超过 0.1% "
                             "⟹ 不是边界抖动而是折版图漂移")
    take = take[agree]

    out = {"time_id": resp["time_id"][agree], "asset_id": resp["asset_id"][agree],
           "target": resp["target"][agree].astype(np.float64),
           "weight": np.maximum(resp["weight"][agree].astype(np.float64), 0.0),
           "fold": resp["fold"][agree].astype(np.int64),
           "aux_all": resp["prediction_responders"][agree], "responder_names": names,
           "prediction_raw": v3["prediction_raw"][take].astype(np.float64),
           "e_lgbm": v3["e_lgbm"][take].astype(np.float64),
           "n_matched": int(len(agree)), "n_dropped_fold_disagreement": n_dropped}
    if args.limit_groups is not None:
        uniq = np.unique(out["time_id"])[: args.limit_groups]
        m = np.isin(out["time_id"], uniq)
        for k in ("time_id", "asset_id", "target", "weight", "fold",
                  "prediction_raw", "e_lgbm"):
            out[k] = out[k][m]
        out["aux_all"] = out["aux_all"][m]
        out["n_matched"] = int(m.sum())
    elif out["n_matched"] != EXPECTED_ROWS:
        raise AssertionError(f"匹配行数 {out['n_matched']:,} != 预期 {EXPECTED_ROWS:,}")
    for k in ("target", "weight", "prediction_raw", "e_lgbm"):
        if not np.all(np.isfinite(out[k])):
            raise AssertionError(f"{k} 含 NaN/inf")
    return out


def moment_rows(y, w, preds, gidx, n_groups):
    """逐 time_id 的充分统计量：[D, v_0..v_{n-1}, G_00, G_01, ..., G_{n-1,n-1}]。"""
    cols = [w * y * y]
    cols += [w * y * p for p in preds]
    cols += [w * preds[i] * preds[j] for i in range(len(preds)) for j in range(len(preds))]
    return np.column_stack([np.bincount(gidx, weights=c, minlength=n_groups) for c in cols])


def split(total: np.ndarray, n: int) -> tuple[float, np.ndarray, np.ndarray]:
    return float(total[0]), total[1:1 + n], total[1 + n:].reshape(n, n)


def solve(total: np.ndarray, n: int) -> np.ndarray:
    _, v, G = split(total, n)
    return np.linalg.solve(G, v)


def frozen_score(total: np.ndarray, coef: np.ndarray) -> float:
    """冻结系数下的 Score（不在评估折上重解 scale）。"""
    D, v, G = split(total, len(coef))
    return float((2 * coef @ v - coef @ G @ coef) / D)


def baseline_peak(total: np.ndarray, n: int) -> float:
    """基准单独、在评估折上重解最优 scale 的 peak = A²/B。"""
    D, v, G = split(total, n)
    A, B = v[0] / D, G[0, 0] / D
    return float(A * A / B) if B > 0 else 0.0


def baseline_frozen_score(past: np.ndarray, ev: np.ndarray, n: int) -> float:
    """基准单独、scale 也只用过去折拟合并冻结 ⟹ 与候选同一个让步口径。"""
    Dp, vp, Gp = split(past, n)
    De, ve, Ge = split(ev, n)
    if Gp[0, 0] <= 0:
        return 0.0
    c0 = vp[0] / Gp[0, 0]
    return float((2 * c0 * ve[0] - c0 * c0 * Ge[0, 0]) / De)


def ab_pair(total: np.ndarray, coef: np.ndarray) -> tuple[float, float]:
    """把组合归一到「基准系数为 1」再读 A、B ⟹ 与基准的 A、B 同口径可比。"""
    D, v, G = split(total, len(coef))
    if coef[0] == 0.0:
        return float("nan"), float("nan")
    c = coef / coef[0]
    return float(c @ v / D), float(c @ G @ c / D)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    data = load_aligned(args)
    y, w, fold = data["target"], data["weight"], data["fold"]
    starts, gidx, n_groups = group_index(data["time_id"])
    group_fold = fold[starts]
    fold_list = sorted(int(f) for f in np.unique(group_fold))
    names = data["responder_names"]
    print(f"{len(y):,} 行 / {n_groups:,} 个 time_id / {len(fold_list)} 折；两缓存对齐 ✅",
          flush=True)

    def aux_col(name: str) -> np.ndarray:
        return data["aux_all"][:, names.index(name)].astype(np.float64)

    rng = np.random.default_rng(args.shuffle_seed)
    perm = np.lexsort((rng.random(len(y)), gidx))      # 组内随机重排（行已按 time_id 排序）
    arms: dict[str, list[np.ndarray]] = {
        "null_frozen_scale": [],
        "responder_00": [aux_col("responder_00")],
        "responder_02": [aux_col("responder_02")],
        "r00_plus_r02": [aux_col("responder_00"), aux_col("responder_02")],
        "negctrl_shuffle": [aux_col("responder_00")[perm]],
        "known_negative_27": [aux_col("responder_27")],
    }

    boot_rng = np.random.default_rng(args.boot_seed)
    results: dict[str, dict[str, Any]] = {}

    for bname, column in BASELINES.items():
        base_pred = data[column]
        results[bname] = {}
        for arm, auxes in arms.items():
            if bname == "pure_e":
                aux_use = [zero_mean_per_time(a, starts, len(y)) for a in auxes]
            else:
                aux_use = auxes
            preds = [base_pred, *aux_use]
            n = len(preds)
            rows = moment_rows(y, w, preds, gidx, n_groups)
            per_fold = {f: rows[group_fold == f].sum(axis=0) for f in fold_list}

            deltas, deltas_frozen, bases, dA, dB = [], [], [], [], []
            frozen: dict[int, np.ndarray] = {}
            for i, f in enumerate(fold_list):
                if i == 0:
                    continue                                   # fold 0 只用于拟合系数
                past = np.sum([per_fold[g] for g in fold_list[:i]], axis=0)
                coef = solve(past, n)
                frozen[f] = coef
                ev = per_fold[f]
                peak_b = baseline_peak(ev, n)
                cand = frozen_score(ev, coef)
                # 同一冻结口径下的基准（只冻结一个 scale、不加 auxiliary）⟹ 剥掉让步
                base_only = baseline_frozen_score(past, ev, n)
                A_c, B_c = ab_pair(ev, coef)
                D, v, G = split(ev, n)
                A_b, B_b = v[0] / D, G[0, 0] / D
                deltas.append(cand - peak_b)
                deltas_frozen.append(cand - base_only)
                bases.append(peak_b)
                dA.append((A_c - A_b) / A_b if A_b else float("nan"))
                dB.append((B_c - B_b) / B_b if B_b else float("nan"))

            deltas = np.array(deltas); bases = np.array(bases)
            deltas_frozen = np.array(deltas_frozen)
            drop = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
            pos = int((deltas > 0).sum())
            rel = float(deltas.mean() / bases.mean())
            mean_dA, mean_dB = float(np.mean(dA)), float(np.mean(dB))

            # 配对 block bootstrap：系数冻结在最后一折，只重采样**评估折**的 time_id
            eval_mask = np.isin(group_fold, fold_list[1:])
            eval_rows = rows[eval_mask]
            m_groups = len(eval_rows)
            prefix = np.vstack([np.zeros(rows.shape[1]), np.cumsum(eval_rows, axis=0)])
            # block 必须显著小于评估折的组数，否则每次抽样都从 0 开始 ⟹ CI 退化成一个点
            block = min(args.block_size, max(m_groups // 10, 1))
            nb = int(np.ceil(m_groups / block))
            coef_last = frozen[fold_list[-1]]
            samples = np.empty(args.n_boot)
            for b in range(args.n_boot):
                st = boot_rng.integers(0, max(m_groups - block, 0) + 1, size=nb)
                sp = np.minimum(st + block, m_groups)
                tot = (prefix[sp] - prefix[st]).sum(axis=0)
                samples[b] = frozen_score(tot, coef_last) - baseline_peak(tot, n)
            ci = np.percentile(samples, [2.5, 50, 97.5])
            floor = float((ci[2] - ci[0]) / 2.0)

            checks = {
                "1_mean_delta_positive": bool(deltas.mean() > 0),
                f"2_at_least_{MIN_POSITIVE_FOLDS}_of_{len(deltas)}_folds_positive":
                    bool(pos >= MIN_POSITIVE_FOLDS),
                "3_survives_drop_best_fold": bool(drop.mean() > 0),
                "4_relative_gain_at_least_3pct": bool(rel >= MIN_RELATIVE_GAIN),
                "5_two_delta_A_exceeds_delta_B": bool(2 * mean_dA > mean_dB),
                "6_paired_bootstrap_ci_lower_bound_positive": bool(ci[0] > 0),
                "7_exceeds_detection_floor": bool(deltas.mean() > floor),
            }
            corr = (float(np.corrcoef(base_pred, aux_use[0])[0, 1]) if aux_use else float("nan"))
            results[bname][arm] = {
                "mean_delta": float(deltas.mean()), "relative": rel,
                "mean_delta_drop_best": float(drop.mean()),
                "mean_delta_vs_frozen_baseline": float(deltas_frozen.mean()),
                "per_fold_delta": [float(x) for x in deltas],
                "baseline_peak_mean": float(bases.mean()),
                "positive_folds": pos, "n_folds": int(len(deltas)),
                "sign_test_p": sign_test_p(pos, len(deltas)),
                "delta_A_relative": mean_dA, "delta_B_relative": mean_dB,
                "corr_with_baseline": corr,
                "frozen_coefficients": {str(f): [float(c) for c in frozen[f]] for f in frozen},
                "paired_bootstrap": {"p2.5": float(ci[0]), "p50": float(ci[1]),
                                     "p97.5": float(ci[2]), "half_width": floor,
                                     "block_size_effective": int(block),
                                     "eval_groups": int(m_groups)},
                "checks": checks, "pass": all(checks.values()),
            }
            print(f"  [{bname:6s}] {arm:18s} Δ折均 {deltas.mean():+.3e}（{rel*100:+.2f}%）"
                  f" 正折 {pos}/{len(deltas)} 检出下限 {floor:.2e} "
                  f"{'PASS' if all(checks.values()) else 'FAIL'}", flush=True)

    # 负控制 / harness 校准
    ncp = {b: results[b]["negctrl_shuffle"]["pass"] for b in BASELINES}
    k27 = {b: results[b]["known_negative_27"]["relative"] for b in BASELINES}
    verdict_arms = ["responder_00", "responder_02", "r00_plus_r02"]
    passed = [(b, a) for b in BASELINES for a in verdict_arms if results[b][a]["pass"]]
    harness_ok = (not any(ncp.values())) and all(v < 0 for v in k27.values())

    payload = {
        "experiment": "horizon_auxiliary_cache_probe",
        "question": "responder_00 / responder_02 的严格 OOF 预测能不能补强 v3 的 target 残差？",
        "why_this_gap_exists": (
            "两者在 Stage B 七项 check 里只有 multi_member_family 为 false（单成员族），"
            "其余六项全过 ⟹ 被启发式而非证据挡在 Stage C 之外，从未被测过"),
        "prior": ("已测的 8 个族可预测性高 8~460× 仍为 −18.81%；responder_00 的可预测性"
                  " 7.7× 偏 market 侧，而 market 侧六条路已全关"),
        "limits": [
            "缓存里的 responder OOF 是 Ridge 强度（与 08-14 同一把尺子，可比），"
            "负结果不证否 LGBM 强度 ⟹ 本脚本是准入筛不是终审",
            "基准在评估折上重解最优 scale，候选用冻结系数 ⟹ 对候选不利；"
            "null_frozen_scale 臂量化该让步，另报 mean_delta_vs_frozen_baseline",
            "基准不含 slow/fast 后处理 ⟹ 与 slow/fast 的交互未验证",
        ],
        "protocol": {
            "resp_cache": str(args.resp_cache), "v3_cache": str(args.v3_cache),
            "rows": int(len(y)), "time_id_groups": int(n_groups),
            "rows_dropped_fold_disagreement": int(data["n_dropped_fold_disagreement"]),
            "fold_disagreement_note": (
                "两缓存的 rolling_time_folds 建在各自的 unique time_id 列表上，折边界错开约 "
                "90 个 time_id；两种标注在因果上都成立，但折号必须唯一 ⟹ 丢掉不一致的行"),
            "folds": fold_list, "eval_folds": fold_list[1:],
            "coefficient_fitting": "扩展窗口，fold k 的系数只用 fold 0..k−1",
            "sampling": "phase_balanced / sample_modulo 5 / train_window 78,960 / embargo 6",
            "min_relative_gain": MIN_RELATIVE_GAIN,
            "block_size": args.block_size, "n_boot": args.n_boot,
            "boot_seed": args.boot_seed, "shuffle_seed": args.shuffle_seed,
        },
        "harness_calibration": {
            "negative_control_passed_gates": ncp,
            "known_negative_27_relative": k27,
            "harness_ok": bool(harness_ok),
            "note": "negctrl 通过门禁、或 responder_27 跑不出负结果 ⟹ 探针失效，不据此裁决",
        },
        "results": results,
        "verdict": {
            "passed_arms": [f"{b}/{a}" for b, a in passed],
            "decision": ("ACCEPT_TO_STAGE2" if passed and harness_ok else
                         "HARNESS_INVALID" if not harness_ok else "REJECT"),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = ["# horizon auxiliary 准入筛（缓存探针）", "",
             f"{len(y):,} 行 / {n_groups:,} 个 time_id / 评估折 {fold_list[1:]}；"
             f"系数只用过去折拟合并冻结。", "",
             "> **为什么这个格子空着**：" + payload["why_this_gap_exists"], "",
             "> **先验**：" + payload["prior"], ""]
    for bname in BASELINES:
        lines += [f"## 基准 `{bname}`（{BASELINES[bname]}）", "",
                  "| 臂 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 检出下限 | 配对 CI | 判定 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|:--:|"]
        for arm in arms:
            r = results[bname][arm]; ci = r["paired_bootstrap"]
            lines.append(
                f"| `{arm}` | {r['mean_delta']:+.3e} | {r['relative']*100:+.2f}% | "
                f"{r['positive_folds']}/{r['n_folds']} | {r['mean_delta_drop_best']:+.3e} | "
                f"{r['delta_A_relative']*100:+.2f}% | {r['delta_B_relative']*100:+.2f}% | "
                f"{ci['half_width']:.2e} | "
                f"[{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] | "
                f"{'✅' if r['pass'] else '❌'} |")
        lines.append("")
    lines += ["## harness 校准", "",
              f"- 负控制（组内打乱）是否通过门禁：{ncp}（应全为 False）",
              f"- 已测族 `responder_27` 相对增量："
              + ", ".join(f"{b} {v*100:+.2f}%" for b, v in k27.items()) + "（应为负）",
              f"- **harness_ok = {harness_ok}**", "",
              "## 限制", ""]
    lines += [f"{i+1}. {t}" for i, t in enumerate(payload["limits"])]
    lines += ["", f"## 裁决：{payload['verdict']['decision']}", "",
              ("通过臂：" + ", ".join(payload["verdict"]["passed_arms"]))
              if passed else "没有任何 responder 臂通过预注册门禁。", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n裁决：{payload['verdict']['decision']}\nwrote {json_path}\nwrote {md_path}",
          flush=True)


if __name__ == "__main__":
    main()
