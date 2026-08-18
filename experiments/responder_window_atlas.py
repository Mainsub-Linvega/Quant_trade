"""responder 窗口图谱：47 个 responder 各自的预测窗口有多长？

## 背景

`v3_temporal_smoothing` 在全分辨率上测出 target 的自相关是
`0.837 / 0.612 / 0.397 / 0.183 / 0.0004`，对**等权 MA(H=5)** 的 RMSE 是 0.025，
对 H=6 是 0.092 —— 而等权 MA(H) 的自相关在 **lag = H 处恰好为 0**，实测 ac(5)=+0.0004。
⟹ target 极可能是某个底层增量 `u` 在**未来 5 步**上的等权平均。

主办方说 responder 「覆盖多个预测窗口」。若某个 `responder_j` 是**同一个 u、但窗口更短**
（长度 `H_j < 5`），就能把 target 拆成 horizon 分量分别定收缩 —— 那是合并拟合在结构上
做不到的纯方差削减。本脚本只做**诊断**，判断这样的 responder 是否存在。

## 两条互相独立的证据

1. **自相关归零点**：等权 MA(H_j) 的自相关在 lag=H_j 处为 0 ⟹ 零点位置直接读出 `H_j`。
2. **与 target 的错位相关呈梯形**：`target_t` 平均 u 在 `t+1..t+5`，`r_{j,t+k}` 平均 u 在
   `t+k+1..t+k+H_j`，两窗重叠长度随 k 变化成梯形 —— **平台宽度 = |5 − H_j|**，底宽 = `5 + H_j`。

两条都指向同一个 `H_j` 才算数；不一致记为「结构不符」。

## 边界与限制

- ⚠️ responder **只在 train 里**（train 375 列含 47 个，test 326 列含 0 个），
  且 `timeseries_api/runner.py:79` 会剥掉任何 `responder_` 开头的列
  ⟹ **它们永远不能当推理输入**，只能当训练目标或结构诊断。
- ⚠️ 「用 responder 换训练目标」已被 A0 否决（外层 −14.44%、0/5 折，
  `responder_targets_phasebal_prodwindow_projection.md`，连 `projection` 臂都测过）。
  本脚本测的是**分解**而不是**替换**，是不同的问题，但先验不乐观。
- 错位按**真实 time_id 步长、同 asset** 对齐（复合键 searchsorted），不是按相邻行位移。
- 逐分区流式累加充分统计量，不整体 materialize（`v2_lgbm` 33 GiB OOM 的教训）；
  代价是丢掉分区边界处约 12 个 time_id 的配对，相对每分区约 10 万个 time_id 可忽略。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/responder_window_atlas.py
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

from src.io import train_files  # noqa: E402

TARGET_H = 5           # 由 v3_temporal_smoothing 的全分辨率自相关拟合得到
MAX_LAG = 12
SHIFTS = list(range(-12, 13))
ZERO_TOLERANCE = 0.05  # |ac| 低于它就算「已归零」


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="responder_window_atlas")
    p.add_argument("--max-lag", type=int, default=MAX_LAG)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


class PairAccumulator:
    """加权相关的充分统计量，可跨分区相加。"""

    __slots__ = ("w", "wx", "wy", "wxx", "wyy", "wxy", "n")

    def __init__(self) -> None:
        self.w = self.wx = self.wy = self.wxx = self.wyy = self.wxy = 0.0
        self.n = 0

    def add(self, weight: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        if not len(x):
            return
        self.w += float(weight.sum())
        self.wx += float(np.dot(weight, x))
        self.wy += float(np.dot(weight, y))
        self.wxx += float(np.dot(weight, x * x))
        self.wyy += float(np.dot(weight, y * y))
        self.wxy += float(np.dot(weight, x * y))
        self.n += len(x)

    def correlation(self) -> float:
        if self.w <= 0 or self.n < 2:
            return float("nan")
        mx, my = self.wx / self.w, self.wy / self.w
        vx = self.wxx / self.w - mx * mx
        vy = self.wyy / self.w - my * my
        cov = self.wxy / self.w - mx * my
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(cov / np.sqrt(vx * vy))


def asset_major_key(asset: np.ndarray, time_id: np.ndarray, span: int) -> np.ndarray:
    """`asset·span + time_id`；键的升序 == (asset, time) 序 ⟹ searchsorted 一次找出全部配对。"""
    return asset.astype(np.int64) * span + time_id.astype(np.int64)


def matched_pairs(sorted_key: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (基准位置, 相隔 shift 的位置)；shift 可正可负。"""
    wanted = sorted_key + shift
    left = np.searchsorted(sorted_key, wanted)
    inside = (left >= 0) & (left < len(sorted_key))
    hit = np.zeros(len(sorted_key), dtype=bool)
    hit[inside] = sorted_key[left[inside]] == wanted[inside]
    base = np.flatnonzero(hit)
    return base, left[base]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"
    md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    import pyarrow.parquet as pq

    first = pq.ParquetFile(train_files(Path(args.data_root))[0])
    responders = sorted(c for c in first.schema_arrow.names if c.startswith("responder_"))
    if not responders:
        raise SystemExit("train 里没有 responder 列")
    print(f"{len(responders)} 个 responder；错位范围 {SHIFTS[0]}..{SHIFTS[-1]}；"
          f"自相关 lag 1..{args.max_lag}", flush=True)

    cross = {j: {k: PairAccumulator() for k in SHIFTS} for j in responders}
    auto = {j: {k: PairAccumulator() for k in range(1, args.max_lag + 1)} for j in responders}
    target_auto = {k: PairAccumulator() for k in range(1, args.max_lag + 1)}
    columns = ["time_id", "asset_id", "weight", "target", *responders]

    for path in train_files(Path(args.data_root)):
        started = time.perf_counter()
        frame = pq.ParquetFile(path).read(columns=columns).to_pandas()
        tid = frame["time_id"].to_numpy(dtype=np.int64)
        aid = frame["asset_id"].to_numpy(dtype=np.int64)
        weight = np.maximum(frame["weight"].to_numpy(dtype=np.float64), 0.0)
        target = frame["target"].to_numpy(dtype=np.float64)
        span = int(tid.max()) + args.max_lag + 20
        order = np.argsort(asset_major_key(aid, tid, span), kind="stable")
        key = asset_major_key(aid[order], tid[order], span)
        w_s, y_s = weight[order], target[order]

        # 配对下标只与 (分区, 位移) 有关，与是哪个 responder 无关 ⟹ 先算一次，47 个列共用
        shift_pairs = {k: matched_pairs(key, k) for k in SHIFTS}
        lag_pairs = {lag: matched_pairs(key, lag) for lag in range(1, args.max_lag + 1)}

        for lag, (base, other) in lag_pairs.items():
            target_auto[lag].add(w_s[base], y_s[base], y_s[other])

        y_finite = np.isfinite(y_s)
        for name in responders:
            r_s = frame[name].to_numpy(dtype=np.float64)[order]
            finite = np.isfinite(r_s)
            for k, (base, other) in shift_pairs.items():
                ok = finite[base] & finite[other] & y_finite[base]
                cross[name][k].add(w_s[base][ok], y_s[base][ok], r_s[other][ok])
            for lag, (base, other) in lag_pairs.items():
                ok = finite[base] & finite[other]
                auto[name][lag].add(w_s[base][ok], r_s[base][ok], r_s[other][ok])
            del r_s
        del frame, tid, aid, weight, target, order, key, w_s, y_s
        print(f"  {path.name}（{time.perf_counter()-started:.0f}s）", flush=True)

    def fit_window(profile: list[float]) -> tuple[int | None, float]:
        """按**整条曲线**拟 H，不用归零点。

        ⚠️ 归零点法（第一个 |ac|<容差 的 lag）在这里会误读：`responder_00` 的 ac 本来就全 ≈0
        （它是单步量），归零点法会报 H=2；而 `responder_02` 的 ac=(0.49, 0.08, ~0) 明明是
        MA(2)，归零点法却因为 ac2=0.08>容差 报成 H=3。整条曲线对 `(H−k)/H` 做 RMSE 拟合
        才是稳的 —— 这正是 target 上验证过的做法（H=5 RMSE 0.025 vs H=6 的 0.092）。
        RMSE 一并返回：拟合差说明它根本不是等权 MA。
        """
        values = np.asarray(profile, dtype=float)
        best_h, best_rmse = None, float("inf")
        for H in range(1, args.max_lag + 1):
            theory = np.array([max(H - k, 0) / H for k in range(1, len(values) + 1)])
            rmse = float(np.sqrt(np.nanmean((values - theory) ** 2)))
            if rmse < best_rmse:
                best_h, best_rmse = H, rmse
        return best_h, best_rmse

    target_profile = [target_auto[k].correlation() for k in range(1, args.max_lag + 1)]
    target_H, target_rmse = fit_window(target_profile)

    rows: list[dict[str, Any]] = []
    for name in responders:
        profile = [auto[name][k].correlation() for k in range(1, args.max_lag + 1)]
        H_j, rmse = fit_window(profile)
        shifted = [cross[name][k].correlation() for k in SHIFTS]
        peak_index = int(np.nanargmax(np.abs(shifted)))
        peak_value = shifted[peak_index]
        # 支撑宽度：|corr| ≥ 峰值 50% 的 shift 个数。两个等权 MA 窗重叠成梯形，
        # 支撑宽度 ≈ H_target + H_j − 1（重叠非零的 shift 个数）。
        threshold = 0.5 * abs(peak_value)
        support = [k for k, v in zip(SHIFTS, shifted) if np.isfinite(v) and abs(v) >= threshold]
        support_width = len(support)
        expected_support = TARGET_H + H_j - 1 if H_j is not None else None
        good_fit = rmse <= 0.06                       # 拟合差 ⟹ 根本不是等权 MA
        consistent = bool(H_j is not None and H_j < TARGET_H and good_fit
                          and expected_support is not None
                          and abs(support_width - expected_support) <= 2)
        rows.append({
            "responder": name, "autocorr": profile,
            "H_estimate": H_j, "H_fit_rmse": rmse, "H_fit_is_equal_weight_MA": bool(good_fit),
            "cross_shift_correlation": shifted,
            "peak_shift": SHIFTS[peak_index], "peak_correlation": peak_value,
            "support_shifts": [int(k) for k in support],
            "support_width": support_width, "expected_support_if_shorter_window": expected_support,
            "shorter_window_candidate": consistent,
        })

    candidates = [r for r in rows if r["shorter_window_candidate"]]
    payload = {
        "experiment": "responder_window_atlas",
        "question": "有没有哪个 responder 是 target 的**更短窗口**版本？",
        "target_autocorr": target_profile, "target_H_estimate": target_H,
        "target_H_fit_rmse": target_rmse,
        "target_H_prior": TARGET_H,
        "zero_tolerance": ZERO_TOLERANCE,
        "criterion": "整条自相关曲线拟合出 H_j < 5 且 RMSE ≤ 0.06（确认是等权 MA），且与 target 的错位相关支撑宽度 ≈ 5 + H_j − 1（±2）",
        "limits": [
            "responder 只在 train 里，runner 会剥掉 responder_ 前缀列 ⟹ 永不能当推理输入",
            "「用 responder 换训练目标」已被 A0 否决（外层 −14.44%、0/5 折）；"
            "本脚本测的是分解而非替换，是不同问题，但先验不乐观",
            "逐分区累加，丢失分区边界约 12 个 time_id 的配对（每分区约 10 万个，可忽略）",
        ],
        "responders": rows,
        "verdict": {"n_candidates": len(candidates),
                    "candidates": [r["responder"] for r in candidates],
                    "decision": ("找到更短窗口候选，可进入 horizon 分解设计"
                                 if candidates else
                                 "没有 responder 同时满足两条证据 ⟹ horizon 分解缺前提，暂不推进")},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# responder 窗口图谱", "",
             f"**target 自相关**（全量、逐 asset、真实步长）：" +
             ", ".join(f"{v:+.3f}" for v in target_profile[:6]) + " …",
             f"⟹ 整条曲线拟合 **H = {target_H}**（RMSE {target_rmse:.3f}；先验 {TARGET_H}，来自 "
             "`v3_temporal_smoothing` 的全分辨率拟合）", "",
             f"判据：{payload['criterion']}", "",
             "| responder | 自相关 lag1..5 | H_j | 拟合RMSE | 峰值 shift | 峰值 corr | 支撑宽 | 预期支撑 | 更短窗口? |",
             "|---|---|---:|---:|---:|---:|---:|---:|:--:|"]
    for r in rows:
        prof = ", ".join(f"{v:+.2f}" for v in r["autocorr"][:5])
        lines.append(
            f"| `{r['responder']}` | {prof} | {r['H_estimate']} | {r['H_fit_rmse']:.3f} | "
            f"{r['peak_shift']} | {r['peak_correlation']:+.3f} | {r['support_width']} | "
            f"{r['expected_support_if_shorter_window']} | "
            f"{'✅' if r['shorter_window_candidate'] else '—'} |")
    lines += ["", "## 判定", "",
              f"满足两条证据的 responder：**{len(candidates)}** 个"
              + (f"（{', '.join(r['responder'] for r in candidates)}）" if candidates else ""),
              "", f"### {payload['verdict']['decision']}", "", "## 限制", ""]
    lines += [f"- {item}" for item in payload["limits"]]
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ntarget H 读数 = {target_H}（先验 {TARGET_H}）；"
          f"更短窗口候选 {len(candidates)} 个")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
