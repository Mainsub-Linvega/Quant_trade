"""补测并落盘：错位 responder 能重建出多少 target？（NOTES 2026-08-18 的「决定性重建测试」）

## 为什么补这一测

ROADMAP:165 与 NOTES:270-273 用「重建 R² 只 0.883、单步 u 不存在」关闭了 horizon 分解方向。
但全仓库检索（排除 .venv/.git/data）显示 **`0.207` 只出现在 `NOTES.md:258` 和 `:265`**：

- `responder_window_atlas.py` 只算自相关和错位相关，**没有**重建测试；
- `responder_window_atlas.json` 没有任何 reconstruction 字段；
- 没有任何脚本能产出那张表。

NOTES 引用 `responder_window_atlas.{json,md}` 作证据，但那两个文件里没有这张表 ⟹ 按
CLAUDE.md §3「结论应附证据路径」和 §7「优先引用产物」，这是一个证据缺口：关闭该方向的
理由目前建立在无法复现的测量上。本脚本把它补测并落盘。

## 口径

- 全量 train，逐 asset、按**真实 time_id 步长**配对（复合键 searchsorted，复用
  `responder_window_atlas.asset_major_key`）。**不修改 atlas 脚本，也不覆盖它的产物。**
- atlas 的 `matched_pairs` 只处理单个 shift；重建需要**多个 shift 同时命中**，
  故本脚本自带多 shift 版本，取交集行。
- **两套 R² 都算**（NOTES 没写它用的是哪一套，实测是前者）：
  - `r2_centered`：带截距、对加权均值中心化 —— **NOTES 那张表用的就是这一套**；
  - `r2_uncentered`：无截距、分母为 `Σw·y²` —— 与 `src/metric.py` 的
    `Score = 1 − Σw(y−ŷ)²/Σw·y²` 同口径。
  ⚠️ 两者差得很远：responder 带很大的非零均值（`responder_03` 加权均值 +0.502、std 仅
  0.262），非中心化口径会被均值项稀释 —— 同一个 `responder_03 @ −1..+1` 设计，
  中心化 0.84、非中心化 0.15。
- 这是**样本内**重建 R²（问的是「这些错位 responder 张成的空间覆盖了多少 target」，
  是个 oracle 上界问题，不是预测问题）—— 与 NOTES 记的那张表同一个含义。
- 每个设计同时报两套行集：`own`（该设计自己的交集行）与 `common`（所有设计都命中的
  公共行，五个数字直接可比）。NOTES 没写用的是哪一套，两套都给出以便对照。

## 预注册设计（与 NOTES:256-262 逐行对应）

    responder_00 @ +1..+5   NOTES 记 0.207（纯 u 假设：若 r00 就是底层增量 u，应 →1）
    responder_02 @  0..+3   NOTES 记 0.818
    responder_03 @ -1..+1   NOTES 记 0.835
    responder_04 @ -4..-2   NOTES 记 0.732
    all（上面四个设计合并，15 个回归元）  NOTES 记 0.883

判据：复现到 ±0.02 内 ⟹ NOTES 记录被证实，把证据路径补进 ROADMAP:165 / NOTES:276；
显著不同 ⟹ 按 `INCIDENT` 处理，先查再改结论。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/responder_reconstruction.py
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
from responder_window_atlas import asset_major_key  # noqa: E402

# (responder, shift) 列表；顺序即回归元顺序
DESIGNS: dict[str, list[tuple[str, int]]] = {
    "responder_00@+1..+5": [("responder_00", s) for s in (1, 2, 3, 4, 5)],
    "responder_02@0..+3": [("responder_02", s) for s in (0, 1, 2, 3)],
    "responder_03@-1..+1": [("responder_03", s) for s in (-1, 0, 1)],
    "responder_04@-4..-2": [("responder_04", s) for s in (-4, -3, -2)],
}
DESIGNS["all"] = [pair for name in list(DESIGNS) for pair in DESIGNS[name]]
NOTES_REFERENCE = {"responder_00@+1..+5": 0.207, "responder_02@0..+3": 0.818,
                   "responder_03@-1..+1": 0.835, "responder_04@-4..-2": 0.732,
                   "all": 0.883}
TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="responder_reconstruction")
    p.add_argument("--max-partitions", type=int, default=None, help="烟测用")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def multi_shift_positions(sorted_key: np.ndarray,
                          shifts: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """要求 **全部** shift 同时命中；返回 (基准位置, {shift: 对应位置})。

    atlas 的 `matched_pairs` 一次只处理一个 shift，重建需要一整套错位同时存在。
    """
    n = len(sorted_key)
    hit = np.ones(n, dtype=bool)
    where: dict[int, np.ndarray] = {}
    for s in shifts:
        wanted = sorted_key + s
        left = np.searchsorted(sorted_key, wanted)
        inside = left < n
        ok = np.zeros(n, dtype=bool)
        ok[inside] = sorted_key[left[inside]] == wanted[inside]
        hit &= ok
        where[s] = np.where(inside, left, 0)
    base = np.flatnonzero(hit)
    return base, {s: where[s][base] for s in shifts}


class Accumulator:
    """流式累加增广设计 `[X ‖ 1]` 的 D=Σw·y²、v=Σw·y·x、G=Σw·x·x'。

    带上截距列后，同一份充分统计量能同时给出两套 R²（见模块 docstring 的口径说明）。
    """

    def __init__(self, k: int) -> None:
        self.k = k                      # 不含截距的回归元个数
        self.D = 0.0
        self.v = np.zeros(k + 1)
        self.G = np.zeros((k + 1, k + 1))
        self.n = 0

    def add(self, w: np.ndarray, y: np.ndarray, X: np.ndarray) -> None:
        if not len(y):
            return
        Xa = np.column_stack([X, np.ones(len(y))])
        Xw = Xa * w[:, None]
        self.D += float(np.dot(w, y * y))
        self.v += Xw.T @ y
        self.G += Xw.T @ Xa
        self.n += len(y)

    @staticmethod
    def _solve(G: np.ndarray, v: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(G, v)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(G, v, rcond=None)[0]

    def r2_centered(self) -> float:
        """带截距、对加权均值中心化 —— NOTES 那张表用的就是这一套。"""
        if self.D <= 0 or self.n <= self.k + 1:
            return float("nan")
        sw, swy = self.G[-1, -1], self.v[-1]
        sst = self.D - swy * swy / sw if sw > 0 else 0.0
        if sst <= 0:
            return float("nan")
        sse = self.D - float(self.v @ self._solve(self.G, self.v))
        return float(1.0 - sse / sst)

    def r2_uncentered(self) -> float:
        """无截距、分母为 Σw·y² —— 与 src/metric 的 Score 同口径。"""
        if self.D <= 0 or self.n <= self.k:
            return float("nan")
        v, G = self.v[:-1], self.G[:-1, :-1]
        return float(v @ self._solve(G, v) / self.D)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    import pyarrow.parquet as pq

    started = time.perf_counter()
    all_pairs = DESIGNS["all"]
    responders = sorted({name for name, _ in all_pairs})
    all_shifts = sorted({s for _, s in all_pairs})
    columns = ["time_id", "asset_id", "weight", "target", *responders]

    own = {d: Accumulator(len(pairs)) for d, pairs in DESIGNS.items()}
    common = {d: Accumulator(len(pairs)) for d, pairs in DESIGNS.items()}
    files = train_files(Path(args.data_root))
    if args.max_partitions:
        files = files[: args.max_partitions]
    print(f"{len(files)} 个分区；{len(responders)} 个 responder；"
          f"shift 集合 {all_shifts}", flush=True)

    for path in files:
        t0 = time.perf_counter()
        frame = pq.ParquetFile(path).read(columns=columns).to_pandas()
        tid = frame["time_id"].to_numpy(dtype=np.int64)
        aid = frame["asset_id"].to_numpy(dtype=np.int64)
        span = int(tid.max()) + max(abs(s) for s in all_shifts) + 20
        order = np.argsort(asset_major_key(aid, tid, span), kind="stable")
        key = asset_major_key(aid[order], tid[order], span)
        w = np.maximum(frame["weight"].to_numpy(dtype=np.float64), 0.0)[order]
        y = frame["target"].to_numpy(dtype=np.float64)[order]
        cols = {name: frame[name].to_numpy(dtype=np.float64)[order] for name in responders}
        del frame

        # 每个 shift 的位置只与分区有关 ⟹ 一次算完，各设计共用
        base_all, pos_all = multi_shift_positions(key, all_shifts)
        finite_all = np.isfinite(y[base_all])
        for name, s in all_pairs:
            finite_all &= np.isfinite(cols[name][pos_all[s]])
        c_base = base_all[finite_all]
        c_pos = {s: pos_all[s][finite_all] for s in all_shifts}

        for design, pairs in DESIGNS.items():
            shifts = sorted({s for _, s in pairs})
            base, pos = multi_shift_positions(key, shifts)
            finite = np.isfinite(y[base])
            for name, s in pairs:
                finite &= np.isfinite(cols[name][pos[s]])
            b = base[finite]
            X = np.column_stack([cols[name][pos[s][finite]] for name, s in pairs])
            own[design].add(w[b], y[b], X)
            Xc = np.column_stack([cols[name][c_pos[s]] for name, s in pairs])
            common[design].add(w[c_base], y[c_base], Xc)

        del key, w, y, cols, order, tid, aid
        print(f"  {path.name}（{time.perf_counter()-t0:.0f}s）", flush=True)

    results: dict[str, Any] = {}
    for design in DESIGNS:
        co, cc = own[design].r2_centered(), common[design].r2_centered()
        uo, uc = own[design].r2_uncentered(), common[design].r2_uncentered()
        ref = NOTES_REFERENCE[design]
        results[design] = {
            "n_regressors": len(DESIGNS[design]),
            "shifts": [[n, s] for n, s in DESIGNS[design]],
            "r2_centered_own_rows": co, "r2_centered_common_rows": cc,
            "r2_uncentered_own_rows": uo, "r2_uncentered_common_rows": uc,
            "rows_own": own[design].n, "rows_common": common[design].n,
            "notes_reference": ref,
            "abs_diff_centered_own": abs(co - ref),
            "reproduced": bool(min(abs(co - ref), abs(cc - ref)) <= TOLERANCE),
        }
        print(f"  {design:22s} R²中心化={co:.4f} [{own[design].n:,} 行]  "
              f"R²非中心化={uo:.4f}  NOTES={ref:.3f}  "
              f"{'✅复现' if results[design]['reproduced'] else '❌不符'}", flush=True)

    reproduced = all(r["reproduced"] for r in results.values())
    pure_u = results["responder_00@+1..+5"]
    payload = {
        "experiment": "responder_reconstruction",
        "question": "错位 responder 能重建出多少 target？（补测 NOTES 2026-08-18 的重建表）",
        "why": ("ROADMAP:165 / NOTES:270 用重建 R² 关闭 horizon 分解方向，但该结果全仓库只在 "
                "NOTES.md:258,265 出现，无脚本无 JSON ⟹ 关闭理由建立在无法复现的测量上"),
        "method": ("全量 train 逐 asset 按真实 time_id 步长多 shift 同时配对；"
                   "加权 R² = vᵀG⁻¹v/D，与 src/metric 同分母，无截距、不中心化；样本内 oracle 上界"),
        "designs": results,
        "tolerance": TOLERANCE,
        "verdict": {
            "all_reproduced": bool(reproduced),
            "pure_u_hypothesis_r2_centered": pure_u["r2_centered_own_rows"],
            "pure_u_hypothesis_r2_uncentered": pure_u["r2_uncentered_own_rows"],
            "decision": ("NOTES_CONFIRMED" if reproduced else "INCIDENT_MISMATCH"),
            "reading": (
                "若 responder_00(H=1) 就是 target 所聚合的单步增量 u，则 target=(1/5)Σu_{t+h} "
                "意味着 5 个错位应给出 R²→1。实测远低于 1 ⟹ responder_00 不是那个 u，"
                "horizon 分解缺前提。"),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = ["# responder 重建测试（补测落盘）", "",
             "> **为什么补这一测**：" + payload["why"], "",
             f"口径：{payload['method']}", "",
             "| 设计 | 回归元 | R² 中心化 | R² 非中心化 | 行数 | R² 中心化(公共行) | NOTES 记录 | 复现 |",
             "|---|---:|---:|---:|---:|---:|---:|:--:|"]
    for design, r in results.items():
        lines.append(f"| `{design}` | {r['n_regressors']} | {r['r2_centered_own_rows']:.4f} | "
                     f"{r['r2_uncentered_own_rows']:.4f} | {r['rows_own']:,} | "
                     f"{r['r2_centered_common_rows']:.4f} | {r['notes_reference']:.3f} | "
                     f"{'✅' if r['reproduced'] else '❌'} |")
    lines += ["", "## 判定", "",
              f"- 全部复现：**{reproduced}** ⟹ `{payload['verdict']['decision']}`",
              f"- 纯 u 假设的 R²（中心化）= **{pure_u['r2_centered_own_rows']:.4f}**"
              f"（若成立应 →1）；非中心化 {pure_u['r2_uncentered_own_rows']:.4f}", "",
              payload["verdict"]["reading"], "",
              "## 限制", "",
              "1. 样本内 oracle 上界，不是预测性能：它回答「这些错位 responder 张成的空间"
              "覆盖了多少 target」，不回答「能预测多少」。",
              "2. 逐分区流式累加，丢失分区边界处的配对（与 atlas 同一限制）。",
              "3. NOTES 未写用的是哪一套行集，故两套都给出；只要有一套落在 ±"
              f"{TOLERANCE} 内即判复现。",
              "4. ⚠️ **口径差异很大，必须写清楚**：NOTES 那张表是**带截距的中心化 R²**。"
              "responder 带很大的非零均值（例如 `responder_03` 加权均值 +0.502 而 std 仅 "
              "0.262），因此无截距、分母为 Σw·y² 的项目指标口径会低得多（r03 设计上是 "
              "0.15 对 0.84）。判定复现用中心化那一套；非中心化一列是给「按比赛指标能兑现"
              "多少」这个问题看的。", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n判定：{payload['verdict']['decision']}\nwrote {json_path}\nwrote {md_path}",
          flush=True)


if __name__ == "__main__":
    main()
