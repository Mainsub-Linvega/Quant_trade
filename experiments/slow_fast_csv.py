"""把 slow/fast 分离施加到一份已有的公榜 CSV 上 —— 纯后处理，不重训、不碰模型产物。

## 为什么这能只改 CSV

1. `data/test/*.parquet` 带 `row_id`/`time_id`/`asset_id`，且 `row_id` 与提交 CSV 逐行对齐；
2. 因果滚动均值是**线性**的 ⟹ `slow(m̂) + slow(ê) = trailing_mean(m̂ + ê)`，
   所以 `v3_slow_variance` 里 M2 的 slow 就是**整份原始预测**的滚动均值 —— 不需要拆分量；
3. 当前 CSV 触限 0 行 ⟹ `raw = pred / prediction_scale` 可精确反解。

## 为什么不能直接搬 OOF 的绝对系数

OOF 上全局最优 scale 是 **0.7296**，而公榜标定的是 **1.16**（差 59%，本项目已知的
本地/公榜尺子分歧）。所以只搬**相对模式**，保留公榜标定的绝对水平：

    scale_slow = 1.16 × c_slow / a_global = 1.16 × 0.2828/0.7296 = 0.4496
    scale_fast = 1.16 × c_fast / a_global = 1.16 × 0.7881/0.7296 = 1.2530

## 口径

窗口按**真实 time_id 步长**定义（不是「多少个观测」）。OOF 上选中的 K=400 采样步，
采样格平均每 `sample_modulo=5` 个真实 time_id 一个点 ⟹ 线上对应 **K_real = 2000**。
全分辨率核对（`v3_fullres_slow_probe_summary.md`）就是按这个换算做的，合并三窗 +5.93%。

⚠️ 本脚本**只生成候选 CSV**，不提交、不碰 `strategies/v3_hybrid/model/`。
按 CLAUDE.md §1.4，正式生成由用户执行；默认 `--dry-run` 只做自检不写文件。

用法：
    # 自检（不写文件）
    .venv/bin/python experiments/slow_fast_csv.py --dry-run
    # 真正生成（用户执行）
    .venv/bin/python experiments/slow_fast_csv.py \\
        --input outputs/submission_mkt_shrunk.csv \\
        --output outputs/submission_mkt_shrunk_slowfast.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# v3_slow_variance_3s480 的 pooled M2 系数与同一份 OOF 上的全局单 scale
OOF_C_SLOW, OOF_C_FAST, OOF_GLOBAL_SCALE = 0.2828, 0.7881, 0.7296
PUBLIC_SCALE = 1.16          # 公榜标定值，也是输入 CSV 用的那个
PREDICTION_CLIP = 0.5
K_REAL_STEPS = 2000          # = OOF 选中的 400 采样步 × sample_modulo 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(_REPO_ROOT / "outputs" / "submission_mkt_shrunk.csv"))
    p.add_argument("--output", default=None)
    p.add_argument("--test-glob", default=str(_REPO_ROOT / "data" / "test" / "*.parquet"))
    p.add_argument("--k-real-steps", type=int, default=K_REAL_STEPS)
    p.add_argument("--prediction-scale", type=float, default=PUBLIC_SCALE)
    p.add_argument("--prediction-clip", type=float, default=PREDICTION_CLIP)
    p.add_argument("--decimals", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="只自检、不写文件")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def causal_trailing_mean(values: np.ndarray, time_id: np.ndarray, asset_id: np.ndarray,
                         window: int) -> np.ndarray:
    """逐 asset、只用**当期之前**、且 time_id 在 `window` 真实步之内的观测求均值。

    与 `v3_fullres_slow_probe.causal_trailing_mean` 同一实现口径：段首无历史时取自身
    （于是 fast=0，不制造假信号）；窗口按真实步长 ⟹ time_id 跳号自然处理。
    """
    order = np.lexsort((time_id, asset_id))
    v, t, a = values[order], time_id[order], asset_id[order]
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    ends = np.r_[starts[1:], len(a)]
    # ⚠️ 累积和必须**逐资产段内从 0 开始**。早先是对整个（asset, time）排序后的数组做一次
    # 全局 cumsum —— 那样第二个资产往后的累积值里含着前面所有资产的总和作为偏移，
    # 两个大数相减的舍入与「从 0 开始累加」不同，实测差 1e-16~1e-15、且**在线端无法复现**
    # （它不可能知道其他资产未来的总和）。段内累积后两端就是同一个算式、逐位相同。
    result = np.empty_like(v)
    for start, end in zip(starts, ends):
        segment = v[start:end]
        cumulative = np.concatenate([[0.0], np.cumsum(segment)])
        local = np.arange(end - start)
        stamps = t[start:end]
        left = np.searchsorted(stamps, stamps - window, side="left")
        count = np.maximum(local - left, 1)
        result[start:end] = (cumulative[local] - cumulative[left]) / count
        result[start] = segment[0]
    out = np.empty_like(values)
    out[order] = result
    return out


def main() -> None:
    args = parse_args()
    scale_slow = args.prediction_scale * OOF_C_SLOW / OOF_GLOBAL_SCALE
    scale_fast = args.prediction_scale * OOF_C_FAST / OOF_GLOBAL_SCALE
    print(f"相对修正后的两个 scale：slow={scale_slow:.4f}，fast={scale_fast:.4f}"
          f"（原单一 {args.prediction_scale}）；K_real={args.k_real_steps}", flush=True)

    submission = pd.read_csv(args.input)
    if list(submission.columns) != ["row_id", "target"]:
        raise SystemExit(f"unexpected submission columns: {list(submission.columns)}")
    files = sorted(glob.glob(args.test_glob))
    if not files:
        raise SystemExit(f"no test parquet matched {args.test_glob}")
    import pyarrow.parquet as pq
    keys = pd.concat([pq.ParquetFile(f).read(columns=["row_id", "time_id", "asset_id"]).to_pandas()
                      for f in files], ignore_index=True)

    merged = submission.merge(keys, on="row_id", how="left", validate="one_to_one")
    if merged["time_id"].isna().any():
        raise AssertionError("some row_id in the submission has no match in the test parquet")
    if len(merged) != len(submission):
        raise AssertionError("merge changed the row count")
    merged = merged.sort_values(["time_id", "asset_id"], kind="stable")

    pred = merged["target"].to_numpy(np.float64)
    clipped_in = int((np.abs(pred) >= args.prediction_clip).sum())
    if clipped_in:
        raise AssertionError(
            f"输入 CSV 有 {clipped_in} 行触限 ⟹ raw 无法精确反解，本变换不适用")
    raw = pred / args.prediction_scale

    time_id = merged["time_id"].to_numpy(np.int64)
    asset_id = merged["asset_id"].to_numpy(np.int64)
    slow = causal_trailing_mean(raw, time_id, asset_id, args.k_real_steps)
    fast = raw - slow
    if float(np.abs(slow + fast - raw).max()) > 1e-12:
        raise AssertionError("slow + fast does not reconstruct raw")

    combined = scale_slow * slow + scale_fast * fast
    out = np.clip(combined, -args.prediction_clip, args.prediction_clip)
    clipped_out = int((np.abs(combined) >= args.prediction_clip).sum())

    if not np.all(np.isfinite(out)):
        raise AssertionError("non-finite value in the transformed prediction")

    print(f"输入 {len(submission):,} 行；max|pred|={np.abs(pred).max():.6f}，"
          f"max|raw|={np.abs(raw).max():.6f}", flush=True)
    slow_share = float(np.var(slow) / np.var(raw))
    print(f"slow 占方差 {slow_share:.4f}；corr(slow, raw)="
          f"{float(np.corrcoef(slow, raw)[0, 1]):.4f}"
          f"  ← OOF 采样格上约 0.17，这里更低是**预期的**：全分辨率窗内点多 5 倍、"
          f"滚动均值噪声更小。系数是在采样格上解出的 ⟹ 施加到全分辨率上偏保守，"
          f"而全分辨率核对（v3_fullres_slow_probe_summary）测的正是这个组合，合并三窗 +5.93%。",
          flush=True)
    print(f"输出 max|pred|={np.abs(out).max():.6f}，触限行数={clipped_out}"
          f"（>0 会让二次式失效、与现有公榜分不可直接比大小）", flush=True)
    print(f"corr(输出, 输入)={float(np.corrcoef(out, pred)[0, 1]):.6f}；"
          f"输出 std / 输入 std = {float(np.std(out) / np.std(pred)):.4f}", flush=True)
    if clipped_out:
        print("⚠️ 输出触限 —— 建议先复核 K 与两个 scale，不要直接提交", flush=True)

    if args.dry_run:
        print("\n--dry-run：未写任何文件。正式生成请由用户执行并指定 --output。", flush=True)
        return

    if args.output is None:
        raise SystemExit("需要 --output（或用 --dry-run 只自检）")
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"{output} 已存在；要覆盖请加 --force")
    merged = merged.assign(target=out)[["row_id", "target"]]
    merged = merged.sort_values("row_id", kind="stable")
    if not np.array_equal(merged["row_id"].to_numpy(), np.sort(submission["row_id"].to_numpy())):
        raise AssertionError("output row_id set differs from the input")
    merged.to_csv(output, index=False, float_format=f"%.{args.decimals}f")
    print(f"wrote {output}（{len(merged):,} 行）", flush=True)


if __name__ == "__main__":
    main()
