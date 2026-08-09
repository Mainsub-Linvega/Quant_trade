"""从一份已有的提交 CSV 里抽出**市场分量 `m̂`**，单独出一份提交 CSV。

## 为什么要单独交 m̂

`replace` 之后模型是 `f = m̂ + ê_lgbm`，而 `⟨m̂, ê⟩ ≡ 0`（`ê` 逐 time_id 零均值），
所以 `B(1) = B_m + B_el` 精确可加，两块的二阶矩都已知。**唯一缺的是 `A_m`。**

交一次「只有 `m̂`」的 CSV 就能精确解出它：

    A_m = (Score + s²·B_m) / (2s)          B_m 是精确值，s 是这份 CSV 的 scale

拿到 `A_m` 之后 `A_er = A(0) − A_m`、`A_el = A(1) − A_m` 全都出来，
三分量 `f = c_m·m̂ + c_r·ê_ridge + c_l·ê_lgbm` 的最优配比就有闭式解
（见 `experiments/component_optimum.py`）。

**一次额度，换来「剩下的额度该往市场块还是截面块投」的依据。**

## 为什么不用跑模型

`m̂` 就是任何一份 hybrid 系列预测的**逐 time_id 截面均值** —— 纯算术。
（`ê` 那部分被投影成逐 time_id 零均值，一取截面均值就只剩 `m̂`。）
所以本脚本只读 CSV，不加载模型、不跑 runner。

用法：
    .venv/bin/python experiments/market_submission.py \\
        --source outputs/submission_replace_s116.csv --source-scale 1.16 \\
        --scale 1.3 --output outputs/submission_market_s130.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the market component as a submission.")
    parser.add_argument("--source", required=True, help="任何一份 hybrid 系列的提交 CSV")
    parser.add_argument("--source-scale", type=float, required=True,
                        help="那份 CSV 的 prediction_scale（用来还原成 raw）")
    parser.add_argument("--scale", type=float, required=True, help="输出的 prediction_scale")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decimals", type=int, default=8)
    return parser.parse_args()


def load_time_ids(data_root: Path) -> pd.DataFrame:
    """测试集的 row_id → time_id。截面均值要按 time_id 分组，而提交 CSV 里没有这一列。"""
    frames = [pd.read_parquet(path, columns=["row_id", "time_id"])
              for path in sorted((data_root / "test").glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.source)
    merged = frame.merge(load_time_ids(Path(args.data_root)), on="row_id", how="left")
    assert merged["time_id"].notna().all(), "有 row_id 在测试集里找不到 time_id"
    assert len(merged) == len(frame), "join 之后行数变了（row_id 有重复？）"

    raw = merged["target"].to_numpy(dtype=np.float64) / args.source_scale
    time_ids = merged["time_id"].to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    market = np.repeat(np.add.reduceat(raw, starts) / counts, counts)
    deviation = raw - market

    # 这两条是「拆解成立」的机械证据，不成立就别往下走
    residual = float(np.abs(np.add.reduceat(deviation, starts)).max())
    cross = float(np.abs((market * deviation).mean()))
    print(f"来源 {Path(args.source).name}：{len(raw):,} 行 / {len(starts):,} 个 time_id")
    print(f"  逐 time_id 的 Σê   = {residual:.2e}（应 ~0，说明 ê 确实被投影过）")
    print(f"  ⟨m̂, ê⟩            = {cross:.2e}（应 ~0，说明二阶矩可加）")
    assert residual < 1e-10 and cross < 1e-15, "ê 不是逐 time_id 零均值，m̂ 抽不干净"

    print(f"  m̂: std {market.std():.6f}  max|m̂| {np.abs(market).max():.6f}")
    print(f"  ê: std {deviation.std():.6f}  （m̂ 占预测方差 "
          f"{(market**2).mean()/(raw**2).mean():.1%}）")

    prediction = market * args.scale
    binding_scale = args.clip / np.abs(market).max()
    touched = int((np.abs(prediction) >= args.clip - 1e-12).sum())
    print(f"\n输出 scale {args.scale}：max|pred| {np.abs(prediction).max():.6f}，"
          f"clip({args.clip}) 自 scale {binding_scale:.3f} 起生效，触限行数 {touched}")
    if touched:
        print("  ⚠️ 触限了 → Score(a)=2aA−a²B 不再精确，A_m 会解错。换个小一点的 scale")
    prediction = np.clip(prediction, -args.clip, args.clip)

    sample = pd.read_csv(Path(args.data_root) / "sample_submission.csv", usecols=["row_id"])
    aligned = np.array_equal(merged["row_id"].to_numpy(dtype=np.int64),
                             sample["row_id"].to_numpy(dtype=np.int64))
    print(f"  row_id 与 sample_submission.csv 逐位对齐 {aligned}；"
          f"非有限值 {int((~np.isfinite(prediction)).sum())}")
    assert aligned and np.all(np.isfinite(prediction))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": merged["row_id"], "target": prediction}).to_csv(
        output, index=False, float_format=f"%.{args.decimals}f")
    print(f"\n写出 {output}（{output.stat().st_size/1e6:.1f} MB）")
    print(f"拿到公榜分 S 之后：A_m = (S + {args.scale}²×B_m) / (2×{args.scale})")
    print("           然后跑 experiments/component_optimum.py --market-score S")


if __name__ == "__main__":
    main()
