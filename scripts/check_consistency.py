"""断言训练侧与推理侧（main.Model.predict）口径一致。

两侧共用 features.py，但截面去均值仍是两条浮点路径（分组 reduceat vs 单批 mean）。
本脚本从真实分区抽若干 time_id，分别走两条路径，逐元素断言在 1e-6 内一致。
任何人改了预处理却只改了一侧，这里会响。

支持两个策略，训练侧参照物不同：

- `v1_ridge`：`train.predict_array`（整块矩阵 + 分组截面去均值）
- `v3_hybrid`：没有对应的整块函数（岭回归冻结、LGBM 只训截面分量），
  所以拿 `main.Model` 的**离线全量口径**做参照 —— 一次喂进全部 time_id、
  用 `cross_sectional_deviation` 分组去均值，对上逐 time_id 顺序喂的在线路径。
  这正是 08-08 那次 `max|Δ|=1.138e-03` 翻车（工程坑第 7 条）能被抓住的那个检查。

用法：
    .venv/bin/python scripts/check_consistency.py [--strategy v1_ridge] [--n-time-ids 50]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io import FEATURE_COLUMNS, train_files       # noqa: E402 —— 要先设好 sys.path
from src.artifact import sha256_file                  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert train/inference prediction parity.")
    parser.add_argument("--strategy", default="v1_ridge", choices=["v1_ridge", "v3_hybrid"])
    parser.add_argument("--backend", default=None, choices=["lightgbm", "numpy"],
                        help="v3_hybrid 专用：树推理走哪条路（默认自动选）")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--partition-index", type=int, default=8)
    parser.add_argument("--n-time-ids", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--model-path", default=None,
                        help="默认取 strategies/<strategy>/model/baseline_model.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_leading_time_ids(path: Path, n_time_ids: int):
    """读取分区开头 n 个完整 time_id 的行（丢掉可能被批边界切开的最后一个）。"""
    import pandas as pd

    columns = ["row_id", "time_id", "asset_id", *FEATURE_COLUMNS]
    frames = []
    seen: set[int] = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
        frame = batch.to_pandas()
        frames.append(frame)
        seen.update(frame["time_id"].unique().tolist())
        if len(seen) > n_time_ids + 1:
            break
    data = pd.concat(frames, ignore_index=True)
    kept = sorted(seen)[: n_time_ids]
    return data[data["time_id"].isin(kept)].reset_index(drop=True)


def load_strategy(name: str):
    """把 strategies/<name> 压进 sys.path 后 import 它的 main / train。

    一次只加载一个策略 —— 两个策略的模块都叫 `main`/`train`，同进程混用会串。
    """
    strategy_dir = _REPO_ROOT / "strategies" / name
    assert strategy_dir.is_dir(), f"策略目录不存在: {strategy_dir}"
    sys.path.insert(0, str(strategy_dir))
    for module in ("main", "train", "features", "lgbm_numpy"):
        sys.modules.pop(module, None)
    return importlib.import_module("main"), importlib.import_module("train")


def _causal_trailing_mean(values, time_id, asset_id, window):
    """复用 `experiments/slow_fast_csv.py` 的**同一个**实现，不另写一份（伤疤规则 §3）。"""
    import importlib.util
    path = _REPO_ROOT / "experiments" / "slow_fast_csv.py"
    spec = importlib.util.spec_from_file_location("_slow_fast_csv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.causal_trailing_mean(values, time_id, asset_id, window)


def main() -> None:
    args = parse_args()
    strategy_main, strategy_train = load_strategy(args.strategy)
    model_path = Path(args.model_path) if args.model_path else (
        _REPO_ROOT / "strategies" / args.strategy / "model" / "baseline_model.json")
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    files = train_files(Path(args.data_root))
    frame = load_leading_time_ids(files[args.partition_index], args.n_time_ids)
    print(f"[{args.strategy}] loaded {len(frame):,} rows / {frame['time_id'].nunique()} "
          f"time_ids from {files[args.partition_index].name}")

    # 训练侧路径：整块矩阵 + 分组截面去均值
    full = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    time_ids = frame["time_id"].to_numpy(dtype=np.int64)
    if args.strategy == "v3_hybrid":
        # 两侧走同一个 backend，否则量到的是「两个后端的差」而不是「两条口径的差」
        pred_train = strategy_train.predict_array(
            model_path.parent, full, time_ids,
            frame["asset_id"].to_numpy(dtype=np.int64),
            backend=args.backend or "lightgbm",
        )
    else:
        pred_train = strategy_train.predict_array(
            artifact,
            full,
            time_ids,
            np.asarray(artifact["selected_indices"], dtype=np.int64),
            float(artifact["prediction_scale"]),
            float(artifact["prediction_clip"]),
        )

    meta_for_slow_fast = {}
    if args.strategy == "v3_hybrid":
        meta_for_slow_fast = json.loads(
            (model_path.parent / "hybrid_meta.json").read_text(encoding="utf-8"))

    # 推理侧路径：官方 API 语义，按 time_id 递增逐批喂 Model
    kwargs = {"backend": args.backend} if args.strategy == "v3_hybrid" else {}
    model = strategy_main.Model(model_path=model_path, **kwargs)
    parts = []
    for time_id in sorted(frame["time_id"].unique()):
        parts.append(model.predict(frame[frame["time_id"] == time_id]))
    pred_infer = np.concatenate(parts)

    if pred_infer.shape != pred_train.shape:
        raise AssertionError("两侧输出长度不一致")

    # ---- slow/fast 是**推理侧后处理**，训练端没有它的概念（它依赖跨 predict 的状态）。
    # 若不在这里补上，本门禁会**永久报红**（实测 9.4e-02）—— 而长期红灯只会让人学会忽略它，
    # 真正的预处理回归反而被掩盖。所以：把同一个离线变换补到训练侧，再比。
    # 补完之后本门禁依然在测它该测的东西（特征/预处理两侧是否同口径）。
    slow_fast_window = int(meta_for_slow_fast.get("slow_fast_window") or 0)
    if slow_fast_window:
        scale = float(meta_for_slow_fast["prediction_scale"])
        clip = float(meta_for_slow_fast["prediction_clip"])
        if float(np.max(np.abs(pred_train))) >= clip:
            raise AssertionError(
                "训练侧预测已触限 ⟹ raw = pred/scale 无法精确反解，本门禁的 slow/fast "
                "补偿不成立；请改用未限幅的训练侧输出")
        raw = pred_train.astype(np.float64) / scale
        slow = _causal_trailing_mean(
            raw, time_ids, frame["asset_id"].to_numpy(dtype=np.int64), slow_fast_window)
        pred_train = np.clip(
            scale * float(meta_for_slow_fast["slow_fast_slow_relative"]) * slow
            + scale * float(meta_for_slow_fast["slow_fast_fast_relative"]) * (raw - slow),
            -clip, clip)
        print(f"（已对训练侧补上 slow/fast：window={slow_fast_window} 真实步）")
    max_diff = float(np.max(np.abs(pred_train.astype(np.float64) - pred_infer.astype(np.float64))))
    print(f"max |train - infer| = {max_diff:.3e}")
    if not np.isfinite(max_diff) or max_diff > args.atol:
        raise AssertionError(
            f"训练与推理口径不一致（max diff {max_diff:.3e} > atol {args.atol:g}）——"
            "有人改了一侧的预处理没改另一侧"
        )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "backend": getattr(model, "backend", None),
                    "model_path": str(model_path),
                    "model_sha256": sha256_file(model_path),
                    "partition": files[args.partition_index].name,
                    "time_ids": int(frame["time_id"].nunique()),
                    "rows": int(len(frame)),
                    "atol": args.atol,
                    "max_abs_diff": max_diff,
                    # 指纹：确认这份报告出自哪一版模型（与验收报告的 candidate_std 对读）
                    "online_pred_std": float(np.std(pred_infer.astype(np.float64))),
                    "online_pred_absmax": float(np.max(np.abs(pred_infer.astype(np.float64)))),
                    "passed": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print("OK: 训练与推理口径一致")


if __name__ == "__main__":
    main()
