"""v3_hybrid 训练：只训 LightGBM 的截面分量，岭回归部分冻结复用生产模型。

## 为什么岭回归不重训

`model/baseline_model.json` 是 `strategies/v1_ridge/` 的生产模型原样拷贝
（sha `c23a8cfb`，公榜实测 0.00187232）。冻结它有两个好处：

1. **唯一的变量是截面分量** —— 与公榜 0.00187232 对比时，差异纯粹来自 LightGBM
2. 那份模型的可复现性已经验收过（严格求解器，7 道门禁）

所以本脚本只产出 LightGBM 部分 + `hybrid_meta.json`。

## 超参从哪来（全部预注册，不在这里搜）

来自 `lgbm_blend_unweighted`（5 折、3 种子、无泄漏口径、判据由 `verdict_of()` 机器判）：

- 候选 `xs_loose`：`num_leaves 63 / lr 0.03 / feature_fraction 0.7 / lambda_l2 1.0`，
  `min_data_in_leaf` 取训练行数的 12000/3.5e6 ≈ 0.343%
- 轮数取**逐折 best_iteration 的折均 160**（逐折范围 44~302）——
  照增补包的做法，不在全量上再搜一次
- 混合 `blend50`：折后 +11.1%，四个臂里最高（`replace` 原始分更高但 ΔB 涨 17.9%，折后只剩 +9.8%）

## 口径

目标 `e = y − 无权截面均值(y)`。**无权**是因为推理端拿不到 weight
（test 无该列，且 runner 的 `forbidden` 会剥掉），详见 `main.py` 的模块注释。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python strategies/v3_hybrid/train.py
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments"),
              str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import apply_robust_transform, cross_sectional_deviation

# ⚠️ 训练侧还要 v1_ridge 的 `train.robust_transform_fit` / `select_features` 与
# `experiments.lgbm_xs.load_rows`，但它们**只在 main() 里 import**。原因：本文件被
# `import train` 加载时模块名就叫 `train`，模块级写 `from train import …` 会导入自己
# （当脚本跑时模块名是 `__main__`，所以以前一直没暴露）。
# 顺带的好处是 `predict_array` 只依赖 numpy + features.py，离线用不必拖进整套训练栈。

# 全部来自 lgbm_blend_unweighted，预注册、不在这里搜
SPEC = {"num_leaves": 63, "learning_rate": 0.03,
        "feature_fraction": 0.7, "lambda_l2": 1.0}
MIN_DATA_FRAC = 12000 / 3_500_000        # lgbm_blend 里 min_data_in_leaf 与行数的比例
NUM_ITERATION = 160                       # 逐折 best_iteration 的折均（范围 44~302）
# 每资产滚动历史特征（`history_peak`，08-11）。history 列**只从选中的 200 列里取**，
# 这样推理端直接复用那 200 列的 lower/upper/center/scale，不必扩推理输入契约。
HISTORY_COUNT = 40                        # 与 walk_forward_history 的 history_feature_count 同
HISTORY_WINDOW = 5                        # 与 AssetHistory.window_size 的历史值同
BLEND_WEIGHT = 0.5                        # blend50：ê 里 LGBM 占一半，不拟合权重
# 行级市场模型（`combo_market_weight`，08-13）。m̂ = (1−λ)·m̂_ridge + λ·m̂_lgbm，
# λ 是先验、不拟合（ROADMAP §5）。设计矩阵 = [raw ‖ xs_dev ‖ history ‖ asset_id]（561 列），
# 即行级岭回归基底的对应物；标签是 **y** 不是 e，取其逐 time_id 无权截面均值当市场分量。
MARKET_LAMBDA = 0.5


def _asset_scaled_zero_mean(values: np.ndarray, asset_ids: np.ndarray, scales: np.ndarray,
                            time_ids: np.ndarray | None = None) -> np.ndarray:
    """Scale the cross block by asset and re-project to zero mean."""
    values = np.asarray(values, dtype=np.float64)
    asset_ids = np.asarray(asset_ids, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.ndim != 1 or len(scales) == 0 or not np.all(np.isfinite(scales)):
        raise ValueError("asset_cross_scales must be a finite non-empty 1D array")
    if asset_ids.min() < 0 or asset_ids.max() >= len(scales):
        raise ValueError("asset_id is outside asset_cross_scales")
    adjusted = values * scales[asset_ids]
    if time_ids is None:
        return adjusted - adjusted.mean()
    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])
    return adjusted - np.repeat(np.add.reduceat(adjusted, starts) / counts, counts)


def predict_array(model_dir, full_features: np.ndarray, time_ids: np.ndarray,
                  asset_ids: np.ndarray, backend: str = "lightgbm") -> np.ndarray:
    """离线全量口径的预测 —— 与 `main.Model.predict` 数学等价，但一次吃全部 time_id。

    存在的理由有两个：

    1. **`scripts/check_consistency.py` 的参照物**。在线路径是逐 time_id 喂进来的，
       离线这条是整块 `cross_sectional_deviation` 分组去均值 —— 两条浮点求和顺序不同。
       08-08 就是在这个对比里抓到 `max|Δ|=1.138e-03`（工程坑第 7 条）。
    2. 想在本地给 v3_hybrid 算分时，不必绕 `timeseries_api` 的顺序 runner。

    ⚠️ 与 `main.py` 的口径必须逐行对应：市场分量走**无权**截面均值、
    ê 投影成无权零均值、限幅只在最后做一次。改了一侧就得改另一侧 —— 所以才有那个检查。
    """
    model_dir = Path(model_dir)
    ridge = json.loads((model_dir / "baseline_model.json").read_text(encoding="utf-8"))
    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))

    starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_ids)])

    def group_mean(values: np.ndarray) -> np.ndarray:
        """逐 time_id 的无权截面均值，广播回每一行。"""
        return np.repeat(np.add.reduceat(values, starts) / counts, counts)

    def stats(artifact: dict) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(artifact[name], dtype=np.float32)
                     for name in ("lower", "upper", "center", "scale"))

    # ---- 岭回归的原始预测（不乘 scale、不 clip）
    selected = np.asarray(ridge["selected_indices"], dtype=np.int64)
    raw = full_features[:, selected].copy()
    lower, upper, center, scale = stats(ridge)
    apply_robust_transform(raw, lower, upper, center, scale)
    dev = cross_sectional_deviation(
        raw, time_ids, str(ridge.get("cross_sectional_scaling", "none")))
    ridge_raw = (np.float32(ridge["intercept"])
                 + np.column_stack([raw, dev]) @ np.asarray(ridge["coef"], dtype=np.float32))
    del raw, dev

    market = group_mean(ridge_raw)
    e_ridge = ridge_raw - market

    # ---- LightGBM 的截面分量，投影成无权零均值
    lgbm_indices = np.array([int(name.split("_")[1]) for name in meta["lgbm_features"]])
    lraw = full_features[:, lgbm_indices].copy()
    lower, upper, center, scale = stats(meta)
    apply_robust_transform(lraw, lower, upper, center, scale)
    blocks = [cross_sectional_deviation(lraw, time_ids)]
    # ---- 每资产滚动历史（有状态）。这里从**空历史**起步，与 main.Model 新建时一致，
    # check_consistency 两侧才可比。history.AssetHistory 的整块调用与逐 time_id 调用
    # 逐位相同（定序直接求和，不用 cumsum），所以这条离线路径不会引入 ulp 级偏差。
    if meta.get("history_positions"):
        from history import history_design_blocks
        hist, _ = history_design_blocks(lraw, asset_ids.astype(np.int64),
                                        meta["history_positions"],
                                        int(meta["history_window"]))
        blocks.extend(hist)
    # ---- 长窗块（有状态，同样从**空历史**起步，与 main.Model 新建时一致）
    long_blocks: list[np.ndarray] = []
    if meta.get("long_window") and meta.get("history_positions"):
        from history import AssetLongWindow
        positions = np.asarray(meta["history_positions"], dtype=np.int64)
        long_blocks = list(AssetLongWindow(feature_count=len(positions),
                                           window=int(meta["long_window"])).transform(
            lraw[:, positions], asset_ids.astype(np.int64)))
    blocks.append(asset_ids.astype(np.float32))       # ⚠️ asset_id 必须留在最后一列
    # ⚠️ 与 train 的组装顺序逐列对应：长窗块只在截面设计里，插在 asset_id 之前
    design = np.column_stack(blocks[:-1] + long_blocks + blocks[-1:])
    # 市场块的设计矩阵只是在同一批块前面多拼一个 raw（与 train 的 market_design 逐列对应）
    market_design = (np.column_stack([lraw, *blocks])
                     if meta.get("market_model_files") else None)
    del lraw, blocks

    def run_forest(names: list[str], matrix: np.ndarray, num_iteration: int) -> np.ndarray:
        paths = [model_dir / name for name in names]
        if backend == "lightgbm":
            import lightgbm as lgb
            total = np.zeros(len(matrix), dtype=np.float64)
            for path in paths:
                total += lgb.Booster(model_file=str(path)).predict(
                    matrix, num_iteration=num_iteration)
        elif backend == "numpy":
            from lgbm_numpy import NumpyForest
            forest = NumpyForest.from_files(paths, num_iteration)
            total = np.concatenate([
                forest.predict_sum(matrix[start:start + count],
                                   asset_ids[start:start + count].astype(np.int64))
                for start, count in zip(starts, counts)])
        else:
            raise ValueError(f"未知 backend {backend!r}")
        return total / len(paths)

    e_lgbm = run_forest(meta["lgbm_model_files"], design, int(meta["num_iteration"]))
    e_lgbm -= group_mean(e_lgbm)
    asset_scales = meta.get("asset_cross_scales")
    if asset_scales is not None:
        e_lgbm = _asset_scaled_zero_mean(e_lgbm, asset_ids, asset_scales, time_ids)

    # ---- 第二个市场分量：行级 LGBM 打 y，取逐 time_id 无权截面均值
    if market_design is not None:
        lam = np.float64(meta.get("market_lambda", 0.0))
        market = (1.0 - lam) * market + lam * group_mean(
            run_forest(meta["market_model_files"], market_design,
                       int(meta.get("market_num_iteration", meta["num_iteration"]))))
        del market_design

    weight = float(meta["blend_weight"])
    blended = market + (1.0 - weight) * e_ridge + weight * e_lgbm
    clip = np.float32(meta["prediction_clip"])
    return np.clip(blended * np.float32(meta["prediction_scale"]), -clip, clip)


def stream_history_blocks(data_root: Path, sample_modulo: int, sampling: str,
                          history_names: list[str], history_stats: tuple[np.ndarray, ...],
                          window: int):
    """流式扫过**每一行**，只留下被采样行的 4 个历史块。

    ⚠️ 历史状态必须在每一行上推进，而 `lgbm_xs.load_rows` 是**逐 batch 先掩码再拼接**、
    未被选中的行根本不进内存 —— 所以历史块没法从它的输出里重建，只能自己再扫一遍。
    好在只需要 history 那几列，一遍很快。

    采样掩码走 `src.io.time_sample_mask`，与 `load_rows` 逐位同口径
    （生产是 `phase_balanced` + modulo 5；写死 `% modulo == 0` 会与训练集错位）。
    """
    import pyarrow.parquet as pq
    from src.io import time_sample_mask, train_files
    from history import AssetHistory

    lower, upper, center, scale = history_stats
    history = AssetHistory(feature_count=len(history_names), window_size=window)
    parts: list[list[np.ndarray]] = [[], [], [], []]
    kept = 0
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=120_000, columns=["time_id", "asset_id", *history_names]):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            current = frame.loc[:, history_names].to_numpy(dtype=np.float32, copy=True)
            apply_robust_transform(current, lower, upper, center, scale)
            blocks = history.transform(current, aid)          # 每一行都推进状态
            mask = time_sample_mask(tid, sample_modulo, sampling=sampling)
            if mask.any():
                for slot, block in zip(parts, blocks):
                    slot.append(block[mask])
                kept += int(mask.sum())
        print(f"  history {path.name}: 累计留下 {kept:,} 行", flush=True)
    return [np.concatenate(slot) for slot in parts]


def stream_long_window_blocks(data_root: Path, sample_modulo: int, sampling: str,
                              history_names: list[str], history_stats: tuple[np.ndarray, ...],
                              window: int):
    """与 `stream_history_blocks` 同构，但产出**长窗**的 2 个块（滚动均值 / 偏离）。

    复用**同一批 history 列与同一套统计量** ⟹ 推理端不必扩输入契约，
    meta 也只多一个 `long_window` 键。

    ⚠️ 同样必须扫过**每一行**推进状态；`AssetLongWindow` 是 O(1)/行，
    离线整块与在线逐 time_id 逐位相同（见其 docstring 与 tests/test_asset_long_window.py）。
    """
    import pyarrow.parquet as pq
    from src.io import time_sample_mask, train_files
    from history import AssetLongWindow

    lower, upper, center, scale = history_stats
    state = AssetLongWindow(feature_count=len(history_names), window=window)
    parts: list[list[np.ndarray]] = [[], []]
    kept = 0
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=120_000, columns=["time_id", "asset_id", *history_names]):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            current = frame.loc[:, history_names].to_numpy(dtype=np.float32, copy=True)
            apply_robust_transform(current, lower, upper, center, scale)
            blocks = state.transform(current, aid)            # 每一行都推进状态
            mask = time_sample_mask(tid, sample_modulo, sampling=sampling)
            if mask.any():
                for slot, block in zip(parts, blocks):
                    slot.append(block[mask])
                kept += int(mask.sum())
        print(f"  long{window} {path.name}: 累计留下 {kept:,} 行", flush=True)
    return [np.concatenate(slot) for slot in parts]


def stream_history_range_blocks(data_root: Path, history_names: list[str],
                                history_stats: tuple[np.ndarray, ...], window: int,
                                train_range: tuple[int, int],
                                valid_range: tuple[int, int]):
    """Advance history on every real row but retain only one fold's train/valid ranges.

    Full-resolution runs cannot materialise four history blocks for all 13.2m rows
    (~8.5GB) and then slice them. This variant preserves identical causal state while
    retaining only the rows used by the current fold.
    """
    import pyarrow.parquet as pq
    from src.io import train_files
    from history import AssetHistory

    lower, upper, center, scale = history_stats
    history = AssetHistory(feature_count=len(history_names), window_size=window)
    train_parts: list[list[np.ndarray]] = [[], [], [], []]
    valid_parts: list[list[np.ndarray]] = [[], [], [], []]
    train_low, train_high = map(int, train_range)
    valid_low, valid_high = map(int, valid_range)
    train_rows = valid_rows = 0
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=120_000, columns=["time_id", "asset_id", *history_names]):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            current = frame.loc[:, history_names].to_numpy(dtype=np.float32, copy=True)
            apply_robust_transform(current, lower, upper, center, scale)
            blocks = history.transform(current, aid)
            train_mask = (tid >= train_low) & (tid <= train_high)
            valid_mask = (tid >= valid_low) & (tid <= valid_high)
            if train_mask.any():
                for slot, block in zip(train_parts, blocks):
                    slot.append(block[train_mask])
                train_rows += int(train_mask.sum())
            if valid_mask.any():
                for slot, block in zip(valid_parts, blocks):
                    slot.append(block[valid_mask])
                valid_rows += int(valid_mask.sum())
        print(f"  history-range {path.name}: train={train_rows:,}, valid={valid_rows:,}",
              flush=True)
    return ([np.concatenate(slot) for slot in train_parts],
            [np.concatenate(slot) for slot in valid_parts])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LightGBM cross-sectional part of v3_hybrid.")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    p.add_argument("--sample-modulo", type=int, default=5)
    p.add_argument("--sampling", default="phase_balanced")
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--history-count", type=int, default=HISTORY_COUNT)
    p.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--long-window", type=int, default=0,
                   help="长窗滚动均值的窗长（观测数）。0 = 关闭 ⟹ 与旧模型逐位相同。"
                        "复用 history 那 40 列，只进**截面**设计，市场设计不变。"
                        "证据：outputs/experiments/long_window_confirm.md（3s480 +7.77%/5-of-5）")
    p.add_argument("--no-history", action="store_true",
                   help="退回到 08-10 那版无历史特征的设计矩阵（回归对照用）")
    p.add_argument("--num-iteration", type=int, default=NUM_ITERATION,
                   help="XS forest boosting rounds")
    p.add_argument("--market-num-iteration", type=int, default=None,
                   help="market forest rounds; defaults to --num-iteration for compatibility")
    # ---- 08-13 两个新开关，默认关闭 ⟹ 不传参数时产物与 08-11 的候选逐位相同
    p.add_argument("--weighted-cross-section", action="store_true",
                   help="截面块 ê 带 sample_weight 训练（与指标 Σw(y−ŷ)² 对齐）。"
                        "本地 5 折 +4.50%%、机制是 B 缩 3.5%% 而非 A 涨（combo_market_weight）")
    p.add_argument("--market-model", action="store_true",
                   help="额外训一个行级 LGBM 打 y，取其截面均值当第二个市场分量。"
                        "⚠️ 它**不带权** —— 带权反而掉 3.5 个百分点（combo_market_weight 的 mkt_wm 格）")
    p.add_argument("--market-lambda", type=float, default=MARKET_LAMBDA)
    # ---- ②类扫描用（08-13）。两片森林至今共用同一组 SPEC，而那组数出自 lgbm_xs 的
    # `xs_loose`，是 2026-08-08 为「截面残差 e / 200 列 dev / 160 轮」挑的 ——
    # 市场模型（标签 y / 561 列 / 480 轮）的超参**一个都没为它选过**。
    p.add_argument("--xs-spec", default=None,
                   help="覆盖截面块超参的 JSON，例如 '{\"num_leaves\":31}'；与 SPEC 合并")
    p.add_argument("--market-spec", default=None, help="同上，作用于市场块")
    p.add_argument("--xs-min-data-scale", type=float, default=1.0,
                   help="截面块 min_data_in_leaf 的倍数（基数是行数 × MIN_DATA_FRAC）")
    p.add_argument("--market-min-data-scale", type=float, default=1.0)
    p.add_argument("--train-only", default="both",
                   choices=["both", "cross_section", "market"],
                   help="只训一片森林，另一片从 --reuse-from 原样复用 —— "
                        "扫②类时只有被调的那片需要重训，成本减半")
    p.add_argument("--reuse-from", default=None,
                   help="被跳过那片森林的来源候选目录。会硬校验选列与预处理统计量逐位相同，"
                        "对不上就退出（否则两片森林不在同一个特征空间里）")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=16)
    # scale 是纯后处理旋钮，最终值由公榜两点法定；这里先写本地最优做占位
    p.add_argument("--prediction-scale", type=float, default=0.856)
    p.add_argument("--prediction-clip", type=float, default=0.5)
    p.add_argument("--allow-production-overwrite", action="store_true",
                   help="显式允许写入 strategies/v3_hybrid/model；默认拒绝。"
                        "与 v1_ridge/train.py 同一条闸门 —— 候选一律走 --model-dir "
                        "outputs/candidates/…，确认上榜后再由 scripts/promote_v3_candidate.py 转正")
    return p.parse_args()


def main() -> None:
    import lightgbm as lgb
    from train import robust_transform_fit, select_features    # v1_ridge 的生产实现
    from lgbm_xs import load_rows                              # 流式加载（带 asset_id）

    args = parse_args()
    market_num_iteration = (args.num_iteration if args.market_num_iteration is None
                            else args.market_num_iteration)
    if args.num_iteration <= 0 or market_num_iteration <= 0:
        raise SystemExit("num iterations must be positive")
    model_dir = Path(args.model_dir)
    production_dir = Path(__file__).resolve().parent / "model"
    # ⚠️ --model-dir 的默认值就是生产目录，跑一次没带参数就把上榜模型覆盖了。
    # v1_ridge/train.py 早有这条闸门，这里 2026-08-13 补上（伤疤清单：产物归属出过事）。
    if model_dir.resolve() == production_dir.resolve() and not args.allow_production_overwrite:
        raise SystemExit(
            "拒绝覆盖正式模型目录；候选请传 --model-dir outputs/candidates/…，"
            "确认上榜后走 scripts/promote_v3_candidate.py 转正，"
            "真要直写请显式加 --allow-production-overwrite")
    model_dir.mkdir(parents=True, exist_ok=True)
    assert (model_dir / "baseline_model.json").exists(), \
        "model/baseline_model.json 缺失 —— 它是 v1_ridge 生产模型的冻结拷贝"

    print(f"加载训练数据（modulo {args.sample_modulo} / {args.sampling}）…", flush=True)
    d = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features, y, w, tid, aid = d["features"], d["target"], d["weight"], d["time_id"], d["asset_id"]
    del d
    # ⚠️ weight 只有训练端有（推理端拿不到），所以它只能进**损失**，绝不能进特征。
    # 截面块带权是 08-13 立的（+4.50%）；市场块**刻意不带权**（带权 −3.5%）。
    sample_weight = np.maximum(w.astype(np.float64), 0.0) if args.weighted_cross_section else None
    del w
    assert np.all(np.diff(tid) >= 0), "行未按 time_id 排序，截面聚合会算错"

    # 目标：无权截面残差
    starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(tid)])
    e = y - np.repeat(np.add.reduceat(y, starts) / counts, counts)
    resid = float(np.abs(np.add.reduceat(e, starts)).max())
    assert resid < 1e-8, f"逐 time_id 的 e 之和不为 0（{resid:.2e}）"
    print(f"{len(tid):,} 行 / {len(starts):,} 个 time_id；截面残差校验 {resid:.2e} ✅", flush=True)

    # 预处理与选列（全量拟合 —— 这是最终模型，没有留出段）
    scratch, stats = robust_transform_fit(features.copy())
    selected = select_features(scratch, e, np.ones_like(e), args.feature_count)
    del scratch
    lgbm_features = [f"feature_{i:03d}" for i in selected]

    raw = features[:, selected].copy()
    del features
    apply_robust_transform(raw, stats["lower"][selected], stats["upper"][selected],
                           stats["center"][selected], stats["scale"][selected])
    dev = cross_sectional_deviation(raw, tid)
    if not args.market_model:
        del raw                                # 市场块要用它当前 200 列，别提前放掉
    blocks = [dev]

    # ---- 每资产滚动历史。history 列在**选中的 200 列之内**按 e 选（无权，与选列同口径），
    # 于是 history 列的统计量已经在 meta 的那 200 个里，推理端不用扩输入契约。
    history_positions: list[int] = []
    if not args.no_history:
        inner = select_features(dev, e, np.ones_like(e), args.history_count)
        history_positions = sorted(int(i) for i in inner)
        history_names = [lgbm_features[i] for i in history_positions]
        sl = np.asarray(history_positions, dtype=np.int64)
        history_stats = tuple(stats[k][selected][sl] for k in ("lower", "upper", "center", "scale"))
        print(f"history 列 {len(history_names)} 个（取自选中的 {args.feature_count} 列，"
              f"窗长 {args.history_window}）；再扫一遍全量建历史块…", flush=True)
        blocks.extend(stream_history_blocks(Path(args.data_root), args.sample_modulo,
                                            args.sampling, history_names, history_stats,
                                            args.history_window))

    # ---- 长窗块（可选）。复用同一批 history 列与统计量，只多一个 meta 键。
    long_blocks: list[np.ndarray] = []
    if args.long_window and history_positions:
        print(f"长窗块 window={args.long_window}（复用同 {len(history_names)} 列）；"
              f"再扫一遍全量…", flush=True)
        long_blocks = list(stream_long_window_blocks(
            Path(args.data_root), args.sample_modulo, args.sampling,
            history_names, history_stats, args.long_window))

    blocks.append(aid.astype(np.float32))          # ⚠️ asset_id 必须留在最后一列
    # ⚠️ 长窗块**只进截面设计**。确认档（long_window_confirm）只测了截面块，
    # 市场块未测 ⟹ `market_design = [raw, *blocks]` 保持一列不动。
    design = np.ascontiguousarray(np.column_stack(blocks[:-1] + long_blocks + blocks[-1:]))
    assert design.shape[0] == len(tid), "历史块与采样矩阵行数不一致 —— 两条读取路径口径不同"
    min_data = max(20, int(round(MIN_DATA_FRAC * len(design))))

    def reuse_forest(prefix: str) -> list[str]:
        """从 --reuse-from 复用一片森林，并硬校验它与本次跑在同一个特征空间里。"""
        source = Path(args.reuse_from)
        source_meta = json.loads((source / "hybrid_meta.json").read_text(encoding="utf-8"))
        if list(source_meta["lgbm_features"]) != lgbm_features:
            raise SystemExit(f"{source} 的选列与本次不同 —— 两片森林会不在同一个特征空间里")
        for key in ("lower", "upper", "center", "scale"):
            if not np.array_equal(np.asarray(source_meta[key], dtype=np.float64),
                                  np.asarray(stats[key][selected], dtype=np.float64)):
                raise SystemExit(f"{source} 的预处理统计量 {key} 与本次不同")
        key = "market_model_files" if prefix.startswith("lgbm_market") else "lgbm_model_files"
        names = list(source_meta[key])
        for name in names:
            if (source / name).resolve() != (model_dir / name).resolve():
                shutil.copy2(source / name, model_dir / name)
        print(f"  复用 {key}（来自 {source.name}）：{names}", flush=True)
        return names

    def train_forest(matrix: np.ndarray, label: np.ndarray, weight, prefix: str,
                     spec: dict, min_data: int, num_iteration: int) -> list[str]:
        """一组超参 × n_seeds 个种子，存盘并返回文件名。asset_id 恒为最后一列。"""
        cat = matrix.shape[1] - 1
        names = []
        for s in range(args.n_seeds):
            params = {
                **spec, "objective": "regression", "metric": "l2", "verbosity": -1,
                "num_threads": args.num_threads, "min_data_in_leaf": min_data,
                "bagging_fraction": 0.7, "bagging_freq": 1,
                "deterministic": True, "force_row_wise": True, "feature_pre_filter": False,
                "seed": args.seed + s, "bagging_seed": args.seed + 1000 + s,
                "feature_fraction_seed": args.seed + 2000 + s,
            }
            t0 = time.perf_counter()
            ds = lgb.Dataset(matrix, label=label, weight=weight, params=params,
                             categorical_feature=[cat], free_raw_data=False)
            booster = lgb.train(params, ds, num_boost_round=num_iteration)
            name = f"{prefix}{args.seed + s}.txt"
            booster.save_model(str(model_dir / name), num_iteration=num_iteration)
            names.append(name)
            print(f"  {prefix} 种子 {args.seed + s}: {time.perf_counter()-t0:.0f}s → {name}", flush=True)
            del booster, ds
        return names

    xs_spec = {**SPEC, **(json.loads(args.xs_spec) if args.xs_spec else {})}
    market_spec = {**SPEC, **(json.loads(args.market_spec) if args.market_spec else {})}
    xs_min_data = max(20, int(round(min_data * args.xs_min_data_scale)))
    market_min_data = max(20, int(round(min_data * args.market_min_data_scale)))

    print(f"截面块设计矩阵 {design.shape[0]:,} × {design.shape[1]}，"
          f"min_data_in_leaf={xs_min_data:,}，{args.n_seeds} 种子 × {args.num_iteration} 轮"
          f"{'，带 sample_weight' if sample_weight is not None else '，无权'}；{xs_spec}", flush=True)
    model_files = (reuse_forest("lgbm_seed") if args.train_only == "market"
                   else train_forest(design, e, sample_weight, "lgbm_seed", xs_spec, xs_min_data,
                                      args.num_iteration))
    train_rows = len(design)          # ⚠️ 先取走再放 —— meta 在函数末尾才写
    del design
    gc.collect()

    # ---- 行级市场模型：同一批块前面再拼上 raw，标签换成 y，**不带权**
    market_files: list[str] = []
    if args.market_model:
        market_design = np.ascontiguousarray(np.column_stack([raw, *blocks]))
        del raw
        gc.collect()
        print(f"市场块设计矩阵 {market_design.shape[0]:,} × {market_design.shape[1]}"
              f"（= raw {len(selected)} 列 + 截面块 {market_design.shape[1] - len(selected)} 列），"
              f"标签 y，无权", flush=True)
        market_files = (reuse_forest("lgbm_market_seed") if args.train_only == "cross_section"
                        else train_forest(market_design, y, None, "lgbm_market_seed",
                                          market_spec, market_min_data,
                                          market_num_iteration))
        del market_design
        gc.collect()
    del dev, blocks, long_blocks

    meta = {
        "strategy": "v3_hybrid_ridge_plus_lgbm_cross_section",
        "blend_weight": BLEND_WEIGHT,
        "blend_note": "最终 ê = (1−w)·ê_ridge + w·ê_lgbm，w=0.5 是先验、不拟合（ROADMAP §5）",
        "num_iteration": args.num_iteration,
        "num_iteration_source": "lgbm_blend_unweighted 逐折 best_iteration 的折均（范围 44~302）",
        "market_num_iteration": int(market_num_iteration),
        "lgbm_model_files": model_files,
        # ---- 截面块是否带权（`combo_market_weight` 的 w_e 格，08-13）
        "cross_section_weighted": bool(sample_weight is not None),
        "cross_section_weighted_note":
            "带权 = 与指标 Σw(y−ŷ)² 对齐。本地 5 折 +4.50%、4/5 折、去最好折 +3.21%；"
            "⚠️ 机制是 ΔB −3.53% 而非 ΔA（+0.08%）—— 减方差不是加信号",
        # ---- 行级市场模型（`combo_market_weight` 的 mkt_we 格，08-13）
        "market_model_files": market_files,
        "market_lambda": float(args.market_lambda) if market_files else 0.0,
        "market_design": "raw ‖ xs_dev ‖ history ‖ asset_id",
        "market_note":
            "m̂ = (1−λ)·m̂_ridge + λ·逐 time_id 无权截面均值(行级 LGBM 打 y)。"
            "λ=0.5 是先验、不拟合（ROADMAP §5，本地实测优于 1.0）。"
            "⚠️ 市场模型**不带权** —— 带权反而从 +15.18% 掉到 +11.63%"
            "（combo_market_weight 的 mkt_wm 格）。"
            "组合 mkt_we 本地 5 折 +18.30%、5/5 折、去最好折 +14.29%、2ΔA>ΔB",
        "lgbm_features": lgbm_features,
        # history 块（`history_peak`，08-11）。下标是**在 lgbm_features 里的位置**（0..199），
        # 不是 323 列里的下标 —— 推理端因此直接复用下面那 200 列的统计量。
        # 设计矩阵列序固定为 [xs_dev ‖ previous ‖ difference ‖ rolling_mean ‖ rolling_deviation ‖ asset_id]，
        # asset_id 永远是最后一列（train/main/lgbm_numpy 三处都假设）。
        "history_positions": history_positions,
        "history_window": int(args.history_window),
        "history_note": ("本地 5 折配对 peak +10.10%、5/5 折、去掉最好一折 +9.10%、"
                         "2ΔA>ΔB（outputs/experiments/history_peak_lgbm_scoped.md）；"
                         "岭回归上同一改动是 −1.97%，只在截面块成立"),
        "lower": stats["lower"][selected].tolist(),
        "upper": stats["upper"][selected].tolist(),
        "center": stats["center"][selected].tolist(),
        "scale": stats["scale"][selected].tolist(),
        "prediction_scale": args.prediction_scale,
        "prediction_clip": args.prediction_clip,
        "scale_note": "scale 是纯后处理旋钮；此处为本地最优占位，最终值由公榜两点法定",
        "cross_sectional_mean": "unweighted",
        "cross_sectional_mean_note":
            "推理端拿不到 weight（test 无该列且 runner 的 forbidden 会剥掉），"
            "所以拆解、投影、训练目标一律无权。已在 lgbm_blend_unweighted 上重测，PASS 成立。",
        "lgbm_params": {**xs_spec, "min_data_in_leaf": xs_min_data,
                        "bagging_fraction": 0.7, "bagging_freq": 1},
        "market_lgbm_params": ({**market_spec, "min_data_in_leaf": market_min_data,
                                "bagging_fraction": 0.7, "bagging_freq": 1}
                               if market_files else None),
        "long_window": int(args.long_window) or None,
        "long_window_note": (
            "长窗滚动均值 + 偏离，复用 history 的那 40 列，**只进截面设计**（市场设计不变）。"
            "缺键或 null ⟹ 完全关闭 ⟹ 与旧模型逐位相同。"
            "证据：outputs/experiments/long_window_confirm.md —— 3s480 确认档 +7.77%、5/5 折、"
            "去最好折 +6.49%、配对 CI 下界 +4.18%，五道门槛全过；但只有 8.7% 检出下限的 0.89× "
            "⟹ PASS_BUT_BELOW_DETECTION_FLOOR，方向可信、幅度测不出。"
            "⚠️ 实现必须用 AssetLongWindow（持久累积和相减，离线/在线逐位相同），"
            "不得用分块重起的 cumsum。"),
        "train_only": args.train_only,
        "reuse_from": args.reuse_from,
        "train_rows": int(train_rows),
        "sample_modulo": args.sample_modulo,
        "sampling": args.sampling,
        "ridge_model_sha_note": "baseline_model.json 是 v1_ridge 生产模型的冻结拷贝，不重训",
    }
    (model_dir / "hybrid_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {model_dir/'hybrid_meta.json'}")
    print(f"模型文件：{model_files}")


if __name__ == "__main__":
    main()
