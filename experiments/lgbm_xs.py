"""LightGBM 能不能吃下截面块 e = y − m？（ROADMAP #4 原定的主攻方向）

背景：`Score = 0.72·R²_m + 0.28·R²_e`（恒等式残差 5.89e-16，见 mt_predictability）。
择时块（85% 的分）两轮 LightGBM 都是 INCONCLUSIVE，已结案（lgbm_mt / lgbm_mt_v2）。
**截面块是被浪费的那一块**：占 28% 方差却只出 15% 的分，`R²_截面`=0.000713
只有 `R²_择时` 的一半 —— 而资产间的非线性 / 交互正是树的天然强项。

注意换算系数是 **0.28**，不是择时块的 0.72。

## 与 lgbm_mt 的关系

**测量口径整套 import 自 `lgbm_mt.py`，不复制粘贴、不重新发明** ——
那把尺子刚被两轮验证过（内层早停 + 无泄漏选择 + 3 种子平均 + pooled Δ +
判据写死进代码）。这里只换数据与目标。

## 两个基准，都要过

截面块没有像 mt_lagged 那样的先验最优 alpha，所以不套用 lgbm_mt 里的
`ridge_fixed=1e8`（那是择时块的先验，换个块就没有依据了），改成：

| 基准 | alpha 怎么定 | 性质 |
|---|---|---|
| `ridge_fair` | 内层验证段上选 | 无泄漏，与 LGBM 的选择流程对等 |
| `ridge_oracle_alpha` | **在外层折均值上取最大** | **故意偏强**（对 ridge 有利），赢它才算真赢 |

用「故意偏强的基准」代替「拍一个魔数」—— 不需要预注册任何常数，且方向保守。

## 数据

`src/io.py` 的 `time_sample_mask` 按 **time_id** 掩码，所以每个选中的 time_id
保留**完整的 15 资产截面** —— 算截面均值必须如此，否则 m_t 是错的。
`sample_modulo=3, sampling="phase_balanced"` → 约 440 万行、10 个相位均匀覆盖。

这里自带一个流式 loader 而不是直接用 `load_time_sample`，只因为后者不返回
`asset_id`；`src/io.py` 被多个已结案实验依赖，不去改它的返回签名。

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/lgbm_xs.py
    # 先试水：--n-folds 2 --max-rounds 300 --report lgbm_xs_trial
输出：outputs/experiments/lgbm_xs.{json,md}（同名已存在需 --force）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, time_sample_mask, train_files
from src.validation import rolling_time_folds
from train import robust_transform_fit, select_features
from features import apply_robust_transform, cross_sectional_deviation
from mt_lagged import weighted_r2
# 测量口径整套复用 lgbm_mt 第二轮的成品
from lgbm_mt import (accumulate, combine, ridge_predictions,
                     score_parts, paired_stats, verdict_of, PUBLIC_REFERENCE)

SHARE_CROSS_SECTION = 0.28      # Score = 0.72·R²_m + 0.28·R²_e —— 这里是截面块

# ⚠️ **不要**沿用 lgbm_mt 的 ALPHAS（1e5~1e11）。那是给择时块设计的 ——
# 那里是 888,315 个**聚合后**的 time_id 行、323 列特征截面均值；这里是**行级**数据、
# 200 列截面去均值 + 15 列 one-hot，量级完全不同。
# 1/20 采样的烟测里，最优 alpha 落在 1e5，也就是那个网格的**最小端** ——
# 基准一旦被压在边界上就是正则不足，LGBM 会赢得莫名其妙。所以向低端扩两个数量级，
# 并在报告里显式检查最优 alpha 是否落在**网格内部**（落在端点 = 本次比较不可信）。
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

# 预注册候选。内层训练段约 350 万行（择时块只有 35 万），所以 min_data_in_leaf
# 必须同比例放大约 12 倍，否则叶子里全是噪声。不做网格搜索 ——
# 树深/轮数/学习率/num_leaves 全是「拟合紧密度」参数，本地会像 alpha 那样量反。
CANDIDATES: dict[str, dict[str, Any]] = {
    "xs_shrunk":   {"num_leaves": 15, "min_data_in_leaf": 100000, "learning_rate": 0.02,
                    "feature_fraction": 0.4, "lambda_l2": 30.0},
    "xs_moderate": {"num_leaves": 31, "min_data_in_leaf": 40000,  "learning_rate": 0.03,
                    "feature_fraction": 0.5, "lambda_l2": 10.0},
    "xs_loose":    {"num_leaves": 63, "min_data_in_leaf": 12000,  "learning_rate": 0.03,
                    "feature_fraction": 0.7, "lambda_l2": 1.0},
}

CHECKPOINTS = (10, 20, 40, 80, 120, 200, 300, 450, 600, 800, 1100, 1500)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Can LightGBM beat ridge on the cross-sectional block?")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--report", default="lgbm_xs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample-modulo", type=int, default=3)
    p.add_argument("--sampling", choices=["periodic", "phase_balanced"], default="phase_balanced")
    p.add_argument("--feature-count", type=int, default=200)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=None)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--inner-frac", type=float, default=0.10)
    p.add_argument("--max-rounds", type=int, default=1500)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--candidates", nargs="*", default=None)
    return p.parse_args()


# ---------------------------------------------------------------- 流式加载

def load_rows(data_root: Path, sample_modulo: int, sampling: str) -> dict[str, np.ndarray]:
    """与 src.io.load_time_sample 同构，额外返回 asset_id。

    逐 batch 读、逐 batch 按 time_id 掩码，**从不整体 materialize** ——
    v2_lgbm/data_utils.py 那个 33 GiB 的 OOM 正是因为它先读全 9 分区再 pd.concat。
    """
    columns = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
                                          ("features", "target", "weight", "time_id", "asset_id")}
    for path in train_files(data_root):
        kept, started = 0, time.perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            mask = time_sample_mask(frame["time_id"].to_numpy(copy=False),
                                    sample_modulo, sampling=sampling)
            if not mask.any():
                continue
            parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float64, copy=True))
            parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float64, copy=True))
            parts["time_id"].append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
            parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int64, copy=True))
            kept += int(mask.sum())
        print(f"  {path.name}: {kept:,} 行 ({time.perf_counter()-started:.0f}s)", flush=True)
    if not parts["features"]:
        raise SystemExit("采样为空")
    return {k: np.concatenate(v) for k, v in parts.items()}


class DiskFeatureSubset:
    """Lazy column subset over a full-schema memmap; avoids a 10GB eager copy."""

    def __init__(self, base: np.memmap, columns: np.ndarray):
        self.base = base
        self.columns = np.asarray(columns, dtype=np.int64)
        self.shape = (base.shape[0], len(self.columns))
        self.dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            if cols != slice(None):
                raise IndexError("DiskFeatureSubset only supports all logical columns")
        else:
            rows = key
        if isinstance(rows, slice):
            return np.asarray(self.base[rows][:, self.columns], dtype=np.float32)
        if np.isscalar(rows):
            return np.asarray(self.base[int(rows), self.columns], dtype=np.float32)
        row_index = np.asarray(rows)
        if row_index.dtype == bool:
            row_index = np.flatnonzero(row_index)
        row_index = row_index.astype(np.int64, copy=False).reshape(-1)
        # Avoid NumPy's paired advanced-index semantics and bound temporary size.
        out = np.empty((len(row_index), len(self.columns)), dtype=np.float32)
        chunk = 50_000
        for start in range(0, len(row_index), chunk):
            stop = min(start + chunk, len(row_index))
            out[start:stop] = self.base[row_index[start:stop]][:, self.columns]
        return out


def load_rows_disk_backed(data_root: Path, sample_modulo: int, sampling: str,
                          cache_dir: Path, *, force: bool = False,
                          feature_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Load sampled rows into disk-backed memmaps instead of Python list/concatenate.

    Full-resolution v3 training has ~13.2m rows × 323 float32 features.  The old
    list-of-batches loader temporarily held the Python parts, the concatenated array,
    pandas batch objects and fold copies at once, which can exceed a 30GB machine.
    This loader performs a cheap row-count pass, allocates fixed memmaps, then fills
    them in one streaming pass.  The returned arrays have the same keys/order as
    ``load_rows`` and are safe to slice normally.
    """
    import json

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    names = ("features", "target", "weight", "time_id", "asset_id")
    selected = (np.arange(len(FEATURE_COLUMNS), dtype=np.int64) if feature_indices is None
               else np.asarray(feature_indices, dtype=np.int64))
    if selected.ndim != 1 or len(selected) == 0 or np.any(selected < 0) or np.any(selected >= len(FEATURE_COLUMNS)):
        raise ValueError("feature_indices must be a non-empty in-range 1D array")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("feature_indices must be unique")
    feature_names = [FEATURE_COLUMNS[int(i)] for i in selected]
    meta = {"data_root": str(Path(data_root).resolve()), "sample_modulo": int(sample_modulo),
            "sampling": str(sampling), "feature_count": len(feature_names),
            "feature_indices": selected.tolist()}
    paths = train_files(data_root)
    if not paths:
        raise SystemExit("no train parquet files")

    reuse_full_schema = False
    if meta_path.exists() and not force:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        compatible = all(existing.get(k) == v for k, v in meta.items())
        existing_count = int(existing.get("feature_count", len(FEATURE_COLUMNS)))
        reuse_full_schema = (existing_count == len(FEATURE_COLUMNS)
                             and len(selected) < existing_count
                             and existing.get("data_root") == meta["data_root"]
                             and int(existing.get("sample_modulo", -1)) == int(sample_modulo)
                             and existing.get("sampling") == sampling)
        if compatible or reuse_full_schema:
            rows = int(existing["rows"])
        else:
            raise SystemExit(f"disk cache metadata mismatch: {meta_path}; pass --force to rebuild")
    else:
        rows = 0
        for path in paths:
            count = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000, columns=["time_id"]):
                tid = batch.column(0).to_numpy(zero_copy_only=False)
                count += int(time_sample_mask(tid, sample_modulo, sampling=sampling).sum())
            rows += count
            print(f"  count {path.name}: {count:,} rows", flush=True)
        if rows <= 0:
            raise SystemExit("disk-backed sample is empty")

    paths_map = {
        "features": cache_dir / "features.f32",
        "target": cache_dir / "target.f64",
        "weight": cache_dir / "weight.f64",
        "time_id": cache_dir / "time_id.i64",
        "asset_id": cache_dir / "asset_id.i64",
    }
    # Reuse a previously built full-schema cache for a fixed production subset.
    # The old cache predates feature_indices metadata, so feature_count is the
    # compatibility signal. The subset is gathered per fold/slice, never eagerly.
    if reuse_full_schema:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        existing_count = int(existing.get("feature_count", len(FEATURE_COLUMNS)))
        if existing_count == len(FEATURE_COLUMNS) and len(selected) < existing_count:
            rows = int(existing["rows"])
            full = np.memmap(paths_map["features"], dtype=np.float32, mode="r",
                             shape=(rows, existing_count))
            arrays = {"features": DiskFeatureSubset(full, selected),
                      "target": np.memmap(paths_map["target"], dtype=np.float64, mode="r", shape=(rows,)),
                      "weight": np.memmap(paths_map["weight"], dtype=np.float64, mode="r", shape=(rows,)),
                      "time_id": np.memmap(paths_map["time_id"], dtype=np.int64, mode="r", shape=(rows,)),
                      "asset_id": np.memmap(paths_map["asset_id"], dtype=np.int64, mode="r", shape=(rows,))}
            return arrays
    mode = "w+" if force or not meta_path.exists() else "r+"
    shapes = {"features": (rows, len(feature_names)), "target": (rows,), "weight": (rows,),
              "time_id": (rows,), "asset_id": (rows,)}
    dtypes = {"features": np.float32, "target": np.float64, "weight": np.float64,
              "time_id": np.int64, "asset_id": np.int64}
    arrays = {name: np.memmap(path, dtype=dtypes[name], mode=mode, shape=shapes[name])
              for name, path in paths_map.items()}
    if mode == "w+" or not meta_path.exists():
        cursor = 0
        columns = ["time_id", "asset_id", "weight", *feature_names, "target"]
        for path in paths:
            kept, started = 0, time.perf_counter()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
                frame = batch.to_pandas()
                mask = time_sample_mask(frame["time_id"].to_numpy(copy=False),
                                        sample_modulo, sampling=sampling)
                if not mask.any():
                    continue
                stop = cursor + int(mask.sum())
                arrays["features"][cursor:stop] = frame.loc[mask, feature_names].to_numpy(
                    dtype=np.float32, copy=True)
                arrays["target"][cursor:stop] = frame.loc[mask, "target"].to_numpy(
                    dtype=np.float64, copy=True)
                arrays["weight"][cursor:stop] = frame.loc[mask, "weight"].to_numpy(
                    dtype=np.float64, copy=True)
                arrays["time_id"][cursor:stop] = frame.loc[mask, "time_id"].to_numpy(
                    dtype=np.int64, copy=True)
                arrays["asset_id"][cursor:stop] = frame.loc[mask, "asset_id"].to_numpy(
                    dtype=np.int64, copy=True)
                cursor = stop; kept += int(mask.sum())
            print(f"  fill {path.name}: {kept:,} rows ({time.perf_counter()-started:.0f}s)", flush=True)
        if cursor != rows:
            raise RuntimeError(f"disk cache fill {cursor:,} != counted {rows:,}")
        for value in arrays.values():
            value.flush()
            # Do not keep the entire freshly-filled 17GB feature file resident in
            # the cgroup page cache before fold training starts. It will be faulted
            # back in only for the current fold slices.
            try:
                import mmap
                value._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        meta.update({"rows": rows, "files": {k: str(v) for k, v in paths_map.items()}})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        for value in arrays.values():
            try:
                import mmap
                value._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
    return arrays


def cross_sectional_target(target: np.ndarray, weight: np.ndarray,
                           time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """e_it = y_it − m_t，m_t = Σᵢwᵢyᵢ/Σᵢwᵢ。返回 (e, 每个 time_id 的行数)。"""
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)])
    total_w = np.add.reduceat(weight, starts)
    m = np.add.reduceat(weight * target, starts) / total_w
    return target - np.repeat(m, counts), counts


# ---------------------------------------------------------------------- 主流程

def main() -> None:
    import lightgbm as lgb

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.report}.json"
    md_path = output_dir / f"{args.report}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"报告已存在：{json_path} / {md_path}。要覆盖请显式加 --force")

    names = list(args.candidates) if args.candidates else list(CANDIDATES)
    if unknown := [n for n in names if n not in CANDIDATES]:
        raise SystemExit(f"未知候选：{unknown}，可选 {list(CANDIDATES)}")

    print(f"加载训练数据（modulo {args.sample_modulo} / {args.sampling}）…", flush=True)
    data = load_rows(Path(args.data_root), args.sample_modulo, args.sampling)
    features, y, w, tid, aid = (data["features"], data["target"], data["weight"],
                                data["time_id"], data["asset_id"])
    del data
    assert np.all(np.diff(tid) >= 0), "行未按 time_id 排序，截面聚合会算错"

    e, counts = cross_sectional_target(y, w, tid)
    uniq_tid = tid[np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]]
    phases = np.bincount(uniq_tid % 10, minlength=10) / len(uniq_tid)
    # 截面完整性：每个 time_id 的加权 e 之和必须为 0，否则 m_t 算错了
    starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]
    resid = float(np.abs(np.add.reduceat(w * e, starts)).max())
    print(f"{len(tid):,} 行 / {len(uniq_tid):,} 个 time_id；"
          f"每 time_id 行数 {counts.min()}~{counts.max()}", flush=True)
    print(f"相位覆盖 {np.round(phases, 3).tolist()}（应各约 0.1）", flush=True)
    print(f"截面完整性 max|Σw·e| = {resid:.2e} {'✅' if resid < 1e-8 else '❌'}", flush=True)
    assert resid < 1e-8, "逐 time_id 的加权 e 之和不为 0 —— 截面被截断了"

    train_window = args.train_window or int(len(uniq_tid) * 4 / 9)
    folds = rolling_time_folds(uniq_tid, args.n_folds, train_window, args.embargo)
    checkpoints = [k for k in CHECKPOINTS if k <= args.max_rounds]
    n_assets = int(aid.max()) + 1

    results: list[dict[str, Any]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        # 行按 time_id 有序，且 fold 的 time_id 是排序后 unique 数组的连续切片
        # → 对应行区间也是连续的，用 searchsorted 取两端即可。
        def rows_of(ids: np.ndarray) -> np.ndarray:
            return np.arange(int(np.searchsorted(tid, ids[0], "left")),
                             int(np.searchsorted(tid, ids[-1], "right")))

        train_rows = rows_of(train_ids)
        valid_rows = rows_of(valid_ids)

        # 内层切分按 time_id 做（不能按行切，否则会把一个截面劈开）
        n_tr_ids = len(train_ids)
        n_inner_ids = max(1, int(n_tr_ids * args.inner_frac))
        inner_valid_ids = train_ids[n_tr_ids - n_inner_ids:]
        inner_train_ids = train_ids[: n_tr_ids - n_inner_ids - args.embargo]
        it0, it1 = np.searchsorted(tid, inner_train_ids[0], "left"), \
            np.searchsorted(tid, inner_train_ids[-1], "right")
        iv0, iv1 = np.searchsorted(tid, inner_valid_ids[0], "left"), \
            np.searchsorted(tid, inner_valid_ids[-1], "right")
        inner_train_rows, inner_valid_rows = np.arange(it0, it1), np.arange(iv0, iv1)
        assert int(tid[inner_valid_rows[-1]]) < int(valid_ids[0]), "内层早停集越过了外层验证段"
        assert int(inner_valid_ids[0]) - int(inner_train_ids[-1]) > args.embargo, "内层 embargo 不足"

        fold: dict[str, Any] = {
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "inner_train_time_range": [int(inner_train_ids[0]), int(inner_train_ids[-1])],
            "inner_valid_time_range": [int(inner_valid_ids[0]), int(inner_valid_ids[-1])],
            "rows": {"train": len(train_rows), "inner_train": len(inner_train_rows),
                     "inner_valid": len(inner_valid_rows), "outer_valid": len(valid_rows)},
            "arms": {},
        }

        # ---- 预处理与选列：统计量与选列**只在内层训练段拟合**
        scratch = features[inner_train_rows].copy()          # robust_transform_fit 会就地改
        scratch, stats = robust_transform_fit(scratch)
        selected = select_features(scratch, e[inner_train_rows], w[inner_train_rows],
                                   args.feature_count)
        del scratch
        # 只把选中的 200 列展开到全体行，省掉 323 列的整份副本
        z = features[:, selected].copy()
        apply_robust_transform(z, stats["lower"][selected], stats["upper"][selected],
                               stats["center"][selected], stats["scale"][selected])
        # e 是纯截面量 —— raw 里的市场分量 x̄_t 在 time_id 内是常数、对 e 无信息，
        # 喂进去只会让树浪费分裂。所以设计矩阵用截面去均值后的特征。
        z = cross_sectional_deviation(z, tid)
        fold["selected_features"] = selected.tolist()

        y_valid, w_valid = e[valid_rows], w[valid_rows]
        y_iv, w_iv = e[inner_valid_rows], w[inner_valid_rows]

        # ---- 岭回归基准：asset_id 用 15 列 one-hot（与 LGBM 的原生 categorical 同信息）
        onehot = np.zeros((len(tid), n_assets), dtype=np.float32)
        onehot[np.arange(len(tid)), aid] = 1.0
        design = np.hstack([z, onehot])
        acc = accumulate(design[inner_train_rows], e[inner_train_rows], w[inner_train_rows])
        preds = ridge_predictions(acc, design[valid_rows], ALPHAS)
        inner_preds = ridge_predictions(acc, design[inner_valid_rows], ALPHAS)
        knob = max(inner_preds, key=lambda k: weighted_r2(y_iv, inner_preds[k], w_iv))
        fold["arms"]["ridge_fair"] = {
            "kind": "ridge", "train_rows": acc["n"],
            "path": {k: score_parts(y_valid, p, w_valid) for k, p in preds.items()},
            "honest": {"knob": knob, **score_parts(y_valid, preds[knob], w_valid)},
        }
        del acc, preds, inner_preds, design, onehot

        # ---- LightGBM：asset_id 作为原生 categorical 追加在最后一列
        x_all = np.hstack([z, aid.astype(np.float32)[:, None]])
        del z
        cat_index = x_all.shape[1] - 1
        x_it = np.ascontiguousarray(x_all[inner_train_rows])
        x_iv = np.ascontiguousarray(x_all[inner_valid_rows])
        x_ov = np.ascontiguousarray(x_all[valid_rows])
        del x_all

        for name in names:
            spec = CANDIDATES[name]
            patience = max(args.early_stopping, int(round(1.0 / spec["learning_rate"])))
            t0 = time.perf_counter()
            sum_path = {str(k): np.zeros(len(y_valid)) for k in checkpoints}
            sum_best = np.zeros(len(y_valid))
            best_iters, trained_iters = [], []
            for s in range(args.n_seeds):
                params = {
                    "objective": "regression", "metric": "l2", "verbosity": -1,
                    "num_threads": args.num_threads, "seed": args.seed + s,
                    "bagging_seed": args.seed + 1000 + s,
                    "feature_fraction_seed": args.seed + 2000 + s,
                    "bagging_fraction": 0.7, "bagging_freq": 1,
                    "deterministic": True, "force_row_wise": True,
                    # 各候选 min_data_in_leaf 不同 → 必须关掉特征预筛选并逐候选重建
                    # Dataset，否则后面的候选会被前一个的阈值筛过的特征集拖累
                    # （lgbm_mt 第一轮就是这么被污染的）
                    "feature_pre_filter": False,
                    **spec,
                }
                ds_tr = lgb.Dataset(x_it, label=e[inner_train_rows], weight=w[inner_train_rows],
                                    params=params, categorical_feature=[cat_index],
                                    free_raw_data=False)
                ds_va = lgb.Dataset(x_iv, label=y_iv, weight=w_iv, reference=ds_tr,
                                    params=params, categorical_feature=[cat_index],
                                    free_raw_data=False)
                booster = lgb.train(params, ds_tr, num_boost_round=args.max_rounds,
                                    valid_sets=[ds_va], valid_names=["inner"],
                                    callbacks=[lgb.early_stopping(patience, verbose=False)])
                trained = booster.current_iteration()
                best = int(booster.best_iteration or trained)
                best_iters.append(best)
                trained_iters.append(trained)
                for k in checkpoints:
                    sum_path[str(k)] += booster.predict(x_ov, num_iteration=min(k, trained))
                sum_best += booster.predict(x_ov, num_iteration=best)
                del booster, ds_tr, ds_va

            n = float(args.n_seeds)
            fold["arms"][name] = {
                "kind": "lgbm", "params": spec, "train_rows": len(inner_train_rows),
                "n_seeds": args.n_seeds, "patience": patience,
                "best_iterations": best_iters, "trained_iterations": trained_iters,
                "best_iteration": int(np.mean(best_iters)),
                "hit_round_budget": any(t >= args.max_rounds for t in trained_iters),
                "fit_seconds": float(time.perf_counter() - t0),
                "path": {str(k): {**score_parts(y_valid, sum_path[str(k)] / n, w_valid),
                                  "effective_iteration": min(k, int(np.mean(trained_iters)))}
                         for k in checkpoints},
                "honest": {"knob": "/".join(map(str, best_iters)),
                           **score_parts(y_valid, sum_best / n, w_valid)},
            }
            print(f"  fold {index:2d} {name:13s} best_iter={best_iters} "
                  f"R²_e@best={fold['arms'][name]['honest']['r2']:+.5f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
            del sum_path, sum_best

        del x_it, x_iv, x_ov
        fold["elapsed_seconds"] = float(time.perf_counter() - started)
        results.append(fold)
        print(f"fold {index:2d} 完成：ridge_fair {fold['arms']['ridge_fair']['honest']['r2']:+.5f} "
              f"({fold['elapsed_seconds']:.0f}s)", flush=True)

    # ---- `ridge_oracle_alpha`：ridge_fair 在**外层折均值**上取最大的那个 alpha。
    # 故意偏强（对 ridge 有利）—— 不需要预注册魔数，且方向保守：赢它才算真赢。
    alpha_keys = sorted(results[0]["arms"]["ridge_fair"]["path"], key=lambda s: float(s))
    curve_rf = {k: float(np.mean([f["arms"]["ridge_fair"]["path"][k]["r2"] for f in results]))
                for k in alpha_keys}
    oracle_alpha = max(curve_rf, key=lambda k: curve_rf[k])
    # 最优 alpha 落在网格端点 = 基准正则不足/过度，被人为压弱 → 整个比较不可信。
    alpha_at_edge = oracle_alpha in (alpha_keys[0], alpha_keys[-1])
    if alpha_at_edge:
        print(f"\n⚠️⚠️ 最优 alpha {oracle_alpha} 落在网格端点 "
              f"[{alpha_keys[0]}, {alpha_keys[-1]}] —— 岭回归基准被人为压弱，"
              f"本次与 LGBM 的比较**不可信**，请先扩 ALPHAS 重跑。\n", flush=True)
    for f in results:
        f["arms"]["ridge_oracle_alpha"] = {
            "kind": "ridge", "train_rows": f["arms"]["ridge_fair"]["train_rows"],
            "path": f["arms"]["ridge_fair"]["path"],
            "honest": {"knob": f"{oracle_alpha}（外层折均值最优，故意偏强）",
                       **f["arms"]["ridge_fair"]["path"][oracle_alpha]},
        }

    # ------------------------------------------------------------- 汇总
    arm_names = ["ridge_fair", "ridge_oracle_alpha", *names]
    summary: dict[str, Any] = {}
    per_fold: dict[str, list[float]] = {}
    for arm in arm_names:
        keys = sorted(results[0]["arms"][arm]["path"], key=lambda s: float(s))
        curve = {k: float(np.mean([f["arms"][arm]["path"][k]["r2"] for f in results])) for k in keys}
        pick = max(curve, key=lambda k: curve[k])
        honest = [f["arms"][arm]["honest"] for f in results]
        series = [h["r2"] for h in honest]
        per_fold[arm] = series
        summary[arm] = {
            "curve": curve, "curve_selected": pick,
            "curve_mean_r2": float(np.mean([f["arms"][arm]["path"][pick]["r2"] for f in results])),
            "honest_knobs": [h["knob"] for h in honest],
            "mean_r2": float(np.mean(series)),
            "pooled_sse": float(sum(h["sse"] for h in honest)),
            "pooled_r2": float(1.0 - sum(h["sse"] for h in honest) / sum(h["sst"] for h in honest)),
            "positive_folds": int(sum(s > 0 for s in series)),
            "A": float(np.mean([h["A"] for h in honest])),
            "B": float(np.mean([h["B"] for h in honest])),
            "oracle_mean_r2": float(np.mean(
                [max(v["r2"] for v in f["arms"][arm]["path"].values()) for f in results])),
        }

    baselines = ["ridge_fair", "ridge_oracle_alpha"]
    total_sst = float(sum(f["arms"]["ridge_fair"]["honest"]["sst"] for f in results))
    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in baselines:
        block: dict[str, Any] = {}
        for arm in arm_names:
            if arm == baseline:
                continue
            deltas = [a - b for a, b in zip(per_fold[arm], per_fold[baseline])]
            st = paired_stats(deltas)
            da = (summary[arm]["A"] - summary[baseline]["A"]) / abs(summary[baseline]["A"])
            db = (summary[arm]["B"] - summary[baseline]["B"]) / abs(summary[baseline]["B"])
            pooled = (summary[baseline]["pooled_sse"] - summary[arm]["pooled_sse"]) / total_sst
            block[arm] = {
                **st,
                "pooled_delta": float(pooled),
                "pooled_score_delta": float(pooled * SHARE_CROSS_SECTION),
                "score_delta": float(st["mean"] * SHARE_CROSS_SECTION),
                "delta_A_pct": da * 100.0, "delta_B_pct": db * 100.0,
                "discounted_mechanism_ok": bool(2.0 * (da / 2.2) > db),
                "detection_floor_r2": None if st["se"] is None else float(2.0 * st["se"]),
                "detection_floor_score": None if st["se"] is None else float(
                    2.0 * st["se"] * SHARE_CROSS_SECTION),
                "detection_floor_pct_of_public": None if st["se"] is None else float(
                    2.0 * st["se"] * SHARE_CROSS_SECTION / PUBLIC_REFERENCE * 100.0),
                "verdict": verdict_of(st, da, db),
            }
        comparisons[baseline] = block

    overall: dict[str, str] = {}
    for arm in names:
        vs = [comparisons[b][arm]["verdict"] for b in baselines]
        overall[arm] = ("INSUFFICIENT_FOLDS" if "INSUFFICIENT_FOLDS" in vs
                        else "PASS" if all(v == "PASS" for v in vs)
                        else "FAIL" if any(v == "FAIL" for v in vs) else "INCONCLUSIVE")

    payload = {
        "question": "LightGBM 能不能吃下截面块 e = y − m？",
        "metric": "样本外 R²_e = 1 − Σw(e−ê)²/Σw·e²",
        "score_mapping": f"Δscore ≈ {SHARE_CROSS_SECTION} × ΔR²_e（截面块占 target 方差 28%）",
        "decision_rule": (
            "对每个基准：PASS 需 t ≥ 2 且 去掉最好一折后 t ≥ 1 且 2·ΔA/2.2 > ΔB。"
            "总判定要求对 ridge_fair（无泄漏）与 ridge_oracle_alpha（外层折均值最优、故意偏强）"
            "两个基准都 PASS。由 verdict_of() 计算写入本文件，不靠人在 md 里下判断。"
        ),
        "baselines": baselines,
        "oracle_alpha": oracle_alpha,
        "oracle_alpha_at_grid_edge": bool(alpha_at_edge),
        "alpha_grid_valid": not alpha_at_edge,
        "overall_verdict": overall if not alpha_at_edge else
            {a: "UNTRUSTWORTHY_ALPHA_GRID" for a in overall},
        "overall_verdict_raw": overall,
        "leakage_controls": {
            "inner_split": f"训练段尾部 {args.inner_frac:.0%} 个 time_id + embargo {args.embargo}",
            "inner_split_is_by_time_id": "按 time_id 切，不按行 —— 否则会把一个截面劈开",
            "outer_valid_never_seen": True,
            "preprocessing_and_selection_fit_on_inner_train_only": True,
            "seed_averaging": f"{args.n_seeds} 个种子各自内层早停后平均外层预测",
            "note_oracle_alpha_is_deliberately_optimistic": True,
        },
        "configuration": {
            "sample_modulo": args.sample_modulo, "sampling": args.sampling,
            "rows": int(len(tid)), "time_ids": int(len(uniq_tid)),
            "phase_coverage": np.round(phases, 4).tolist(),
            "feature_count": args.feature_count, "n_assets": n_assets,
            # 截面不总是满 15 个资产（实测 p008 有 6.3% 的 time_id 不足 15，最少 2 个）。
            # **不做过滤** —— 生产的 cross_sectional_deviation 也不过滤，口径必须一致。
            # 但要知道：2 资产的截面上 e₁ 与 e₂ 完全共线，只剩 1 个自由度，本质不可预测。
            "cross_section_size_counts": {int(k): int(v) for k, v in
                                          zip(*np.unique(counts, return_counts=True))},
            "n_folds": args.n_folds, "train_window": train_window, "embargo": args.embargo,
            "inner_frac": args.inner_frac, "alphas": ALPHAS,
            "max_rounds": args.max_rounds, "n_seeds": args.n_seeds, "seed": args.seed,
            "public_reference_score": PUBLIC_REFERENCE,
            "candidates": {n: CANDIDATES[n] for n in names},
        },
        "summary": summary, "comparisons": comparisons, "folds": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- markdown
    def fmt(v, spec=".2f"):
        return "n/a" if v is None else format(v, spec)

    lines = [
        "# LightGBM vs 岭回归：截面块 e = y − m",
        "",
        f"{len(tid):,} 行 / {len(uniq_tid):,} 个 time_id"
        f"（modulo {args.sample_modulo} / {args.sampling}，10 个相位均匀覆盖），"
        f"{args.n_folds} 折滚动 + embargo {args.embargo}。",
        f"`R²_e = 1 − Σw(e−ê)²/Σw·e²`；换算总分 **`Δscore ≈ {SHARE_CROSS_SECTION} × ΔR²_e`**"
        "（注意不是择时块的 0.72）。",
        "",
        "**设计**：`cross_sectional_deviation` 作用在内层训练段选出的 "
        f"{args.feature_count} 列上 —— `e` 是纯截面量，raw 里的市场分量在 time_id 内是常数、"
        "对它无信息。`asset_id` 给 LGBM 用原生 categorical、给 ridge 用 one-hot"
        "（同信息，各用各的自然编码）。",
        "",
        "**两个基准都要过**：`ridge_fair`（内层选 alpha，无泄漏）与 "
        f"`ridge_oracle_alpha`（alpha={oracle_alpha}，在**外层折均值**上取最大，"
        "**故意偏强**）。赢一个不算数。",
        "",
    ] + ([f"> ⚠️ **最优 alpha `{oracle_alpha}` 落在网格端点 "
          f"[`{alpha_keys[0]}`, `{alpha_keys[-1]}`]** —— 岭回归基准被人为压弱，"
          f"本报告的比较结论**不可信**，需先扩 `ALPHAS` 重跑。", ""] if alpha_at_edge else []) + [
        "## 总判定",
        "",
        "| 候选 | vs `ridge_fair` | vs `ridge_oracle_alpha` | **总判定** |",
        "|---|:--:|:--:|:--:|",
    ] + [
        f"| `{a}` | {comparisons['ridge_fair'][a]['verdict']} | "
        f"{comparisons['ridge_oracle_alpha'][a]['verdict']} | **{overall[a]}** |" for a in names
    ] + ["", "## 各臂表现", "",
         "| 臂 | 无泄漏旋钮 | 平均 R²_e | pooled | 为正折数 | curve 峰值 | 逐折 oracle |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for arm in arm_names:
        s = summary[arm]
        uniq = sorted(set(s["honest_knobs"]))
        knob = "/".join(uniq) if len(uniq) <= 2 else f"逐折不同（{len(uniq)} 种）"
        lines.append(f"| `{arm}` | {knob} | {s['mean_r2']:+.5f} | {s['pooled_r2']:+.5f} | "
                     f"{s['positive_folds']}/{len(results)} | {s['curve_mean_r2']:+.5f} | "
                     f"{s['oracle_mean_r2']:+.5f} |")

    for baseline in baselines:
        lines += ["", f"## 相对 `{baseline}` 的配对 Δ —— **判据是 t**", "",
                  "| 臂 | 判定 | mean(Δ) | SE | **t** | 中位数 | 正折 | 去掉最好一折 t | pooled Δ | ΔA | ΔB | 机制 |",
                  "|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|"]
        for arm, c in comparisons[baseline].items():
            badge = {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
                     "INSUFFICIENT_FOLDS": "🚫 折数不足"}.get(c["verdict"], "⚠️ 未定")
            lines.append(
                f"| `{arm}` | {badge} | {c['mean']:+.5f} | {fmt(c['se'], '.5f')} | "
                f"**{fmt(c['t'], '+.2f')}** | {c['median']:+.5f} | "
                f"{c['positive_folds']}/{c['n_folds']} | {fmt(c['drop_best_t'], '+.2f')} | "
                f"{c['pooled_delta']:+.5f} | {c['delta_A_pct']:+.1f}% | {c['delta_B_pct']:+.1f}% | "
                f"{'✅' if c['discounted_mechanism_ok'] else '❌'} |")

    flat = [c for b in comparisons.values() for c in b.values()]
    fl = [c["detection_floor_score"] for c in flat if c["detection_floor_score"] is not None]
    pc = [c["detection_floor_pct_of_public"] for c in flat
          if c["detection_floor_pct_of_public"] is not None]
    lines += ["",
              (f"**检出下限**：t=2 需要 Δscore ≥ {min(fl):.5f}，即相对公榜 {PUBLIC_REFERENCE} "
               f"至少 **+{min(pc):.0f}%**。INCONCLUSIVE 是「测不出来」，不是「没效果」。"
               f"截面块的换算系数只有 {SHARE_CROSS_SECTION}，所以同样的 ΔR² 折到总分上"
               "比择时块小 2.6 倍 —— 这一块天生更难测出来。")
              if fl else "**折数不足以定义标准误，本报告不能用于下结论。**",
              "", "## 轮数 / alpha 曲线（全折均值 R²_e）", ""]
    for arm in arm_names:
        lines.append(f"- `{arm}`：" + "  ".join(f"{k}={v:+.5f}"
                                                for k, v in summary[arm]["curve"].items()))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 各臂（无泄漏口径）===")
    for arm in arm_names:
        s = summary[arm]
        print(f"  {arm:20s} 平均 R²_e={s['mean_r2']:+.5f}  pooled={s['pooled_r2']:+.5f}  "
              f"(curve {s['curve_mean_r2']:+.5f} @ {s['curve_selected']})")
    for baseline in baselines:
        print(f"\n=== 相对 {baseline} 的配对 Δ ===")
        for arm, c in comparisons[baseline].items():
            print(f"  {arm:20s} {c['verdict']:<18s} mean={c['mean']:+.5f}  "
                  f"t={fmt(c['t'], '+.2f')}  正折 {c['positive_folds']}/{c['n_folds']}  "
                  f"去最好一折 t={fmt(c['drop_best_t'], '+.2f')}  "
                  f"机制{'✅' if c['discounted_mechanism_ok'] else '❌'}")
    if fl:
        print(f"\n检出下限：Δscore ≥ {min(fl):.5f}（公榜的 +{min(pc):.0f}%）")
    print("\n=== 总判定（两个基准都要过）===")
    for arm, v in overall.items():
        print(f"  {arm:15s} {v}")
    passed = [a for a, v in overall.items() if v == "PASS"]
    print(f"结论：{'PASS —— ' + '/'.join(passed) if passed else '无一组过关'}")
    print(f"\n产物：{json_path}\n     {md_path}")


if __name__ == "__main__":
    main()
