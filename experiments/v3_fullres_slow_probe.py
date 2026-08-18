"""P1 口径核对：慢分量在**全分辨率**下还成立吗？

`v3_slow_variance` 的 trailing mean 走**采样格**（`phase_balanced` + modulo 5，
每 ~5 个真实 time_id 才有一个点），而生产推理每个 time_id 都出一次预测。
`K=400 采样步 ≈ 2000 真实步` —— 两者估计的是同一个慢分量，但不是同一个估计量。
本项目被这类口径错配烧过（伤疤规则 §3；`v3_temporal_smoothing` 里 lag 的同型陷阱），
所以必须实测，不能靠「方向有利」推断。

## 受控比较

**系数不在探针窗里拟合。** 直接取 `v3_slow_variance` 里 fold 4 那一行的系数
（它们是在 fold 0..3 的**采样格**上解出来的），冻结后拿到探针窗上评分。
这正是部署时的处境：离线在采样 OOF 上定系数，线上在全分辨率上算 slow/fast。

    A 组：全分辨率  —— 窗内每个真实 time_id 都有预测，K 按**真实步**取
    B 组：采样格    —— 同一窗内只保留采样行，K 按**采样步**取（= v3_slow_variance 的口径）

唯一变量就是 trailing mean 的分辨率。两组一致 ⟹ 采样格口径没骗人。

⚠️ 第一版设计把探针窗对半切、在前半重拟系数 —— 2,000 个采样 time_id 上拟两个高度共线的
系数，得到的比值在 −0.457 ~ +1.655 之间乱跳，相对增益 ±100%。那不是「口径不一致」，
是探针本身没有功效。冻结系数就是为了把拟合噪声整个拿掉；剩下的评估噪声由 block bootstrap 量化。

训练严格沿用生产口径：在**采样**训练窗上训，复用 `v3_production_oof` 的
`fit_predict_lgbm` / `fit_ridge` / `select_features` / `robust_transform_fit` 与同一套
`XS_SPEC` / `MARKET_SPEC` / 轮数 / 种子数。history 状态在**每一行**上推进，
内部仍调用生产的 `strategies/v3_hybrid/history.py:AssetHistory`
（`strategies/v3_hybrid/train.py` 不修改）。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/v3_fullres_slow_probe.py --fold 4
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
# ⚠️ 顺序与 v3_production_oof.py 逐字一致，不能改：v1_ridge 与 v3_hybrid 都有 `train.py`、
# `features.py`，但只有 v1_ridge 的 `train` 提供 robust_transform_fit / select_features。
# v3_hybrid 只能 **append** 到末尾（为了 `history`），抢到前面就会解析成另一个 train。
# —— CLAUDE.md 长期伤疤规则 §4。
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
_V3_PATH = str(_REPO_ROOT / "strategies" / "v3_hybrid")
if _V3_PATH not in sys.path:
    sys.path.append(_V3_PATH)

from features import apply_robust_transform, cross_sectional_deviation  # noqa: E402
from history_peak import fit_ridge, ridge_designs  # noqa: E402
from lgbm_xs import load_rows  # noqa: E402
from src.io import FEATURE_COLUMNS, time_sample_mask, train_files  # noqa: E402
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402
from v3_production_oof import (FEATURE_COUNT, HISTORY_COUNT, HISTORY_WINDOW,  # noqa: E402
                               MARKET_LAMBDA, MARKET_MIN_DATA_SCALE, MARKET_SPEC,
                               REFERENCE_TRAIN_WINDOW, RIDGE_ALPHA, XS_SPEC,
                               fit_predict_lgbm, group_mean)

SAMPLE_MODULO = 5
SAMPLING = "phase_balanced"
# 采样步 K 与真实步 K 的对应关系：采样格上平均每 SAMPLE_MODULO 个真实 time_id 取一个点
K_SAMPLED = [100, 200, 400, 800]
K_REAL = [k * SAMPLE_MODULO for k in K_SAMPLED]
SELECTED_K_SAMPLED = 400          # v3_slow_variance 在主 cache 与复现 cache 上都选中它


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default=None)
    p.add_argument("--fold", type=int, default=4)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--train-window", type=int, default=REFERENCE_TRAIN_WINDOW)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--probe-time-ids", type=int, default=20_000,
                   help="验证段末尾取多少个**连续真实** time_id")
    p.add_argument("--slow-variance-json",
                   default=str(_REPO_ROOT / "outputs" / "experiments" / "v3_slow_variance_3s480.json"),
                   help="冻结系数的来源：那里 fold k 的系数只用 fold 0..k−1 拟合")
    p.add_argument("--cache", default=None,
                   help="探针窗预测的 npz 缓存；存在就跳过训练，便于改评估不重训")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--num-iteration", type=int, default=480)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def stream_history_two_masks(data_root: Path, history_names: list[str],
                             history_stats: tuple[np.ndarray, ...], window: int,
                             probe_low: int, probe_high: int):
    """一次扫过**每一行**推进历史状态，同时留下两套行：采样行 + 探针窗全分辨率行。

    这是 `strategies/v3_hybrid/train.py:stream_history_blocks` 的本地变体，
    **只改「保留哪些行」**；状态推进仍然调用生产的同一个 `AssetHistory`
    （伤疤规则 §3：特征实现必须唯一）。生产训练路径本身不修改。
    """
    import pyarrow.parquet as pq
    from history import AssetHistory

    lower, upper, center, scale = history_stats
    history = AssetHistory(feature_count=len(history_names), window_size=window)
    sampled_parts: list[list[np.ndarray]] = [[], [], [], []]
    probe_parts: list[list[np.ndarray]] = [[], [], [], []]
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=120_000, columns=["time_id", "asset_id", *history_names]):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            aid = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            current = frame.loc[:, history_names].to_numpy(dtype=np.float32, copy=True)
            apply_robust_transform(current, lower, upper, center, scale)
            blocks = history.transform(current, aid)          # 每一行都推进状态
            sampled = time_sample_mask(tid, SAMPLE_MODULO, sampling=SAMPLING)
            probe = (tid >= probe_low) & (tid <= probe_high)
            for slot, block in zip(sampled_parts, blocks):
                slot.append(block[sampled])
            for slot, block in zip(probe_parts, blocks):
                slot.append(block[probe])
    return ([np.concatenate(slot) for slot in sampled_parts],
            [np.concatenate(slot) for slot in probe_parts])


def load_probe_rows(data_root: Path, low: int, high: int) -> dict[str, np.ndarray]:
    """探针窗内的**全部**行（全分辨率），含 323 个特征列。"""
    import pyarrow.parquet as pq

    columns = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
                                          ("features", "target", "weight", "time_id", "asset_id")}
    for path in train_files(data_root):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=columns):
            frame = batch.to_pandas()
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            mask = (tid >= low) & (tid <= high)
            if not mask.any():
                continue
            parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float64, copy=True))
            parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float64, copy=True))
            parts["time_id"].append(tid[mask].copy())
            parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int64, copy=True))
    return {k: np.concatenate(v) for k, v in parts.items()}


# ------------------------------------------------------- slow/fast 与闭式解

def causal_trailing_mean(values, time_id, asset_id, window: int) -> np.ndarray:
    """逐 asset、只用当期之前的观测求均值；窗口按**真实 time_id 步长**定义。

    与 `v3_slow_variance` 的差别只有一处：那里的窗口按「多少个观测」，
    这里按「多少个真实 time_id」—— 因为全分辨率下两者才是同一件事。
    """
    out = np.empty_like(values)
    order = np.lexsort((time_id, asset_id))
    v, t, a = values[order], time_id[order], asset_id[order]
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    ends = np.r_[starts[1:], len(a)]
    cumulative = np.concatenate([[0.0], np.cumsum(v)])
    result = np.empty_like(v)
    for start, end in zip(starts, ends):
        index = np.arange(start, end)
        # 段内 t 单调 ⟹ 用 searchsorted 找「t - window」的位置，得到窗口左端
        left = start + np.searchsorted(t[start:end], t[index] - window, side="left")
        count = np.maximum(index - left, 1)
        result[index] = (cumulative[index] - cumulative[left]) / count
        result[start] = v[start]
    out[order] = result
    return out


def group_moment_rows(y, w, slow, fast, time_id) -> np.ndarray:
    """逐 time_id 的 6 个矩：D、v(2)、G 上三角(3)。bootstrap 只对这些行重采样。"""
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    index = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(time_id)]))
    n = len(starts)
    columns = [w * y * y, w * y * slow, w * y * fast,
               w * slow * slow, w * slow * fast, w * fast * fast]
    return np.column_stack([np.bincount(index, weights=c, minlength=n) for c in columns])


def score_frozen(totals: np.ndarray, coefficients: np.ndarray, scale: float) -> dict[str, float]:
    """冻结系数下的 Score：候选 = M2 两系数，基线 = 同样冻结的全局单 scale。"""
    D = float(totals[0])
    v = totals[1:3].astype(np.float64)
    G = np.array([[totals[3], totals[4]], [totals[4], totals[5]]], dtype=np.float64)
    u = np.ones(2)
    candidate = float((2.0 * coefficients @ v - coefficients @ G @ coefficients) / D)
    baseline = float((2.0 * scale * (u @ v) - scale * scale * (u @ G @ u)) / D)
    return {"baseline": baseline, "candidate": candidate, "delta": candidate - baseline,
            "relative": (candidate - baseline) / abs(baseline) if baseline != 0 else float("nan")}


def bootstrap_relative(rows: np.ndarray, coefficients: np.ndarray, scale: float,
                       block_size: int, n_boot: int, seed: int) -> dict[str, float]:
    """对**绝对 delta** 做 bootstrap，再按**固定的** pooled 基线归一。

    ⚠️ 不能在每个 replicate 里各自除以自己的基线：探针窗的基线 Score 很小，
    重采样时会掠过 0，比值随之炸到几百个百分点，CI 完全没有意义（第一版就是这样）。
    """
    n_groups = len(rows)
    n_blocks = int(np.ceil(n_groups / block_size))
    prefix = np.vstack([np.zeros(rows.shape[1]), np.cumsum(rows, axis=0)])
    pooled_baseline = abs(score_frozen(rows.sum(axis=0), coefficients, scale)["baseline"])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n_groups - block_size, 0) + 1, size=n_blocks)
        stops = np.minimum(starts + block_size, n_groups)
        totals = (prefix[stops] - prefix[starts]).sum(axis=0)
        samples.append(score_frozen(totals, coefficients, scale)["delta"])
    array = np.asarray(samples, dtype=float)
    out = {f"delta_{k}": float(np.nanpercentile(array, q))
           for k, q in (("p2.5", 2.5), ("p50", 50.0), ("p97.5", 97.5))}
    out.update({k.replace("delta_", ""): v / pooled_baseline if pooled_baseline > 0 else float("nan")
                for k, v in out.items()})
    return out


def frozen_coefficients(slow_variance_json: Path, fold: int) -> dict[str, dict[str, Any]]:
    """从 v3_slow_variance 取出「在 fold 0..fold-1 上解出、用于评估 fold」的那组系数。"""
    payload = json.loads(slow_variance_json.read_text(encoding="utf-8"))
    by_k = payload["arms"]["production"]["by_K"]
    out = {}
    for key, entry in by_k.items():
        rows = [r for r in entry["out_of_sample"]["M2_slow_fast"]["folds"] if r["fold"] == fold]
        if rows:
            out[key] = {"coefficients": np.asarray(rows[0]["coefficients"], dtype=np.float64),
                        "baseline_scale": float(rows[0]["baseline_scale"])}
    return out


def main() -> None:
    args = parse_args()
    label = args.label or f"v3_fullres_slow_probe_fold{args.fold}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    md_path = output_dir / f"{label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    data_root = Path(args.data_root)
    cache_path = Path(args.cache) if args.cache else (
        _REPO_ROOT / "outputs" / "cache" / f"{label}_predictions.npz")

    if cache_path.exists():
        print(f"reusing cached probe predictions: {cache_path}", flush=True)
        with np.load(cache_path, allow_pickle=False) as c:
            y_pb, w_pb = c["target"], c["weight"]
            tid_pb, aid_pb = c["time_id"], c["asset_id"]
            raw = c["prediction_raw"]
            probe_low, probe_high = int(c["probe_window"][0]), int(c["probe_window"][1])
    else:
        y_pb, w_pb, tid_pb, aid_pb, raw, probe_low, probe_high = train_and_predict(args, data_root)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, target=y_pb, weight=w_pb, time_id=tid_pb,
                            asset_id=aid_pb, prediction_raw=raw,
                            probe_window=np.array([probe_low, probe_high], dtype=np.int64))
        print(f"wrote probe prediction cache: {cache_path}", flush=True)

    evaluate_and_report(args, label, json_path, md_path, y_pb, w_pb, tid_pb, aid_pb, raw,
                        probe_low, probe_high, started)


def train_and_predict(args, data_root: Path):
    started = time.perf_counter()
    print(f"loading sampled data: modulo {SAMPLE_MODULO}/{SAMPLING}", flush=True)
    data = load_rows(data_root, SAMPLE_MODULO, SAMPLING)
    time_ids = data["time_id"]
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(unique_time_ids, args.n_folds, args.train_window, args.embargo)
    train_ids, valid_ids = folds[args.fold]
    probe_high = int(valid_ids[-1])
    probe_low = probe_high - args.probe_time_ids + 1
    print(f"fold {args.fold}: train {len(train_ids):,} sampled tids, "
          f"probe window [{probe_low}, {probe_high}] = {args.probe_time_ids:,} real time_ids",
          flush=True)

    train_mask = np.isin(time_ids, train_ids)
    train_features = data["features"][train_mask].copy()
    y_tr = data["target"][train_mask].astype(np.float64)
    w_tr = np.maximum(data["weight"][train_mask].astype(np.float64), 0.0)
    tid_tr, aid_tr = time_ids[train_mask], data["asset_id"][train_mask]
    tr_starts = np.r_[0, np.flatnonzero(tid_tr[1:] != tid_tr[:-1]) + 1]
    tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)

    print("loading full-resolution probe window", flush=True)
    probe = load_probe_rows(data_root, probe_low, probe_high)
    print(f"  probe rows: {len(probe['target']):,}", flush=True)

    transformed_train, stats = robust_transform_fit(train_features)
    transformed_probe = probe["features"]
    apply_robust_transform(transformed_probe, stats["lower"], stats["upper"],
                           stats["center"], stats["scale"])

    ridge_selected = select_features(transformed_train, y_tr, w_tr, FEATURE_COUNT)
    fold_alpha = RIDGE_ALPHA * len(train_ids) / REFERENCE_TRAIN_WINDOW
    ridge = fit_ridge(ridge_designs(transformed_train, tid_tr, ridge_selected, None),
                      y_tr, w_tr, fold_alpha)
    ridge_raw = ridge.predict(
        ridge_designs(transformed_probe, probe["time_id"], ridge_selected, None)).astype(np.float64)
    pb_starts = np.r_[0, np.flatnonzero(probe["time_id"][1:] != probe["time_id"][:-1]) + 1]
    pb_counts = np.diff(np.r_[pb_starts, len(probe["time_id"])]).astype(np.float64)
    market_ridge = group_mean(ridge_raw, pb_starts, pb_counts)
    del ridge, ridge_raw
    gc.collect()

    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    xs_selected = select_features(transformed_train, e_tr, np.ones_like(e_tr), FEATURE_COUNT)
    xs_tr = cross_sectional_deviation(transformed_train[:, xs_selected].copy(), tid_tr)
    xs_pb = cross_sectional_deviation(transformed_probe[:, xs_selected].copy(), probe["time_id"])
    history_positions = np.sort(select_features(xs_tr, e_tr, np.ones_like(e_tr),
                                                HISTORY_COUNT).astype(np.int64))
    history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
    history_stats = tuple(stats[key][xs_selected[history_positions]]
                          for key in ("lower", "upper", "center", "scale"))

    print("streaming causal history (one pass, two row sets)", flush=True)
    sampled_blocks, probe_blocks = stream_history_two_masks(
        data_root, history_names, history_stats, HISTORY_WINDOW, probe_low, probe_high)
    history_tr = [block[train_mask] for block in sampled_blocks]
    del sampled_blocks
    gc.collect()

    lgbm_args = SimpleNamespace(n_seeds=args.n_seeds, seed=args.seed,
                                num_threads=args.num_threads, num_iteration=args.num_iteration)
    d_tr_xs = np.ascontiguousarray(np.column_stack([xs_tr, *history_tr, aid_tr.astype(np.float32)]))
    d_pb_xs = np.ascontiguousarray(np.column_stack(
        [xs_pb, *probe_blocks, probe["asset_id"].astype(np.float32)]))
    e_pred = fit_predict_lgbm(d_tr_xs, e_tr, w_tr, d_pb_xs, lgbm_args, "cross", XS_SPEC,
                              num_iteration=args.num_iteration)
    e_hat = e_pred - group_mean(e_pred, pb_starts, pb_counts)
    del d_tr_xs, d_pb_xs, e_pred
    gc.collect()

    d_tr_market = np.ascontiguousarray(np.column_stack(
        [transformed_train[:, xs_selected], xs_tr, *history_tr, aid_tr.astype(np.float32)]))
    d_pb_market = np.ascontiguousarray(np.column_stack(
        [transformed_probe[:, xs_selected], xs_pb, *probe_blocks,
         probe["asset_id"].astype(np.float32)]))
    market_pred = fit_predict_lgbm(d_tr_market, y_tr, None, d_pb_market, lgbm_args, "market",
                                   MARKET_SPEC, MARKET_MIN_DATA_SCALE,
                                   num_iteration=args.num_iteration)
    market_lgbm = group_mean(market_pred, pb_starts, pb_counts)
    del d_tr_market, d_pb_market, market_pred, transformed_train, transformed_probe
    gc.collect()

    m_hat = (1.0 - MARKET_LAMBDA) * market_ridge + MARKET_LAMBDA * market_lgbm
    raw = m_hat + e_hat
    y_pb = probe["target"]
    w_pb = np.maximum(probe["weight"], 0.0)
    tid_pb, aid_pb = probe["time_id"], probe["asset_id"]
    print(f"probe prediction ready ({time.perf_counter() - started:.0f}s)", flush=True)
    return y_pb, w_pb, tid_pb, aid_pb, raw, probe_low, probe_high


def evaluate_and_report(args, label, json_path, md_path, y_pb, w_pb, tid_pb, aid_pb, raw,
                        probe_low, probe_high, started) -> None:
    frozen = frozen_coefficients(Path(args.slow_variance_json), args.fold)
    sampled_mask = time_sample_mask(tid_pb, SAMPLE_MODULO, sampling=SAMPLING)
    rows: list[dict[str, Any]] = []
    for k_sampled, k_real in zip(K_SAMPLED, K_REAL):
        key = str(k_sampled)
        if key not in frozen:
            continue
        c, scale = frozen[key]["coefficients"], frozen[key]["baseline_scale"]

        # A 组：全分辨率，窗口按真实步
        slow_full = causal_trailing_mean(raw, tid_pb, aid_pb, k_real)
        rows_full = group_moment_rows(y_pb, w_pb, slow_full, raw - slow_full, tid_pb)
        full = score_frozen(rows_full.sum(axis=0), c, scale)
        full["bootstrap"] = bootstrap_relative(rows_full, c, scale, args.block_size,
                                               args.n_boot, args.boot_seed)
        # B 组：同一窗只留采样行（= v3_slow_variance 的口径）
        sm = sampled_mask
        slow_sampled = causal_trailing_mean(raw[sm], tid_pb[sm], aid_pb[sm], k_real)
        rows_sampled = group_moment_rows(y_pb[sm], w_pb[sm], slow_sampled,
                                         raw[sm] - slow_sampled, tid_pb[sm])
        sampled = score_frozen(rows_sampled.sum(axis=0), c, scale)
        sampled["bootstrap"] = bootstrap_relative(rows_sampled, c, scale, args.block_size,
                                                  args.n_boot, args.boot_seed)

        rows.append({"K_sampled_steps": k_sampled, "K_real_steps": k_real,
                     "frozen_coefficients": [float(x) for x in c],
                     "frozen_baseline_scale": scale,
                     "full_resolution": full, "sampled_grid": sampled})
        print(f"  K real={k_real:>5} (sampled {k_sampled:>4}): "
              f"full {full['relative']*100:+7.2f}% "
              f"[{full['bootstrap']['p2.5']*100:+.1f}%, {full['bootstrap']['p97.5']*100:+.1f}%] | "
              f"sampled {sampled['relative']*100:+7.2f}% "
              f"[{sampled['bootstrap']['p2.5']*100:+.1f}%, "
              f"{sampled['bootstrap']['p97.5']*100:+.1f}%]", flush=True)

    selected = str(SELECTED_K_SAMPLED)
    primary = next((r for r in rows if str(r["K_sampled_steps"]) == selected), rows[-1])
    sign_agreement = sum(1 for r in rows
                         if np.sign(r["full_resolution"]["relative"])
                         == np.sign(r["sampled_grid"]["relative"]))
    full_positive_count = sum(1 for r in rows if r["full_resolution"]["relative"] > 0)
    sampled_positive_count = sum(1 for r in rows if r["sampled_grid"]["relative"] > 0)
    full_never_worse = sum(1 for r in rows
                           if r["full_resolution"]["relative"] >= r["sampled_grid"]["relative"])
    full_positive_at_selected = primary["full_resolution"]["relative"] > 0
    ci_excludes_zero = primary["full_resolution"]["bootstrap"]["p2.5"] > 0
    payload = {
        "experiment": "v3_fullres_slow_probe",
        "question": "慢分量降权在**全分辨率** trailing mean 下还成立吗？",
        "design": "系数从 v3_slow_variance 的 fold 4 行冻结取来（在 fold 0..3 的采样格上解出），"
                  "探针窗内不做任何拟合；唯一变量是 trailing mean 的分辨率",
        "config": {"fold": args.fold, "probe_time_ids": args.probe_time_ids,
                   "probe_window": [probe_low, probe_high],
                   "num_iteration": args.num_iteration, "n_seeds": args.n_seeds,
                   "sample_modulo": SAMPLE_MODULO, "sampling": SAMPLING,
                   "block_size": args.block_size, "n_boot": args.n_boot},
        "probe_rows": {"full_resolution": int(len(y_pb)), "sampled_grid": int(sampled_mask.sum())},
        "K_rows": rows,
        "verdict": {
            "selected_K_sampled": selected,
            "full_resolution_relative_at_selected_K": primary["full_resolution"]["relative"],
            "full_resolution_positive_at_selected_K": bool(full_positive_at_selected),
            "bootstrap_ci_excludes_zero": bool(ci_excludes_zero),
            "sign_agreement_across_K": f"{sign_agreement}/{len(rows)}",
            "full_resolution_positive_K_count": f"{full_positive_count}/{len(rows)}",
            "sampled_grid_positive_K_count": f"{sampled_positive_count}/{len(rows)}",
            "full_resolution_not_worse_than_sampled": f"{full_never_worse}/{len(rows)}",
            "decision": ("全分辨率下同向且可分辨" if full_positive_at_selected and ci_excludes_zero
                         else "全分辨率下同向，但本窗口功效不足以分辨"
                         if full_positive_at_selected
                         else "全分辨率下反向 —— v3_slow_variance 结论需重估"),
            "power_note": "单个 20,000 真实 time_id 的窗口只有主实验约 4% 的数据量；"
                          "CI 跨 0 是预期之内，本探针只用来判断**换口径会不会翻向**，"
                          "不用来独立确认效应大小。",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# P1 口径核对：慢分量在全分辨率下还成立吗", "",
             f"fold {args.fold}，探针窗 `[{probe_low}, {probe_high}]` = "
             f"{args.probe_time_ids:,} 个连续真实 time_id；"
             f"全分辨率 {len(y_pb):,} 行 vs 采样格 {int(sampled_mask.sum()):,} 行。", "",
             f"**设计**：{payload['design']}。", "",
             "> ⚠️ 第一版曾在探针窗内对半切、前半重拟系数 —— 2,000 个采样 time_id 上拟两个高度"
             "共线的系数，比值在 −0.457 ~ +1.655 之间乱跳、相对增益 ±100%。那是探针没有功效，"
             "不是口径不一致。现版本冻结系数，只剩评估噪声，并由 block bootstrap 量化。", "",
             "| K（真实步） | K（采样步） | 冻结 c_slow/c_fast | 全分辨率 相对增益 | 95% CI | 采样格 相对增益 | 95% CI |",
             "|---:|---:|---:|---:|---|---:|---|"]
    for r in rows:
        f_, s_ = r["full_resolution"], r["sampled_grid"]
        c = r["frozen_coefficients"]
        lines.append(f"| {r['K_real_steps']} | {r['K_sampled_steps']} | "
                     f"{c[0]/c[1]:.3f} | {f_['relative']*100:+.2f}% | "
                     f"[{f_['bootstrap']['p2.5']*100:+.1f}%, {f_['bootstrap']['p97.5']*100:+.1f}%] | "
                     f"{s_['relative']*100:+.2f}% | "
                     f"[{s_['bootstrap']['p2.5']*100:+.1f}%, {s_['bootstrap']['p97.5']*100:+.1f}%] |")
    v = payload["verdict"]
    lines += ["", "## 判定", "",
              f"- 主实验选中的 K（采样步 {selected}）上，全分辨率相对增益 "
              f"**{v['full_resolution_relative_at_selected_K']*100:+.2f}%**",
              f"- 全分辨率为正的 K 数：**{v['full_resolution_positive_K_count']}**；"
              f"采样格为正的 K 数：{v['sampled_grid_positive_K_count']}",
              f"- 全分辨率不劣于采样格的 K 数：**{v['full_resolution_not_worse_than_sampled']}**"
              "（换口径若有害，这一栏会低）",
              f"- 跨 K 的符号一致性：{v['sign_agreement_across_K']}",
              f"- bootstrap 95% CI 是否排除 0：{'是' if v['bootstrap_ci_excludes_zero'] else '**否**'}",
              "", f"### 结论：{v['decision']}", "", f"> {v['power_note']}", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{json.dumps(payload['verdict'], ensure_ascii=False, indent=2)}")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
