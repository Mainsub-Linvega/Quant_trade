"""滞后的特征截面均值对 m_t 有没有增量预测力？（ROADMAP #3 的诊断）

背景：85% 的分来自择时块（Score = 0.72·R²_m + 0.28·R²_e，见 mt_predictability），
而用**当期**特征截面均值线性预测 m_t 已近饱和（样本外 R²_m = 0.00177）。
但 m_t 的自相关 ac1=0.836、到 lag 5~6 才归零 —— 时序结构存在却从没被用过。

**为什么不能改 mt_predictability.py 来做这件事**：它用 sample_modulo=10 加载，
相邻样本相差 10 个真实 time_id，而 m_t 的自相关在 lag 5~6 就归零 ——
那个口径下的「lag 1」其实是真实的 lag 10，早出了相关性范围，必然假阴性。
所以这里用**全时间分辨率**。

**为什么每个臂都要扫一遍 alpha**：本地尺子在「拟合紧密度」维度上已被证明会量反
（公榜说 alpha 2e6 比 5e5 好 23.8%，三把本地尺子都说反话）。加滞后列 = 增加灵活度，
同属该维度。所以判据是「整条 alpha 曲线上移」，不是某个 alpha 上的胜负。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/mt_lagged.py
输出：outputs/experiments/mt_lagged.{json,md}
缓存：outputs/cache/mt_aggregates.npz（约 1.2G，删了会自动重建）
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
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, train_files
from src.validation import rolling_time_folds
from history_features import iter_complete_time_batches  # 处理跨 batch 的 time_id 切分
from train import robust_transform_fit  # 预处理复用生产的唯一实现
from features import apply_robust_transform

ALPHAS = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]

# 每个臂 = 用哪些滞后。roll6 单独处理（lag 1..6 的均值，针对 m_t 自相关线性归零到 6 的形状）
ARMS: dict[str, dict[str, Any]] = {
    "now": {"lags": ()},
    "lag1": {"lags": (1,)},
    "lag3": {"lags": (1, 2, 3)},
    "roll6": {"lags": (), "rolling": 6},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Do lagged cross-sectional feature means predict m_t?")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--cache", default=str(_REPO_ROOT / "outputs" / "cache" / "mt_aggregates.npz"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--train-window", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=6)
    return parser.parse_args()


# ---------------------------------------------------------------- 流式聚合

def build_aggregates(data_root: Path) -> dict[str, np.ndarray]:
    """一次扫过全部训练数据，只留下每个 time_id 一行的聚合量。

    W_t = Σᵢwᵢ，m_t = Σᵢwᵢyᵢ/W_t，X̄_t[j] = Σᵢwᵢxᵢⱼ/W_t。
    20 GB 行级数据 → 888,315 × 323 的矩阵（约 1.15 GB）。
    非有限值按生产口径先归零再加权平均（apply_robust_transform 的第一步也是这个）。
    """
    columns = ["time_id", "weight", "target", *FEATURE_COLUMNS]
    tid_parts, w_parts, m_parts, x_parts = [], [], [], []
    for path in train_files(data_root):
        kept = 0
        started = time.perf_counter()
        for frame in iter_complete_time_batches(path, columns):
            tid = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]
            weight = frame["weight"].to_numpy(dtype=np.float64, copy=False)
            target = frame["target"].to_numpy(dtype=np.float64, copy=False)
            features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
            np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            total_w = np.add.reduceat(weight, starts)
            tid_parts.append(tid[starts])
            w_parts.append(total_w)
            m_parts.append(np.add.reduceat(weight * target, starts) / total_w)
            x_parts.append(
                (np.add.reduceat(features * weight[:, None], starts, axis=0)
                 / total_w[:, None]).astype(np.float32)
            )
            kept += len(starts)
        print(f"aggregated {path.name}: {kept:,} time_ids ({time.perf_counter()-started:.0f}s)", flush=True)

    return {
        "time_id": np.concatenate(tid_parts),
        "weight": np.concatenate(w_parts),
        "m": np.concatenate(m_parts),
        "xbar": np.concatenate(x_parts),
    }


def load_or_build(cache: Path, data_root: Path, rebuild: bool) -> dict[str, np.ndarray]:
    if cache.exists() and not rebuild:
        print(f"loading cached aggregates from {cache}", flush=True)
        with np.load(cache) as handle:
            return {k: handle[k] for k in ("time_id", "weight", "m", "xbar")}
    agg = build_aggregates(data_root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **agg)
    print(f"cached aggregates → {cache}", flush=True)
    return agg


# ---------------------------------------------------------------- 设计矩阵

def make_design(
    xbar: np.ndarray, time_id: np.ndarray, rows: np.ndarray,
    lags: tuple[int, ...], rolling: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """给定行下标，拼出 [当期 ‖ 各滞后块]。

    ⚠️ time_id 不连续（p003/p004/p007 内部有断点，间隔出现 7/8/10/19/23/80），
    所以滞后一律按**真实时间差**校验：只有 time_id[i] − time_id[i−k] == k 时该块才填值，
    否则置 0。这样滞后列永远只来自 time_id ≤ t−k 的数据，因果性成立。
    """
    width = xbar.shape[1]
    blocks = 1 + len(lags) + (1 if rolling else 0)
    design = np.zeros((len(rows), width * blocks), dtype=np.float32)
    design[:, :width] = xbar[rows]
    coverage: dict[str, float] = {}

    for index, lag in enumerate(lags, start=1):
        source = rows - lag
        ok = source >= 0
        ok[ok] &= (time_id[rows[ok]] - time_id[source[ok]]) == lag
        design[np.flatnonzero(ok), width * index: width * (index + 1)] = xbar[source[ok]]
        coverage[f"lag{lag}"] = float(ok.mean())

    if rolling:
        source = rows - rolling
        ok = source >= 0
        # 严格递增的整数序列上，跨 rolling 步的差恰为 rolling ⟺ 中间每步都是 1
        ok[ok] &= (time_id[rows[ok]] - time_id[source[ok]]) == rolling
        valid_rows = np.flatnonzero(ok)
        accumulator = np.zeros((len(valid_rows), width), dtype=np.float32)
        for step in range(1, rolling + 1):
            accumulator += xbar[rows[valid_rows] - step]
        design[valid_rows, width * (1 + len(lags)):] = accumulator / rolling
        coverage[f"roll{rolling}"] = float(ok.mean())

    return design, coverage


# ---------------------------------------------------------------- 加权岭回归（正规方程）

def normal_equations(
    design: np.ndarray, target: np.ndarray, weight: np.ndarray, chunk: int = 50_000
) -> tuple[np.ndarray, np.ndarray]:
    """分块累积 A = XᵀWX 与 b = XᵀWy（float64）。累积一次，alpha 网格就白送。"""
    width = design.shape[1]
    gram = np.zeros((width, width), dtype=np.float64)
    moment = np.zeros(width, dtype=np.float64)
    for start in range(0, len(design), chunk):
        block = design[start:start + chunk].astype(np.float64)
        block_weight = weight[start:start + chunk]
        weighted = block * block_weight[:, None]
        gram += block.T @ weighted
        moment += weighted.T @ target[start:start + chunk]
    return gram, moment


def standardise(design: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """加权列均值与标准差 —— 只给 self_check 用。

    ⚠️ 不要拿它预处理真实特征：X̄ 里存在训练段方差小到 6.1e-15 的列，
    按标准差归一化后，验证段一旦漂移，z 会冲到 1e5 量级 —— 爆的是输入不是系数，
    再大的 alpha 也压不住（第一版就是这么得出 R²=−8e7 的）。
    真实特征一律走 robust_transform_fit：按训练分位数裁剪 → 除 IQR → 裁到 ±10。
    """
    total = weight.sum()
    centred = design.astype(np.float64) - (design.astype(np.float64) * weight[:, None]).sum(axis=0) / total
    mean = (design.astype(np.float64) * weight[:, None]).sum(axis=0) / total
    var = (centred**2 * weight[:, None]).sum(axis=0) / total
    return mean, np.sqrt(np.maximum(var, 1e-12))


def weighted_r2(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.dot(weight, actual * actual))
    if denominator <= 0:
        return 0.0
    return float(1.0 - np.dot(weight, (actual - predicted) ** 2) / denominator)


def self_check() -> None:
    """正规方程与 sklearn.Ridge 对齐（验收标准 3）。"""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 25)).astype(np.float32)
    y = (x[:, :3] @ np.array([0.4, -0.3, 0.2]) + 0.5 * rng.standard_normal(4000))
    w = np.abs(rng.standard_normal(4000)) + 0.1
    mean, sd = standardise(x, w)
    z = ((x - mean) / sd).astype(np.float32)
    y_mean = float(np.dot(w, y) / w.sum())
    gram, moment = normal_equations(z, y - y_mean, w)
    alpha = 25.0
    mine = np.linalg.solve(gram + alpha * np.eye(len(gram)), moment)
    theirs = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky").fit(
        z.astype(np.float64), y - y_mean, sample_weight=w
    ).coef_
    relative = float(np.abs(mine - theirs).max() / np.abs(theirs).max())
    print(f"self-check 正规方程 vs sklearn.Ridge: 相对差 {relative:.2e} "
          f"{'✅' if relative < 1e-6 else '❌'}", flush=True)
    assert relative < 1e-6, "正规方程实现与 sklearn 不一致"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    self_check()

    agg = load_or_build(Path(args.cache), Path(args.data_root), args.rebuild_cache)
    time_id, weight, m_series, xbar = agg["time_id"], agg["weight"], agg["m"], agg["xbar"]
    order = np.argsort(time_id, kind="stable")
    time_id, weight, m_series, xbar = time_id[order], weight[order], m_series[order], xbar[order]
    total = len(time_id)
    print(f"{total:,} 个 time_id，{xbar.shape[1]} 列特征截面均值", flush=True)

    train_window = args.train_window or int(total * 4 / 9)
    folds = rolling_time_folds(time_id, args.n_folds, train_window, args.embargo)

    results: list[dict[str, Any]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        # time_id 已排序，rolling_time_folds 返回的就是它的切片 → searchsorted 直接给下标
        train_rows = np.searchsorted(time_id, train_ids)
        valid_rows = np.searchsorted(time_id, valid_ids)
        assert np.array_equal(time_id[train_rows], train_ids)
        assert np.array_equal(time_id[valid_rows], valid_ids)
        fold: dict[str, Any] = {
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "train_rows": len(train_rows), "valid_rows": len(valid_rows),
            "arms": {},
        }
        # 预处理统计量只在训练段拟合，然后套到整条 X̄ 上（滞后块要和当期块同口径）。
        # 走生产的 robust_transform_fit：裁到训练分位数 → 除 IQR → 裁到 ±10。
        # 最后那道 ±10 是关键 —— 没有它，验证段的低方差列会把输入炸到 1e5。
        _, stats = robust_transform_fit(xbar[train_rows].copy())
        scaled = apply_robust_transform(
            xbar.copy(), stats["lower"], stats["upper"], stats["center"], stats["scale"]
        )

        for arm, spec in ARMS.items():
            lags = tuple(spec["lags"])
            rolling = int(spec.get("rolling", 0))
            z_train, coverage = make_design(scaled, time_id, train_rows, lags, rolling)
            z_valid, _ = make_design(scaled, time_id, valid_rows, lags, rolling)
            w_train, w_valid = weight[train_rows], weight[valid_rows]
            y_train, y_valid = m_series[train_rows], m_series[valid_rows]
            offset = float(np.dot(w_train, y_train) / w_train.sum())
            gram, moment = normal_equations(z_train, y_train - offset, w_train)

            scores = {}
            eye = np.eye(len(gram))
            for alpha in ALPHAS:
                beta = np.linalg.solve(gram + alpha * eye, moment)
                scores[f"{alpha:.0e}"] = weighted_r2(y_valid, z_valid @ beta + offset, w_valid)
            fold["arms"][arm] = {
                "r2": scores, "columns": int(z_train.shape[1]), "coverage": coverage,
                "max_abs_input": float(np.abs(z_valid).max()),  # 应 ≤10，超了说明预处理没生效
            }
            del z_train, z_valid, gram
        del scaled

        fold["elapsed_seconds"] = float(time.perf_counter() - started)
        results.append(fold)
        best = {a: max(fold["arms"][a]["r2"].values()) for a in ARMS}
        print(f"fold {index:2d}: " + "  ".join(f"{a}={best[a]:+.5f}" for a in ARMS)
              + f"  ({fold['elapsed_seconds']:.0f}s)", flush=True)

    summary = {
        arm: {
            key: {
                "mean": float(np.mean([f["arms"][arm]["r2"][key] for f in results])),
                "positive_folds": int(sum(f["arms"][arm]["r2"][key] > 0 for f in results)),
            }
            for key in (f"{a:.0e}" for a in ALPHAS)
        }
        for arm in ARMS
    }

    payload = {
        "question": "滞后的特征截面均值对 m_t 有增量预测力吗？",
        "metric": "样本外 R²_m = 1 − ΣW(m−m̂)²/ΣW·m²",
        "score_mapping": "Δscore ≈ 0.72 × ΔR²_m（择时块占 target 方差 72%）",
        "configuration": {
            "n_folds": args.n_folds, "train_window": train_window,
            "embargo": args.embargo, "total_time_ids": total, "alphas": ALPHAS,
            "arms": {a: {"lags": list(s["lags"]), "rolling": s.get("rolling", 0)} for a, s in ARMS.items()},
        },
        "summary": summary,
        "folds": results,
    }
    (output_dir / "mt_lagged.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 滞后的特征截面均值能否增量预测 m_t",
        "",
        f"全时间分辨率（{total:,} 个 time_id），{args.n_folds} 折滚动 + embargo {args.embargo}。",
        "样本外 `R²_m = 1 − ΣW(m−m̂)²/ΣW·m²`；换算到总分 `Δscore ≈ 0.72 × ΔR²_m`。",
        "",
        "滞后按**真实时间差**校验（time_id 不连续，p003/p004/p007 有断点），不满足的行置 0。",
        "",
        "## 样本外 R²_m（每格：均值 / 为正的折数）",
        "",
        "| 臂 | 列数 | " + " | ".join(f"α={a:.0e}" for a in ALPHAS) + " |",
        "|---|---:|" + "---:|" * len(ALPHAS),
    ]
    for arm in ARMS:
        cells = []
        for alpha in ALPHAS:
            cell = summary[arm][f"{alpha:.0e}"]
            cells.append(f"{cell['mean']:+.5f} / {cell['positive_folds']}")
        lines.append(f"| `{arm}` | {results[0]['arms'][arm]['columns']} | " + " | ".join(cells) + " |")

    base = max(summary["now"][f"{a:.0e}"]["mean"] for a in ALPHAS)
    lines += ["", "## 相对 `now` 臂的增量", "",
              "| 臂 | 最好的 R²_m | ΔR²_m | 换算到总分 |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        best = max(summary[arm][f"{a:.0e}"]["mean"] for a in ALPHAS)
        lines.append(f"| `{arm}` | {best:+.5f} | {best - base:+.5f} | {(best - base) * 0.72:+.6f} |")
    lines += [
        "",
        "## 怎么读",
        "",
        "- **整条 alpha 曲线上移** → 滞后信息有料，下一步才是接进 train.py 的择时块",
        "- **只在小 alpha 处上移** → 大概率是过拟合，与 alpha 那次同源，不予采信",
        "  （本地尺子在「拟合紧密度」维度上已被证明会量反，见 NOTES.md）",
        "- **完全不动** → 择时块的线性路到头，收益只能寄望于非线性",
    ]
    (output_dir / "mt_lagged.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({arm: max(summary[arm][f"{a:.0e}"]["mean"] for a in ALPHAS) for arm in ARMS},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
