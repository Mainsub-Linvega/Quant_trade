"""LightGBM 能不能在择时块 m_t 上打赢岭回归的线性天花板？（ROADMAP #4 的主攻）

背景：`Score = 0.72·R²_m + 0.28·R²_e`（恒等式残差 5.89e-16，见 mt_predictability），
**85% 的分来自择时块**。而 NOTES 已经把这块的线性路走完了：
用当期特征截面均值线性回归 m_t，七档 alpha 扫下来峰值 R²_m = +0.00085（mt_lagged 的 now 臂）；
加滞后输入被 mt_lagged 证伪（表面增量全在去掉最好一折后翻负）。
剩下的只有一条没试过的路 —— **非线性**。

为什么这一刀先切择时块而不是截面块：
1. 数据已经缓存好（outputs/cache/mt_aggregates.npz，888,315 × 323），不用重扫 17G parquet；
2. 它是**全时间分辨率**的 —— 全部 10 个相位一个不落，所以承载 85% 分数的这块
   **根本不存在 ROADMAP #2 那个相位错配问题**（那是 v1_ridge 的 sample_modulo=5 才有的）；
3. 1.15G 数据，分钟级一轮，而截面块是 1322 万行。

## 三条必须守住的口径（否则数就是假的）

**A. 早停集绝不能是外层验证段。** 从训练段尾部切 `--inner-frac` 做内层早停集，
中间再空出 embargo 个 time_id。外层验证段全程对训练不可见。

**B. 岭回归对照臂必须吃到和 LGBM 一样的数据。** 所以 gram 分三段（内层训练 / embargo 间隙 /
内层验证）分别累积再相加 —— gram、XᵀWy、XᵀW1、Σw、Σwy 全是可加的，
一次扫描同时给出 `ridge_fair`（只吃内层训练段，与 LGBM 同数据）和
`ridge_full`（吃满整个训练段，与 mt_lagged 口径一致，用于对照历史结论）。

**C. 轮数曲线白送。** 一次拟合后用 `predict(num_iteration=k)` 在一串 k 上打分，
拿到整条轮数路径 —— 和岭回归「累积一次 gram、alpha 网格白送」是同一个手法。
判据看**整条曲线**，不看某个轮数上的胜负（本地尺子在「拟合紧密度」维度会量反，见 NOTES）。

## 判据（照搬 ab_decomposition / mt_lagged 的规矩）

- 逐折**配对 Δ**，不看绝对分
- **去掉最好的一折后仍为正** —— mt_lagged 的教训，表面增量当初全在这一步翻负
- **A/B 分解**：`A = ΣW·m·f/ΣW·m²`、`B = ΣW·f²/ΣW·m²`，本地 ΔA 打 2.2 折后判 `2·ΔA > ΔB`
  （增加容量的改动一律靠 A↑ 起作用 → 重罚；LGBM 是本项目最大的一次容量增加）

用法：
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/lgbm_mt.py
    # 先试水：--n-folds 2 --max-rounds 300 --report lgbm_mt_trial
输出：outputs/experiments/lgbm_mt.{json,md}（同名已存在需 --force）
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

from src.validation import rolling_time_folds
from train import robust_transform_fit          # 预处理复用生产的唯一实现
from features import apply_robust_transform
from mt_lagged import load_or_build, normal_equations, weighted_r2

ALPHAS = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]   # 与 mt_lagged 同一档，结论可直接对读

# 预注册的固定 alpha —— 第三个岭回归臂 `ridge_fixed` 用它，**完全不做选择**。
# 出处：mt_lagged 已发表的结论（now 臂七档扫下来在 α=1e8 见顶），是**先验**，
# 不是在本次外层验证段上选出来的。
#
# 为什么必须有这个臂：`ridge_fair` 在内层验证段上选 alpha，而 alpha 网格是 7 个点、
# 跨十倍数量级，选错一档代价巨大 —— 试跑里它的无泄漏均值 −0.00009、自己的曲线峰值
# +0.00075。LGBM 的轮数是细粒度的、早停也更平滑，两者的**选择过程方差不对等**，
# 会把「选择噪声」冒充成「模型好坏」，白送 LGBM 一截。
# 所以判据要求对**两个基准都过关**才算 PASS。
FIXED_ALPHA = 1e8

# 预注册超参 —— **不做网格搜索**。树深/轮数/学习率/num_leaves 全是「拟合紧密度」参数，
# 本地网格会像 alpha 那样量反（公榜说 2e6 比 5e5 好 23.8%，三把本地尺子都说反话）。
# 目标 R² 只有 ~1e-3 量级，信噪比极低，所以四组统一走
# 「小叶子数 + 大 min_data_in_leaf + 强 L2 + 低学习率」，靠这四组张开容量轴。
CANDIDATES: dict[str, dict[str, Any]] = {
    "ultra_shrunk": {"num_leaves": 7,  "min_data_in_leaf": 20000, "learning_rate": 0.01,
                     "feature_fraction": 0.3, "lambda_l2": 100.0},
    "shrunk":       {"num_leaves": 15, "min_data_in_leaf": 8000,  "learning_rate": 0.02,
                     "feature_fraction": 0.4, "lambda_l2": 30.0},
    "moderate":     {"num_leaves": 31, "min_data_in_leaf": 3000,  "learning_rate": 0.03,
                     "feature_fraction": 0.5, "lambda_l2": 10.0},
    "loose":        {"num_leaves": 63, "min_data_in_leaf": 1000,  "learning_rate": 0.03,
                     "feature_fraction": 0.7, "lambda_l2": 1.0},
    # 第二轮新增的两格：第一轮唯一指向「还有东西」的线索全在收缩轴最外侧
    # （ultra_shrunk 的 ΔB 只有 +2.7%，其余三组 +46%~+56%；且它到 800 轮仍在爬），
    # 说明预注册网格的起点偏松了。内层训练段约 355k 行，所以 min_data_in_leaf
    # 50000/100000 把有效叶子数分别压到约 7/3，与 num_leaves 5/3 相互印证。
    "hyper_shrunk":   {"num_leaves": 5, "min_data_in_leaf": 50000,  "learning_rate": 0.005,
                       "feature_fraction": 0.2, "lambda_l2": 300.0},
    "extreme_shrunk": {"num_leaves": 3, "min_data_in_leaf": 100000, "learning_rate": 0.005,
                       "feature_fraction": 0.15, "lambda_l2": 1000.0},
}

CHECKPOINTS = (10, 20, 40, 80, 120, 200, 300, 450, 600, 800, 1200, 1800, 2400, 3000)

N_SEEDS = 3          # 三种子预测平均：削掉 bagging/feature_fraction 的抖动，
                     # 也让 LGBM 臂与（无抖动的）岭回归臂在确定性上对等。
                     # 顺带这就是真要上线的形态（主办方基线也是 3 种子）。
PUBLIC_REFERENCE = 0.00186805     # 当前公榜分，只用于把检出下限换算成百分比


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Does LightGBM beat the linear ceiling on m_t?")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--cache", default=str(_REPO_ROOT / "outputs" / "cache" / "mt_aggregates.npz"))
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--report", default="lgbm_mt_v2", help="产物文件名（不含扩展名）")
    p.add_argument("--force", action="store_true", help="覆盖同名报告")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--train-window", type=int, default=None)
    p.add_argument("--embargo", type=int, default=6)
    p.add_argument("--inner-frac", type=float, default=0.10,
                   help="训练段尾部留作内层早停集的比例")
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--early-stopping", type=int, default=50,
                   help="早停耐心的下限；实际取 max(该值, round(1/lr)) —— "
                        "固定 50 对 lr=0.005 等于提前掐断")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--candidates", nargs="*", default=None,
                   help="只跑其中几组（默认全部）")
    return p.parse_args()


# ------------------------------------------------------- 可加的正规方程累积量

def accumulate(design: np.ndarray, target: np.ndarray, weight: np.ndarray,
               chunk: int = 50_000) -> dict[str, Any]:
    """分块累积 XᵀWX / XᵀWy / XᵀW1 / Σw / Σwy —— **全部可加**。

    这样把训练段切成 [内层训练 ‖ embargo 间隙 ‖ 内层验证] 三片各累一次，
    相加就能同时得到「只吃内层训练」和「吃满整段」两套岭回归解，不用扫两遍。
    截距走 offset 技巧（与 mt_lagged.normal_equations 一致）：
    moment = XᵀWy − offset·XᵀW1，其中 offset = Σwy/Σw。
    """
    width = design.shape[1]
    out = {
        "gram": np.zeros((width, width), dtype=np.float64),
        "xtwy": np.zeros(width, dtype=np.float64),
        "xtw1": np.zeros(width, dtype=np.float64),
        "sw": 0.0, "swy": 0.0, "n": int(len(design)),
    }
    for start in range(0, len(design), chunk):
        block = design[start:start + chunk].astype(np.float64)
        bw = weight[start:start + chunk]
        by = target[start:start + chunk]
        weighted = block * bw[:, None]
        out["gram"] += block.T @ weighted
        out["xtwy"] += weighted.T @ by
        out["xtw1"] += weighted.sum(axis=0)
        out["sw"] += float(bw.sum())
        out["swy"] += float(np.dot(bw, by))
    return out


def combine(*accs: dict[str, Any]) -> dict[str, Any]:
    out = {
        "gram": sum(a["gram"] for a in accs),
        "xtwy": sum(a["xtwy"] for a in accs),
        "xtw1": sum(a["xtw1"] for a in accs),
        "sw": sum(a["sw"] for a in accs),
        "swy": sum(a["swy"] for a in accs),
        "n": sum(a["n"] for a in accs),
    }
    return out


def ridge_predictions(acc: dict[str, Any], z_valid: np.ndarray,
                      alphas: list[float]) -> dict[str, np.ndarray]:
    """一次累积 → 整条 alpha 路径的验证段预测。"""
    offset = acc["swy"] / acc["sw"]
    moment = acc["xtwy"] - offset * acc["xtw1"]
    eye = np.eye(len(acc["gram"]))
    preds = {}
    for alpha in alphas:
        beta = np.linalg.solve(acc["gram"] + alpha * eye, moment)
        preds[f"{alpha:.0e}"] = z_valid @ beta + offset
    return preds


# ------------------------------------------------------------------ A/B 分解

def ab_terms(actual: np.ndarray, predicted: np.ndarray,
             weight: np.ndarray) -> tuple[float, float]:
    """A = ΣW·m·f/ΣW·m²（信号对齐）、B = ΣW·f²/ΣW·m²（预测方差）。峰值 = A²/B。"""
    denom = float(np.dot(weight, actual * actual))
    if denom <= 0:
        return 0.0, 0.0
    return (float(np.dot(weight, actual * predicted)) / denom,
            float(np.dot(weight, predicted * predicted)) / denom)


def score_parts(actual: np.ndarray, predicted: np.ndarray,
                weight: np.ndarray) -> dict[str, float]:
    """一次算齐 r2 / A / B / sse / sst。

    sse、sst 是为 **pooled Δ** 留的：`rolling_time_folds` 的验证段互不重叠，
    所以逐折的加权平方和可直接相加 —— `pooled R² = 1 − ΣSSE/ΣSST`。
    sst 只依赖标签，与臂无关，配对天然成立：
    `pooled Δ = (ΣSSE_base − ΣSSE_arm) / ΣSST`。
    mean-of-folds 会被单折异常值拉走（第一轮 fold 3 就占了 mean 的 80%），
    pooled 按数据量加权，两个口径同时看才安全。
    """
    sse = float(np.dot(weight, (actual - predicted) ** 2))
    sst = float(np.dot(weight, actual * actual))
    a, b = ab_terms(actual, predicted, weight)
    return {"r2": 0.0 if sst <= 0 else 1.0 - sse / sst,
            "A": a, "B": b, "sse": sse, "sst": sst}


def paired_stats(deltas: list[float]) -> dict[str, Any]:
    """配对 Δ 的统计量。**t 才是判据，mean 的正负不是。**

    第一轮四个臂的 mean(Δ) 有三个为正，但 t 全在 ±1.1 以内 —— 看 mean 会得出
    完全相反的结论。同时报「去掉最好一折」后的 t（`mt_lagged` 的教训：
    表面增量当初全部来自单独一折）。
    """
    x = np.asarray(deltas, dtype=np.float64)

    def _t(v: np.ndarray) -> tuple[float | None, float | None]:
        """样本量 < 2 时标准误未定义 —— 返回 None，**不要退化成 0**。

        0 会被 verdict_of 当成一个真实的、刚好不显著的 t 值悄悄吞掉；
        None 会让判定显式变成 INSUFFICIENT_FOLDS。
        """
        if len(v) < 2:
            return None, None
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        return (se, float(v.mean() / se)) if se > 0 else (0.0, None)

    se, t = _t(x)
    drop = np.sort(x)[:-1]
    se_d, t_d = _t(drop)
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "se": se, "t": t,
        "positive_folds": int((x > 0).sum()), "n_folds": int(len(x)),
        "drop_best_mean": float(drop.mean()) if len(drop) else None,
        "drop_best_se": se_d, "drop_best_t": t_d,
        "per_fold": [float(v) for v in x],
    }


def verdict_of(stats: dict[str, Any], da: float, db: float) -> str:
    """**预注册判据，机器判。**

    伤疤清单里有一条「报告写 Accepted: True，但与实际代码无机械联系」——
    所以结论由这个函数算出来直接写进 json，不靠我在 md 里下判断。

    PASS 需要三条同时成立：
      1. t ≥ 2                      —— 效应超过这个设计的检出下限
      2. 去掉最好一折后 t ≥ 1        —— 不是单折撑起来的
      3. 2·ΔA/2.2 > ΔB              —— A/B 机制对（本地 ΔA 系统性高估 2.2 倍，只折 ΔA）

    折数不足以定义标准误时判 INSUFFICIENT_FOLDS —— 试跑（2 折）会走到这里，
    不能让它看起来像一次真正的「未定」。
    """
    if stats["t"] is None or stats["drop_best_t"] is None:
        return "INSUFFICIENT_FOLDS"
    if stats["t"] <= -2.0:
        return "FAIL"
    if stats["t"] >= 2.0 and stats["drop_best_t"] >= 1.0 and 2.0 * (da / 2.2) > db:
        return "PASS"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ self-check

def self_check() -> None:
    """两条：可加累积 == 一次性累积；正规方程 == sklearn.Ridge。"""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 25)).astype(np.float32)
    y = x[:, :3] @ np.array([0.4, -0.3, 0.2]) + 0.5 * rng.standard_normal(4000)
    w = np.abs(rng.standard_normal(4000)) + 0.1
    alpha = 25.0

    whole = accumulate(x, y, w)
    pieces = combine(accumulate(x[:1500], y[:1500], w[:1500]),
                     accumulate(x[1500:2600], y[1500:2600], w[1500:2600]),
                     accumulate(x[2600:], y[2600:], w[2600:]))
    add_err = max(float(np.abs(whole[k] - pieces[k]).max()) for k in ("gram", "xtwy", "xtw1"))
    add_err = max(add_err, abs(whole["sw"] - pieces["sw"]), abs(whole["swy"] - pieces["swy"]))
    scale = float(np.abs(whole["gram"]).max())
    print(f"self-check 累积量可加性: 绝对差 {add_err:.2e} (gram 量级 {scale:.2e}) "
          f"{'✅' if add_err / scale < 1e-12 else '❌'}", flush=True)
    assert add_err / scale < 1e-12, "累积量不可加，分片相加的岭回归解不可信"

    # 与 mt_lagged 的 normal_equations + sklearn 三方对齐
    offset = float(np.dot(w, y) / w.sum())
    gram_ref, moment_ref = normal_equations(x, y - offset, w)
    mine = np.linalg.solve(whole["gram"] + alpha * np.eye(25),
                           whole["xtwy"] - offset * whole["xtw1"])
    ref = np.linalg.solve(gram_ref + alpha * np.eye(25), moment_ref)
    theirs = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky").fit(
        x.astype(np.float64), y - offset, sample_weight=w).coef_
    rel = max(float(np.abs(mine - ref).max()), float(np.abs(mine - theirs).max())) / float(np.abs(theirs).max())
    print(f"self-check 正规方程 vs mt_lagged vs sklearn: 相对差 {rel:.2e} "
          f"{'✅' if rel < 1e-6 else '❌'}", flush=True)
    assert rel < 1e-6, "正规方程实现与既有实现/sklearn 不一致"


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

    self_check()

    agg = load_or_build(Path(args.cache), Path(args.data_root), args.rebuild_cache)
    time_id, weight, m_series, xbar = agg["time_id"], agg["weight"], agg["m"], agg["xbar"]
    order = np.argsort(time_id, kind="stable")
    time_id, weight, m_series, xbar = time_id[order], weight[order], m_series[order], xbar[order]
    total = len(time_id)
    print(f"{total:,} 个 time_id，{xbar.shape[1]} 列特征截面均值（全部 10 个相位）", flush=True)

    names = list(args.candidates) if args.candidates else list(CANDIDATES)
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise SystemExit(f"未知候选：{unknown}，可选 {list(CANDIDATES)}")

    train_window = args.train_window or int(total * 4 / 9)
    folds = rolling_time_folds(time_id, args.n_folds, train_window, args.embargo)
    checkpoints = [k for k in CHECKPOINTS if k <= args.max_rounds]

    results: list[dict[str, Any]] = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        train_rows = np.searchsorted(time_id, train_ids)
        valid_rows = np.searchsorted(time_id, valid_ids)
        assert np.array_equal(time_id[train_rows], train_ids)
        assert np.array_equal(time_id[valid_rows], valid_ids)

        # 内层切分：训练段尾部 inner_frac 做早停集，与内层训练段之间再空 embargo 个 time_id。
        # 外层验证段全程不参与训练与轮数选择。
        n_train = len(train_rows)
        n_inner_valid = max(1, int(n_train * args.inner_frac))
        inner_valid_rows = train_rows[n_train - n_inner_valid:]
        gap_rows = train_rows[n_train - n_inner_valid - args.embargo: n_train - n_inner_valid]
        inner_train_rows = train_rows[: n_train - n_inner_valid - args.embargo]

        fold: dict[str, Any] = {
            "fold": index,
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "inner_train_time_range": [int(time_id[inner_train_rows[0]]), int(time_id[inner_train_rows[-1]])],
            "inner_valid_time_range": [int(time_id[inner_valid_rows[0]]), int(time_id[inner_valid_rows[-1]])],
            "rows": {"train": n_train, "inner_train": len(inner_train_rows),
                     "gap": len(gap_rows), "inner_valid": len(inner_valid_rows),
                     "outer_valid": len(valid_rows)},
            "arms": {},
        }
        # 无泄漏断言：内层验证段必须整段早于外层验证段
        assert int(time_id[inner_valid_rows[-1]]) < int(valid_ids[0]), "内层早停集越过了外层验证段"

        # 预处理统计量只在**内层训练段**拟合（LGBM 与 ridge_fair 共用），再套到全序列上。
        _, stats = robust_transform_fit(xbar[inner_train_rows].copy())
        scaled = apply_robust_transform(
            xbar.copy(), stats["lower"], stats["upper"], stats["center"], stats["scale"])

        y_valid, w_valid = m_series[valid_rows], weight[valid_rows]
        z_valid = scaled[valid_rows]

        z_iv = scaled[inner_valid_rows]
        y_iv, w_iv = m_series[inner_valid_rows], weight[inner_valid_rows]

        # ---- 岭回归对照臂：三片累积 → fair（同数据）与 full（吃满训练段）两套解
        acc_inner = accumulate(scaled[inner_train_rows], m_series[inner_train_rows], weight[inner_train_rows])
        acc_gap = accumulate(scaled[gap_rows], m_series[gap_rows], weight[gap_rows])
        acc_iv = accumulate(scaled[inner_valid_rows], m_series[inner_valid_rows], weight[inner_valid_rows])
        for arm_name, acc in (("ridge_fair", acc_inner),
                              ("ridge_full", combine(acc_inner, acc_gap, acc_iv))):
            preds = ridge_predictions(acc, z_valid, ALPHAS)
            entry: dict[str, Any] = {
                "kind": "ridge",
                "train_rows": acc["n"],
                "path": {k: score_parts(y_valid, p, w_valid) for k, p in preds.items()},
                "honest": None,
            }
            # 只有 ridge_fair 没吃过内层验证段，才有资格在上面选 alpha。
            # ridge_full 吃满整个训练段（含内层验证），选出来会有偏 —— 它只留曲线做对照。
            if arm_name == "ridge_fair":
                inner_preds = ridge_predictions(acc, z_iv, ALPHAS)
                knob = max(inner_preds, key=lambda k: weighted_r2(y_iv, inner_preds[k], w_iv))
                entry["honest"] = {"knob": knob, **score_parts(y_valid, preds[knob], w_valid)}
            fold["arms"][arm_name] = entry

            # ridge_fixed：与 ridge_fair 同数据同拟合，只是 alpha 预注册死、**不做任何选择** →
            # 零选择方差、零泄漏。系数由同一次 gram 累积白送，不额外花代价。
            if arm_name == "ridge_fair":
                fixed_key = f"{FIXED_ALPHA:.0e}"
                fold["arms"]["ridge_fixed"] = {
                    "kind": "ridge", "train_rows": acc["n"],
                    "path": entry["path"],          # 同一条 alpha 路径，共享
                    "honest": {"knob": f"{fixed_key}（预注册）",
                               **score_parts(y_valid, preds[fixed_key], w_valid)},
                }
        del acc_inner, acc_gap, acc_iv

        # ---- LightGBM 臂
        x_it = np.ascontiguousarray(scaled[inner_train_rows])
        x_iv = np.ascontiguousarray(z_iv)
        x_ov = np.ascontiguousarray(z_valid)

        for name in names:
            spec = CANDIDATES[name]
            # 早停耐心随学习率放大：固定 50 对 lr=0.005 只相当于 0.25 个「lr 单位」的进展，
            # 等于提前掐断。lr 0.03→50、0.01→100、0.005→200。
            patience = max(args.early_stopping, int(round(1.0 / spec["learning_rate"])))
            t0 = time.perf_counter()

            # **三种子平均**：bagging_fraction / feature_fraction 让单种子自带抖动，
            # 而岭回归基准没有 —— 第一轮等于让树带着噪声跟一把干净的尺子比。
            # 三个种子各自在内层验证段早停，再把外层预测平均（这也正是要上线的形态）。
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
                    # 各候选的 min_data_in_leaf 不同，而 Dataset 的特征预筛选是按构造时的
                    # min_data_in_leaf 定死的 —— 复用同一个 Dataset 会让后面的候选被前一个的
                    # 阈值筛过的特征集拖累（第一轮就是这么污染的、数已作废）。
                    # 关掉预筛选，并且每个候选/每个种子都重建 Dataset，两道保险。
                    "feature_pre_filter": False,
                    **spec,
                }
                ds_train = lgb.Dataset(x_it, label=m_series[inner_train_rows],
                                       weight=weight[inner_train_rows],
                                       params=params, free_raw_data=False)
                ds_valid = lgb.Dataset(x_iv, label=y_iv, weight=w_iv,
                                       reference=ds_train, params=params, free_raw_data=False)
                booster = lgb.train(
                    params, ds_train, num_boost_round=args.max_rounds,
                    valid_sets=[ds_valid], valid_names=["inner"],
                    callbacks=[lgb.early_stopping(patience, verbose=False)],
                )
                trained = booster.current_iteration()
                best = int(booster.best_iteration or trained)
                best_iters.append(best)
                trained_iters.append(trained)
                # checkpoint 一律取 min(k, trained)，保证每折的路径键完全一致 ——
                # 否则各折早停轮数不同，取交集可能为空（第一轮就是这么崩的）。
                for k in checkpoints:
                    sum_path[str(k)] += booster.predict(x_ov, num_iteration=min(k, trained))
                sum_best += booster.predict(x_ov, num_iteration=best)
                del booster, ds_train, ds_valid

            n = float(args.n_seeds)
            path = {}
            for k in checkpoints:
                path[str(k)] = {**score_parts(y_valid, sum_path[str(k)] / n, w_valid),
                                "effective_iteration": min(k, int(np.mean(trained_iters)))}
            # honest = 各种子在**内层**验证段上早停选出的轮数，外层从未参与选择
            fold["arms"][name] = {
                "kind": "lgbm", "params": spec, "train_rows": len(inner_train_rows),
                "n_seeds": args.n_seeds, "patience": patience,
                "best_iterations": best_iters, "trained_iterations": trained_iters,
                "best_iteration": int(np.mean(best_iters)),
                "hit_round_budget": any(t >= args.max_rounds for t in trained_iters),
                "fit_seconds": float(time.perf_counter() - t0),
                "path": path,
                "honest": {"knob": "/".join(map(str, best_iters)),
                           **score_parts(y_valid, sum_best / n, w_valid)},
            }
            print(f"  fold {index:2d} {name:15s} best_iter={best_iters} "
                  f"R²_m@best={fold['arms'][name]['honest']['r2']:+.5f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
            del sum_path, sum_best

        del x_it, x_iv, x_ov, scaled, z_valid, z_iv
        fold["elapsed_seconds"] = float(time.perf_counter() - started)
        results.append(fold)
        rf = max(v["r2"] for v in fold["arms"]["ridge_full"]["path"].values())
        print(f"fold {index:2d} 完成：ridge_full 峰值 {rf:+.5f}  ({fold['elapsed_seconds']:.0f}s)", flush=True)

    # ------------------------------------------------------------- 汇总
    arm_names = ["ridge_fixed", "ridge_fair", "ridge_full", *names]

    # **headline 用 honest**：容量旋钮（alpha / 轮数）一律在**内层**验证段上选，
    # 外层验证段从未参与选择。curve 只作诊断 —— 它在外层均值上取最大，是乐观有偏的。
    summary: dict[str, Any] = {}
    per_fold: dict[str, list[float]] = {}
    for arm in arm_names:
        arm_keys = sorted(results[0]["arms"][arm]["path"], key=float)
        curve = {k: float(np.mean([f["arms"][arm]["path"][k]["r2"] for f in results]))
                 for k in arm_keys}
        pick = max(curve, key=lambda k: curve[k])
        honest = [f["arms"][arm].get("honest") for f in results]
        has_honest = all(h is not None for h in honest)
        series = [h["r2"] for h in honest] if has_honest else [f["arms"][arm]["path"][pick]["r2"] for f in results]
        if has_honest:
            per_fold[arm] = series
        summary[arm] = {
            "curve": curve,
            "curve_selected": pick,
            "curve_mean_r2": float(np.mean([f["arms"][arm]["path"][pick]["r2"] for f in results])),
            "has_honest_selection": has_honest,
            "honest_knobs": [h["knob"] for h in honest] if has_honest else None,
            "mean_r2": float(np.mean(series)),
            "positive_folds": int(sum(s > 0 for s in series)),
            "A": float(np.mean([(h["A"] if has_honest else f["arms"][arm]["path"][pick]["A"])
                                for h, f in zip(honest, results)])),
            "B": float(np.mean([(h["B"] if has_honest else f["arms"][arm]["path"][pick]["B"])
                                for h, f in zip(honest, results)])),
            "oracle_mean_r2": float(np.mean(
                [max(v["r2"] for v in f["arms"][arm]["path"].values()) for f in results])),
        }
        # pooled：验证段互不重叠，逐折平方和直接相加 → 整个验证区间上的单一 R²_m。
        # 与 mean-of-folds 并列看：后者会被单折异常值拉走（第一轮 fold 3 占了 mean 的 80%）。
        if has_honest:
            summary[arm]["pooled_sse"] = float(sum(h["sse"] for h in honest))
            summary[arm]["pooled_r2"] = float(
                1.0 - summary[arm]["pooled_sse"] / sum(h["sst"] for h in honest))

    # **两个基准都要过**：`ridge_fixed` 零选择方差（alpha 预注册），
    # `ridge_fair` 与 LGBM 的选择流程对等。只赢其中一个都不算数 ——
    # 只赢 ridge_fair 很可能是赢在 alpha 网格太粗导致的选择噪声上。
    baselines = ["ridge_fixed", "ridge_fair"]
    total_sst = float(sum(f["arms"][baselines[0]]["honest"]["sst"] for f in results))
    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in baselines:
        block: dict[str, Any] = {}
        for arm in arm_names:
            if arm == baseline or arm not in per_fold:
                continue      # ridge_full 吃过内层验证段，没有无泄漏选择，不进配对判据
            deltas = [a - b for a, b in zip(per_fold[arm], per_fold[baseline])]
            stats_d = paired_stats(deltas)
            da = (summary[arm]["A"] - summary[baseline]["A"]) / abs(summary[baseline]["A"])
            db = (summary[arm]["B"] - summary[baseline]["B"]) / abs(summary[baseline]["B"])
            pooled_delta = (summary[baseline]["pooled_sse"] - summary[arm]["pooled_sse"]) / total_sst
            block[arm] = {
                **stats_d,
                "pooled_delta": float(pooled_delta),
                "pooled_score_delta": float(pooled_delta * 0.72),
                "score_delta": float(stats_d["mean"] * 0.72),
                "delta_A_pct": da * 100.0,
                "delta_B_pct": db * 100.0,
                # 2·ΔA > ΔB 才涨峰值；本地 ΔA 系统性高估 2.2 倍 → 只折 ΔA
                "discounted_mechanism_ok": bool(2.0 * (da / 2.2) > db),
                # 检出下限：这个设计能分辨的最小效应
                "detection_floor_r2": None if stats_d["se"] is None else float(2.0 * stats_d["se"]),
                "detection_floor_score": None if stats_d["se"] is None else float(2.0 * stats_d["se"] * 0.72),
                "detection_floor_pct_of_public": None if stats_d["se"] is None else float(
                    2.0 * stats_d["se"] * 0.72 / PUBLIC_REFERENCE * 100.0),
                # **预注册判据，机器判**（见 verdict_of 的 docstring）
                "verdict": verdict_of(stats_d, da, db),
            }
        comparisons[baseline] = block

    overall: dict[str, str] = {}
    for arm in names:
        vs = [comparisons[b][arm]["verdict"] for b in baselines if arm in comparisons[b]]
        if not vs or "INSUFFICIENT_FOLDS" in vs:
            overall[arm] = "INSUFFICIENT_FOLDS"
        elif all(v == "PASS" for v in vs):
            overall[arm] = "PASS"
        elif any(v == "FAIL" for v in vs):
            overall[arm] = "FAIL"
        else:
            overall[arm] = "INCONCLUSIVE"

    payload = {
        "question": "LightGBM 能不能在择时块 m_t 上打赢岭回归的线性天花板？",
        "metric": "样本外 R²_m = 1 − ΣW(m−m̂)²/ΣW·m²",
        "score_mapping": "Δscore ≈ 0.72 × ΔR²_m（择时块占 target 方差 72%）",
        "baseline_arm": baseline,
        "decision_rule": (
            "对**每个基准**：PASS 需三条同时成立 —— t ≥ 2；去掉最好一折后 t ≥ 1；2·ΔA/2.2 > ΔB。"
            "t ≤ −2 判 FAIL，其余 INCONCLUSIVE。"
            "总判定（overall）要求对 ridge_fixed 与 ridge_fair **两个基准都 PASS**："
            "前者零选择方差（alpha 预注册 1e8），后者与 LGBM 选择流程对等；"
            "只赢 ridge_fair 很可能是赢在 alpha 网格太粗导致的选择噪声上。"
            "全部由 verdict_of() 计算并写入本文件，不靠人在 md 里下判断。"
        ),
        "baselines": baselines,
        "overall_verdict": overall,
        "leakage_controls": {
            "inner_early_stopping_split": f"训练段尾部 {args.inner_frac:.0%} + embargo {args.embargo}",
            "outer_valid_never_seen_in_training": True,
            "hyperparameter_selection": "岭回归的 alpha 与 LGBM 的轮数一律在内层验证段上选，逐折独立",
            "ridge_fair_sees_same_rows_as_lgbm": True,
            "seed_averaging": f"{args.n_seeds} 个种子各自内层早停后平均外层预测",
        },
        "configuration": {
            "n_folds": args.n_folds, "train_window": train_window, "embargo": args.embargo,
            "inner_frac": args.inner_frac, "total_time_ids": total,
            "alphas": ALPHAS, "max_rounds": args.max_rounds,
            "early_stopping_floor": args.early_stopping,
            "early_stopping_rule": "max(--early-stopping, round(1/lr))",
            "n_seeds": args.n_seeds, "seed": args.seed,
            "public_reference_score": PUBLIC_REFERENCE,
            "candidates": {n: CANDIDATES[n] for n in names},
        },
        "summary": summary,
        "comparisons": comparisons,
        "folds": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- markdown
    lines = [
        "# LightGBM vs 岭回归：择时块 m_t",
        "",
        f"全时间分辨率 {total:,} 个 time_id（**全部 10 个相位**），"
        f"{args.n_folds} 折滚动 + embargo {args.embargo}。",
        "`R²_m = 1 − ΣW(m−m̂)²/ΣW·m²`；换算到总分 `Δscore ≈ 0.72 × ΔR²_m`。",
        "",
        f"**无泄漏口径**：早停集取训练段尾部 {args.inner_frac:.0%} 并与内层训练段隔开 "
        f"{args.embargo} 个 time_id；岭回归的 alpha 与 LGBM 的轮数**都在内层验证段上选**，"
        "外层验证段全程不参与训练与选择。",
        f"`ridge_fair` 与 LGBM 吃**完全相同的行**；`ridge_full` 额外吃满整个训练段（对读 mt_lagged）。",
        f"LGBM 每组 **{args.n_seeds} 个种子**各自内层早停后平均外层预测 —— "
        "削掉 bagging/feature_fraction 的抖动，也是要上线的形态。",
        "",
        "## 预注册判据（由代码判，不由我判）",
        "",
        "```",
        "对每个基准：PASS ⟺ t ≥ 2  且  去掉最好一折后 t ≥ 1  且  2·ΔA/2.2 > ΔB",
        "            FAIL ⟺ t ≤ −2      其余一律 INCONCLUSIVE",
        "总判定       ⟺ 对 ridge_fixed 与 ridge_fair 两个基准都 PASS",
        "```",
        "",
        f"**为什么要两个基准**：`ridge_fixed` 的 alpha 预注册死在 {FIXED_ALPHA:.0e}"
        "（出处是 mt_lagged 已发表的结论，先验，不是在本次外层选的），**零选择方差**；"
        "`ridge_fair` 则与 LGBM 的选择流程对等。alpha 网格 7 个点跨十倍数量级、选错一档代价巨大，"
        "而 LGBM 的轮数是细粒度的 —— 两者选择过程方差不对等，"
        "**只赢 `ridge_fair` 很可能是赢在选择噪声上，不算数**。",
        "",
        "## 总判定",
        "",
        "| 候选 | vs `ridge_fixed` | vs `ridge_fair` | **总判定** |",
        "|---|:--:|:--:|:--:|",
    ] + [
        f"| `{a}` | {comparisons['ridge_fixed'][a]['verdict']} | "
        f"{comparisons['ridge_fair'][a]['verdict']} | **{overall[a]}** |"
        for a in names if a in comparisons["ridge_fixed"]
    ] + [
        "",
        "## 各臂表现",
        "",
        "`平均 R²_m` = **无泄漏**口径（容量旋钮在内层验证段上选）。",
        "后两列是乐观有偏的诊断量，**不能拿来下结论**：`curve 峰值` 在外层均值上选旋钮，",
        "`oracle` 更进一步逐折在外层上选。三者的差距就是「本地选参能偷到多少分」。",
        "",
        "| 臂 | 无泄漏选出的旋钮 | 平均 R²_m | pooled R²_m | 为正折数 | curve 峰值 | 逐折 oracle |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arm_names:
        s = summary[arm]
        if not s["has_honest_selection"]:
            knob, pooled = "—（吃过内层，无资格）", "—"
        else:
            uniq = sorted(set(s["honest_knobs"]))
            knob = "/".join(uniq) if len(uniq) <= 3 else f"逐折不同（{len(uniq)} 种）"
            pooled = f"{s['pooled_r2']:+.5f}"
        lines.append(f"| `{arm}` | {knob} | {s['mean_r2']:+.5f} | {pooled} | "
                     f"{s['positive_folds']}/{len(results)} | {s['curve_mean_r2']:+.5f} | "
                     f"{s['oracle_mean_r2']:+.5f} |")

    def fmt(v, spec=".2f"):
        return "n/a" if v is None else format(v, spec)

    for baseline in baselines:
        lines += ["", f"## 相对 `{baseline}` 的配对 Δ —— **判据是 t，不是 mean 的正负**", "",
                  "| 臂 | 判定 | mean(Δ) | SE | **t** | 中位数 | 正折 | 去掉最好一折 t | pooled Δ | ΔA | ΔB | 机制 |",
                  "|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|"]
        for arm, c in comparisons[baseline].items():
            badge = {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
                     "INSUFFICIENT_FOLDS": "🚫 折数不足"}.get(c["verdict"], "⚠️ 未定")
            lines.append(
                f"| `{arm}` | {badge} | {c['mean']:+.5f} | {fmt(c['se'], '.5f')} | "
                f"**{fmt(c['t'], '+.2f')}** | "
                f"{c['median']:+.5f} | {c['positive_folds']}/{c['n_folds']} | "
                f"{fmt(c['drop_best_t'], '+.2f')} | "
                f"{c['pooled_delta']:+.5f} | {c['delta_A_pct']:+.1f}% | {c['delta_B_pct']:+.1f}% | "
                f"{'✅' if c['discounted_mechanism_ok'] else '❌'} |")

    flat = [c for block in comparisons.values() for c in block.values()]
    floors = [c["detection_floor_score"] for c in flat if c["detection_floor_score"] is not None]
    pcts = [c["detection_floor_pct_of_public"] for c in flat
            if c["detection_floor_pct_of_public"] is not None]
    lines += [
        "",
        (f"**这个设计的检出下限**：t=2 需要 Δscore ≥ {min(floors):.5f}，"
         f"即相对公榜 {PUBLIC_REFERENCE} 至少 **+{min(pcts):.0f}%**。"
         "比这小的真实提升，本地分辨不了 —— 结果落在 INCONCLUSIVE 就是这个意思，"
         "**不等于「没效果」，是「测不出来」**。")
        if floors else "**折数不足以定义标准误，本报告不能用于下结论。**",
        "",
        "## 怎么读",
        "",
        "- **mean(Δ) 为正不算数**，第一轮四个臂里三个 mean 为正但 t 全在 ±1.1 以内。",
        "- **去掉最好一折后 t 仍要 ≥1**：`mt_lagged` 的表面增量当初全部来自单独一折，",
        "  第一轮的 fold 3 也占了 mean 的 80%。",
        "- **pooled Δ 与 mean(Δ) 要同号**。不同号说明存在折间权重效应，两个都别信。",
        "- LGBM 是本项目最大的一次**容量增加**，按 `ab_decomposition` 的规则一律靠 A↑ 起作用 → 重罚。",
        "- 择时块是全相位的，所以这里的结论**不受 ROADMAP #2 相位错配影响**。",
        "",
        "## 轮数 / alpha 曲线（全折均值 R²_m）",
        "",
    ]
    for arm in arm_names:
        curve = summary[arm]["curve"]
        lines.append(f"- `{arm}`：" + "  ".join(f"{k}={v:+.5f}" for k, v in curve.items()))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 各臂（无泄漏口径）===")
    for arm in arm_names:
        s = summary[arm]
        tag = "无泄漏" if s["has_honest_selection"] else "仅曲线"
        pooled = f"pooled={s['pooled_r2']:+.5f}  " if s["has_honest_selection"] else ""
        print(f"  {arm:15s} [{tag}] 平均 R²_m={s['mean_r2']:+.5f}  {pooled}"
              f"(curve {s['curve_mean_r2']:+.5f} @ {s['curve_selected']}, "
              f"oracle {s['oracle_mean_r2']:+.5f})")
    def sfmt(v, spec="+.2f"):
        return "  n/a" if v is None else format(v, spec)

    for baseline in baselines:
        print(f"\n=== 相对 {baseline} 的配对 Δ —— 判据是 t ===")
        for arm, c in comparisons[baseline].items():
            print(f"  {arm:15s} {c['verdict']:<18s} mean={c['mean']:+.5f}  t={sfmt(c['t'])}  "
                  f"中位={c['median']:+.5f}  正折 {c['positive_folds']}/{c['n_folds']}  "
                  f"去最好一折 t={sfmt(c['drop_best_t'])}  pooled={c['pooled_delta']:+.5f}  "
                  f"机制{'✅' if c['discounted_mechanism_ok'] else '❌'}")
    flat2 = [c for block in comparisons.values() for c in block.values()]
    f2 = [c["detection_floor_score"] for c in flat2 if c["detection_floor_score"] is not None]
    p2 = [c["detection_floor_pct_of_public"] for c in flat2
          if c["detection_floor_pct_of_public"] is not None]
    if f2:
        print(f"\n检出下限：Δscore ≥ {min(f2):.5f}（相对公榜 {PUBLIC_REFERENCE} 的 +{min(p2):.0f}%）"
              f" —— INCONCLUSIVE 意思是「测不出来」，不是「没效果」")
    print("\n=== 总判定（ridge_fixed 与 ridge_fair 两个基准都要过）===")
    for arm, v in overall.items():
        print(f"  {arm:15s} {v}")
    passed = [a for a, v in overall.items() if v == "PASS"]
    print(f"结论：{'PASS —— ' + '/'.join(passed) if passed else '无一组过关'}"
          + ("  ⚠️ 折数不足，本次不能下结论" if "INSUFFICIENT_FOLDS" in overall.values() else ""))
    print(f"\n产物：{json_path}\n     {md_path}")


if __name__ == "__main__":
    main()
