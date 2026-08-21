"""函数类探针：在**与生产截面块逐列相同**的输入上，非树的函数类能拿到多少？

## 问的问题

诊断 D 记录了三次失败：`history` 列 40→80 公榜 **0.00%**、`market_asset_panel`
**−7.24%**（ΔA +16.88% 而 **ΔB +49.44%**）、`peer_leadlag` −2.31%/−1.73%。
三次都是「ΔB 涨幅是 ΔA 的三倍」—— 那是**函数类吃不下这种输入**的指纹，
不是「时间维/截面维没有信息」。本脚本换一个函数类，输入一列不改，看能不能拿到东西。

## 为什么是 RFF

随机傅里叶特征 + 岭回归是**光滑、全局、各向同性**的 —— 与 LightGBM 的
轴对齐分段常数在函数空间里尽可能地不同（这正是集成需要的 ρ 小）。
且纯 numpy/scipy、推理端就是一次矩阵乘，若真过门槛，交付成本几乎为零。

## 关键实现约束

1. **永不 materialize 完整 Z**（2.1m × 2048 × 8B = 34 GB）。分块累加
   `ZᵀWZ`(2048²) 与 `ZᵀWy`，Gram 走 **float64**（实测只比 float32 贵 40s/折，
   而 float32 的 Gram 相对误差 5.8e-07 会吃掉预注册阶梯最小的那两档 alpha）。
2. **alpha 在训练折内层选**（后 20% time_id），**绝不看 validation**。
   利用 `ZᵀWZ` 可加：内层两半各累一份，全部 alpha 的内层得分都是闭式，
   不需要重扫数据。
3. **基准强度**：`r` 的分母必须是生产强度的 `e_lgbm`（3 种子 × 480 轮）。
   P8 就是栽在这里 —— 拿 1s160 弱基准算 oracle 得 +6.97%，实测只有 +0.026%，
   差 270 倍，根因就是基准弱了 1.51×。
4. **行对齐**：按 `(fold, time_id, asset_id)` join，不假设顺序；并断言
   自算的 `e_va` 与 cache 里的 `e_target` 数值一致 —— 这一条同时验证了
   fold 版图、采样掩码和截面去均值三处口径。

判据先于结果落盘在 `outputs/experiments/function_class_probe_plan.json`，
其 sha256 记进本脚本的产物。**不搜带宽、不搜 D、不换标签、不加列。**
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import apply_robust_transform, cross_sectional_deviation  # noqa: E402
from lgbm_xs import load_rows  # noqa: E402
from mt_predictability import group_starts  # noqa: E402
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402
from strategies.v3_hybrid.train import stream_history_blocks  # noqa: E402
from v3_production_oof import group_mean, row_slice  # noqa: E402

# ---- 预注册常量（与 function_class_probe_plan.json 一一对应，改这里必须同步改那里）----
PLAN_PATH = _REPO_ROOT / "outputs" / "experiments" / "function_class_probe_plan.json"
CACHE_PATH = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"
FEATURE_COUNT = 200
HISTORY_COUNT = 40
HISTORY_WINDOW = 5
TRAIN_WINDOW = 78_960
EMBARGO = 6
N_FOLDS = 5
SAMPLE_MODULO = 5
SAMPLING = "phase_balanced"
D_RFF = 2048
SEED = 20260821
PCA_COMPONENTS = 64
N_ASSETS = 15
ALPHA_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
INNER_VALID_FRACTION = 0.2
BANDWIDTH_SUBSAMPLE = 20_000
GRAM_CHUNK = 50_000
GATE_RHO_MARGIN = 0.05
GATE_MIN_BLEND_GAIN = 0.03
GATE_MIN_POSITIVE_FOLDS = 4


# ------------------------------------------------------------------ 指标

def weighted_ic(label: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> tuple[float, float, float, float]:
    """返回 (IC, A, B, D)。IC = A/sqrt(B·D)，与 src/metric.py 的 peak = A²/B 同族。"""
    a = float(np.dot(weight * label, pred))
    b = float(np.dot(weight * pred, pred))
    d = float(np.dot(weight * label, label))
    ic = a / np.sqrt(b * d) if b > 0 and d > 0 else 0.0
    return ic, a, b, d


def weighted_corr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """加权相关 —— 集成公式用的就是这个内积，不能用普通 Pearson。"""
    num = float(np.dot(weight * x, y))
    den = np.sqrt(float(np.dot(weight * x, x)) * float(np.dot(weight * y, y)))
    return num / den if den > 0 else 0.0


def blend_gain_ic(r: float, rho: float) -> float:
    """两分量最优配比后的 IC 相对增益。零增益边界恰好是 r = ρ。

    IC²_blend / IC²_1 = (1 + r² − 2ρr)/(1 − ρ²)   （多元相关的标准式）
    """
    if abs(rho) >= 1.0:
        return 0.0
    ratio = (1.0 + r * r - 2.0 * rho * r) / (1.0 - rho * rho)
    return float(np.sqrt(max(ratio, 0.0)) - 1.0)


# ------------------------------------------------------------------ RFF

def rff_gram(design: np.ndarray, weight: np.ndarray, label: np.ndarray,
             proj: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """分块累加 ZᵀWZ 与 ZᵀWy，**从不 materialize 完整 Z**。

    `Z = sqrt(2/D)·cos(X·proj + phase)`。返回 (ZtZ, Zty, Σw·y²)。
    """
    d = proj.shape[1]
    gram = np.zeros((d, d), dtype=np.float64)
    rhs = np.zeros(d, dtype=np.float64)
    scale = np.sqrt(2.0 / d)
    for start in range(0, len(design), GRAM_CHUNK):
        stop = min(start + GRAM_CHUNK, len(design))
        z = np.cos(design[start:stop] @ proj + phase, dtype=np.float32)
        z = z.astype(np.float64) * scale
        zw = z * weight[start:stop, None]
        gram += zw.T @ z
        rhs += zw.T @ label[start:stop]
        del z, zw
    return gram, rhs, float(np.dot(weight * label, label))


def linear_gram(design: np.ndarray, weight: np.ndarray,
                label: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """阴性对照臂的正规方程，分块累加，口径与 rff_gram 完全一致。"""
    d = design.shape[1]
    gram = np.zeros((d, d), dtype=np.float64)
    rhs = np.zeros(d, dtype=np.float64)
    for start in range(0, len(design), GRAM_CHUNK):
        stop = min(start + GRAM_CHUNK, len(design))
        block = design[start:stop].astype(np.float64)
        blockw = block * weight[start:stop, None]
        gram += blockw.T @ block
        rhs += blockw.T @ label[start:stop]
        del block, blockw
    return gram, rhs, float(np.dot(weight * label, label))


def linear_predict(design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.empty(len(design), dtype=np.float64)
    for start in range(0, len(design), GRAM_CHUNK):
        stop = min(start + GRAM_CHUNK, len(design))
        out[start:stop] = design[start:stop].astype(np.float64) @ beta
    return out


def rff_predict(design: np.ndarray, proj: np.ndarray, phase: np.ndarray,
                beta: np.ndarray) -> np.ndarray:
    d = proj.shape[1]
    scale = np.sqrt(2.0 / d)
    out = np.empty(len(design), dtype=np.float64)
    for start in range(0, len(design), GRAM_CHUNK):
        stop = min(start + GRAM_CHUNK, len(design))
        z = np.cos(design[start:stop] @ proj + phase, dtype=np.float32)
        out[start:stop] = (z.astype(np.float64) * scale) @ beta
        del z
    return out


def solve_ridge(gram: np.ndarray, rhs: np.ndarray, alpha_rel: float) -> np.ndarray:
    """alpha 用 trace(gram)/d 归一化 ⟹ 阶梯与行数、权重尺度无关。"""
    d = gram.shape[0]
    penalty = alpha_rel * float(np.trace(gram)) / d
    return np.linalg.solve(gram + penalty * np.eye(d), rhs)


def pick_alpha(gram_a: np.ndarray, rhs_a: np.ndarray,
               gram_b: np.ndarray, rhs_b: np.ndarray, dss_b: float) -> tuple[float, list[dict]]:
    """内层选 alpha：前 80% 拟合、后 20% 打分。闭式，不重扫数据。

    在内层验证段上：A = β·rhs_b、B = βᵀ·gram_b·β ⟹ peak = A²/(B·D) 全部是二次型。
    """
    trace = []
    best = (None, -np.inf)
    for alpha in ALPHA_LADDER:
        beta = solve_ridge(gram_a, rhs_a, alpha)
        a = float(np.dot(beta, rhs_b))
        b = float(beta @ gram_b @ beta)
        peak = (a * a / (b * dss_b)) if b > 0 and dss_b > 0 else -np.inf
        trace.append({"alpha_relative": alpha, "inner_peak": peak})
        if peak > best[1]:
            best = (alpha, peak)
    return float(best[0]), trace


def median_bandwidth(design: np.ndarray, rng: np.random.Generator) -> float:
    """中位数启发式：σ = 成对欧氏距离的中位数。唯一决策点，不搜索。"""
    take = min(BANDWIDTH_SUBSAMPLE, len(design))
    idx = rng.choice(len(design), size=take, replace=False)
    sample = design[np.sort(idx)].astype(np.float64)
    pairs = rng.choice(take, size=(4096, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    diff = sample[pairs[:, 0]] - sample[pairs[:, 1]]
    return float(np.median(np.sqrt(np.einsum("ij,ij->i", diff, diff))))


# ------------------------------------------------------------------ 主流程

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="function_class_probe")
    p.add_argument("--stage1", action="store_true",
                   help="只跑 fold 0（预注册的降级路径：设计矩阵构建超时才用）")
    p.add_argument("--bandwidth-diagnostic", action="store_true",
                   help="在 fold 0 额外跑 {0.5x, 2.0x} 两格。诊断用，**不作选参依据**")
    p.add_argument("--dry-run", action="store_true", help="只打印配置与判据，不读数据")
    p.add_argument("--force", action="store_true",
                   help="允许覆盖已有产物（CLAUDE.md §5.10：不得静默覆盖）")
    return p.parse_args()


def build_fold_design(features, target, weight, time_ids, asset_ids,
                      train_ids, valid_ids, data_root, history_cache):
    """逐折复刻 v3_production_oof.py 的截面块设计矩阵构造（不训练任何树）。

    每一步都只在训练段内拟合：robust_transform_fit / select_features / history 选列。
    """
    tr, va = row_slice(time_ids, train_ids), row_slice(time_ids, valid_ids)
    y_tr, y_va = target[tr], target[va]
    w_tr, w_va = weight[tr], weight[va]
    tid_tr, tid_va = time_ids[tr], time_ids[va]
    aid_tr, aid_va = asset_ids[tr], asset_ids[va]
    tr_starts, va_starts = group_starts(tid_tr), group_starts(tid_va)
    tr_counts = np.diff(np.r_[tr_starts, len(tid_tr)]).astype(np.float64)
    va_counts = np.diff(np.r_[va_starts, len(tid_va)]).astype(np.float64)

    transformed_train, stats = robust_transform_fit(features[tr].copy())
    transformed_valid = features[va].copy()
    apply_robust_transform(transformed_valid, stats["lower"], stats["upper"],
                           stats["center"], stats["scale"])

    e_tr = y_tr - group_mean(y_tr, tr_starts, tr_counts)
    e_va = y_va - group_mean(y_va, va_starts, va_counts)
    unit = np.ones_like(e_tr)
    xs_selected = select_features(transformed_train, e_tr, unit, FEATURE_COUNT)
    xs_tr = cross_sectional_deviation(transformed_train[:, xs_selected].copy(), tid_tr)
    xs_va = cross_sectional_deviation(transformed_valid[:, xs_selected].copy(), tid_va)

    history_positions = np.sort(
        select_features(xs_tr, e_tr, unit, HISTORY_COUNT).astype(np.int64))
    history_names = [f"feature_{int(i):03d}" for i in xs_selected[history_positions]]
    history_stats = tuple(stats[key][xs_selected[history_positions]]
                          for key in ("lower", "upper", "center", "scale"))
    key = tuple(history_names)
    if key not in history_cache:
        history_cache.clear()          # 一次只留一份：4 × 2.65m × 40 × 4B = 1.7 GB
        history_cache[key] = stream_history_blocks(
            Path(data_root), SAMPLE_MODULO, SAMPLING, history_names,
            history_stats, HISTORY_WINDOW)
    all_history = history_cache[key]

    design_tr = np.column_stack([xs_tr, *[block[tr] for block in all_history]])
    design_va = np.column_stack([xs_va, *[block[va] for block in all_history]])
    del transformed_train, transformed_valid, xs_tr, xs_va
    gc.collect()
    return {"design_tr": design_tr, "design_va": design_va,
            "e_tr": e_tr, "e_va": e_va, "w_tr": w_tr, "w_va": w_va, "y_va": y_va,
            "tid_va": tid_va, "aid_va": aid_va, "aid_tr": aid_tr,
            "va_starts": va_starts, "va_counts": va_counts, "tid_tr": tid_tr}


def standardise(design_tr: np.ndarray, design_va: np.ndarray,
                aid_tr: np.ndarray, aid_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """训练段拟合的标准化 + asset one-hot。

    asset_id 对树是 categorical，对光滑核必须 one-hot —— 原样喂整数会让核
    把 asset 3 和 asset 4 当成「相邻」，那是无意义的几何。
    """
    mean = design_tr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = design_tr.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd <= 0] = 1.0
    n_assets = int(max(aid_tr.max(), aid_va.max())) + 1
    if n_assets != N_ASSETS:
        raise AssertionError(f"asset 数 {n_assets} != 常量 N_ASSETS={N_ASSETS}，PCA 臂的切片会错位")
    # ⚠️ 内存：`np.column_stack([(design-mean)/sd, onehot])` 会同时活着**两份**
    # 完整设计矩阵（各 3.1 GB）。改成先开好目标数组、再逐块就地写，峰值省一整份。
    out = []
    for design, aid in ((design_tr, aid_tr), (design_va, aid_va)):
        rows, cols = design.shape
        buf = np.empty((rows, cols + n_assets), dtype=np.float32)
        for start in range(0, rows, 200_000):
            stop = min(start + 200_000, rows)
            np.subtract(design[start:stop], mean, out=buf[start:stop, :cols])
            np.divide(buf[start:stop, :cols], sd, out=buf[start:stop, :cols])
        buf[:, cols:] = 0.0
        buf[np.arange(rows), cols + aid] = 1.0
        out.append(buf)
    return out[0], out[1]


def fit_arm(name: str, design_tr, design_va, e_tr, w_tr, tid_tr,
            proj: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """一个臂：内层选 alpha → 整训练折重拟合 → 预测 validation。

    `proj=None` 表示线性对照臂（阴性对照：若它也过门槛，说明测到的不是函数类）。
    """
    if proj is None:
        # ⚠️ 同样分块：整份 `d.astype(np.float64)` 是 2.1m × 375 × 8B = 6.3 GB，还要两份。
        gram_fn = linear_gram
        predict_fn = linear_predict
    else:
        phase = rng.uniform(0.0, 2.0 * np.pi, proj.shape[1]).astype(np.float32)
        gram_fn = lambda d, w, y: rff_gram(d, w, y, proj, phase)  # noqa: E731
        predict_fn = lambda d, beta: rff_predict(d, proj, phase, beta)  # noqa: E731

    # 内层切分：训练段**后** INNER_VALID_FRACTION 的 time_id（时序，不随机）。
    # ⚠️ tid_tr 有序 ⟹ 用**切片**而不是布尔掩码：后者会整份拷贝 3.1 GB 的设计矩阵，
    # 而且要拷两次（inner 与 ~inner），30 GB 无 swap 的机器扛不住。
    uniq = np.unique(tid_tr)
    cut_id = uniq[int(len(uniq) * (1.0 - INNER_VALID_FRACTION))]
    cut = int(np.searchsorted(tid_tr, cut_id, side="left"))
    if not (0 < cut < len(tid_tr)):
        raise AssertionError("内层切分为空 —— 训练段 time_id 可能未排序")
    gram_a, rhs_a, _ = gram_fn(design_tr[:cut], w_tr[:cut], e_tr[:cut])
    gram_b, rhs_b, dss_b = gram_fn(design_tr[cut:], w_tr[cut:], e_tr[cut:])
    alpha, trace = pick_alpha(gram_a, rhs_a, gram_b, rhs_b, dss_b)
    beta = solve_ridge(gram_a + gram_b, rhs_a + rhs_b, alpha)
    del gram_a, gram_b
    gc.collect()
    return predict_fn(design_va, beta), {"arm": name, "alpha_relative": alpha,
                                         "alpha_trace": trace}


def main() -> None:
    args = parse_args()
    plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    print(f"预注册判据 {PLAN_PATH.name}  sha256={plan_sha}")
    print(f"臂：{[a['name'] for a in plan['arms']]}   门槛 {len(plan['gates'])} 道")
    for gate in plan["gates"]:
        print(f"  {gate['id']}. {gate['name']}: {gate['rule']}")
    print(f"D={D_RFF} seed={SEED} alpha 阶梯={ALPHA_LADDER} PCA={PCA_COMPONENTS}")
    if args.dry_run:
        print("\n--dry-run：未读数据。")
        return

    started = time.perf_counter()
    # ⚠️ CLAUDE.md §8.4：各策略都有同名的 features/train/main。`stream_history_blocks`
    # 内部是裸的 `from history import AssetHistory`，只有 v3_hybrid 有这个模块。
    # 必须 **append** 而不是 insert —— insert 会把 v3_hybrid 的 features/train 顶到
    # v1_ridge 前面，本脚本上面 import 的那两个就换了实现。与
    # v3_production_oof.py:264 同一惯例。
    v3_path = str(_REPO_ROOT / "strategies" / "v3_hybrid")
    if v3_path not in sys.path:
        sys.path.append(v3_path)

    from src.oof_cache import assert_reproducible_cache
    assert_reproducible_cache(CACHE_PATH)
    cache = np.load(CACHE_PATH)
    print(f"基准 cache：{CACHE_PATH.name}（e_lgbm = 3 种子 × 480 轮生产强度）")

    rows = load_rows(Path(args.data_root), SAMPLE_MODULO, SAMPLING)
    features, target = rows["features"], rows["target"]
    weight, time_ids, asset_ids = rows["weight"], rows["time_id"], rows["asset_id"]
    order = np.argsort(time_ids, kind="stable")
    features, target = features[order], target[order]
    weight, time_ids, asset_ids = weight[order], time_ids[order], asset_ids[order]
    unique_time_ids = np.unique(time_ids)
    folds = rolling_time_folds(unique_time_ids, N_FOLDS, TRAIN_WINDOW, EMBARGO)
    print(f"{len(target):,} 行 / {len(unique_time_ids):,} 个采样 time_id / {len(folds)} 折"
          f"（{time.perf_counter()-started:.0f}s）", flush=True)

    # cache 侧按 (time_id, asset_id) 建索引，供逐折 join
    cache_key = cache["time_id"] * 16 + cache["asset_id"]
    cache_order = np.argsort(cache_key, kind="stable")
    cache_key_sorted = cache_key[cache_order]

    rng = np.random.default_rng(SEED)
    history_cache: dict = {}
    fold_rows: list[dict] = []
    selected = folds[:1] if args.stage1 else folds
    for index, (train_ids, valid_ids) in enumerate(selected):
        t0 = time.perf_counter()
        f = build_fold_design(features, target, weight, time_ids, asset_ids,
                              train_ids, valid_ids, args.data_root, history_cache)
        design_tr, design_va = standardise(f["design_tr"], f["design_va"],
                                           f["aid_tr"], f["aid_va"])
        del f["design_tr"], f["design_va"]
        gc.collect()
        print(f"fold {index}: train {len(f['e_tr']):,} / valid {len(f['e_va']):,}，"
              f"设计 {design_tr.shape[1]} 列（{time.perf_counter()-t0:.0f}s）", flush=True)

        # ---- join：拿到同样这些行上的生产 e_lgbm 与 e_target
        probe_key = f["tid_va"] * 16 + f["aid_va"]
        pos = np.searchsorted(cache_key_sorted, probe_key)
        if pos.max() >= len(cache_key_sorted) or not np.array_equal(
                cache_key_sorted[np.minimum(pos, len(cache_key_sorted) - 1)], probe_key):
            raise AssertionError(f"fold {index}: validation 行在基准 cache 里找不到对应行")
        take = cache_order[pos]
        e_lgbm = cache["e_lgbm"][take]
        # ⚠️ **不要**拿 cache 的 `e_target` 做对齐检查 —— 它是全 NaN。
        # `v3_production_oof.py:511` 把它写成占位 `e_tr[:0]`，紧接着 512 行
        # `if name == "e_target": continue` 又跳过赋值 ⟹ 那一列从初始化的 NaN
        # 起就没被写过。`src/oof_cache.py:19` 仍把它列在 COMPONENT_COLUMNS 里，
        # 是个埋着没爆的雷（当前没有任何实验消费它）。
        # 改用**确实写入了**的 target/weight 对齐：两者都是同一份 parquet 的
        # float64 拷贝，逐位相同才算 join 正确。
        y_drift = float(np.max(np.abs(cache["target"][take] - f["y_va"])))
        w_drift = float(np.max(np.abs(cache["weight"][take] - f["w_va"])))
        if not (y_drift == 0.0 and w_drift == 0.0):
            raise AssertionError(
                f"fold {index}: join 后 target 差 {y_drift:.3e}、weight 差 {w_drift:.3e}（应为 0）"
                " ⟹ fold 版图 / 采样掩码 / 行序三者之一不同口径，比较无效")
        if not np.all(cache["fold"][take] == index):
            raise AssertionError(f"fold {index}: join 到的行 fold 编号不一致")
        print(f"  join 通过：{len(take):,} 行，target/weight 逐位相同", flush=True)

        sigma = median_bandwidth(design_tr, rng)
        n_in = design_tr.shape[1]
        arms: dict[str, np.ndarray | None] = {
            "linear": None,
            "rff_full": (rng.standard_normal((n_in, D_RFF)) / sigma).astype(np.float32),
        }
        # PCA 臂：在 360 个数值列上取训练段前 64 主成分，再接回 asset one-hot
        # ⚠️ 内存：`numeric.astype(np.float64)` 是 2.1m × 360 × 8B = 6 GB。
        # 协方差只需要 360×360，逐块累加即可，峰值降到一个 chunk。
        n_num = design_tr.shape[1] - N_ASSETS
        cov = np.zeros((n_num, n_num), dtype=np.float64)
        for start in range(0, len(design_tr), 200_000):
            block = design_tr[start:min(start + 200_000, len(design_tr)), :n_num]
            cov += block.T.astype(np.float64) @ block.astype(np.float64)
        cov /= len(design_tr)
        _, evecs = np.linalg.eigh(cov)
        basis = np.ascontiguousarray(evecs[:, -PCA_COMPONENTS:], dtype=np.float32)
        del cov, evecs
        pca_tr = np.empty((len(design_tr), PCA_COMPONENTS + N_ASSETS), dtype=np.float32)
        pca_va = np.empty((len(design_va), PCA_COMPONENTS + N_ASSETS), dtype=np.float32)
        for src, dst in ((design_tr, pca_tr), (design_va, pca_va)):
            for start in range(0, len(src), 200_000):
                stop = min(start + 200_000, len(src))
                dst[start:stop, :PCA_COMPONENTS] = src[start:stop, :n_num] @ basis
                dst[start:stop, PCA_COMPONENTS:] = src[start:stop, n_num:]
        pc_sd = np.maximum(pca_tr[:, :PCA_COMPONENTS].std(axis=0), 1e-6)
        pca_tr[:, :PCA_COMPONENTS] /= pc_sd
        pca_va[:, :PCA_COMPONENTS] /= pc_sd
        sigma_pca = median_bandwidth(pca_tr, rng)

        row: dict = {"fold": index, "n_valid": int(len(take)),
                     "sigma": sigma, "sigma_pca": sigma_pca, "arms": {}}
        ic_base, _, _, _ = weighted_ic(f["e_va"], e_lgbm, f["w_va"])
        row["ic_e_lgbm"] = ic_base
        for name, proj in arms.items():
            t1 = time.perf_counter()
            pred, info = fit_arm(name, design_tr, design_va, f["e_tr"], f["w_tr"],
                                 f["tid_tr"], proj, rng)
            # 复刻生产：validation 上投影成逐 time_id 无权零均值（v3_production_oof.py:461）
            pred = pred - group_mean(pred, f["va_starts"], f["va_counts"])
            ic, _, _, _ = weighted_ic(f["e_va"], pred, f["w_va"])
            rho = weighted_corr(pred, e_lgbm, f["w_va"])
            r = ic / ic_base if ic_base != 0 else 0.0
            row["arms"][name] = {**info, "ic": ic, "r": r, "rho": rho,
                                 "blend_gain_ic": blend_gain_ic(r, rho),
                                 "seconds": time.perf_counter() - t1}
            print(f"  {name:10s} IC={ic:+.5f} r={r:+.4f} rho={rho:+.4f} "
                  f"blend={100*blend_gain_ic(r, rho):+.2f}%  alpha={info['alpha_relative']:.0e} "
                  f"({time.perf_counter()-t1:.0f}s)", flush=True)

        # PCA 臂单独跑（输入不同，不能共用 design_tr）
        t1 = time.perf_counter()
        proj_pca = (rng.standard_normal((pca_tr.shape[1], D_RFF)) / sigma_pca).astype(np.float32)
        pred, info = fit_arm("rff_pca64", pca_tr, pca_va, f["e_tr"], f["w_tr"],
                             f["tid_tr"], proj_pca, rng)
        pred = pred - group_mean(pred, f["va_starts"], f["va_counts"])
        ic, _, _, _ = weighted_ic(f["e_va"], pred, f["w_va"])
        rho = weighted_corr(pred, e_lgbm, f["w_va"])
        r = ic / ic_base if ic_base != 0 else 0.0
        row["arms"]["rff_pca64"] = {**info, "ic": ic, "r": r, "rho": rho,
                                    "blend_gain_ic": blend_gain_ic(r, rho),
                                    "seconds": time.perf_counter() - t1}
        print(f"  {'rff_pca64':10s} IC={ic:+.5f} r={r:+.4f} rho={rho:+.4f} "
              f"blend={100*blend_gain_ic(r, rho):+.2f}%  alpha={info['alpha_relative']:.0e} "
              f"({time.perf_counter()-t1:.0f}s)", flush=True)

        fold_rows.append(row)
        del design_tr, design_va, pca_tr, pca_va, arms, f
        gc.collect()

    report = build_report(fold_rows, plan_sha, args, time.perf_counter() - started)
    write_outputs(Path(args.output_dir), args.label, report, force=args.force)


def build_report(fold_rows: list[dict], plan_sha: str, args, elapsed: float) -> dict:
    """按预注册的五道门槛逐臂机判。"""
    arm_names = ["linear", "rff_full", "rff_pca64"]
    summary: dict = {}
    linear_gate1 = False
    for name in arm_names:
        rs = np.array([row["arms"][name]["r"] for row in fold_rows])
        rhos = np.array([row["arms"][name]["rho"] for row in fold_rows])
        gains = np.array([row["arms"][name]["blend_gain_ic"] for row in fold_rows])
        pooled_r, pooled_rho = float(rs.mean()), float(rhos.mean())
        pooled_gain = blend_gain_ic(pooled_r, pooled_rho)
        positive = int((rs > rhos).sum())
        if len(gains) > 1:
            keep = np.argsort(gains)[:-1]
            drop_best = blend_gain_ic(float(rs[keep].mean()), float(rhos[keep].mean()))
        else:
            drop_best = float("nan")
        gates = {
            "1_pooled_r_exceeds_rho_plus_margin": bool(pooled_r > pooled_rho + GATE_RHO_MARGIN),
            "2_implied_blend_gain_at_least_3pct_ic": bool(pooled_gain >= GATE_MIN_BLEND_GAIN),
            "3_at_least_4_of_5_folds_r_exceeds_rho": bool(positive >= min(GATE_MIN_POSITIVE_FOLDS, len(rs))),
            "4_survives_drop_best_fold": bool(drop_best >= GATE_MIN_BLEND_GAIN) if len(gains) > 1 else None,
        }
        if name == "linear":
            linear_gate1 = gates["1_pooled_r_exceeds_rho_plus_margin"]
        summary[name] = {"pooled_r": pooled_r, "pooled_rho": pooled_rho,
                         "pooled_blend_gain_ic": pooled_gain, "positive_folds": positive,
                         "n_folds": len(rs), "drop_best_blend_gain_ic": drop_best,
                         "gates": gates}
    for name in ("rff_full", "rff_pca64"):
        summary[name]["gates"]["5_linear_control_does_not_pass_gate_1"] = bool(not linear_gate1)
        summary[name]["passed"] = all(v for v in summary[name]["gates"].values() if v is not None)
    summary["linear"]["passed"] = None          # 对照臂不参与裁决
    verdict = "PASS" if any(summary[n].get("passed") for n in ("rff_full", "rff_pca64")) else "FAIL"
    if linear_gate1:
        verdict = "INVALID_LINEAR_CONTROL_PASSED"
    return {"experiment": "function_class_probe", "plan_sha256": plan_sha,
            "baseline_cache": CACHE_PATH.name, "stage1_only": bool(args.stage1),
            "config": {"D_RFF": D_RFF, "seed": SEED, "pca_components": PCA_COMPONENTS,
                       "alpha_ladder": list(ALPHA_LADDER), "n_folds": len(fold_rows),
                       "train_window": TRAIN_WINDOW, "embargo": EMBARGO,
                       "sample_modulo": SAMPLE_MODULO, "sampling": SAMPLING},
            "folds": fold_rows, "summary": summary, "verdict": verdict,
            "elapsed_seconds": elapsed}


def write_outputs(output_dir: Path, label: str, report: dict, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in (output_dir / f"{label}.json", output_dir / f"{label}.md") if p.exists()]
    if existing and not force:
        raise SystemExit(f"产物已存在：{existing}；要覆盖请显式加 --force（CLAUDE.md §5.10）")
    (output_dir / f"{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# 函数类探针（`{label}`）", "",
             f"> 预注册判据 sha256 `{report['plan_sha256']}`，先于结果落盘。",
             f"> 基准：`{report['baseline_cache']}` 的 `e_lgbm`（3 种子 × 480 轮生产强度）。", ""]
    if report["stage1_only"]:
        lines += ["⚠️ **Stage 1 单折**（预注册的降级路径），不构成五折裁决。", ""]
    lines += ["| 臂 | pooled r | pooled ρ | 隐含集成增益 (IC) | r>ρ 折数 | 去最好折 | 判定 |",
              "|---|--:|--:|--:|--:|--:|:--:|"]
    for name in ("rff_full", "rff_pca64", "linear"):
        s = report["summary"][name]
        mark = {True: "✅", False: "❌", None: "对照"}[s["passed"]]
        drop = "—" if not np.isfinite(s["drop_best_blend_gain_ic"]) else f"{100*s['drop_best_blend_gain_ic']:+.2f}%"
        lines.append(f"| `{name}` | {s['pooled_r']:+.4f} | {s['pooled_rho']:+.4f} | "
                     f"**{100*s['pooled_blend_gain_ic']:+.2f}%** | {s['positive_folds']}/{s['n_folds']} | "
                     f"{drop} | {mark} |")
    lines += ["", "> 零增益边界是 `r = ρ`。`r ≤ ρ` ⟹ 集成增益恰好为 0（CLAUDE.md §8.6 的精确形式）。", ""]
    for name in ("rff_full", "rff_pca64"):
        lines += [f"### `{name}` 逐门槛", ""]
        for gate, ok in report["summary"][name]["gates"].items():
            lines.append(f"- {'✅' if ok else '❌'} {gate}")
        lines.append("")
    lines += ["### 逐折", "", "| fold | IC(e_lgbm) | " +
              " | ".join(f"r({n})" for n in ("linear", "rff_full", "rff_pca64")) + " | " +
              " | ".join(f"ρ({n})" for n in ("linear", "rff_full", "rff_pca64")) + " |",
              "|---|--:|" + "--:|" * 6]
    for row in report["folds"]:
        lines.append(f"| {row['fold']} | {row['ic_e_lgbm']:+.5f} | " +
                     " | ".join(f"{row['arms'][n]['r']:+.4f}" for n in ("linear", "rff_full", "rff_pca64")) +
                     " | " +
                     " | ".join(f"{row['arms'][n]['rho']:+.4f}" for n in ("linear", "rff_full", "rff_pca64")) + " |")
    lines += ["", f"## 裁决：{report['verdict']}", ""]
    (output_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / (label + '.json')}\nwrote {output_dir / (label + '.md')}")


if __name__ == "__main__":
    main()
