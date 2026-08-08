"""P0 细粒度验证：按 time_id 滚动切 fold + embargo。

三种用法：

1. 单配置（默认，与本次改动前逐位一致）——写 walk_forward_rolling.{json,md}
       OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
         experiments/walk_forward_rolling.py

2. 配对 A/B（P0-2）——同一批 fold 上跑多个具名配置，只看每 fold 的 Δ。
   绝对分的 fold 间离散度来自行情本身（10 折从 2.6e-4 到 2.6e-3，SE 2.1e-4），
   加 fold 数压不下去；配对减法才能把行情共同项消掉。写 ab_<label>.{json,md}
       ... --configs baseline,scale_auto --label scale_auto

   自检（配对机制本身有没有错）：同一配置跑两遍且关掉 fit 缓存，Δ 必须恒为 0
       ... --configs baseline,baseline --disable-fit-cache --label selfcheck

3. 噪声地板（P0-3）——同一个 A/B 换一套 fold 边界再跑一遍，
   两次 mean(Δ) 的差就是检出下限
       ... --configs baseline,scale_auto --label scale_auto_offhalf --fold-offset half
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.io import FEATURE_COLUMNS, train_files
from src.metric import weighted_zero_mean_r2
from src.validation import rolling_fold_chunk_size, rolling_time_folds
from train import fit_model, predict_array

READ_COLUMNS = ["time_id", "asset_id", "weight", *FEATURE_COLUMNS, "target"]

# 生产训练段的采样后 time_id 数：约 40 万原始 time_id / sample_modulo 5。
# fold 的 alpha 按窗口比例缩放，保证每行的正则强度与生产一致。
PROD_SAMPLED_WINDOW = 400_000 // 5

AUTO_INNER = "auto_inner"

# 一个配置 = BASE_CONFIG 加上若干覆盖项。配置名进 git、进报告 —— 跑过的实验
# 必须能凭名字回溯到具体口径（伤疤清单 #2：报告结论与实际代码没有机械联系）。
BASE_CONFIG: dict[str, object] = {
    "feature_count": 200,
    "ridge_alpha": 2_000_000.0,  # 生产口径；运行时按 train_window 再缩放
    "prediction_scale": 0.5,  # float，或 AUTO_INNER（在内层 fold 上估 a*）
    "prediction_clip": 0.5,
    "zero_intercept": False,
    "design_basis": "raw_dev",  # 或 mean_dev：[截面均值 ‖ deviation]
    "market_alpha_ratio": 1.0,  # 择时分量的正则倍数，只在 mean_dev 下生效
    "cross_sectional_scaling": "none",  # 或 std：deviation 再除以截面标准差
    # 历史 A/B 产物使用旧 LSQR 停止条件；锁死在配置中，避免求解器修复
    # 被误算成某个建模旋钮的收益。新求解器应注册独立配置比较。
    "ridge_tol": 1e-4,
    "ridge_max_iter": 100,
}

CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {},  # = 已提交的 walk_forward_rolling.json 那一版口径
    "production_legacy": {"prediction_scale": 1.13},
    "production_strict": {
        "prediction_scale": 1.13,
        "ridge_tol": 1e-8,
        "ridge_max_iter": 2000,
    },
    "scale_auto": {"prediction_scale": AUTO_INNER},
    "zero_intercept": {"zero_intercept": True},
    "strict_solver": {"ridge_tol": 1e-8, "ridge_max_iter": 2000},
    # 固定 scale 扫描：与 baseline 共用同一次拟合 → 噪声地板 ~1e-5，最灵敏的一类 A/B。
    # 公榜两点定抛物线算出最优 scale ≈1.13，远大于上线的 0.6424，本地要复核。
    "scale075": {"prediction_scale": 0.75},
    "scale100": {"prediction_scale": 1.0},
    "scale125": {"prediction_scale": 1.25},
    "scale150": {"prediction_scale": 1.5},
    # alpha 扫描：最优 scale 显著大于 1，是「正则过重、预测被压扁」的典型症状
    "alpha_quarter": {"ridge_alpha": 500_000.0},
    "alpha_half": {"ridge_alpha": 1_000_000.0},
    "alpha_double": {"ridge_alpha": 4_000_000.0},
    # 特征数扫描：预选是按单变量相关性挑的，看看这一步到底在帮忙还是在扔信息
    "feat50": {"feature_count": 50},
    "feat100": {"feature_count": 100},
    "feat300": {"feature_count": 300},
    "feat323": {"feature_count": 323},  # = 不做预选
    # 生产候选：去掉预选 × 调高 scale。scale 是后处理旋钮，四个 f323_* 共用同一次拟合。
    "f323_s075": {"feature_count": 323, "prediction_scale": 0.75},
    "f323_s100": {"feature_count": 323, "prediction_scale": 1.0},
    "f323_s125": {"feature_count": 323, "prediction_scale": 1.25},
    # P2-1：换成 [截面均值 ‖ deviation] 基底，对择时分量单独加罚。
    # r=1 不等于 baseline —— 基底一换，同一个 alpha 罚的东西就变了。
    "meandev_r1": {"design_basis": "mean_dev"},
    "meandev_r3": {"design_basis": "mean_dev", "market_alpha_ratio": 3.0},
    "meandev_r10": {"design_basis": "mean_dev", "market_alpha_ratio": 10.0},
    "meandev_r30": {"design_basis": "mean_dev", "market_alpha_ratio": 30.0},
}

# 过度收缩联合网格。alpha 太大、预选太狠、scale 太小这三个发现都在修同一个毛病
# （模型被压得太扁），收益不可加，必须放在同一个网格里比。
# scale 是后处理旋钮 → 每个 (feature_count, alpha) 只拟合一次，18 个臂只要 6 次拟合。
for _fc in (200, 323):
    for _alpha, _atag in ((250_000.0, "a8"), (500_000.0, "a4"),
                          (1_000_000.0, "a2"), (2_000_000.0, "a1"),
                          (4_000_000.0, "x2"), (8_000_000.0, "x4")):
        for _scale in (0.5, 0.75, 1.0, 1.25):
            CONFIGS[f"g{_fc}_{_atag}_s{int(_scale * 100):03d}"] = {
                "feature_count": _fc,
                "ridge_alpha": _alpha,
                "prediction_scale": _scale,
            }
            # 截面标准差归一化：固定容量的表示变换（列数不变），按 A/B 分解的规则
            # 属于「不靠 A↑」那一类，迁移性预期更好。
            CONFIGS[f"x{_fc}_{_atag}_s{int(_scale * 100):03d}"] = {
                "feature_count": _fc,
                "ridge_alpha": _alpha,
                "prediction_scale": _scale,
                "cross_sectional_scaling": "std",
            }

# 影响 fit 的参数（决定能不能共用同一次拟合）。其余参数都是 fit 之后才起作用的
# 旋钮：只差这些的两个臂共用同一个 artifact，Δ 就纯粹是那个旋钮的效果。
FIT_KEYS = ("feature_count", "ridge_alpha", "design_basis", "market_alpha_ratio",
            "cross_sectional_scaling", "ridge_tol", "ridge_max_iter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling time_id-level walk-forward validation.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--train-window", type=int, default=None,
                        help="Training window in unique sampled time_ids. Default: ~4/9 of total (≈4 partitions).")
    parser.add_argument("--embargo", type=int, default=6)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--configs", default="baseline",
                        help=f"逗号分隔的配置名，第一个是 baseline 臂。可选：{', '.join(CONFIGS)}")
    parser.add_argument("--label", default=None,
                        help="A/B 模式的实验名，输出到 ab_<label>.{json,md}；多配置时必填。")
    parser.add_argument("--fold-offset", default="0",
                        help="把 fold 边界整体右移 N 个采样 time_id；'half' = 预留范围的上限。"
                             "基准与平移版都预留约半个旧验证段，保证折数与长度相同。")
    parser.add_argument("--holdout-phase", type=int, default=None,
                        help="相位隔离：额外加载 time_id %% sample_modulo == P 的行，"
                             "训练只用余数 0、验证只用余数 P。用来量「模型没见过的相位」上的表现 —— "
                             "生产就是这个处境（训练只覆盖 2 个相位，测试集有全部 10 个）。")
    parser.add_argument("--disable-fit-cache", action="store_true",
                        help="每个臂都重新拟合一次，即使参数相同（配对自检要用）。")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的 A/B 报告。")
    return parser.parse_args()


def resolve_config(name: str) -> dict[str, object]:
    if name not in CONFIGS:
        raise SystemExit(f"未知配置 '{name}'；可选：{', '.join(CONFIGS)}")
    config = dict(BASE_CONFIG)
    config.update(CONFIGS[name])
    return config


def arm_labels(names: list[str]) -> list[str]:
    """重名的臂加 #i 后缀（配对自检会故意把同一个配置写两遍）。"""
    totals = Counter(names)
    seen: Counter[str] = Counter()
    labels = []
    for name in names:
        if totals[name] > 1:
            labels.append(f"{name}#{seen[name]}")
            seen[name] += 1
        else:
            labels.append(name)
    return labels


def load_all_sampled(
    files: list[Path], sample_modulo: int, residues: tuple[int, ...] = (0,)
) -> dict[str, np.ndarray]:
    """保留 time_id % sample_modulo ∈ residues 的行。

    residues=(0,) 是历史行为。给多个余数是为了相位隔离实验：把训练相位和评估相位
    分开（见 --holdout-phase），这样才能量出「在模型没见过的相位上」的表现。
    """
    parts: dict[str, list[np.ndarray]] = {
        "features": [], "target": [], "weight": [], "time_id": [], "asset_id": [],
    }
    for path in files:
        kept = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=120_000, columns=READ_COLUMNS):
            frame = batch.to_pandas()
            phase = frame["time_id"].to_numpy(copy=False) % sample_modulo
            mask = phase == residues[0]
            for residue in residues[1:]:
                mask |= phase == residue
            if not mask.any():
                continue
            parts["features"].append(frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            parts["target"].append(frame.loc[mask, "target"].to_numpy(dtype=np.float32, copy=True))
            parts["weight"].append(frame.loc[mask, "weight"].to_numpy(dtype=np.float32, copy=True))
            parts["time_id"].append(frame.loc[mask, "time_id"].to_numpy(dtype=np.int64, copy=True))
            parts["asset_id"].append(frame.loc[mask, "asset_id"].to_numpy(dtype=np.int8, copy=True))
            kept += int(mask.sum())
        print(f"loaded {path.name}: {kept:,} sampled rows", flush=True)
    return {name: np.concatenate(values) for name, values in parts.items()}


def optimal_scale(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    """闭式最优缩放 a* = Σwyf / Σwf²，夹到 [0, 2]（护栏与 train.py 一致）。"""
    w64 = np.maximum(weight.astype(np.float64), 0.0)
    y64 = target.astype(np.float64)
    f64 = prediction.astype(np.float64)
    value = float(np.dot(w64, y64 * f64) / max(float(np.dot(w64, f64 * f64)), 1e-30))
    return max(0.0, min(value, 2.0))


def estimate_inner_scale(
    data: dict[str, np.ndarray],
    all_time_ids: np.ndarray,
    train_ids: np.ndarray,
    valid_len: int,
    embargo: int,
    feature_count: int,
    fold_alpha: float,
    design_basis: str = "raw_dev",
    market_alpha_ratio: float = 1.0,
    cross_sectional_scaling: str = "none",
    ridge_tol: float = 1e-4,
    ridge_max_iter: int = 100,
    phase_split: tuple[int, int, int] | None = None,
) -> tuple[float, dict[str, object]]:
    """在训练段内部再切一层 fold 来估 a*，绝不碰外层验证段。

    为什么不能在训练段样本内估：ridge 在样本内已经拟合了 y，样本内的 a* 会趋近 1，
    比拍脑袋的 0.5 还糟。a* 必须来自一段模型没见过的数据。
    为什么不能像现在的 train.py 那样在打分的那批数据上估：a* 的定义就是让那批数据
    分数最大的值，那样测出来的收益是上界不是估计。

    已知偏差：内层模型只看到外层训练窗口的约 87%，模型稍弱 → a* 估得偏小 →
    由此测出的 auto-scale 收益是偏保守的下界。
    """
    inner_valid_start = len(train_ids) - valid_len
    inner_train_end = inner_valid_start - embargo
    if inner_train_end <= 0:
        raise ValueError("训练窗口装不下内层 fold（train_window 太小或 valid 段太长）")
    inner_train_ids = train_ids[:inner_train_end]
    inner_valid_ids = train_ids[inner_valid_start:]
    # 内层窗口更短，alpha 同比例缩小，保持每行正则强度一致。
    inner_alpha = fold_alpha * len(inner_train_ids) / len(train_ids)

    inner_train_set = np.isin(all_time_ids, inner_train_ids)
    inner_valid_set = np.isin(all_time_ids, inner_valid_ids)
    if phase_split:
        # 内层也照外层的规矩切相位：训练相位估出来的 a* 要用在验证相位上
        inner_train_set &= (all_time_ids % phase_split[0]) == phase_split[1]
        inner_valid_set &= (all_time_ids % phase_split[0]) == phase_split[2]
    artifact, selected = fit_model(
        data["features"][inner_train_set],  # fit_model 就地改这份副本，用完即弃
        data["target"][inner_train_set],
        data["weight"][inner_train_set],
        data["time_id"][inner_train_set],
        feature_count,
        inner_alpha,
        design_basis=design_basis,
        market_alpha_ratio=market_alpha_ratio,
        cross_sectional_scaling=cross_sectional_scaling,
        ridge_tol=ridge_tol,
        ridge_max_iter=ridge_max_iter,
    )
    raw_prediction = predict_array(
        artifact,
        data["features"][inner_valid_set],
        data["time_id"][inner_valid_set],
        selected,
        prediction_scale=1.0,
        prediction_clip=1e9,
    )
    scale = optimal_scale(
        data["target"][inner_valid_set], raw_prediction, data["weight"][inner_valid_set]
    )
    diagnostics = {
        "scale": scale,
        "inner_train_range": [int(inner_train_ids[0]), int(inner_train_ids[-1])],
        "inner_valid_range": [int(inner_valid_ids[0]), int(inner_valid_ids[-1])],
        "inner_train_time_ids": int(len(inner_train_ids)),
        "inner_valid_time_ids": int(len(inner_valid_ids)),
        "inner_ridge_alpha": float(inner_alpha),
    }
    del artifact, selected, raw_prediction, inner_train_set, inner_valid_set
    gc.collect()
    return scale, diagnostics


def sign_test_p(positive: int, negative: int) -> float:
    """双侧符号检验 p 值；零差值按定义剔除。"""
    total = positive + negative
    if total == 0:
        return 1.0
    extreme = max(positive, total - positive)
    tail = sum(math.comb(total, j) for j in range(extreme, total + 1))
    return min(1.0, 2.0 * tail / (2**total))


def summarise(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "std": std,
        "se": std / math.sqrt(count) if count > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run_folds(
    data: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    arms: list[tuple[str, dict[str, object]]],
    embargo: int,
    fold_alpha_factor: float,
    use_fit_cache: bool,
    phase_split: tuple[int, int, int] | None = None,
) -> list[dict[str, object]]:
    """每折跑完所有臂。同一折内，参数相同的 fit 只做一次并在臂之间共享。

    phase_split = (modulus, train_residue, valid_residue) 时，训练行再按
    time_id %% modulus == train_residue 过滤、验证行按 == valid_residue 过滤，
    于是验证段落在模型完全没见过的相位上。None 时行为与历史完全一致。
    """
    all_time_ids = data["time_id"]
    phase = all_time_ids % phase_split[0] if phase_split else None
    fold_results: list[dict[str, object]] = []

    for index, (train_ids, valid_ids) in enumerate(folds):
        started = time.perf_counter()
        train_set = np.isin(all_time_ids, train_ids)
        valid_set = np.isin(all_time_ids, valid_ids)
        if phase_split:
            train_set &= phase == phase_split[1]
            valid_set &= phase == phase_split[2]

        # predict_array 只读输入（内部 [:, selected].copy()），验证段特征可以跨臂共用。
        v_features = data["features"][valid_set]
        v_target = data["target"][valid_set]
        v_weight = data["weight"][valid_set]
        v_time = data["time_id"][valid_set]
        w64 = np.maximum(v_weight.astype(np.float64), 0.0)
        y64 = v_target.astype(np.float64)
        score_denominator = float(np.dot(w64, y64 * y64))

        fit_cache: dict[tuple, tuple[dict[str, object], np.ndarray]] = {}
        scale_cache: dict[tuple, tuple[float, dict[str, object]]] = {}
        scores: dict[str, float] = {}
        auto_scale_info: dict[str, object] = {}
        fit_diagnostics: dict[str, object] = {}

        for label, config in arms:
            arm_started = time.perf_counter()
            fold_alpha = float(config["ridge_alpha"]) * fold_alpha_factor
            # 必须由 FIT_KEYS 生成 —— 早先这里硬编码成 (feature_count, ridge_alpha)，
            # 结果 design_basis / market_alpha_ratio 不同的臂全部命中同一份缓存，
            # Δ 恒等于 0（见 ab_meandev.json 那次作废的运行）。
            fit_key = tuple(config[name] for name in FIT_KEYS)

            if config["prediction_scale"] == AUTO_INNER:
                # 关掉缓存时 a* 也重估一遍，--disable-fit-cache 才是名副其实的
                # 「两个臂各自独立走完整条路径」。
                if not use_fit_cache or fit_key not in scale_cache:
                    scale_cache[fit_key] = estimate_inner_scale(
                        data, all_time_ids, train_ids, len(valid_ids), embargo,
                        int(config["feature_count"]), fold_alpha,
                        str(config["design_basis"]), float(config["market_alpha_ratio"]),
                        str(config["cross_sectional_scaling"]),
                        float(config["ridge_tol"]), int(config["ridge_max_iter"]),
                        phase_split,
                    )
                scale, diagnostics = scale_cache[fit_key]
                auto_scale_info[label] = diagnostics
            else:
                scale = float(config["prediction_scale"])

            if use_fit_cache and fit_key in fit_cache:
                artifact, selected = fit_cache[fit_key]
            else:
                # fit_model 就地改 features（robust_transform_fit 直接改数组），
                # 所以每次拟合都必须重新切一份训练段副本 —— 复用会让第二个臂
                # 在「已经变换过」的数据上拟合，错得无声无息。
                artifact, selected = fit_model(
                    data["features"][train_set],
                    data["target"][train_set],
                    data["weight"][train_set],
                    data["time_id"][train_set],
                    int(config["feature_count"]),
                    fold_alpha,
                    design_basis=str(config["design_basis"]),
                    market_alpha_ratio=float(config["market_alpha_ratio"]),
                    cross_sectional_scaling=str(config["cross_sectional_scaling"]),
                    ridge_tol=float(config["ridge_tol"]),
                    ridge_max_iter=int(config["ridge_max_iter"]),
                )
                gc.collect()
                if use_fit_cache:
                    fit_cache[fit_key] = (artifact, selected)

            arm_artifact = artifact
            if config["zero_intercept"]:
                # 浅拷贝：直接改缓存里那份会串味到共用同一次拟合的其他臂。
                arm_artifact = dict(artifact)
                arm_artifact["intercept"] = 0.0

            prediction = predict_array(
                arm_artifact, v_features, v_time, selected,
                scale, float(config["prediction_clip"]),
            )
            scores[label] = float(weighted_zero_mean_r2(v_target, prediction, v_weight))
            fit_diagnostics[label] = {
                "selected_indices": [int(value) for value in selected],
                "ridge_n_iter": int(artifact.get("ridge_n_iter", -1)),
                "ridge_tol": float(config["ridge_tol"]),
                "ridge_max_iter": int(config["ridge_max_iter"]),
            }
            print(
                f"fold {index:2d} [{label}]: score={scores[label]:.8f} scale={scale:.6f} "
                f"({time.perf_counter() - arm_started:.1f}s)",
                flush=True,
            )
            del prediction
            if not use_fit_cache:
                del artifact, selected
                gc.collect()

        fold_results.append({
            "fold": index,
            "train_time_range": [int(train_ids[0]), int(train_ids[-1])],
            "valid_time_range": [int(valid_ids[0]), int(valid_ids[-1])],
            "embargo_gap": int(valid_ids[0] - train_ids[-1]),
            "train_rows": int(train_set.sum()),
            "valid_rows": int(valid_set.sum()),
            "scores": scores,
            "score_denominator": score_denominator,
            "auto_scale": auto_scale_info,
            "fit_diagnostics": fit_diagnostics,
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        del v_features, v_target, v_weight, v_time, train_set, valid_set
        fit_cache.clear()
        scale_cache.clear()
        gc.collect()

    return fold_results


def write_single_report(
    output_dir: Path, fold_results: list[dict[str, object]], arm_label: str,
    config: dict[str, object], requested_n_folds: int, actual_n_folds: int,
    train_window: int, embargo: int, sample_modulo: int, fold_alpha: float,
    offset: int, reserved_offset: int,
) -> None:
    """单配置模式：保持本次改动前的 json/md 结构与数值不变。"""
    scores = np.array([float(f["scores"][arm_label]) for f in fold_results])
    denominators = np.array([float(f["score_denominator"]) for f in fold_results])
    stats = summarise(scores)
    pooled_score = (
        float(np.average(scores, weights=denominators))
        if float(denominators.sum()) > 0.0 else 0.0
    )
    configuration = {
        "validation_grid_version": 2,
        "requested_n_folds": requested_n_folds,
        "actual_n_folds": actual_n_folds,
        "train_window": train_window,
        "embargo": embargo,
        "reserved_offset": reserved_offset,
        "sample_modulo": sample_modulo,
        "feature_count": config["feature_count"],
        "ridge_alpha": fold_alpha,
        "ridge_tol": config["ridge_tol"],
        "ridge_max_iter": config["ridge_max_iter"],
        "prediction_scale": config["prediction_scale"],
        "prediction_clip": config["prediction_clip"],
    }
    if offset:  # offset=0 时不写这个键，保证与已提交的 json 结构逐字一致
        configuration["fold_offset"] = offset

    payload = {
        "metric": "weighted_zero_mean_r2",
        "configuration": configuration,
        "summary": {
            "mean_score": stats["mean"],
            "pooled_score": pooled_score,
            "std_score": stats["std"],
            "se_score": stats["se"],
            "min_score": stats["min"],
            "max_score": stats["max"],
            "positive_folds": int((scores > 0).sum()),
            "total_folds": len(scores),
        },
        "folds": [
            {
                "fold": f["fold"],
                "train_time_range": f["train_time_range"],
                "valid_time_range": f["valid_time_range"],
                "embargo_gap": f["embargo_gap"],
                "train_rows": f["train_rows"],
                "valid_rows": f["valid_rows"],
                "score": float(f["scores"][arm_label]),
                "score_denominator": f["score_denominator"],
                "elapsed_seconds": f["elapsed_seconds"],
            }
            for f in fold_results
        ],
    }
    (output_dir / "walk_forward_rolling.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Rolling time_id-level walk-forward validation",
        "",
        f"train_window={train_window:,}, embargo={embargo}, "
        f"sample_modulo={sample_modulo}, n_folds={actual_n_folds}"
        + (f", fold_offset={offset:,}" if offset else ""),
        "",
        "| Fold | Train range | Valid range | Embargo | Score |",
        "|---:|---|---|---:|---:|",
    ]
    for f in payload["folds"]:
        tr, vr = f["train_time_range"], f["valid_time_range"]
        lines.append(
            f"| {f['fold']} | [{tr[0]:,}, {tr[1]:,}] | [{vr[0]:,}, {vr[1]:,}] "
            f"| {f['embargo_gap']} | {f['score']:.8f} |"
        )
    lines += [
        "",
        f"**Mean**: {stats['mean']:.8f}, **Pooled**: {pooled_score:.8f}, **SE**: {stats['se']:.8f}, "
        f"**Positive folds**: {int((scores > 0).sum())}/{len(scores)}",
    ]
    (output_dir / "walk_forward_rolling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "mean_score": stats["mean"],
        "pooled_score": pooled_score,
        "se_score": stats["se"],
        "positive_folds": int((scores > 0).sum()),
        "total_folds": len(scores),
    }, indent=2))


CAVEAT = (
    "滚动 fold 的训练窗口互相重叠约 87%，各 fold 的 Δ 是正相关的，"
    "上面这个 SE 偏乐观；判断阈值以 P0-3 噪声地板（--fold-offset half 重跑的 mean(Δ) 漂移量）为准。"
)


def write_ab_report(
    output_dir: Path, label: str, fold_results: list[dict[str, object]],
    arms: list[tuple[str, dict[str, object]]], requested_n_folds: int,
    actual_n_folds: int, train_window: int, embargo: int, sample_modulo: int,
    fold_alpha_factor: float, offset: int, reserved_offset: int, chunk_size: int,
    use_fit_cache: bool, force: bool,
) -> dict[str, object]:
    json_path = output_dir / f"ab_{label}.json"
    md_path = output_dir / f"ab_{label}.md"
    if not force:
        for path in (json_path, md_path):
            if path.exists():
                raise SystemExit(f"{path} 已存在；换个 --label 或显式加 --force（会覆盖旧实验记录）")

    baseline_label = arms[0][0]
    arm_names = [name for name, _ in arms]
    per_arm = {
        name: np.array([float(f["scores"][name]) for f in fold_results]) for name in arm_names
    }
    denominators = np.array([float(f["score_denominator"]) for f in fold_results])
    pooled_by_arm = {
        name: (
            float(np.average(values, weights=denominators))
            if float(denominators.sum()) > 0.0 else 0.0
        )
        for name, values in per_arm.items()
    }

    paired: dict[str, object] = {}
    for name in arm_names[1:]:
        deltas = per_arm[name] - per_arm[baseline_label]
        stats = summarise(deltas)
        positive = int((deltas > 0).sum())
        negative = int((deltas < 0).sum())
        paired[name] = {
            "mean_delta": stats["mean"],
            "std_delta": stats["std"],
            "se_delta": stats["se"],
            # SE=0（例如配对自检里两臂完全相同）时 t 无定义，写 null 而不是 Infinity
            # —— Infinity 不是合法 JSON，别的工具读不了这份记录。
            "t_stat": stats["mean"] / stats["se"] if stats["se"] > 0 else None,
            "min_delta": stats["min"],
            "max_delta": stats["max"],
            "positive_folds": positive,
            "negative_folds": negative,
            "zero_folds": len(deltas) - positive - negative,
            "sign_test_p": sign_test_p(positive, negative),
            "pooled_delta": pooled_by_arm[name] - pooled_by_arm[baseline_label],
        }

    payload = {
        "metric": "weighted_zero_mean_r2",
        "mode": "paired_ab",
        "label": label,
        "baseline_arm": baseline_label,
        "configuration": {
            "validation_grid_version": 2,
            "requested_n_folds": requested_n_folds,
            "actual_n_folds": actual_n_folds,
            "train_window": train_window,
            "embargo": embargo,
            "fold_offset": offset,
            "reserved_offset": reserved_offset,
            "valid_chunk_size": chunk_size,
            "sample_modulo": sample_modulo,
            "fold_alpha_factor": fold_alpha_factor,
            "fit_cache": use_fit_cache,
        },
        "arms": {name: config for name, config in arms},
        "absolute_reference": {
            name: {**summarise(values), "pooled_score": pooled_by_arm[name]}
            for name, values in per_arm.items()
        },
        "paired_delta": paired,
        "folds": fold_results,
        "caveat": CAVEAT,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# A/B paired rolling walk-forward: {label}",
        "",
        f"n_folds={actual_n_folds}, train_window={train_window:,}, embargo={embargo}, "
        f"sample_modulo={sample_modulo}, fold_offset={offset:,}, fit_cache={use_fit_cache}",
        "",
        f"baseline arm = `{baseline_label}`",
        "",
        "| Fold | " + " | ".join(f"`{name}`" for name in arm_names)
        + " | " + " | ".join(f"Δ {name}" for name in arm_names[1:]) + " |",
        "|---:|" + "---:|" * (2 * len(arm_names) - 1),
    ]
    for f in fold_results:
        row = [str(f["fold"])]
        row += [f"{float(f['scores'][name]):.8f}" for name in arm_names]
        row += [
            f"{float(f['scores'][name]) - float(f['scores'][baseline_label]):+.3e}"
            for name in arm_names[1:]
        ]
        lines.append("| " + " | ".join(row) + " |")
    mean_row = ["**mean**"]
    mean_row += [f"{per_arm[name].mean():.8f}" for name in arm_names]
    mean_row += [f"**{paired[name]['mean_delta']:+.3e}**" for name in arm_names[1:]]
    lines.append("| " + " | ".join(mean_row) + " |")

    lines += ["", "## 配对 delta（结论看这里）", ""]
    for name in arm_names[1:]:
        stats = paired[name]
        t_text = f"{stats['t_stat']:.2f}" if stats["t_stat"] is not None else "n/a"
        lines.append(
            f"- `{name}` vs `{baseline_label}`: mean(Δ)={stats['mean_delta']:+.3e}, "
            f"SE={stats['se_delta']:.3e}, t={t_text}, "
            f"同号 {max(stats['positive_folds'], stats['negative_folds'])}/{len(fold_results)}"
            f"（正 {stats['positive_folds']} / 负 {stats['negative_folds']} / 零 {stats['zero_folds']}），"
            f"符号检验 p={stats['sign_test_p']:.4f}"
        )
    lines += [
        "",
        f"> {CAVEAT}",
        "",
        "## 绝对分（参考值，不作结论依据）",
        "",
        "| Arm | mean | pooled | SE | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in arm_names:
        stats = summarise(per_arm[name])
        lines.append(
            f"| `{name}` | {stats['mean']:.8f} | {pooled_by_arm[name]:.8f} | {stats['se']:.8f} | "
            f"{stats['min']:.8f} | {stats['max']:.8f} |"
        )
    auto_names = [
        name for name in arm_names if any(name in f["auto_scale"] for f in fold_results)
    ]
    if auto_names:
        lines += [
            "",
            "## 内层 fold 估出的 a*（auto_inner 臂）",
            "",
            "| Fold | " + " | ".join(f"a* `{name}`" for name in auto_names) + " |",
            "|---:|" + "---:|" * len(auto_names),
        ]
        for f in fold_results:
            row = [str(f["fold"])]
            for name in auto_names:
                info = f["auto_scale"].get(name)
                row.append(f"{float(info['scale']):.6f}" if info else "—")
            lines.append("| " + " | ".join(row) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    console = {
        "label": label,
        "baseline_arm": baseline_label,
        "paired_delta": {
            name: {
                "mean_delta": stats["mean_delta"],
                "se_delta": stats["se_delta"],
                "t_stat": stats["t_stat"],
                "positive_folds": stats["positive_folds"],
                "negative_folds": stats["negative_folds"],
                "sign_test_p": stats["sign_test_p"],
            }
            for name, stats in paired.items()
        },
        "report": str(md_path),
    }
    print(json.dumps(console, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = train_files(data_root)

    config_names = [name.strip() for name in args.configs.split(",") if name.strip()]
    if not config_names:
        raise SystemExit("--configs 不能为空")
    if len(config_names) > 1 and not args.label:
        raise SystemExit("多配置（A/B）模式必须给 --label，输出会写到 ab_<label>.{json,md}")
    labels = arm_labels(config_names)
    arms = [(label, resolve_config(name)) for label, name in zip(labels, config_names)]

    phase_split = None
    residues: tuple[int, ...] = (0,)
    if args.holdout_phase is not None:
        if not 0 < args.holdout_phase < args.sample_modulo:
            raise SystemExit(f"--holdout-phase 必须在 1..{args.sample_modulo - 1} 之间")
        residues = (0, args.holdout_phase)
        phase_split = (args.sample_modulo, 0, args.holdout_phase)
        print(f"相位隔离：训练用 time_id%{args.sample_modulo}==0，"
              f"验证用 =={args.holdout_phase}（模型没见过）", flush=True)

    print("loading all partitions...", flush=True)
    data = load_all_sampled(files, args.sample_modulo, residues)
    unique_time_ids = np.unique(data["time_id"])
    print(f"total rows: {len(data['time_id']):,}, unique time_ids: {len(unique_time_ids):,}", flush=True)

    train_window = args.train_window
    if train_window is None:
        train_window = int(len(unique_time_ids) * 4 / 9)

    unreserved_chunk = rolling_fold_chunk_size(
        len(unique_time_ids), args.n_folds, train_window, args.embargo
    )
    reserved_offset = unreserved_chunk // 2
    chunk_size = rolling_fold_chunk_size(
        len(unique_time_ids), args.n_folds, train_window, args.embargo, reserved_offset
    )
    if args.fold_offset == "half":
        offset = reserved_offset
    else:
        offset = int(args.fold_offset)
    if not 0 <= offset <= reserved_offset:
        raise SystemExit(f"--fold-offset 必须在 0..{reserved_offset}，或使用 'half'")
    print(
        f"valid chunk_size={chunk_size:,} sampled time_ids "
        f"(--fold-offset half → {reserved_offset:,}); using offset={offset:,}",
        flush=True,
    )

    folds = rolling_time_folds(
        unique_time_ids, args.n_folds, train_window, args.embargo,
        offset, reserved_offset,
    )
    print(f"generated {len(folds)} folds (embargo={args.embargo})", flush=True)

    # Scale alpha: match per-row regularisation to the production fit.
    # Production: modulo=5, ~400K raw time_ids → 80K sampled → ~1.2M rows.
    # 用「实际参与训练的 time_id 数」而不是 train_window —— 切相位后前者只有后者的一半，
    # 不改的话 alpha 会大一倍。不切相位时两者相等，所以历史结果逐位不变。
    effective_train_ids = len(folds[0][0])
    if phase_split:
        effective_train_ids = int((folds[0][0] % phase_split[0] == phase_split[1]).sum())
    fold_alpha_factor = effective_train_ids / PROD_SAMPLED_WINDOW
    if phase_split:
        print(f"每折实际训练 time_id 数 {effective_train_ids:,}（train_window {train_window:,} 的一半）"
              f" → alpha 因子 {fold_alpha_factor:.4f}", flush=True)
    print(
        "arms: " + ", ".join(
            f"{label}(alpha={float(config['ridge_alpha']) * fold_alpha_factor:,.0f})"
            for label, config in arms
        ),
        flush=True,
    )

    fold_results = run_folds(
        data, folds, arms, args.embargo, fold_alpha_factor,
        not args.disable_fit_cache, phase_split,
    )

    if args.label:
        payload = write_ab_report(
            output_dir, args.label, fold_results, arms, args.n_folds, len(folds),
            train_window, args.embargo, args.sample_modulo, fold_alpha_factor,
            offset, reserved_offset, chunk_size, not args.disable_fit_cache, args.force,
        )
        if phase_split:
            # 报告写完后补记相位口径，免得日后拿它和不切相位的结果直接比
            path = output_dir / f"ab_{args.label}.json"
            payload["configuration"]["holdout_phase"] = args.holdout_phase
            payload["configuration"]["train_residue"] = 0
            payload["configuration"]["effective_train_time_ids"] = effective_train_ids
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        label, config = arms[0]
        write_single_report(
            output_dir, fold_results, label, config, args.n_folds, len(folds),
            train_window, args.embargo, args.sample_modulo,
            float(config["ridge_alpha"]) * fold_alpha_factor, offset, reserved_offset,
        )


if __name__ == "__main__":
    main()
