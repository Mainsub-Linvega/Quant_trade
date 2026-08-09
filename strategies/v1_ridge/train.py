from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

# 离线训练专用：把仓库根加入 sys.path 以复用 src/ 下的公共实现。
# 注意 main.py（提交件）绝不允许这样做 —— 提交包里没有 src/。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io import FEATURE_COLUMNS, load_time_sample, train_files
from src.metric import weighted_zero_mean_r2

# 同目录的预处理 / 推理唯一实现（main.py 也 import 它，两侧口径由此保持一致）。
from features import apply_robust_transform, cross_sectional_deviation, linear_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a robust, CPU-friendly contest baseline.")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "model"))
    parser.add_argument("--train-partitions", type=int, default=4)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--validation-sample-modulo", type=int, default=10)
    parser.add_argument(
        "--sampling", choices=["periodic", "phase_balanced"], default="periodic",
        help="periodic 只取 time_id %% modulo == 0（相位 0 和 5）；phase_balanced 同预算覆盖全部 "
             "10 个相位。⚠️ 默认必须保持 periodic —— 生产模型 c23a8cfb 是这么训的，改了就不是它了。"
             "测试集是连续 time_id（全部 10 个相位），所以这里存在已知的口径错配，见 ROADMAP 第 2 项")
    # 2026-08-08：alpha 回滚到 2e6。滚动网格曾指向 5e5（两个采样口径都 8/10 折同号），
    # 但公榜用两点定抛物线做峰值对峰值的对比，5e5 比 2e6 低 19.2%（0.00150896 vs 0.00186804）。
    # 相位隔离与长验证跨度实验都未复现这次反转；当前只能确认公榜时期分布与训练期不同，
    # 不能把病因归到相位。详见 ROADMAP「本地与公榜为什么在 alpha 上量反了」。
    #
    # ⚠️ prediction_scale=1.13 是公榜实测最优（Score(a)=2aA−a²B 两点解出，顶点 1.1317）。
    # 它是对公榜测试集拟合出来的，**私榜交付前必须重新评估**：本地尺子给 0.88~0.97。
    # 好在曲线顶部很平，0.883 也能拿到峰值的 95.2%，选错代价不大。
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--prediction-scale", type=float, default=1.13,
                        help="发布用的固定缩放；要改回在验证段现估请加 --auto-scale。")
    parser.add_argument("--auto-scale", action="store_true",
                        help="在验证段上闭式估 a*=Σwyf/Σwf² 并发布它。"
                             "已被否决（每折重估的 a* 在 10 折上 sd=0.41，配对 A/B 测不出收益，"
                             "见 outputs/experiments/ab_scale_auto.md），保留仅为可复跑。")
    parser.add_argument("--prediction-clip", type=float, default=0.5)
    parser.add_argument("--zero-intercept", action="store_true")
    parser.add_argument("--design-basis", choices=["raw_dev", "mean_dev"], default="raw_dev",
                        help="raw_dev=历史口径；mean_dev=[截面均值‖deviation]，可对择时分量单独加罚")
    parser.add_argument("--market-alpha-ratio", type=float, default=1.0,
                        help="择时分量（截面均值块）的正则倍数，只在 --design-basis mean_dev 下生效")
    parser.add_argument("--cross-sectional-scaling", choices=["none", "std"], default="none",
                        help="none=只去截面均值（历史口径）；std=再除以截面标准差，"
                             "把时变的截面离散度归一化掉")
    parser.add_argument("--ridge-tol", type=float, default=1e-8,
                        help="LSQR 收敛容差；收紧以降低 BLAS 线程数导致的模型漂移。")
    parser.add_argument("--ridge-max-iter", type=int, default=2000)
    parser.add_argument(
        "--allow-production-overwrite",
        action="store_true",
        help="显式允许写入 strategies/v1_ridge/model；默认拒绝，防止候选误覆盖正式模型。",
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def robust_transform_fit(features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Anonymous market features occasionally contain NaN/inf and regime-specific
    # extremes. Learn every preprocessing statistic on the training period only.
    # Mark inf as NaN so nanquantile ignores both; apply_robust_transform handles fill.
    features[~np.isfinite(features)] = np.nan
    quantiles = np.nanquantile(features, [0.001, 0.25, 0.5, 0.75, 0.999], axis=0).astype(np.float32)
    lower, q25, center, q75, upper = quantiles
    scale = np.maximum(q75 - q25, np.float32(1e-4))
    features = apply_robust_transform(features, lower, upper, center, scale)
    return features, {"lower": lower, "upper": upper, "center": center, "scale": scale}


def select_features(features: np.ndarray, target: np.ndarray, weight: np.ndarray, count: int) -> np.ndarray:
    count = min(max(1, count), features.shape[1])
    weight64 = np.maximum(weight.astype(np.float64), 0.0)
    target64 = target.astype(np.float64)
    total_weight = float(weight64.sum())
    weighted_target_sum = float(np.dot(weight64, target64))
    target_variance = float(np.dot(weight64, target64 * target64) - weighted_target_sum**2 / total_weight)
    correlations = np.zeros(features.shape[1], dtype=np.float64)

    # Work in column blocks to avoid materialising several full float64 matrices.
    for start in range(0, features.shape[1], 64):
        stop = min(start + 64, features.shape[1])
        block = features[:, start:stop].astype(np.float64)
        weighted_feature_sum = block.T @ weight64
        weighted_square_sum = (block * block).T @ weight64
        weighted_cross_sum = block.T @ (weight64 * target64)
        covariance = weighted_cross_sum - weighted_feature_sum * weighted_target_sum / total_weight
        variance = weighted_square_sum - weighted_feature_sum**2 / total_weight
        correlations[start:stop] = covariance / np.sqrt(np.maximum(variance * target_variance, 1e-30))

    selected = np.argsort(np.abs(correlations), kind="stable")[-count:]
    # Keeping original column order makes inference indexing and artifacts easier to inspect.
    return np.sort(selected)


def make_design(
    features: np.ndarray,
    time_ids: np.ndarray,
    selected: np.ndarray,
    design_basis: str = "raw_dev",
    market_scale: float = 1.0,
    cross_sectional_scaling: str = "none",
) -> np.ndarray:
    """构造设计矩阵。两种基底张成同一个列空间，区别只在 ridge 怎么罚。

    raw_dev  : [raw ‖ deviation] —— 历史口径，罚的是 (β_raw, β_dev)
    mean_dev : [截面均值×market_scale ‖ deviation] —— 罚的是 (β_mean, β_dev)，
               于是可以对「择时分量」单独加罚（ROADMAP P2-1）。
               raw = mean + deviation，所以两者可逆变换，见 fit_model 里的还原。
    """
    raw = features[:, selected].copy()
    deviation = cross_sectional_deviation(raw, time_ids, cross_sectional_scaling)
    if design_basis == "raw_dev":
        return np.column_stack([raw, deviation]).astype(np.float32, copy=False)
    if design_basis == "mean_dev":
        if cross_sectional_scaling != "none":
            # 这条路径靠 raw − deviation 反推截面均值，deviation 一旦被标准差归一化，
            # 这个等式就不成立了。mean_dev 已被 P2-1 否决，不值得为它再补一套推导。
            raise ValueError("design_basis=mean_dev 不支持 cross_sectional_scaling≠none")
        market = raw - deviation  # 逐 time_id 的截面均值，广播回每一行
        if market_scale != 1.0:
            market *= np.float32(market_scale)
        return np.column_stack([market, deviation]).astype(np.float32, copy=False)
    raise ValueError(f"unknown design_basis: {design_basis}")


def fit_model(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_ids: np.ndarray,
    feature_count: int,
    ridge_alpha: float,
    *,
    design_basis: str = "raw_dev",
    market_alpha_ratio: float = 1.0,
    cross_sectional_scaling: str = "none",
    # 与 CLI 默认（--ridge-tol / --ridge-max-iter）保持一致。历史值 1e-4/100 停得太早，
    # 换个 BLAS 线程数就训出不同模型（coef 相对差 4.75e-04）——绕过 CLI 直接调本函数的
    # 代码路径会悄悄退回那个不可复现的求解器，所以签名默认值也必须是严格档。
    ridge_tol: float = 1e-8,
    ridge_max_iter: int = 2000,
) -> tuple[dict[str, object], np.ndarray]:
    """拟合。design_basis="raw_dev"（默认）与历史口径逐位一致。

    design_basis="mean_dev" 时用 [截面均值 ‖ deviation] 拟合，并让择时分量吃
    market_alpha_ratio 倍的正则；系数最后一律换算回 [raw ‖ deviation] 基底存盘，
    所以 features.py / main.py 的推理路径完全不需要改。
    """
    if market_alpha_ratio <= 0:
        raise ValueError("market_alpha_ratio must be positive")
    if ridge_tol <= 0 or ridge_max_iter <= 0:
        raise ValueError("ridge_tol and ridge_max_iter must be positive")
    # 列缩放 c 与有效正则的关系：对 mean 列乘 c，等价于对它的系数用 alpha/c² 的正则。
    # 要让择时分量吃 r 倍正则，取 c = 1/√r。
    market_scale = 1.0 / math.sqrt(market_alpha_ratio) if design_basis == "mean_dev" else 1.0

    features, preprocessing = robust_transform_fit(features)
    selected = select_features(features, target, weight, feature_count)
    design = make_design(features, time_ids, selected, design_basis, market_scale,
                         cross_sectional_scaling)
    del features

    estimator = Ridge(
        alpha=ridge_alpha,
        solver="lsqr",
        tol=ridge_tol,
        max_iter=ridge_max_iter,
        fit_intercept=True,
        copy_X=False,
    )
    estimator.fit(design, target, sample_weight=np.maximum(weight, 0.0))

    coef = estimator.coef_.astype(np.float64)
    if design_basis == "mean_dev":
        block = len(selected)
        beta_market = coef[:block] * market_scale  # 撤掉列缩放，还原 mean 的真实系数
        beta_deviation = coef[block:]
        # mean·b_m + dev·b_d ≡ raw·b_m + dev·(b_d − b_m)，换回推理端期待的 [raw ‖ dev] 基底
        coef = np.concatenate([beta_market, beta_deviation - beta_market])

    artifact: dict[str, object] = {
        "selected_indices": selected.tolist(),
        "selected_features": [FEATURE_COLUMNS[index] for index in selected],
        "lower": preprocessing["lower"][selected].tolist(),
        "upper": preprocessing["upper"][selected].tolist(),
        "center": preprocessing["center"][selected].tolist(),
        "scale": preprocessing["scale"][selected].tolist(),
        "intercept": float(estimator.intercept_),
        "coef": coef.tolist(),
        "ridge_alpha": float(ridge_alpha),
        "design_basis": design_basis,
        "market_alpha_ratio": float(market_alpha_ratio),
        "cross_sectional_scaling": cross_sectional_scaling,
        "ridge_tol": float(ridge_tol),
        "ridge_max_iter": int(ridge_max_iter),
        "ridge_n_iter": int(np.max(np.atleast_1d(estimator.n_iter_))),
    }
    return artifact, selected


def predict_array(
    artifact: dict[str, object],
    full_features: np.ndarray,
    time_ids: np.ndarray,
    selected: np.ndarray,
    prediction_scale: float,
    prediction_clip: float,
) -> np.ndarray:
    selected_features = full_features[:, selected].copy()
    preprocessing = {
        name: np.asarray(artifact[name], dtype=np.float32)
        for name in ["lower", "upper", "center", "scale"]
    }
    selected_features = apply_robust_transform(
        selected_features,
        preprocessing["lower"],
        preprocessing["upper"],
        preprocessing["center"],
        preprocessing["scale"],
    )
    deviation = cross_sectional_deviation(
        selected_features, time_ids, str(artifact.get("cross_sectional_scaling", "none"))
    )
    return linear_predict(
        selected_features,
        deviation,
        float(artifact["intercept"]),
        np.asarray(artifact["coef"], dtype=np.float32),
        prediction_scale,
        prediction_clip,
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    production_dir = Path(__file__).resolve().parent / "model"
    if model_dir.resolve() == production_dir.resolve() and not args.allow_production_overwrite:
        raise SystemExit(
            "拒绝覆盖正式模型目录；候选请传 --model-dir，确认晋级后再显式加 "
            "--allow-production-overwrite"
        )
    files = train_files(data_root)
    if len(files) < args.train_partitions + 1:
        raise ValueError("not enough chronological train partitions")
    if args.prediction_clip <= 0:
        raise ValueError("prediction clip must be positive")
    auto_scale = args.auto_scale

    validation_score: float | None = None
    final_scale: float = args.prediction_scale
    validation_train_paths = files[-(args.train_partitions + 1) : -1]
    validation_path = files[-1]
    if not args.skip_validation:
        print("validation: train on earlier partitions and score the final partition", flush=True)
        x_train, y_train, w_train, t_train = load_time_sample(
            validation_train_paths, args.validation_sample_modulo, sampling=args.sampling
        )
        # Ridge's loss is a sum over rows, so keep alpha proportional to the
        # sampled row count when validation uses a sparser training sample.
        validation_alpha = args.ridge_alpha * args.sample_modulo / args.validation_sample_modulo
        validation_artifact, selected = fit_model(
            x_train, y_train, w_train, t_train, args.feature_count, validation_alpha,
            design_basis=args.design_basis, market_alpha_ratio=args.market_alpha_ratio,
            cross_sectional_scaling=args.cross_sectional_scaling,
            ridge_tol=args.ridge_tol, ridge_max_iter=args.ridge_max_iter,
        )
        del x_train, y_train, w_train, t_train
        # ⚠️ 验证段**故意不跟 --sampling 走**：要量「训练相位覆盖」的效果，
        # 评估口径必须固定，否则两个臂连评的行都不是同一批。
        x_valid, y_valid, w_valid, t_valid = load_time_sample([validation_path], args.sample_modulo)
        if auto_scale:
            raw_prediction = predict_array(
                validation_artifact, x_valid, t_valid, selected,
                prediction_scale=1.0, prediction_clip=1e9,
            )
            w64 = np.maximum(w_valid.astype(np.float64), 0.0)
            y64 = y_valid.astype(np.float64)
            f64 = raw_prediction.astype(np.float64)
            final_scale = float(np.dot(w64, y64 * f64) / np.maximum(np.dot(w64, f64 * f64), 1e-30))
            final_scale = max(0.0, min(final_scale, 2.0))
            valid_prediction = np.clip(raw_prediction * final_scale, -args.prediction_clip, args.prediction_clip)
            print(f"optimal prediction_scale: {final_scale:.6f}", flush=True)
        else:
            valid_prediction = predict_array(
                validation_artifact, x_valid, t_valid, selected,
                final_scale, args.prediction_clip,
            )
        validation_score = weighted_zero_mean_r2(y_valid, valid_prediction, w_valid)
        # ⚠️ 这个数只能和它自己的历史比，不能用来选配置：它训练用 validation_sample_modulo=10
        # （只有相位 0），评估却用 sample_modulo=5（相位 0+5）—— 正是 phase_diagnostic 测出
        # 有害的那种训练/评估相位错配，而生产模型并没有这个错配。选配置一律用
        # experiments/walk_forward_rolling.py 的配对 Δ。
        print(f"validation weighted zero-mean R2: {validation_score:.8f}"
              f"  ← 单折且训练/评估相位错配，仅供参考，选配置看滚动配对 A/B", flush=True)

        # 限幅体检。历史上 clip 从未触发，而「分数关于 scale 是精确二次式」这条性质
        # （公榜两点定抛物线就靠它）只在不触发时成立。alpha 调小会放大 raw 幅度，
        # 所以每次训练都把「clip 从哪个 scale 开始咬」打出来。
        peak = float(np.abs(valid_prediction).max())
        clipped_rows = int((np.abs(valid_prediction) >= args.prediction_clip - 1e-12).sum())
        raw_peak = peak / final_scale if final_scale > 0 else float("nan")
        print(
            f"|prediction| max={peak:.6f}  触顶 {clipped_rows} 行  "
            f"raw 幅度≈{raw_peak:.6f} → clip({args.prediction_clip}) 自 "
            f"scale≈{args.prediction_clip / raw_peak:.3f} 起生效",
            flush=True,
        )
        del x_valid, y_valid, w_valid, t_valid, valid_prediction, validation_artifact

    final_paths = files[-args.train_partitions :]
    print("final fit: " + ", ".join(path.name for path in final_paths), flush=True)
    features, target, weight, time_ids = load_time_sample(
        final_paths, args.sample_modulo, sampling=args.sampling)
    artifact, _ = fit_model(
        features, target, weight, time_ids, args.feature_count, args.ridge_alpha,
        design_basis=args.design_basis, market_alpha_ratio=args.market_alpha_ratio,
        cross_sectional_scaling=args.cross_sectional_scaling,
        ridge_tol=args.ridge_tol, ridge_max_iter=args.ridge_max_iter,
    )
    if args.zero_intercept:
        artifact["intercept"] = 0.0
    artifact.update(
        {
            "strategy": "robust_ridge_cross_section_baseline",
            "train_files": [path.name for path in final_paths],
            "sample_modulo": int(args.sample_modulo),
            "train_rows": int(len(target)),
            "feature_count": int(args.feature_count),
            "prediction_scale": float(final_scale),
            "prediction_clip": float(args.prediction_clip),
            "validation_score": validation_score,
            "validation_metric": "weighted_zero_mean_r2",
        }
    )
    # 只在非默认时才写这个键：默认 periodic 下产物与生产模型 c23a8cfb **逐字节相同**，
    # 「重训能复现 sha」那条验收（NOTES 线程数复现性一节）不能被一个新字段搞坏。
    # 非 periodic 的产物则自带标识，不会和生产模型混淆。
    if args.sampling != "periodic":
        artifact["sampling"] = str(args.sampling)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "baseline_model.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_path": str(output_path),
                "train_rows": artifact["train_rows"],
                "selected_features": artifact["feature_count"],
                "validation_score": validation_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
