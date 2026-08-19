"""Stage, validate, and optionally atomically promote a v3_hybrid model candidate.

The default action is safe: build a separate staging directory and write an auditable manifest. Production
is changed only with the explicit pair ``--activate --allow-production-overwrite``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "strategies" / "v3_hybrid"
PRODUCTION = STRATEGY / "model"

# 公榜 0.0032523499（ledger 2026-08-11）那份 CSV 的**完整**配置。
#
# ⚠️⚠️ 它**不等于** `outputs/candidates/*/hybrid_meta.json` 里落盘的东西 ——
# 候选目录写的是 `blend_weight 0.5` / `prediction_scale 0.856`（train.py 的本地占位值），
# 而所有公榜好成绩都是 `experiments/variant_submission.py --blend-weight 1.0 --scale 1.16`
# 在**临时副本**上覆写出来的，生产 meta 从来没被同步过。
#
# 2026-08-13 用还留着的 `outputs/submission_hist_r480_s116.csv` 前 200 个 time_id 对拍确认：
# `blend_weight=1.0` 的 `max|Δ| = 5.0e-09`（正好是 8 位小数的舍入底噪），
# `blend_weight=0.5` 差 **1.21e-01**（corr 0.9616）。
# ⟹ 转正/打包必须按本表覆写并校验，否则交出去的不是榜上那个模型。
#
# 2026-08-13 晚更新：交付基线换成 `v3_hybrid_mkt_shrunk`（公榜 0.0039977510）。
# 相对上一版多了两个**结构开关**，它们和 `blend_weight` 一样是「设错就交出另一个模型」
# 级别的东西，因此一并进校验表：
#   - `market_lambda` 0.5    行级 LGBM 打 y 的那个市场分量占多少（0 = 整片森林白跑）
#   - `cross_section_weighted` true   截面块是否带 sample_weight 训练
# ⚠️ `market_model_count` 单独列出来：`market_model_files` 为空时 λ 再对也没用。
PUBLIC_BASELINE = {
    "blend_weight": 1.0,
    "num_iteration": 480,
    "history_window": 5,
    "history_positions_count": 40,
    "prediction_scale": 1.16,
    "n_seeds": 3,
    "market_lambda": 0.5,
    "market_model_count": 3,
    "cross_section_weighted": True,
    # ⚠️ 2026-08-18 补：slow/fast 转正后，这三个键**就是**公榜 0.0041150085 与前一版
    # 0.0039977510 的**全部差别**（6 片森林 + 冻结岭回归 hash 逐字节相同、未重训）。
    # 在此之前 PUBLIC_BASELINE 里没有它们 ⟹ 打包时丢键或写错值不会被任何门禁发现，
    # 交出去的会是低 2.93% 的旧模型 —— 正是 CLAUDE.md §8.2 那一类事故。
    "slow_fast_window": 2000,
    "slow_fast_slow_relative": 0.387609649122807,
    "slow_fast_fast_relative": 1.0801809210526316,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage/validate/promote a v3_hybrid candidate.")
    parser.add_argument("--candidate", default=str(ROOT / "outputs" / "candidates" /
                                                    "v3_hybrid_mkt_shrunk"))
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--scale", type=float, default=PUBLIC_BASELINE["prediction_scale"])
    parser.add_argument("--blend-weight", type=float, default=PUBLIC_BASELINE["blend_weight"],
                        help="ê 里 LGBM 占的比重；1.0 = replace，就是公榜那份的口径。"
                             "候选目录落盘的是 0.5（本地占位），**不要**沿用它")
    parser.add_argument("--n-seeds", type=int, choices=[2, 3], default=PUBLIC_BASELINE["n_seeds"])
    # ⚠️ 2026-08-19 补：`strategies/v3_hybrid/train.py` 的 CLI 里**根本没有 slow/fast 概念**
    # ⟹ 任何重训候选的 meta 都不会带这三个键；而 `main.py:222` 是
    # `PredictionTrail(int(window)) if window else None`，缺键会**静默关掉** slow/fast，
    # 退回单一 scale 1.16 的旧模型（公榜低 2.93%）。在此之前唯一的补法是手改候选 JSON。
    # 现在与 --scale / --blend-weight 同一条路：由 staging 写进 meta。
    #   • 沿用当前标定  = 什么都不传（默认即 PUBLIC_BASELINE）
    #   • 用新 OOF 重标定 = experiments/v3_slow_variance.py 算出两个 relative 后显式传入
    parser.add_argument("--slow-fast-window", type=int,
                        default=PUBLIC_BASELINE["slow_fast_window"],
                        help="逐 asset 因果滚动均值的窗口（真实 time_id 步）")
    parser.add_argument("--slow-fast-slow-relative", type=float,
                        default=PUBLIC_BASELINE["slow_fast_slow_relative"],
                        help="慢块 scale 相对 prediction_scale 的乘数")
    parser.add_argument("--slow-fast-fast-relative", type=float,
                        default=PUBLIC_BASELINE["slow_fast_fast_relative"],
                        help="快块 scale 相对 prediction_scale 的乘数")
    parser.add_argument("--off-baseline", action="store_true",
                        help="显式允许 staging 配置偏离公榜基线（例如 2 种子的超时退路）。"
                             "默认拒绝 —— 偏离必须是有意的")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--allow-production-overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_files(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.iterdir() if path.is_file() and path.name != "promotion_manifest.json")


def _float_matches(actual: Any, expected: Any) -> bool:
    """数值键的容差比较；**缺键或非数值一律判不匹配** —— 丢键必须是失败，不是静默通过。"""
    try:
        return abs(float(actual) - float(expected)) < 1e-12
    except (TypeError, ValueError):
        return False


def validate_meta(meta: dict[str, Any], *, scale: float, n_seeds: int, blend_weight: float,
                  off_baseline: bool = False) -> None:
    """staging meta 必须与**请求的**配置逐项一致（结构项则对 PUBLIC_BASELINE）。"""
    checks = {
        "num_iteration_is_480": meta.get("num_iteration") == PUBLIC_BASELINE["num_iteration"],
        "history_window_is_5": meta.get("history_window") == PUBLIC_BASELINE["history_window"],
        "history_positions_is_40":
            len(meta.get("history_positions") or []) == PUBLIC_BASELINE["history_positions_count"],
        "prediction_scale_matches": abs(float(meta.get("prediction_scale", float("nan"))) - scale) < 1e-12,
        # ⚠️ 这一条是 2026-08-13 补的：原来整个脚本没有任何地方看过 blend_weight，
        # 候选目录的 0.5 会被原样带进生产，而榜上那份是 1.0。
        "blend_weight_matches": abs(float(meta.get("blend_weight", float("nan"))) - blend_weight) < 1e-12,
        "model_file_count_matches": len(meta.get("lgbm_model_files") or []) == n_seeds,
        # ⚠️ 2026-08-13 晚补：新架构的两个结构开关，设错同样等于交出另一个模型。
        # `market_lambda=0` 会让整片 1440 棵树的市场森林白跑而**不报任何错**。
        "market_lambda_matches":
            abs(float(meta.get("market_lambda", 0.0)) - PUBLIC_BASELINE["market_lambda"]) < 1e-12,
        # ⚠️ 与**请求的** n_seeds 比，不是与常量 3 比 —— `--n-seeds 2` 是合法的超时退路，
        # 「偏离公榜基线的种子数」由 check_against_public_baseline 单独把关。
        "market_model_count_matches": len(meta.get("market_model_files") or []) == n_seeds,
        "cross_section_weighted_matches":
            bool(meta.get("cross_section_weighted", False)) == PUBLIC_BASELINE["cross_section_weighted"],
        # ⚠️ 2026-08-18 补：slow/fast 三个键同属模型身份（见 PUBLIC_BASELINE 处的注释）。
        # 有意偏离（例如 8/23 回补数据后重训出不带 slow/fast 的候选）走 --off-baseline。
        **{f"{key}_matches": _float_matches(meta.get(key), PUBLIC_BASELINE[key])
           for key in ("slow_fast_window", "slow_fast_slow_relative",
                       "slow_fast_fast_relative")},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed and not off_baseline:
        raise ValueError(f"candidate metadata failed: {', '.join(failed)}"
                         "\n有意为之请显式加 --off-baseline")
    if failed:
        print(f"⚠️ 已按 --off-baseline 放行 meta 偏离：{', '.join(failed)}", flush=True)


def check_against_public_baseline(*, scale: float, n_seeds: int, blend_weight: float,
                                  off_baseline: bool) -> list[str]:
    """请求的配置与公榜那份的差异。默认拒绝偏离 —— 偏离必须是有意按下的。"""
    requested = {"blend_weight": blend_weight, "prediction_scale": scale, "n_seeds": n_seeds}
    drift = [f"{key}: {value} != 公榜基线 {PUBLIC_BASELINE[key]}"
             for key, value in requested.items()
             if abs(float(value) - float(PUBLIC_BASELINE[key])) > 1e-12]
    if drift and not off_baseline:
        raise ValueError(
            "staging 配置偏离公榜 0.0041150085 那份（2026-08-18 slow/fast 转正后）：\n  "
            + "\n  ".join(drift)
            + "\n有意为之请显式加 --off-baseline")
    return drift


def slow_fast_defaults() -> dict[str, float]:
    """slow/fast 三键的默认值 —— 唯一定义仍是 `PUBLIC_BASELINE`，这里只是取一份视图。"""
    return {key: PUBLIC_BASELINE[key] for key in
            ("slow_fast_window", "slow_fast_slow_relative", "slow_fast_fast_relative")}


def stage_candidate(candidate: Path, destination: Path, *, scale: float, n_seeds: int,
                    blend_weight: float = PUBLIC_BASELINE["blend_weight"],
                    slow_fast: dict[str, float] | None = None,
                    force: bool = False, off_baseline: bool = False) -> dict[str, Any]:
    required = [candidate / "baseline_model.json", candidate / "hybrid_meta.json"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"candidate is incomplete: {candidate}")
    if destination.exists():
        if not force:
            raise FileExistsError(f"staging exists: {destination}; pass --force")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    original_meta = json.loads((candidate / "hybrid_meta.json").read_text(encoding="utf-8"))
    selected_models = list(original_meta["lgbm_model_files"])[:n_seeds]
    # ⚠️ 2026-08-13 晚补：原来这里**只复制截面块的森林**，`market_model_files` 一个都没搬 ——
    # staging 目录因此是残缺的，`Model()` 一构造就 FileNotFoundError。
    # （这个 bug 是被 validate_staging 的 Model 构造抓出来的；若当时没有那道校验，
    #   残缺目录会一路走到 --activate。）
    selected_market = list(original_meta.get("market_model_files") or [])[:n_seeds]
    for name in selected_models + selected_market:
        if not (candidate / name).is_file():
            raise FileNotFoundError(candidate / name)
    shutil.copy2(candidate / "baseline_model.json", destination / "baseline_model.json")
    for name in selected_models + selected_market:
        shutil.copy2(candidate / name, destination / name)
    slow_fast = dict(slow_fast_defaults() if slow_fast is None else slow_fast)
    meta = dict(original_meta)
    meta["prediction_scale"] = float(scale)
    # ⚠️ 候选目录落盘的是 0.5（train.py 的先验占位），榜上那份是 1.0 —— 必须显式覆写。
    meta["blend_weight"] = float(blend_weight)
    # ⚠️ 2026-08-19 补：train.py 不认识 slow/fast，重训候选一定缺这三个键，
    # 而缺键会被 main.py 静默降级。与 scale/blend_weight 同一条路，由 staging 写入。
    meta["slow_fast_window"] = int(slow_fast["slow_fast_window"])
    meta["slow_fast_slow_relative"] = float(slow_fast["slow_fast_slow_relative"])
    meta["slow_fast_fast_relative"] = float(slow_fast["slow_fast_fast_relative"])
    meta.setdefault("slow_fast_note",
                    "由 scripts/promote_v3_candidate.py 在 staging 时写入；"
                    "train.py 不产出这三个键。默认值 = PUBLIC_BASELINE（公榜 0.0041150085 那份）")
    meta["lgbm_model_files"] = selected_models
    if selected_market:
        meta["market_model_files"] = selected_market
    meta["promotion_note"] = ("Staged by scripts/promote_v3_candidate.py; source artifacts are unchanged. "
                              f"scale={scale}, blend_weight={blend_weight}, seeds={n_seeds}, "
                              f"slow_fast={slow_fast}")
    validate_meta(meta, scale=scale, n_seeds=n_seeds, blend_weight=blend_weight,
                  off_baseline=off_baseline)
    (destination / "hybrid_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(candidate.resolve()), "destination": str(destination.resolve()),
        "configuration": {"prediction_scale": scale, "blend_weight": blend_weight,
                          "num_iteration": meta.get("num_iteration"), "n_seeds": n_seeds,
                          **slow_fast},
        "public_baseline": dict(PUBLIC_BASELINE),
        "source_meta": {key: original_meta.get(key) for key in ("blend_weight", "prediction_scale",
                                                                "num_iteration")},
        "source_files": {path.name: sha256_file(path) for path in model_files(candidate)},
        "staged_files": {path.name: sha256_file(path) for path in model_files(destination)},
    }
    (destination / "promotion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_model_class():
    sys.path.insert(0, str(STRATEGY))
    for name in ("main", "features", "history", "lgbm_numpy"):
        sys.modules.pop(name, None)
    try:
        return importlib.import_module("main").Model
    finally:
        sys.path.remove(str(STRATEGY))


SMOKE_SEED = 20260813
SMOKE_TIME_IDS = (100, 101)          # 两个 time_id：第一个无历史，第二个才吃到 AssetHistory


def smoke_frames(model) -> list[pd.DataFrame]:
    """固定种子的合成截面。

    2026-08-13 之前这里喂的是全 0 特征。那样 15 行经稳健变换后**完全相同**
    ⟹ `ê_ridge ≡ 0`、`ldev ≡ 0`，整条链路里只剩 `asset_id` 在驱动树 ——
    截面去均值、岭回归的 dev 半边、history 的 4 个块全部退化成常数，**等于没测**。
    （它对 `blend_weight` 仍是敏感的，实测 blend 0 vs 1 差 1.63e-02；
    真正的漏洞是当时**根本没有任何一处断言过 `blend_weight`**，
    而且两个 backend 只比了预测的**均值**。）

    改成在**原始量纲**上按 `center + scale·N(0,1)` 造数（scale 是训练期 IQR），
    变换后落在 ±3 的正常区间，截面结构与 history 才真的被走到。
    """
    stats = {name: (float(c), float(s)) for name, c, s in
             zip(model.ridge_features, model.r_center, model.r_scale)}
    stats.update({name: (float(c), float(s)) for name, c, s in
                  zip(model.lgbm_features, model.l_center, model.l_scale)})
    rng = np.random.default_rng(SMOKE_SEED)
    rows = 15
    frames = []
    for time_id in SMOKE_TIME_IDS:
        columns = {name: (center + scale * rng.normal(0.0, 1.0, rows)).astype(np.float32)
                   for name, (center, scale) in
                   ((name, stats.get(name, (0.0, 1.0))) for name in model.feature_columns)}
        frames.append(pd.DataFrame({
            "row_id": np.arange(rows) + time_id * rows,
            "time_id": np.full(rows, time_id, dtype=np.int64),
            "asset_id": np.arange(rows, dtype=np.int64),
            **columns,
        }))
    return frames


def predict_frames(Model, model_dir: Path, backend: str | None,
                   blend_weight: float | None = None,
                   market_lambda: float | None = None) -> np.ndarray:
    """新建一个 Model（history 状态从空起步）跑完两个 time_id，返回 30 个预测。"""
    model = Model(model_dir, backend=backend)
    if blend_weight is not None:
        model.blend_weight = float(blend_weight)
    if market_lambda is not None:
        model.market_lambda = float(market_lambda)
    return np.concatenate([np.asarray(model.predict(frame), dtype=np.float64)
                           for frame in smoke_frames(model)])


def validate_staging(model_dir: Path, *, off_baseline: bool = False) -> dict[str, Any]:
    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    validate_meta(meta, scale=float(meta["prediction_scale"]),
                  n_seeds=len(meta["lgbm_model_files"]),
                  blend_weight=float(meta["blend_weight"]),
                  off_baseline=off_baseline)
    Model = load_model_class()

    backends: dict[str, Any] = {}
    values: dict[str, np.ndarray] = {}
    for backend in ("numpy", "lightgbm"):
        found = predict_frames(Model, model_dir, backend)
        if found.shape != (30,) or not np.all(np.isfinite(found)):
            raise AssertionError(f"{backend} smoke returned invalid predictions")
        values[backend] = found
        backends[backend] = {"rows": int(len(found)), "max_abs": float(np.abs(found).max()),
                             "mean": float(found.mean()), "std": float(found.std())}
    difference = float(np.max(np.abs(values["numpy"] - values["lightgbm"])))
    if difference > 1e-10:                       # 与 main.py 的开机对拍同一个常数同一个理由
        raise AssertionError(f"numpy/lightgbm staging smoke mismatch: max|Δ| = {difference:.3e}")

    # ---- blend_weight 是不是真的接上了。烟测若对它不敏感，上面那条校验就形同虚设。
    sensitivity = float(np.max(np.abs(
        predict_frames(Model, model_dir, "lightgbm", blend_weight=0.0)
        - predict_frames(Model, model_dir, "lightgbm", blend_weight=1.0))))
    if not np.isfinite(sensitivity) or sensitivity < 1e-6:
        raise AssertionError(
            f"blend_weight 对预测无影响（max|Δ| = {sensitivity:.3e}）—— "
            "要么烟测输入退化，要么这个旋钮没接上；两种都必须停下来查")

    # ---- market_lambda 同理：λ=0 会让整片 1440 棵树的市场森林白跑而不报任何错，
    # 所以「校验它」只有在「它确实影响预测」时才有意义。
    market_sensitivity = float(np.max(np.abs(
        predict_frames(Model, model_dir, "lightgbm", market_lambda=0.0)
        - predict_frames(Model, model_dir, "lightgbm", market_lambda=1.0))))
    if not np.isfinite(market_sensitivity) or market_sensitivity < 1e-6:
        raise AssertionError(
            f"market_lambda 对预测无影响（max|Δ| = {market_sensitivity:.3e}）—— "
            "市场森林没接上，或烟测输入退化")

    model = Model(model_dir, backend="lightgbm")
    if abs(model.blend_weight - float(meta["blend_weight"])) > 1e-12:
        raise AssertionError("Model 读到的 blend_weight 与 meta 不一致")
    if abs(model.market_lambda - float(meta.get("market_lambda", 0.0))) > 1e-12:
        raise AssertionError("Model 读到的 market_lambda 与 meta 不一致")

    result = {"passed": True, "backends": backends,
              "numpy_vs_lightgbm_max_abs": difference,
              "blend_weight": float(meta["blend_weight"]),
              "blend_weight_sensitivity_max_abs": sensitivity,
              "market_lambda": float(meta.get("market_lambda", 0.0)),
              "market_lambda_sensitivity_max_abs": market_sensitivity,
              "cross_section_weighted": bool(meta.get("cross_section_weighted", False)),
              "files": {path.name: sha256_file(path) for path in model_files(model_dir)}}
    # ⚠️ 生产目录**没有** promotion_manifest.json —— `activate_staging` 故意用
    # ignore_patterns 把它排除在外（manifest 不该进提交包）。而这里原来无条件读写它，
    # 于是 `--activate` 结尾那次 `validate_staging(PRODUCTION)` 必然 FileNotFoundError。
    # 这条路以前从没跑过，所以一直没暴露：激活本身已经完成，只有确认那一句没打出来。
    # ⟹ 有 manifest 就记录，没有就跳过（校验本身照跑）。
    manifest_path = model_dir / "promotion_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation"] = result
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    else:
        result["manifest_recorded"] = False
    return result


def activate_staging(staging: Path, production: Path, backup_root: Path) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"model_before_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    incoming = production.parent / f".{production.name}.incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    shutil.copytree(staging, incoming, ignore=shutil.ignore_patterns("promotion_manifest.json"))
    try:
        os.replace(production, backup)
        os.replace(incoming, production)
    except Exception:
        if not production.exists() and backup.exists():
            os.replace(backup, production)
        if incoming.exists():
            shutil.rmtree(incoming)
        raise
    return backup


def main() -> None:
    args = parse_args()
    if args.activate and not args.allow_production_overwrite:
        raise SystemExit("--activate requires --allow-production-overwrite")
    drift = check_against_public_baseline(scale=args.scale, n_seeds=args.n_seeds,
                                          blend_weight=args.blend_weight,
                                          off_baseline=args.off_baseline)
    if drift:
        print("⚠️ 已按 --off-baseline 放行的偏离：\n  " + "\n  ".join(drift))
    candidate = Path(args.candidate)
    stage_dir = Path(args.stage_dir) if args.stage_dir else (
        ROOT / "outputs" / "promotions"
        / f"v3_hybrid_s{args.scale:g}_w{args.blend_weight:g}_{args.n_seeds}seed"
    )
    # ⚠️ 文档里的用法是两步：先跑一次只 staging、看一眼，再加 --activate 转正。
    # 但第二步会**再次**走到 stage_candidate，撞上「目录已存在」的防覆盖闸门。
    # 加 --force 能过，代价是把你刚检查过的那份删掉重建 —— 语义就从
    # 「激活我看过的那个」变成「激活一个新构建的」，在打包当天是个陷阱。
    # 改成：目录已存在且不是 --force 时，**复用并复验**它，同时核对它确实出自同一个候选。
    reused = stage_dir.exists() and not args.force
    if reused:
        manifest_path = stage_dir / "promotion_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"{stage_dir} 已存在但没有 promotion_manifest.json —— "
                             "不是本脚本产出的目录，拒绝复用；换 --stage-dir 或加 --force")
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["source"]
        if Path(recorded).resolve() != candidate.resolve():
            raise SystemExit(f"{stage_dir} 是从 {recorded} 建的，与 --candidate "
                             f"{candidate} 不符 —— 拒绝激活错的东西；要重建请加 --force")
        print(f"复用已存在的 staging（出自 {Path(recorded).name}），只复验不重建；"
              f"要重建请加 --force")
    else:
        stage_candidate(candidate, stage_dir, scale=args.scale, n_seeds=args.n_seeds,
                        blend_weight=args.blend_weight,
                        slow_fast={"slow_fast_window": args.slow_fast_window,
                                   "slow_fast_slow_relative": args.slow_fast_slow_relative,
                                   "slow_fast_fast_relative": args.slow_fast_fast_relative},
                        force=args.force, off_baseline=args.off_baseline)
    validation = validate_staging(stage_dir, off_baseline=args.off_baseline)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"{'revalidated' if reused else 'staged and validated'}: {stage_dir}")
    if args.activate:
        backup = activate_staging(stage_dir, PRODUCTION, ROOT / "outputs" / "promotions" / "backups")
        validate_staging(PRODUCTION, off_baseline=args.off_baseline)
        print(f"production activated; backup: {backup}")


if __name__ == "__main__":
    main()
