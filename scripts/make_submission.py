"""生成私榜提交 zip：只读复制 + 校验 + 压缩，不修改策略源目录。

入包内容：策略根目录下的所有 `*.py`（除 `train.py`）+ `model/`。
校验项：main.py 在包根、Model 可实例化、predict 返回长度正确且全为有限浮点。

用法：
    .venv/bin/python scripts/make_submission.py [--strategy v1_ridge]
输出：
    outputs/<strategy>_submission_<YYYYMMDD>.zip
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 训练侧模块，不进提交包（依赖 src/ 与 sklearn，评测端没有）
EXCLUDED_MODULES = {"train.py"}


def check_v3_hybrid_meta(model_dir: Path, *, off_baseline: bool) -> dict:
    """打包 v3_hybrid 前，把 `hybrid_meta.json` 对公榜那份的配置校一遍。

    存在的理由：`outputs/candidates/*/hybrid_meta.json` 落盘的是
    `blend_weight 0.5` / `prediction_scale 0.856`（train.py 的本地占位值），
    而公榜 0.0032523499 那份是 `variant_submission.py --blend-weight 1.0 --scale 1.16`
    在临时副本上覆写出来的。两者差 1.21e-01（2026-08-13 用留档 CSV 对拍确认）。
    直接拿候选目录打包 = 交出去的不是榜上那个模型，而本脚本原来的烟测查不到这一层。

    期望值以 `scripts/promote_v3_candidate.PUBLIC_BASELINE` 为唯一定义，不在这里再抄一份。
    """
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from promote_v3_candidate import PUBLIC_BASELINE
    finally:
        sys.path.remove(str(_REPO_ROOT / "scripts"))

    meta = json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))
    found = {
        "blend_weight": float(meta.get("blend_weight", float("nan"))),
        "num_iteration": meta.get("num_iteration"),
        "history_window": meta.get("history_window"),
        "history_positions_count": len(meta.get("history_positions") or []),
        "prediction_scale": float(meta.get("prediction_scale", float("nan"))),
        "n_seeds": len(meta.get("lgbm_model_files") or []),
        "market_lambda": float(meta.get("market_lambda", 0.0)),
        "market_model_count": len(meta.get("market_model_files") or []),
        "cross_section_weighted": bool(meta.get("cross_section_weighted", False)),
    }
    # ⚠️ 这张表与 PUBLIC_BASELINE 是**两处派生同一份口径**，08-13 就因为只改了
    # promote_v3_candidate 那边、漏了这里，打包时直接 KeyError。
    # 与其靠记性，不如让不一致当场炸出来并说清楚该改哪。
    missing = set(PUBLIC_BASELINE) - set(found)
    extra = set(found) - set(PUBLIC_BASELINE)
    if missing or extra:
        raise SystemExit(
            "check_v3_hybrid_meta 的取值表与 PUBLIC_BASELINE 不同步："
            f"{'缺 ' + ', '.join(sorted(missing)) if missing else ''}"
            f"{'；多 ' + ', '.join(sorted(extra)) if extra else ''}"
            " —— 往 PUBLIC_BASELINE 加键时，这里的 found 也要同步加一项")

    def differs(key: str) -> bool:
        expected, actual = PUBLIC_BASELINE[key], found[key]
        if isinstance(expected, float) and isinstance(actual, float):
            return not abs(actual - expected) < 1e-12       # NaN 也走这一支 → 判为偏离
        return actual != expected

    drift = [f"{key}: {found[key]!r} != 公榜基线 {PUBLIC_BASELINE[key]!r}"
             for key in PUBLIC_BASELINE if differs(key)]
    print("model/hybrid_meta.json: " + ", ".join(f"{k}={v!r}" for k, v in found.items()))
    if drift:
        message = "提交包的 meta 偏离公榜 0.0039977510 那份：\n  " + "\n  ".join(drift)
        if not off_baseline:
            raise SystemExit(message + "\n有意为之请显式加 --off-baseline")
        print(f"⚠️ 已按 --off-baseline 放行：\n  " + "\n  ".join(drift))
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a strategy directory for private submission.")
    parser.add_argument("--strategy", default="v1_ridge")
    parser.add_argument("--model-dir", default=None,
                        help="Optional model artifact directory. Strategy Python files still come from "
                             "strategies/<strategy>; useful for packaging a validated staging model "
                             "without promoting production first.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs"))
    parser.add_argument("--date-tag", default=None, help="Override YYYYMMDD output tag for reproducible drills.")
    parser.add_argument("--off-baseline", action="store_true",
                        help="显式允许包内 meta 偏离公榜基线（例如 2 种子的超时退路）。"
                             "只对 v3_hybrid 生效；默认拒绝")
    return parser.parse_args()


def smoke_test(package_dir: Path) -> None:
    """从打包目录加载 main.py，模拟官方评测的加载与一次 predict。"""
    import pandas as pd

    main_path = package_dir / "main.py"
    assert main_path.exists(), "main.py 必须在提交包根目录"

    sys.path.insert(0, str(package_dir))
    try:
        spec = importlib.util.spec_from_file_location("submission_main", main_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model()

        rows = 15
        frame = pd.DataFrame(
            {
                "row_id": np.arange(rows),
                "time_id": np.zeros(rows, dtype=np.int64),
                "asset_id": np.arange(rows, dtype=np.int8),
                **{column: np.zeros(rows, dtype=np.float32) for column in model.feature_columns},
            }
        )
        prediction = np.asarray(model.predict(frame))
        assert prediction.shape == (rows,), f"predict 返回形状 {prediction.shape} != ({rows},)"
        assert np.all(np.isfinite(prediction)), "predict 返回了非有限值"
    finally:
        sys.path.remove(str(package_dir))
        sys.modules.pop("submission_main", None)


def main() -> None:
    args = parse_args()
    strategy_dir = _REPO_ROOT / "strategies" / args.strategy
    assert strategy_dir.is_dir(), f"策略目录不存在: {strategy_dir}"

    date_tag = args.date_tag or datetime.date.today().strftime("%Y%m%d")
    output_dir = Path(args.output_dir)
    staging = output_dir / f"{args.strategy}_submission_{date_tag}"
    zip_path = output_dir / f"{args.strategy}_submission_{date_tag}.zip"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # 复制策略根目录下的所有 .py（不递归）+ model/ 产物。
    # 排除 train.py —— 它依赖 src/ 与 sklearn，评测端两者都没有。
    # 用「除 train.py 外全收」而不是写死清单，是因为写死过一次就漏过一次：
    # v3_hybrid 加了 lgbm_numpy.py，而清单还停在 (main.py, features.py)。
    sources = sorted(path for path in strategy_dir.glob("*.py")
                     if path.name not in EXCLUDED_MODULES)
    assert any(path.name == "main.py" for path in sources), "策略目录里没有 main.py"
    for path in sources:
        shutil.copy2(path, staging / path.name)
    model_dir = Path(args.model_dir) if args.model_dir else (strategy_dir / "model")
    if model_dir.is_dir():
        shutil.copytree(model_dir, staging / "model",
                        ignore=shutil.ignore_patterns("promotion_manifest.json",
                                                     "consistency_*.json"))

    # ⚠️ 配置校验必须在烟测**之前** —— 烟测只查形状与有限性，交错模型照样能过。
    if args.strategy == "v3_hybrid":
        check_v3_hybrid_meta(staging / "model", off_baseline=args.off_baseline)

    smoke_test(staging)

    def package_files():
        for item in sorted(staging.rglob("*")):
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                yield item

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in package_files():
            archive.write(item, item.relative_to(staging))

    print(f"OK: {zip_path}")
    print("包内文件:")
    for item in package_files():
        print(f"  {item.relative_to(staging)}")


if __name__ == "__main__":
    main()
