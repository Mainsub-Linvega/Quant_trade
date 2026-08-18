"""出一份 v3_hybrid 变体的公榜 CSV，**不碰生产模型产物**。

## 为什么要有这个

`blend_weight` / `num_iteration` / `prediction_scale` 都写在
`strategies/v3_hybrid/model/hybrid_meta.json` 里，而官方 runner 是
`Model()` 无参构造 —— 想换一个值只能改那份 json。但那是已提交模型的产物，
**在公榜确认之前不该动**（先测后转正）。

所以这里的做法是：把策略目录**只读复制**到临时目录，只改那份**拷贝**的 meta，
再拿**官方 `runner.run_strategy`** 去跑。生产的 `model/` 一个字节都不动，
而口径完全走官方路径、不会漂。

## 8 位小数

主办方示例是 8 位小数（64.1 MB）；`to_csv` 直写 float64 是 97.6 MB，多出来的全是垃圾位
（工程坑第 3 条）。runner 写的是全精度，所以这里再过一道格式化。

用法：
    .venv/bin/python experiments/variant_submission.py --blend-weight 1.0 --scale 1.16 \\
        --output outputs/submission_replace_s116.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "timeseries_api")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from runner import run_strategy                       # 官方 runner，只读引用

STRATEGY_DIR = _REPO_ROOT / "strategies" / "v3_hybrid"
EXCLUDED_MODULES = {"train.py"}                       # 与 scripts/make_submission.py 同口径


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce a v3_hybrid variant submission CSV.")
    parser.add_argument("--blend-weight", type=float, default=None,
                        help="ê 里 LGBM 占的比重；1.0 = replace，默认沿用 meta 的 0.5")
    parser.add_argument("--num-iteration", type=int, default=None,
                        help="用前 k 棵树；不能超过模型文件里的棵数（免训练地减轮数）")
    parser.add_argument("--scale", type=float, default=None, help="prediction_scale")
    parser.add_argument("--model-dir", default=None,
                        help="换一套模型产物（默认用生产的 strategies/v3_hybrid/model）。"
                             "重训出来的候选放在 outputs/candidates/ 下，用这个参数指过去 —— "
                             "推理代码仍然取自 strategies/v3_hybrid/*.py，只换 model/。")
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--decimals", type=int, default=8)
    parser.add_argument("--backend", default=None, choices=["lightgbm", "numpy"],
                        help="⚠️ 官方 runner 用 `Model()` **无参**构造，这里传不进去 —— "
                             "本脚本会直接报错而不是静默忽略。要对拍后端请直接构造 Model("
                             "model_dir, backend=...) 逐 time_id 喂，见 NOTES 的双后端门禁。")
    return parser.parse_args()


def stage(overrides: dict[str, float | int], destination: Path,
          model_dir: Path | None = None) -> Path:
    """只读复制策略目录到 destination，并把 overrides 写进**拷贝**的 hybrid_meta.json。"""
    package = destination / "v3_hybrid"
    package.mkdir(parents=True)
    for path in sorted(STRATEGY_DIR.glob("*.py")):
        if path.name not in EXCLUDED_MODULES:
            shutil.copy2(path, package / path.name)
    shutil.copytree(model_dir or (STRATEGY_DIR / "model"), package / "model")

    meta_path = package / "model" / "hybrid_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        print(f"  {key}: {meta[key]} → {value}")
        meta[key] = value
    meta["variant_note"] = ("由 experiments/variant_submission.py 生成的临时变体，"
                            "不是生产模型；生产 meta 未被改动")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return package


def check(frame: pd.DataFrame, clip: float) -> None:
    """提交前体检：行数、row_id 对齐、有限性、限幅。"""
    sample = pd.read_csv(_REPO_ROOT / "data" / "sample_submission.csv", usecols=["row_id"])
    values = frame["target"].to_numpy(dtype=np.float64)
    aligned = np.array_equal(frame["row_id"].to_numpy(dtype=np.int64),
                             sample["row_id"].to_numpy(dtype=np.int64))
    touched = int((np.abs(values) >= clip - 1e-12).sum())
    print(f"  行数 {len(frame):,}（sample {len(sample):,}）| row_id 逐位对齐 {aligned}")
    print(f"  非有限值 {int((~np.isfinite(values)).sum())} | max|pred| {np.abs(values).max():.4f} "
          f"| 触 clip({clip}) 的行 {touched}")
    assert aligned, "row_id 与 sample_submission.csv 对不齐"
    assert np.all(np.isfinite(values)), "有非有限值"
    if touched:
        print("  ⚠️ 有行触到限幅 → Score(a)=2aA−a²B 的二次式不再精确，两点法会失真")


def main() -> None:
    args = parse_args()
    overrides: dict[str, float | int] = {}
    if args.blend_weight is not None:
        overrides["blend_weight"] = args.blend_weight
    if args.num_iteration is not None:
        overrides["num_iteration"] = args.num_iteration
    if args.scale is not None:
        overrides["prediction_scale"] = args.scale
    # `--model-dir` 本身就是一个变体来源：拿另一套模型产物、用它自己的 meta 跑官方 runner。
    # 这个守卫写在 --model-dir 之前，当时「变体」只能靠覆盖 meta 产生，现在不成立了。
    if not overrides and args.model_dir is None:
        raise SystemExit("至少要覆盖一个参数或指定 --model-dir，否则直接跑 runner 就行了")

    if args.backend is not None:
        # 曾经这个参数被**声明但从未使用** —— 于是 `--backend numpy` 跑出来的其实是
        # lightgbm，两次结果 100% 逐位相同、耗时也一样，差点被当成「双后端对拍通过」。
        # 静默忽略比不支持危险得多，所以这里显式拒绝。
        raise SystemExit(
            "--backend 传不进官方 runner（它用 Model() 无参构造）。"
            "要对拍后端请直接构造 Model(model_dir, backend='numpy') 逐 time_id 喂。")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="v3_hybrid_variant_") as workspace:
        workspace = Path(workspace)
        print("暂存策略副本并改写 meta："
              + (f"（模型产物来自 {args.model_dir}）" if args.model_dir else ""))
        package = stage(overrides, workspace,
                        Path(args.model_dir) if args.model_dir else None)

        # 打印**实际入包**的模型身份 —— 08-13 的事故正是「候选 meta 与实际跑的不一致」。
        staged_meta = json.loads((package / "model" / "hybrid_meta.json").read_text(encoding="utf-8"))
        identity = {k: staged_meta.get(k) for k in
                    ("blend_weight", "num_iteration", "prediction_scale", "prediction_clip",
                     "market_lambda", "cross_section_weighted")}
        identity["n_lgbm_files"] = len(staged_meta.get("lgbm_model_files", []))
        identity["n_market_files"] = len(staged_meta.get("market_model_files", []))
        scales = staged_meta.get("asset_cross_scales")
        identity["asset_cross_scales"] = (f"{len(scales)} 个资产" if scales else "无")
        print("  实际入包的模型身份：" + ", ".join(f"{k}={v}" for k, v in identity.items()))

        raw_path = workspace / "raw.csv"
        print(f"\n跑官方 runner（全量测试集）…", flush=True)
        result = run_strategy(
            data_root=args.data_root, strategy_dir=str(package), output_path=str(raw_path),
            split="test", model_init_timeout_seconds=None, per_step_timeout_seconds=None,
            total_timeout_seconds=None, timeout_policy="zero_step",
        )
        timing = result.timing
        print(f"  status={result.status} rows={result.rows:,} "
              f"predict_total={timing.predict_total_seconds:.1f}s "
              f"({timing.predict_total_seconds/60:.2f} 分钟) "
              f"calls={timing.predict_calls:,} timeouts={timing.predict_timeout_count} "
              f"max_step={timing.max_predict_seconds:.3f}s")
        for message in result.messages:
            print(f"  [{message.level}] {message.code}: {message.message}")

        frame = pd.read_csv(raw_path)

    # clip 必须取**入包那份**的值。原来读的是生产 meta —— 换 --model-dir 时那是另一个模型，
    # 候选若用了不同的 clip，体检就会按错的阈值判（同 08-13 事故的那一类错配）。
    print("\n提交前体检：")
    check(frame, float(staged_meta["prediction_clip"]))

    frame.to_csv(output, index=False, float_format=f"%.{args.decimals}f")
    print(f"\n写出 {output}（{output.stat().st_size/1e6:.1f} MB，{args.decimals} 位小数）")
    print("⚠️ 公榜提交由你执行；生产 model/ 未被改动，可用 "
          "`git diff --stat strategies/v3_hybrid/model/` 确认")


if __name__ == "__main__":
    main()
