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
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 训练侧模块，不进提交包（依赖 src/ 与 sklearn，评测端没有）
EXCLUDED_MODULES = {"train.py"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a strategy directory for private submission.")
    parser.add_argument("--strategy", default="v1_ridge")
    parser.add_argument("--model-dir", default=None,
                        help="Optional model artifact directory. Strategy Python files still come from "
                             "strategies/<strategy>; useful for packaging a validated staging model "
                             "without promoting production first.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs"))
    parser.add_argument("--date-tag", default=None, help="Override YYYYMMDD output tag for reproducible drills.")
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
