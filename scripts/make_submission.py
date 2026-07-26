"""生成私榜提交 zip：只读复制 + 校验 + 压缩，不修改策略源目录。

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a strategy directory for private submission.")
    parser.add_argument("--strategy", default="v1_ridge")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs"))
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

    date_tag = datetime.date.today().strftime("%Y%m%d")
    output_dir = Path(args.output_dir)
    staging = output_dir / f"{args.strategy}_submission_{date_tag}"
    zip_path = output_dir / f"{args.strategy}_submission_{date_tag}.zip"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # 只复制推理所需内容：main.py + features.py + model/ 产物。
    # train.py 不进包 —— 它依赖 src/ 与 sklearn，评测端两者都没有。
    for name in ("main.py", "features.py"):
        shutil.copy2(strategy_dir / name, staging / name)
    model_dir = strategy_dir / "model"
    if model_dir.is_dir():
        shutil.copytree(model_dir, staging / "model")

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
