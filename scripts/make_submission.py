"""生成私榜提交 zip：只读复制 + 校验 + 压缩，不修改策略源目录。

入包内容：`SUBMISSION_MODULES` 声明的那几个 `*.py` + `model/`；
声明集与 `main.py` 的 AST import 闭包双向对拍，缺模块和多模块都当场失败。
校验项：main.py 在包根、Model 可实例化、predict 返回长度正确且全为有限浮点。

用法：
    .venv/bin/python scripts/make_submission.py [--strategy v1_ridge]
输出：
    outputs/<strategy>_submission_<YYYYMMDD>.zip
"""

from __future__ import annotations

import argparse
import ast
import datetime
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 明确**不**入包的模块，连同理由。策略目录下的每个 .py 都必须被分类：
# 要么在 SUBMISSION_MODULES 里（入包），要么在这里（不入包）。
# 出现没被分类的新文件一律硬失败 —— 偏离必须是按下去的，不是漏掉的。
EXCLUDED_MODULES: dict[str, str] = {
    "train.py": "训练侧，依赖 src/ 与 sklearn，评测端两者都没有",
    # ⚠️ 2026-08-19：它此前是**入包**的（旧口径「除 train.py 外全收」），
    # 而它只被 experiments/temporal_multiscale.py 与 tests 使用，
    # main.py 的 import 闭包根本够不到 ⟹ 研究改动会白白改变提交包的字节。
    "temporal.py": "V4-T 研究模块，不在 main.py 的 import 闭包里",
}

# 提交包的**内容身份**：包里该有哪些 .py，唯一定义在这里。
# 与 `promote_v3_candidate.PUBLIC_BASELINE`（模型身份）分工对称 ——
# 一个管「装的是不是榜上那个模型」，一个管「装的是不是该装的那些代码」，
# 两者都由 `scripts/audit_submission_zip.py` 消费，不在别处抄第二份。
#
# ⚠️ 这张表**不是唯一依据**。`main()` 里那条注释记着「写死清单曾漏过 lgbm_numpy.py」，
# 所以它只是一份**被校验的声明**：`resolve_local_modules()` 用 AST 求出 main.py 的
# 本地 import 闭包与它对拍，两边不一致当场退出 —— 既不会漏模块，也不会多塞模块。
SUBMISSION_MODULES: dict[str, frozenset[str]] = {
    "v1_ridge": frozenset({"main.py", "features.py"}),
    "v3_hybrid": frozenset({"main.py", "features.py", "lgbm_numpy.py", "history.py"}),
}


def resolve_local_modules(strategy_dir: Path) -> set[str]:
    """`main.py` 靠 import 真正能拉起来的本地模块闭包（含 `main.py` 自己）。

    只跟进「策略目录里确实存在同名 .py」的名字，所以 `main.py` 里那句延迟的
    `import lightgbm` 会被自动忽略（评测端由 pip 提供，不该进包）。
    用 `ast.walk` 而不是只看顶层，函数体内的 import 同样算数。
    """
    local = {path.stem for path in strategy_dir.glob("*.py")}
    seen: set[str] = set()
    stack = ["main"]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        tree = ast.parse((strategy_dir / f"{current}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # 相对 import（level>0）在提交包里不成立：包根是平铺的顶层模块
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in local and root not in seen:
                    stack.append(root)
    return {f"{name}.py" for name in seen}


def check_submission_modules(strategy: str, strategy_dir: Path) -> list[str]:
    """定下入包的 .py 清单，并把三种偏离都变成硬失败。

    ⚠️ 2026-08-19 补的这道闸门。原实现是「除 `train.py` 外全收 `*.py`」，于是纯研究模块
    `temporal.py` 也进了私榜包 —— 它不在 `main.py` 的 import 闭包里，却会因为研究改动
    **改变提交包的字节**；而当时的审计只查缺文件、不查多文件 ⟹ 全程没有任何东西报警。
    多塞模块还有第二层风险：包内 .py 在评测端就是 `sys.path` 上的顶层名字，
    哪天出现一个叫 `types.py` / `logging.py` 的研究文件就会遮蔽标准库。
    """
    declared = SUBMISSION_MODULES.get(strategy)
    present = {path.name for path in strategy_dir.glob("*.py")}
    if declared is None:
        print(f"⚠️ {strategy} 未在 SUBMISSION_MODULES 里声明，"
              f"按旧口径「除 {'/'.join(sorted(EXCLUDED_MODULES))} 外全收」打包")
        return sorted(present - set(EXCLUDED_MODULES))

    reachable = resolve_local_modules(strategy_dir)
    problems: list[str] = []
    if declared != reachable:
        problems.append(
            "声明集与 main.py 的 import 闭包不一致："
            f"闭包里有而未声明 {sorted(reachable - declared) or '—'}；"
            f"声明了但闭包里够不到 {sorted(declared - reachable) or '—'}")
    missing = sorted(declared - present)
    if missing:
        problems.append(f"声明了但策略目录里不存在：{missing}")
    unclassified = sorted(present - declared - set(EXCLUDED_MODULES))
    if unclassified:
        problems.append(
            f"策略目录里有未分类的模块：{unclassified} —— main.py 需要它就加进 "
            f"SUBMISSION_MODULES['{strategy}']，是研究代码就加进 EXCLUDED_MODULES 并写明理由")
    if problems:
        raise SystemExit("提交包内容校验失败：\n  " + "\n  ".join(problems))
    return sorted(declared)


def _as_float(value) -> float:
    """缺键/非数值一律落成 NaN，让 differs() 判为偏离 —— 丢键必须是失败，不是静默通过。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
        # ⚠️ 2026-08-18 补：slow/fast 三个键是公榜 0.0041150085 与 0.0039977510 的全部差别。
        # `float(None)` 会 TypeError，所以缺键先落成 NaN，由下面的 differs() 判为偏离 ——
        # **丢键必须是失败，不是静默通过**。
        "slow_fast_window": _as_float(meta.get("slow_fast_window")),
        "slow_fast_slow_relative": _as_float(meta.get("slow_fast_slow_relative")),
        "slow_fast_fast_relative": _as_float(meta.get("slow_fast_fast_relative")),
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
        if isinstance(actual, float) and isinstance(expected, (int, float)) \
                and not isinstance(expected, bool):
            return not abs(actual - float(expected)) < 1e-12  # NaN 也走这一支 → 判为偏离
        return actual != expected

    drift = [f"{key}: {found[key]!r} != 公榜基线 {PUBLIC_BASELINE[key]!r}"
             for key in PUBLIC_BASELINE if differs(key)]
    print("model/hybrid_meta.json: " + ", ".join(f"{k}={v!r}" for k, v in found.items()))
    if drift:
        message = ("提交包的 meta 偏离公榜 0.0041150085 那份"
                   "（2026-08-18 slow/fast 转正后）：\n  " + "\n  ".join(drift))
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

    # 入包的 .py 由 SUBMISSION_MODULES 声明，再与 main.py 的 AST import 闭包双向对拍。
    # 旧口径是「除 train.py 外全收」—— 它确实漏不掉模块（当年写死清单漏过 lgbm_numpy.py，
    # 那条教训成立），但也拦不住**多余**模块：08-19 研究模块 temporal.py 就是这样进的包。
    # 现在两个方向都有门。
    names = check_submission_modules(args.strategy, strategy_dir)
    assert "main.py" in names, "策略目录里没有 main.py"
    print("入包模块: " + ", ".join(names))
    for name in names:
        shutil.copy2(strategy_dir / name, staging / name)
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
