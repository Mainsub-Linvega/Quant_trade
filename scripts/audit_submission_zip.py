"""Audit a private-submission zip without extracting it into the repository.

⚠️ 2026-08-18 补强了三个洞（都属于「审计通过但交出去的不是榜上那个模型」）：

1. 原来只核 `lgbm_model_files` 在不在包里，**不核 `market_model_files`** ——
   市场森林是架构的一半（公榜 +21.99% 的来源），漏打包照样 PASS。
2. 原来不核 `main.py` 无条件 import 的三个模块（`features` / `lgbm_numpy` / `history`），
   少任何一个都会让 `Model` 无法初始化 ⟹ 整份提交判无效。
3. 原来不核 slow/fast。`main.py:222` 是 `PredictionTrail(int(window)) if window else None` ——
   `slow_fast_window` 缺失时 slow/fast 被**静默关掉**、不报错，交出去的就是低 2.93% 的旧模型。
   `--expect-public-baseline` 会拿 `promote_v3_candidate.PUBLIC_BASELINE` 全表核对。

⚠️ 2026-08-19 又补一个方向相反的洞：原来只查**缺**文件，不查**多**文件。
打包那边当时是「除 `train.py` 外全收 `*.py`」，于是纯研究模块 `temporal.py` 混进了包 ——
它不在 `main.py` 的 import 闭包里，却会因为研究改动改变提交包字节。
入包清单现在由 `make_submission.SUBMISSION_MODULES` 唯一定义，本脚本派生并双向核对。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 包内容身份的唯一定义在 make_submission 里（那边还会拿 main.py 的 AST import 闭包
# 双向对拍）。这里**派生**而不是再抄一张表 —— 08-13 就因为两处手抄同一份口径、
# 只改了一处而当场 KeyError。
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from make_submission import SUBMISSION_MODULES

# main.py 顶层无条件 import features / lgbm_numpy / history ⟹ 少一个就装不起来
DECLARED_MODULES = SUBMISSION_MODULES["v3_hybrid"]
REQUIRED = {*DECLARED_MODULES, "model/baseline_model.json", "model/hybrid_meta.json"}
FORBIDDEN_NAMES = {"train.py"}
FORBIDDEN_PREFIXES = ("src/", "data/", "outputs/", ".git/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit submission zip contents and model metadata.")
    parser.add_argument("zip_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-scale", type=float, default=None)
    parser.add_argument("--expected-iterations", type=int, default=None)
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--expect-public-baseline", action="store_true",
                        help="拿 promote_v3_candidate.PUBLIC_BASELINE 全表核对 meta —— "
                             "包含 slow/fast 三个键。打私榜包前应当加上")
    return parser.parse_args()


def public_baseline_drift(meta: dict) -> list[str]:
    """meta 与公榜那份模型身份的逐键差异（空列表 = 完全一致）。"""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from promote_v3_candidate import PUBLIC_BASELINE
    finally:
        sys.path.remove(str(_REPO_ROOT / "scripts"))

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    found = {
        "blend_weight": as_float(meta.get("blend_weight")),
        "num_iteration": meta.get("num_iteration"),
        "history_window": meta.get("history_window"),
        "history_positions_count": len(meta.get("history_positions") or []),
        "prediction_scale": as_float(meta.get("prediction_scale")),
        "n_seeds": len(meta.get("lgbm_model_files") or []),
        "market_lambda": as_float(meta.get("market_lambda")),
        "market_model_count": len(meta.get("market_model_files") or []),
        "cross_section_weighted": bool(meta.get("cross_section_weighted", False)),
        "slow_fast_window": as_float(meta.get("slow_fast_window")),
        "slow_fast_slow_relative": as_float(meta.get("slow_fast_slow_relative")),
        "slow_fast_fast_relative": as_float(meta.get("slow_fast_fast_relative")),
        # 取原值不走 as_float：基线是 None，None==None 才算不偏离
        "long_window": meta.get("long_window"),
    }
    unknown = set(PUBLIC_BASELINE) - set(found)
    if unknown:
        raise SystemExit(f"audit 的取值表缺 PUBLIC_BASELINE 的键：{', '.join(sorted(unknown))}"
                         " —— 往 PUBLIC_BASELINE 加键时这里也要同步加")

    def differs(key: str) -> bool:
        expected, actual = PUBLIC_BASELINE[key], found[key]
        if isinstance(actual, float) and isinstance(expected, (int, float)) \
                and not isinstance(expected, bool):
            return not abs(actual - float(expected)) < 1e-12   # NaN（缺键）也判为偏离
        return actual != expected

    return [f"{key}: {found[key]!r} != 公榜基线 {PUBLIC_BASELINE[key]!r}"
            for key in PUBLIC_BASELINE if differs(key)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def audit(path: Path, expected_scale: float | None = None,
          expected_iterations: int | None = None, expected_seeds: int | None = None,
          expect_public_baseline: bool = False) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        duplicates = sorted({name for name in names if names.count(name) > 1})
        missing = sorted(REQUIRED - set(names))
        forbidden = sorted(name for name in names
                           if Path(name).name in FORBIDDEN_NAMES
                           or name.startswith(FORBIDDEN_PREFIXES)
                           or "__pycache__" in Path(name).parts or name.endswith(".pyc"))
        # ⚠️ 2026-08-19 补：原来只查「缺文件」，**不查多文件**。而打包那边当时是
        # 「除 train.py 外全收 *.py」⟹ 研究模块 temporal.py 一路混进私榜包，
        # 08-19 对它的研究改动直接改变了提交包的字节，全程没有任何门禁出声。
        # 包内的 .py 在评测端就是 sys.path 上的顶层名字，多塞一个还可能遮蔽标准库。
        unexpected_modules = sorted(name for name in names
                                    if name.endswith(".py") and name not in DECLARED_MODULES)
        meta = json.loads(archive.read("model/hybrid_meta.json")) if not missing else {}
        model_files = list(meta.get("lgbm_model_files") or [])
        market_files = list(meta.get("market_model_files") or [])
        absent_models = sorted(name for name in model_files if f"model/{name}" not in names)
        # ⚠️ 市场森林此前完全没被核过 —— 它是架构的一半，漏打包会静默降级
        absent_market = sorted(name for name in market_files if f"model/{name}" not in names)
        drift = public_baseline_drift(meta) if (expect_public_baseline and not missing) else []
        checks = {
            "required_files_present": not missing,
            "no_unexpected_modules": not unexpected_modules,
            "no_forbidden_files": not forbidden,
            "no_duplicate_entries": not duplicates,
            "all_declared_models_present": not absent_models,
            "all_declared_market_models_present": not absent_market,
            "prediction_scale_matches": (expected_scale is None or
                                           abs(float(meta.get("prediction_scale", float("nan")))
                                               - expected_scale) < 1e-12),
            "iterations_match": (expected_iterations is None or
                                  meta.get("num_iteration") == expected_iterations),
            "seed_count_matches": (expected_seeds is None or len(model_files) == expected_seeds),
            "matches_public_baseline": (not expect_public_baseline) or not drift,
        }
        return {
            "zip": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "files": names, "missing": missing, "forbidden": forbidden,
            "unexpected_modules": unexpected_modules,
            "duplicates": duplicates, "absent_declared_models": absent_models,
            "absent_declared_market_models": absent_market,
            "public_baseline_drift": drift,
            "meta_summary": {"prediction_scale": meta.get("prediction_scale"),
                             "num_iteration": meta.get("num_iteration"),
                             "history_window": meta.get("history_window"),
                             "history_positions": len(meta.get("history_positions") or []),
                             "blend_weight": meta.get("blend_weight"),
                             "market_lambda": meta.get("market_lambda"),
                             "cross_section_weighted": meta.get("cross_section_weighted"),
                             "slow_fast_window": meta.get("slow_fast_window"),
                             "slow_fast_slow_relative": meta.get("slow_fast_slow_relative"),
                             "slow_fast_fast_relative": meta.get("slow_fast_fast_relative"),
                             "long_window": meta.get("long_window"),
                             "lgbm_model_files": model_files,
                             "market_model_files": market_files},
            "checks": checks, "passed": all(checks.values()),
        }


def main() -> None:
    args = parse_args()
    result = audit(Path(args.zip_path), args.expected_scale, args.expected_iterations,
                   args.expected_seeds, expect_public_baseline=args.expect_public_baseline)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
