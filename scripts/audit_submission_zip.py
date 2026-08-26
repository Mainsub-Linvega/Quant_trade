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

⚠️ 2026-08-25 补第三类洞：**「该查的项本身漏了一条」**。主办方 08-23 新文档
`submission_and_evaluation.md:53` 的「最终交付要求」第 3 条明写 ZIP 必须包含
`requirements.txt`，而 `REQUIRED` 里从来没有它 ⟹ `20260824.zip` 只有 12 个文件、
本脚本 11/11 全过。现在非 .py 交付物也由 `make_submission.SUBMISSION_EXTRA_FILES`
派生，并额外核「它覆盖住 main.py 真正 import 的第三方包」「版本与评测机实测一致」。
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
from make_submission import (IMPORT_TO_DISTRIBUTION, REQUIREMENTS_NAME, SUBMISSION_EXTRA_FILES,
                             SUBMISSION_MODULES, _normalize_distribution,
                             eval_environment_versions, inspect_requirements,
                             resolve_third_party_imports)

# main.py 顶层无条件 import features / lgbm_numpy / history ⟹ 少一个就装不起来
DECLARED_MODULES = SUBMISSION_MODULES["v3_hybrid"]
# ⚠️ 2026-08-25：非 .py 的交付物同样**派生**而不是手抄。此前 REQUIRED 里没有
# `requirements.txt` ⟹ 审计 11/11 全过，却漏掉主办方 08-23 新文档「最终交付要求」
# 第 3 条这条明写的硬要求（`20260824.zip` 只有 12 个文件）。
DECLARED_EXTRA_FILES = SUBMISSION_EXTRA_FILES["v3_hybrid"]
REQUIRED = {*DECLARED_MODULES, *DECLARED_EXTRA_FILES,
            "model/baseline_model.json", "model/hybrid_meta.json"}
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
    parser.add_argument("--off-env-baseline", action="store_true",
                        help="显式允许 requirements.txt 的版本偏离评测机实测环境；"
                             "与 make_submission.py 的同名参数配对使用，默认拒绝")
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


def requirements_drift(archive: zipfile.ZipFile) -> tuple[list[str], list[str], dict]:
    """包里的 `requirements.txt` 够不够、以及是不是真从评测机来的。

    返回 `(硬问题, 环境漂移, 摘要)`。解析与判据全部复用 `make_submission` ——
    不在这里抄第二份（这个文件顶上那条注释记着两处手抄同一口径的下场）。

    ⚠️ 第三方 import 根取自**仓库里的** `strategies/v3_hybrid`，与 `DECLARED_MODULES`
    同源。「包里的 .py 是不是就是那份声明」由 `required_files_present` 与
    `no_unexpected_modules` 两道 check 单独把关，所以这里不必再解压一次。
    """
    if REQUIREMENTS_NAME not in archive.namelist():
        return [f"包里没有 {REQUIREMENTS_NAME}"], [], {}
    blob = archive.read(REQUIREMENTS_NAME)
    report = inspect_requirements(blob.decode("utf-8"),
                                  resolve_third_party_imports(_REPO_ROOT / "strategies"
                                                              / "v3_hybrid"),
                                  eval_environment_versions())
    summary = report["summary"]
    return report["problems"], report["env_drift"], {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "entry_count": summary["entry_count"],
        "pinned_versions": {name: summary["pins"].get(_normalize_distribution(name))
                            for name in sorted(IMPORT_TO_DISTRIBUTION.values())},
        "direct_reference_count": len(summary["direct_reference_lines"]),
        "option_line_count": len(summary["option_lines"]),
        "team_path_lines": summary["team_path_lines"],
    }


def frozen_ridge_drift(archive: zipfile.ZipFile) -> list[str]:
    """包里的 `model/baseline_model.json` 是不是榜上那份冻结岭回归。

    ⚠️ 2026-08-24 补。它是**文件身份**，不是 meta 标量 —— meta 里没有任何字段承载它，
    所以 `public_baseline_drift` 那条路看不见它（同理它也不该进 `PUBLIC_BASELINE`，
    否则现存那份已过全部门禁的 20260819.zip 会因为 meta 缺键而当场判 FAIL）。

    为什么要查：`v3_hybrid/train.py:132` 是 `market = group_mean(ridge_raw)`，市场块是
    `m̂ = (1−λ)·m̂_ridge + λ·m̂_lgbm`（λ=0.5）⟹ **换岭回归 = 换市场块 = 交出另一个模型**，
    而 08-24 之前没有任何门禁会发现这件事。
    """
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from promote_v3_candidate import PRODUCTION_RIDGE_SHA256
    finally:
        sys.path.remove(str(_REPO_ROOT / "scripts"))
    name = "model/baseline_model.json"
    if name not in archive.namelist():
        return [f"包里没有 {name}"]
    actual = hashlib.sha256(archive.read(name)).hexdigest()
    if actual == PRODUCTION_RIDGE_SHA256:
        return []
    return [f"{name} sha256 {actual[:16]}… != 冻结岭回归 {PRODUCTION_RIDGE_SHA256[:16]}…"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def audit(path: Path, expected_scale: float | None = None,
          expected_iterations: int | None = None, expected_seeds: int | None = None,
          expect_public_baseline: bool = False, off_env_baseline: bool = False) -> dict:
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
        # ⚠️ 2026-08-25：原来是 `if not missing`，即**任何**必需文件缺失都会把 meta 清空。
        # 往 REQUIRED 里加 requirements.txt 之后，那会让存量包的 meta_summary 整块变空、
        # public_baseline_drift 从 3 条虚涨成 13 条 —— 缺一件交付物不该污染另一件的读数。
        has_meta = "model/hybrid_meta.json" in names
        meta = json.loads(archive.read("model/hybrid_meta.json")) if has_meta else {}
        model_files = list(meta.get("lgbm_model_files") or [])
        market_files = list(meta.get("market_model_files") or [])
        absent_models = sorted(name for name in model_files if f"model/{name}" not in names)
        # ⚠️ 市场森林此前完全没被核过 —— 它是架构的一半，漏打包会静默降级
        absent_market = sorted(name for name in market_files if f"model/{name}" not in names)
        drift = public_baseline_drift(meta) if (expect_public_baseline and has_meta) else []
        req_problems, req_env_drift, req_summary = requirements_drift(archive)
        ridge_drift = frozen_ridge_drift(archive) if expect_public_baseline else []
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
            "frozen_ridge_matches": (not expect_public_baseline) or not ridge_drift,
            # 交付要求第 3 条。无条件核，**不设豁免开关** —— 加开关就是再造一个
            # 「审计过了但缺硬要求」的洞，那正是本次要修的东西。
            "requirements_covers_dependencies": not req_problems,
            "requirements_matches_eval_env": off_env_baseline or not req_env_drift,
        }
        return {
            "zip": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "files": names, "missing": missing, "forbidden": forbidden,
            "unexpected_modules": unexpected_modules,
            "duplicates": duplicates, "absent_declared_models": absent_models,
            "absent_declared_market_models": absent_market,
            "public_baseline_drift": drift,
            "frozen_ridge_drift": ridge_drift,
            "requirements_problems": req_problems,
            "requirements_env_drift": req_env_drift,
            "requirements_summary": req_summary,
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
                   args.expected_seeds, expect_public_baseline=args.expect_public_baseline,
                   off_env_baseline=args.off_env_baseline)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
