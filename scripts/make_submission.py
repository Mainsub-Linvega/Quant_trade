"""生成私榜提交 zip：只读复制 + 校验 + 压缩，不修改策略源目录。

入包内容：`SUBMISSION_MODULES` 声明的那几个 `*.py` + `SUBMISSION_EXTRA_FILES` 声明的
非 .py 交付物（v3_hybrid = `requirements.txt`）+ `model/`；
声明集与 `main.py` 的 AST import 闭包双向对拍，缺模块和多模块都当场失败。
校验项：main.py 在包根、Model 可实例化、predict 返回长度正确且全为有限浮点、
`requirements.txt` 覆盖住真实第三方依赖且版本与评测机实测一致。

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
import re
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

REQUIREMENTS_NAME = "requirements.txt"

# 提交包里**非 .py** 的交付物，按策略声明。与 `SUBMISSION_MODULES` 同一套纪律：
# 每个策略都必须被显式分类，空集也要写出来 —— 偏离必须是按下去的，不是漏掉的。
#
# ⚠️ 2026-08-25 补。主办方 08-23 新文档 `submission_and_evaluation.md:53`
# 「最终交付要求」第 3 条：**ZIP 必须包含 `requirements.txt`**。
# 这条要求在旧的 `competition_description.md` / `data_description.md` 里出现 0 次 ——
# 08-24 做更新包审计时抓了新文档的评分公式与 80/20 规则，却没把它的**打包要求**
# 与打包代码对一遍，于是 `20260824.zip` 只有 12 个文件、审计 11/11 全过，
# 而它缺一条明写的硬要求。
SUBMISSION_EXTRA_FILES: dict[str, frozenset[str]] = {
    "v1_ridge": frozenset(),          # 已退役，不再打私榜包；显式声明为空，不是漏掉
    "v3_hybrid": frozenset({REQUIREMENTS_NAME}),
}

# import 根 → 发行包名。两者并不总是同名（`sklearn` 装的是 `scikit-learn`），
# 所以不能靠字符串相等猜。查不到的根一律**硬失败**：将来往策略里引入一个新第三方包，
# 必须有人在这里按一下，才可能通过打包 —— 新依赖不能静默溜进交付物。
IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "numpy": "numpy",
    "lightgbm": "lightgbm",
}

# 归属检查用的**已知真值**（CLAUDE.md 伤疤规则 11）。这份 JSON 是 08-23 在
# **主办方真实评测机**（JupyterHub，`/home/jovyan/Quant_trade`）上落的盘，
# `environment` 块记着那台机器上实际装着的 python / numpy / lightgbm 版本。
# 它是一件**独立于 requirements.txt 的产物** ⟹ 拿本机 `.venv` 的 freeze 冒充
# 评测环境 freeze 时，这道门会当场炸（本机 numpy 2.5.1 ≠ 评测机 1.24.3）。
EVAL_ENV_EVIDENCE = _REPO_ROOT / "outputs" / "cloud" / "delivery_cloud_py311_4t.json"

# 文档「最终交付要求」第 7 条：不得写死 `/home/jovyan` 或队伍专属绝对路径。
# conda 的 `pip freeze` 常见 `pkg @ file:///croot/...`、`file:///opt/conda/...` ——
# 那些是**构建根**不是队伍路径，只记录不拦；`/home/<user>/`、`/Users/` 才拦。
#
# ⚠️ 2026-08-27：conda-forge 的构建根偏偏长成 `/home/conda/feedstock_root/`，
# 会被 `/home/<user>/` 命中。评测机 base 里的 numpy 正是这一形状 ⟹ 一份**合法**的
# 评测机 freeze 会被误判成「写死队伍路径」。构建根按前缀显式豁免，而不是放宽 `/home/`：
# `/home/jovyan/...` 仍然必须拦得住。
_BUILD_ROOT = re.compile(r"file://+/home/conda/feedstock_root/")
_TEAM_PATH = re.compile(r"(/home/[^/\s]+/|/Users/[^/\s]+/)")


def _walk_imports(strategy_dir: Path) -> tuple[set[str], set[str]]:
    """从 `main.py` 出发遍历一次 AST，同时收下两样东西。

    返回 `(本地模块闭包, 闭包里出现过的非本地 import 根)`。

    ⚠️ 两者必须来自**同一次**遍历。分成两个函数各走一遍 AST 迟早会有一处漏改，
    而那正是这个文件反复吃过的亏（08-13 两处手抄同一份口径、只改了一处）。
    """
    local = {path.stem for path in strategy_dir.glob("*.py")}
    seen: set[str] = set()
    foreign: set[str] = set()
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
                if root in local:
                    if root not in seen:
                        stack.append(root)
                else:
                    foreign.add(root)
    return {f"{name}.py" for name in seen}, foreign


def resolve_local_modules(strategy_dir: Path) -> set[str]:
    """`main.py` 靠 import 真正能拉起来的本地模块闭包（含 `main.py` 自己）。

    只跟进「策略目录里确实存在同名 .py」的名字，所以 `main.py` 里那句延迟的
    `import lightgbm` 会被自动忽略（评测端由 pip 提供，不该进包）。
    用 `ast.walk` 而不是只看顶层，函数体内的 import 同样算数。
    """
    return _walk_imports(strategy_dir)[0]


def resolve_third_party_imports(strategy_dir: Path) -> set[str]:
    """提交包在评测端真正需要 pip 提供的那些顶层包。

    就是 `resolve_local_modules()` 忽略掉的那一半 —— 同一次遍历的另一个出口，
    再剔除标准库（`json` / `pathlib` / `re` / `__future__` 都在这里被滤掉）。
    2026-08-25 实测 v3_hybrid 恰为 `{numpy, lightgbm}`。

    为什么要**现算**而不是维护一张清单：`requirements.txt` 是一份从评测机 freeze 来的
    外部文件，我们无法控制它的内容，只能核「它有没有覆盖住我们真正 import 的东西」。
    这个集合必须随代码自动变化，否则新加一个依赖时门禁会继续绿着。
    """
    return {root for root in _walk_imports(strategy_dir)[1]
            if root not in sys.stdlib_module_names}


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
        # ⚠️ 2026-08-21 补：长窗块。榜上那份没有它 ⟹ 基线是 None。
        # 这里取**原值**不走 _as_float —— None==None 才算不偏离；带长窗的候选会被判 drift，
        # 必须显式 `--off-baseline` 才放行（有意偏离的出口）。
        "long_window": meta.get("long_window"),
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


# `pip freeze` 的两种行型。conda base 环境里两种都会出现：pip 装的给 `name==version`，
# conda 装的常给 `name @ file:///.../name-version-cp311-...whl` 这样的直接引用。
_REQ_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
_REQ_DIRECT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)")


def _normalize_distribution(name: str) -> str:
    """PEP 503 规范化：`Foo_Bar.baz` 与 `foo-bar-baz` 是同一个包，不能靠字面比。"""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _version_from_url(distribution: str, url: str) -> str | None:
    """从直接引用的 URL 里抠出版本，抠不出来返回 `None`。

    conda 装的包在 `pip freeze` 里没有 `==`，版本只藏在文件名中
    （`file:///croot/numpy_.../dist/numpy-1.24.3-cp311-cp311-linux_x86_64.whl`）。
    ⚠️ 只认「路径段里那个名字规范化后**等于**目标包名」的候选 —— 否则
    `numpy_and_numpy_base_1708638617955` 这种构建目录会被当成版本 `1708638617955`。
    抠不出来时返回 `None`，由调用方判为「版本不可核」而**不是**当成核过了。
    """
    for match in re.finditer(r"([A-Za-z0-9._-]+)-(\d[A-Za-z0-9._!+]*)", url):
        if _normalize_distribution(match.group(1)) == distribution:
            return match.group(2)
    return None


def analyze_requirements(text: str) -> dict:
    """把 `pip freeze` 的输出解析成可审计的结构。纯函数 —— 打包端与审计端共用这一份。"""
    pins: dict[str, str | None] = {}
    unparsable: list[str] = []
    direct_lines: list[str] = []
    option_lines: list[str] = []
    team_path_lines: list[str] = []
    entries = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries += 1
        if _TEAM_PATH.search(_BUILD_ROOT.sub("file:///<conda-forge-build-root>/", line)):
            team_path_lines.append(line)
        if line.startswith("-"):        # `-e ...`、`--index-url ...` 之类的选项行
            option_lines.append(line)
            continue
        pinned = _REQ_PINNED.match(line)
        direct = _REQ_DIRECT.match(line)
        if pinned:
            pins[_normalize_distribution(pinned.group(1))] = pinned.group(2)
        elif direct:
            name = _normalize_distribution(direct.group(1))
            pins[name] = _version_from_url(name, direct.group(2))
            direct_lines.append(line)
        else:
            unparsable.append(line)
    return {"entry_count": entries, "pins": pins, "unparsable": unparsable,
            "direct_reference_lines": direct_lines, "option_lines": option_lines,
            "team_path_lines": team_path_lines}


def eval_environment_versions(path: Path = EVAL_ENV_EVIDENCE) -> dict[str, str]:
    """评测机上**实际装着**的第三方包版本，取自云端交付验证 JSON 的 `environment` 块。

    这是归属检查的已知真值。文件不存在时返回空字典，由调用方判为「无法核」——
    静默跳过等于把门禁关掉，那正是本次要修的那类洞。
    """
    if not path.exists():
        return {}
    environment = (json.loads(path.read_text(encoding="utf-8")).get("environment") or {})
    return {root: str(environment[root]) for root in IMPORT_TO_DISTRIBUTION
            if environment.get(root)}


def inspect_requirements(text: str, third_party_roots: set[str],
                         eval_versions: dict[str, str]) -> dict:
    """核 `requirements.txt`：既核「够不够」，也核「是不是从评测机来的」。纯函数。

    分两档返回：
    - `problems`：一律硬失败（文件空、行解析不了、写死队伍绝对路径、缺我们真正 import 的包）。
    - `env_drift`：版本与评测机真值对不上 —— 可以用 `--off-env-baseline` 显式放行
      （base 环境真升级过时），但默认拒绝。这是 CLAUDE.md 伤疤规则 11 要的那种
      **能失败、且独立于被测量本身**的归属检查：拿本机 `.venv` 的 freeze 冒充
      评测环境 freeze，会在这里当场炸。
    """
    summary = analyze_requirements(text)
    problems: list[str] = []
    env_drift: list[str] = []

    # 判据是「有没有依赖条目」而不是 `text.strip()` —— 一份只剩注释的文件同样什么都没记录。
    if summary["entry_count"] == 0:
        problems.append(f"{REQUIREMENTS_NAME} 是空的（0 条依赖条目）")
    if summary["unparsable"]:
        problems.append(f"有 {len(summary['unparsable'])} 行不是合法的依赖声明："
                        + "; ".join(summary["unparsable"][:5]))
    if summary["team_path_lines"]:
        problems.append(
            "写死了队伍专属绝对路径（交付要求第 7 条）："
            + "; ".join(summary["team_path_lines"][:5]))

    for root in sorted(third_party_roots):
        distribution = IMPORT_TO_DISTRIBUTION.get(root)
        if distribution is None:
            problems.append(
                f"第三方 import 根 `{root}` 没在 IMPORT_TO_DISTRIBUTION 里登记 —— "
                "新依赖必须有人显式按一下，才能确认它在评测环境里装着")
            continue
        name = _normalize_distribution(distribution)
        if name not in summary["pins"]:
            problems.append(f"`{root}` 是 main.py 的 import 闭包里真正用到的第三方包，"
                            f"但 {REQUIREMENTS_NAME} 里没有 `{distribution}`")
            continue
        expected = eval_versions.get(root)
        actual = summary["pins"][name]
        if expected is None:
            env_drift.append(f"{distribution}: 评测环境真值缺这一项"
                             f"（{EVAL_ENV_EVIDENCE.name} 的 environment 块）⟹ 版本无法核")
        elif actual is None:
            env_drift.append(f"{distribution}: 声明为直接引用、读不出版本 ⟹ 无法与评测机的 "
                             f"{expected} 对拍")
        elif actual != expected:
            env_drift.append(f"{distribution}=={actual} != 评测机实测 {expected}"
                             f"（{EVAL_ENV_EVIDENCE.name}）")

    return {"summary": summary, "problems": problems, "env_drift": env_drift}


def check_requirements(strategy: str, strategy_dir: Path, *, off_env_baseline: bool) -> dict:
    """打包前的依赖闸门。见 `SUBMISSION_EXTRA_FILES` 的注释。"""
    declared = SUBMISSION_EXTRA_FILES.get(strategy)
    if declared is None:
        raise SystemExit(
            f"{strategy} 未在 SUBMISSION_EXTRA_FILES 里分类 —— 要么声明它需要哪些非 .py "
            f"交付物，要么显式写成 frozenset()。不分类就打包 = 又一次「审计全过但缺硬要求」")
    if REQUIREMENTS_NAME not in declared:
        print(f"ℹ️ {strategy} 未声明 {REQUIREMENTS_NAME}，跳过依赖校验")
        return {}

    path = strategy_dir / REQUIREMENTS_NAME
    if not path.exists():
        raise SystemExit(
            f"缺 {path} —— 主办方 08-23 新文档「最终交付要求」第 3 条：ZIP 必须包含 "
            f"{REQUIREMENTS_NAME}。\n"
            f"⚠️ 它必须在**主办方 JupyterHub** 上生成，本机 freeze 不算数：\n"
            f"    cd ~/submit && python -m pip freeze > {REQUIREMENTS_NAME}\n"
            f"再把它放到 {path}")

    report = inspect_requirements(path.read_text(encoding="utf-8"),
                                  resolve_third_party_imports(strategy_dir),
                                  eval_environment_versions())
    summary = report["summary"]
    print(f"{REQUIREMENTS_NAME}: {summary['entry_count']} 条依赖，"
          f"其中直接引用 {len(summary['direct_reference_lines'])} 条；"
          + ", ".join(f"{d}={summary['pins'].get(_normalize_distribution(d))!r}"
                      for d in sorted(IMPORT_TO_DISTRIBUTION.values())))
    if report["problems"]:
        raise SystemExit(f"{REQUIREMENTS_NAME} 校验失败：\n  "
                         + "\n  ".join(report["problems"]))
    if report["env_drift"]:
        message = (f"{REQUIREMENTS_NAME} 与评测机实测环境不符"
                   f"（{EVAL_ENV_EVIDENCE}）：\n  " + "\n  ".join(report["env_drift"]))
        if not off_env_baseline:
            raise SystemExit(message + "\n最可能的原因：这份 freeze 是在本机跑的。"
                             "\nbase 环境确实升级过请显式加 --off-env-baseline"
                             "（并重跑一次云端交付验证刷新真值）")
        print("⚠️ 已按 --off-env-baseline 放行：\n  " + "\n  ".join(report["env_drift"]))
    return report


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
    parser.add_argument("--off-env-baseline", action="store_true",
                        help="显式允许 requirements.txt 的版本偏离评测机实测环境"
                             "（outputs/cloud/delivery_cloud_py311_4t.json）。"
                             "base 环境真升级过才用；默认拒绝")
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
    extras = sorted(SUBMISSION_EXTRA_FILES.get(args.strategy, frozenset()))
    print("入包其它交付物: " + (", ".join(extras) or "—"))
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

    # 依赖闸门同理放在烟测之前：烟测跑在**本机**解释器上，requirements.txt 写什么它都能过。
    check_requirements(args.strategy, strategy_dir, off_env_baseline=args.off_env_baseline)
    for name in sorted(SUBMISSION_EXTRA_FILES.get(args.strategy, frozenset())):
        shutil.copy2(strategy_dir / name, staging / name)

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
