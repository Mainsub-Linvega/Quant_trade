"""Gate and orchestrate fixed-structure retraining after an organizer data refresh.

The command refuses to train unless ``audit_data_release.py`` proves that the training split changed.
Default mode is a dry-run command plan. ``--execute`` runs the two existing production trainers into a new
candidate directory and never overwrites ``strategies/*/model``.

⚠️ 2026-08-19：命令计划此前**不复现当前生产架构** —— 缺 ``--weighted-cross-section`` /
``--market-model`` / ``--market-lambda`` / ``--market-spec`` / ``--market-min-data-scale``，
跑出来的是 08-11 那版结构（公榜 0.0032523499，比生产低 21.99%）。现在这些项由
``production_structure()`` 从生产 ``hybrid_meta.json`` 派生，并与 ``PUBLIC_BASELINE`` 对拍。
slow/fast 三键不在这里 —— ``train.py`` 没有那个概念，由
``scripts/promote_v3_candidate.py`` 在 staging 时写入。

⚠️⚠️ 2026-08-24 两处修正（8/23 回补包实际是**增量包**，与预注册假设对不上）：

1. **``--role`` 成为必填门禁。** RUNBOOK D1 写死「决策期重训必须止于
   ``time_id 1,045,889``，训进密封段则 D2 之后的一切比较全部作废 —— 而且**不会报错**」，
   但 ``strategies/{v1_ridge,v3_hybrid}/train.py`` 都没有时间截断参数，
   ``src/io.py:20`` 按 manifest 顺序整分区读 ⟹ 此前这条纪律**没有任何机械手段**。
   现在训练段边界由 ``scripts/build_extended_data_root.py`` 写进
   ``<data-root>/root_identity.json``，本脚本读它并与
   ``outputs/experiments/sealed_period_plan.json`` 对拍；缺文件、role 不符、
   边界不符一律**拒绝生成计划**（fail closed）。

2. **删掉了「重训岭回归」那条命令。** 原计划第一条是
   ``v1_ridge/train.py --train-partitions 999``，有两个问题：
   - 它**跑不起来**：``v1_ridge/train.py:261`` 是
     ``if len(files) < args.train_partitions + 1: raise``，而 9/11/12 恒 ``< 1000``，
     且这一句在 ``--skip-validation`` 分支**之前** ⟹ 立刻抛
     ``ValueError("not enough chronological train partitions")``。
     「08-18 干跑验证过」只覆盖了 dry-run 打印，从未执行过。
   - 它**不该跑**：``v3_hybrid/train.py:607`` 自己写着岭回归是「冻结拷贝，不重训」，
     ledger 从 08-08 起每一版 v3 都是逐位复用 ⟹ 同时换岭回归会把「扩展数据的增量」
     和「换市场块」两件事混在一起（CLAUDE.md §5.2 一次只回答一个问题）。
   现在改为把生产的冻结岭回归**原样拷进候选目录**，并记录 sha256；
   身份由 ``promote_v3_candidate.PRODUCTION_RIDGE_SHA256`` 把关。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_META = ROOT / "strategies" / "v3_hybrid" / "model" / "hybrid_meta.json"
PRODUCTION_RIDGE = ROOT / "strategies" / "v3_hybrid" / "model" / "baseline_model.json"
SEALED_PLAN = ROOT / "outputs" / "experiments" / "sealed_period_plan.json"

# role → 该数据根的 train 段 time_id 上界该等于什么。
#   decision      止于密封段之前（值从 sealed_period_plan.json 读，不在这里写死）
#   extended_full 用满 100% 数据，只用于 D4.5 最终交付件（那一份**不参与任何比较**）
ROLE_DECISION = "decision"
ROLE_FULL = "extended_full"
# 本地公榜协议的训练根：止于 888,479 ⟹ 整个公榜窗口对它都是样本外
ROLE_ORIGINAL = "original"

# 市场块超参里只有这四项由 --market-spec 传；min_data_in_leaf 走倍数（见下）。
MARKET_SPEC_KEYS = ("num_leaves", "learning_rate", "feature_fraction", "lambda_l2")

# 市场块 min_data_in_leaf 的**倍数**。meta 里落的是解析后的绝对值（75580），
# 而 train.py 收的是倍数，且扩展数据后训练行数会变 ⟹ 只能传倍数。
# 用生产数逐位核对过：
#   MIN_DATA_FRAC × train_rows = (12000/3.5e6) × 2,645,530 = 9070 = lgbm_params.min_data_in_leaf
#   9070 × 8.333                                            = 75580 = market_lgbm_params.min_data_in_leaf
# 出处：experiments/ledger.csv 2026-08-13 `v3_hybrid_mkt_shrunk`（公榜 +0.77%）。
MARKET_MIN_DATA_SCALE = 8.333

# 与 PUBLIC_BASELINE 逐项对齐的结构键（8/23 之前就要红，而不是训练几小时之后才红）
BASELINE_CHECKED_KEYS = ("market_lambda", "market_model_count", "cross_section_weighted",
                         "num_iteration", "n_seeds", "history_window",
                         "history_positions_count",
                         # ⚠️ 2026-08-23 补：长窗块 08-21 转正（公榜 +1.662%），但这张表
                         # 与 production_structure() 都没跟上 ⟹ 重训计划不传 --long-window，
                         # 而 train.py 的默认是 0（＝关闭）⟹ D1 会训出一个没有长窗的候选。
                         # 转正门禁最终会拦下，但那是在几小时训练之后。
                         # 覆盖由 tests/test_model_identity_key_coverage.py 机械保证。
                         "long_window")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain fixed v3 structure after a verified data refresh.")
    parser.add_argument("--audit", required=True, help="Updated audit JSON containing comparison to baseline.")
    parser.add_argument("--data-root", required=True,
                        help="必须是 build_extended_data_root.py 产出的派生根（带 root_identity.json）")
    parser.add_argument("--role", required=True, choices=(ROLE_DECISION, ROLE_FULL, ROLE_ORIGINAL),
                        help=f"{ROLE_DECISION}=决策期（止于密封段前，用于 D1/D2 比较）；"
                             f"{ROLE_FULL}=100% 数据（只用于 D4.5 最终交付件）；"
                             f"{ROLE_ORIGINAL}=止于 888,479（本地公榜协议，公榜窗口全样本外）")
    parser.add_argument("--candidate-dir", default=str(ROOT / "outputs" / "candidates" /
                                                        "v3_hybrid_extended_fixed"))
    parser.add_argument("--lgbm-feature-count", type=int, default=200)
    parser.add_argument("--sample-modulo", type=int, default=5)
    parser.add_argument("--num-iteration", type=int, default=480)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--prediction-scale", type=float, default=1.16)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def production_structure(meta_path: Path = PRODUCTION_META) -> dict:
    """从生产 `hybrid_meta.json` 读出「固定结构重训」必须复现的那几项。

    ⚠️ 2026-08-19 之前没有这个函数，`command_plan` 里那条 v3_hybrid 命令只传了
    轮数 / 种子数 / 特征数 / history —— `--weighted-cross-section` 与 `--market-model`
    两个 store_true **一个都没传**（默认 False）。于是 8/23 跑出来的所谓「固定结构」候选
    根本没有行级市场森林、截面块也不带权，等于退回 08-11 那版架构
    （公榜 0.0032523499，比当前生产低 21.99%）。转正门禁确实会拦住它，
    但那是在几小时训练之后 —— 8/23 到 8/31 只有 8 天。

    结构从生产产物**派生**而不是在这里再抄一份常量：CLAUDE.md §7
    「不在多处手工维护同一个数字」。
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    market_params = meta.get("market_lgbm_params") or {}
    missing_spec = [key for key in MARKET_SPEC_KEYS if key not in market_params]
    if missing_spec:
        raise SystemExit(f"生产 meta 的 market_lgbm_params 缺 {missing_spec}；"
                         "无法复现市场块容量，重训计划拒绝生成")
    return {
        "cross_section_weighted": bool(meta.get("cross_section_weighted", False)),
        "market_model_count": len(meta.get("market_model_files") or []),
        "market_lambda": float(meta.get("market_lambda", 0.0)),
        "market_spec": {key: market_params[key] for key in MARKET_SPEC_KEYS},
        "num_iteration": meta.get("num_iteration"),
        "history_window": meta.get("history_window"),
        "history_positions_count": len(meta.get("history_positions") or []),
        "n_seeds": len(meta.get("lgbm_model_files") or []),
        # 取**原值**、不做 int() 兜底 —— None / 0 / 512 必须可区分：
        # `main.py` 缺键或 0 都是「关掉长窗」，而 512 是榜上那份。
        # 与 audit_submission_zip.public_baseline_drift 对该键「不走 as_float」同口径。
        "long_window": meta.get("long_window"),
    }


def assert_matches_public_baseline(structure: dict) -> None:
    """生产 meta 与 `PUBLIC_BASELINE` 哪天分家，必须在这里当场炸。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from promote_v3_candidate import PUBLIC_BASELINE
    finally:
        sys.path.remove(str(ROOT / "scripts"))

    def differs(key: str) -> bool:
        expected, actual = PUBLIC_BASELINE[key], structure[key]
        if isinstance(expected, bool) or isinstance(actual, bool):
            return bool(actual) != bool(expected)
        if isinstance(expected, (int, float)):
            try:
                return not abs(float(actual) - float(expected)) < 1e-12
            except (TypeError, ValueError):
                return True
        return actual != expected

    drift = [f"{key}: 生产 meta {structure[key]!r} != PUBLIC_BASELINE {PUBLIC_BASELINE[key]!r}"
             for key in BASELINE_CHECKED_KEYS if differs(key)]
    if drift:
        raise SystemExit("生产模型结构与公榜基线不一致，重训计划拒绝生成：\n  "
                         + "\n  ".join(drift))


def command_plan(args: argparse.Namespace, structure: dict | None = None) -> list[list[str]]:
    python = str(ROOT / ".venv" / "bin" / "python")
    candidate = str(Path(args.candidate_dir))
    if structure is None:
        structure = production_structure()
    assert_matches_public_baseline(structure)
    # 08-13 起生产架构里的两个结构开关 + 市场块容量收缩。它们都是 store_true / 可选值，
    # 不传就是**另一个模型**，而不是「用默认值」。
    structure_flags: list[str] = []
    if structure["cross_section_weighted"]:
        structure_flags.append("--weighted-cross-section")
    if structure.get("long_window"):
        # 值从生产 meta 派生，不在这里写常量（CLAUDE.md §7「不在多处手工维护同一个数字」）
        structure_flags += ["--long-window", str(int(structure["long_window"]))]
    if structure["market_model_count"]:
        structure_flags += [
            "--market-model",
            "--market-lambda", str(structure["market_lambda"]),
            "--market-spec", json.dumps(structure["market_spec"], sort_keys=True,
                                        separators=(",", ":")),   # 无空格 ⟹ 复制到 shell 也不用再引
            "--market-min-data-scale", str(MARKET_MIN_DATA_SCALE),
        ]
    # ⚠️ 2026-08-24：这里**只剩一条命令**。原来的第一条
    # `v1_ridge/train.py --train-partitions 999` 已删除 —— 它既跑不起来
    # （`v1_ridge/train.py:261` 的 `len(files) < train_partitions + 1` 恒真，
    #  且在 `--skip-validation` 之前就 raise），也不该跑
    # （`v3_hybrid/train.py:607`：岭回归是冻结拷贝，不重训）。
    # 冻结岭回归由 `main()` 在跑命令之前拷进候选目录，见那里。
    return [
        [python, str(ROOT / "strategies" / "v3_hybrid" / "train.py"),
         "--data-root", args.data_root, "--model-dir", candidate,
         "--sample-modulo", str(args.sample_modulo), "--sampling", "phase_balanced",
         "--feature-count", str(args.lgbm_feature_count), "--history-count", "40",
         "--history-window", "5", "--num-iteration", str(args.num_iteration),
         "--n-seeds", str(args.n_seeds), "--prediction-scale", str(args.prediction_scale),
         *structure_flags],
    ]


def load_root_identity(data_root: Path, role: str) -> dict:
    """数据根的训练段边界门禁 —— **fail closed**：读不到就拒绝，不猜。

    这是 RUNBOOK D1「训练段必须止于 1,045,889」的机械手段。原文自己写着训进密封段
    「不会报错」，所以这道门必须在**生成计划时**就响，而不是等训练几小时之后。
    """
    identity_path = data_root / "root_identity.json"
    if not identity_path.is_file():
        raise SystemExit(
            f"{data_root} 里没有 root_identity.json ⟹ 拒绝生成计划。\n"
            f"训练段边界必须是可核的：先跑\n"
            f"  .venv/bin/python scripts/build_extended_data_root.py --execute\n"
            f"再用 outputs/data_roots/{role} 作 --data-root。\n"
            f"⚠️ 直接拿 data/ 或回补包原目录重训会静默训进密封段（RUNBOOK D1）。")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("role") != role:
        raise SystemExit(f"--role {role} 与数据根声明的 role={identity.get('role')!r} 不符："
                         f"{identity_path}")

    actual = identity.get("train_time_id_max")
    if role == ROLE_DECISION:
        if not SEALED_PLAN.is_file():
            raise SystemExit(f"找不到 {SEALED_PLAN} —— 决策期边界不能靠本脚本猜")
        geometry = json.loads(SEALED_PLAN.read_text(encoding="utf-8")).get("geometry") or {}
        expected = geometry.get("decision_train_time_id_max")
        if not isinstance(expected, int):
            raise SystemExit(f"{SEALED_PLAN} 的 geometry.decision_train_time_id_max 缺失")
        if actual != expected:
            raise SystemExit(
                f"⚠️⚠️ 决策期数据根的训练段止于 time_id {actual}，密封期计划要求 {expected}。\n"
                f"差值 {(actual or 0) - expected} —— 训进密封段等于把测试集喂给模型，"
                f"D2 之后的一切比较全部作废。拒绝生成计划。")
    elif actual is None:
        raise SystemExit(f"{identity_path} 没有 train_time_id_max，无法核对边界")
    return identity


def validate_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison") or {}
    train = comparison.get("splits", {}).get("train", {})
    train_changed = bool(train.get("added") or train.get("removed") or train.get("modified")
                         or train.get("row_delta"))
    if not comparison.get("changed") or not train_changed:
        raise SystemExit("audit does not prove a changed training split; fixed retraining is blocked")
    return comparison


def frozen_ridge_sha256() -> str:
    """生产冻结岭回归的 sha256，并当场与公榜身份常量对拍。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from promote_v3_candidate import PRODUCTION_RIDGE_SHA256, sha256_file
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    if not PRODUCTION_RIDGE.is_file():
        raise SystemExit(f"生产冻结岭回归不在盘上：{PRODUCTION_RIDGE}")
    actual = sha256_file(PRODUCTION_RIDGE)
    if actual != PRODUCTION_RIDGE_SHA256:
        raise SystemExit(
            f"生产 baseline_model.json 的 sha256 {actual[:16]}… 与公榜身份常量 "
            f"{PRODUCTION_RIDGE_SHA256[:16]}… 不符 ⟹ 拒绝生成计划。\n"
            f"岭回归自 2026-08-08 起逐位冻结；它变了说明生产目录被动过。")
    return actual


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    identity = load_root_identity(data_root, args.role)
    comparison = validate_audit(Path(args.audit))
    candidate = Path(args.candidate_dir)
    structure = production_structure()
    ridge_sha = frozen_ridge_sha256()
    plan = command_plan(args, structure)
    print(json.dumps({"audit": args.audit, "comparison": comparison,
                      "role": args.role, "data_root": str(data_root),
                      "train_time_id_max": identity.get("train_time_id_max"),
                      "train_rows": identity.get("train_rows"),
                      "train_partitions": identity.get("train_partitions"),
                      "truncated_member": identity.get("truncated_member"),
                      "candidate_dir": str(candidate),
                      "frozen_ridge_sha256": ridge_sha,
                      "production_structure": structure, "commands": plan,
                      "execute": args.execute}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if candidate.exists():
        if not args.force:
            raise SystemExit(f"candidate exists: {candidate}; pass --force")
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)
    # ⚠️ 必须在跑 v3_hybrid/train.py **之前**放进去：`train.py:402` 断言它存在。
    shutil.copy2(PRODUCTION_RIDGE, candidate / "baseline_model.json")
    print(f"冻结岭回归已拷入候选目录（sha256 {ridge_sha[:16]}…）：不重训", flush=True)
    for command in plan:
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"fixed-structure extended-data candidate: {candidate}")


if __name__ == "__main__":
    main()
