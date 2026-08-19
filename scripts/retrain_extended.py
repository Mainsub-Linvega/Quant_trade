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
                         "history_positions_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain fixed v3 structure after a verified data refresh.")
    parser.add_argument("--audit", required=True, help="Updated audit JSON containing comparison to baseline.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--candidate-dir", default=str(ROOT / "outputs" / "candidates" /
                                                        "v3_hybrid_extended_fixed"))
    parser.add_argument("--ridge-alpha", type=float, default=2_000_000.0)
    parser.add_argument("--ridge-feature-count", type=int, default=200)
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
    if structure["market_model_count"]:
        structure_flags += [
            "--market-model",
            "--market-lambda", str(structure["market_lambda"]),
            "--market-spec", json.dumps(structure["market_spec"], sort_keys=True,
                                        separators=(",", ":")),   # 无空格 ⟹ 复制到 shell 也不用再引
            "--market-min-data-scale", str(MARKET_MIN_DATA_SCALE),
        ]
    return [
        [python, str(ROOT / "strategies" / "v1_ridge" / "train.py"),
         "--data-root", args.data_root, "--model-dir", candidate,
         "--train-partitions", "999", "--sample-modulo", str(args.sample_modulo),
         "--validation-sample-modulo", "10", "--sampling", "phase_balanced",
         "--feature-count", str(args.ridge_feature_count), "--ridge-alpha", str(args.ridge_alpha),
         "--prediction-scale", str(args.prediction_scale), "--skip-validation"],
        [python, str(ROOT / "strategies" / "v3_hybrid" / "train.py"),
         "--data-root", args.data_root, "--model-dir", candidate,
         "--sample-modulo", str(args.sample_modulo), "--sampling", "phase_balanced",
         "--feature-count", str(args.lgbm_feature_count), "--history-count", "40",
         "--history-window", "5", "--num-iteration", str(args.num_iteration),
         "--n-seeds", str(args.n_seeds), "--prediction-scale", str(args.prediction_scale),
         *structure_flags],
    ]


def validate_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison") or {}
    train = comparison.get("splits", {}).get("train", {})
    train_changed = bool(train.get("added") or train.get("removed") or train.get("modified")
                         or train.get("row_delta"))
    if not comparison.get("changed") or not train_changed:
        raise SystemExit("audit does not prove a changed training split; fixed retraining is blocked")
    return comparison


def main() -> None:
    args = parse_args()
    comparison = validate_audit(Path(args.audit))
    candidate = Path(args.candidate_dir)
    structure = production_structure()
    plan = command_plan(args, structure)
    print(json.dumps({"audit": args.audit, "comparison": comparison,
                      "candidate_dir": str(candidate),
                      "production_structure": structure, "commands": plan,
                      "execute": args.execute}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if candidate.exists():
        if not args.force:
            raise SystemExit(f"candidate exists: {candidate}; pass --force")
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True)
    for command in plan:
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"fixed-structure extended-data candidate: {candidate}")


if __name__ == "__main__":
    main()
