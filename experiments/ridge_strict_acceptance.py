"""汇总 Strict Ridge 的机械验收门禁。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact import sha256_file

OUTPUTS = ROOT / "outputs" / "experiments"
CANDIDATE = ROOT / "outputs" / "candidates" / "v1_ridge_strict" / "baseline_model.json"
PRODUCTION = ROOT / "strategies" / "v1_ridge" / "model" / "baseline_model.json"


def load(name: str) -> dict:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def fold_gate(payload: dict) -> tuple[bool, dict]:
    expected_legacy = {"ridge_tol": 1e-4, "ridge_max_iter": 100, "prediction_scale": 1.13}
    expected_strict = {"ridge_tol": 1e-8, "ridge_max_iter": 2000, "prediction_scale": 1.13}
    arms = payload.get("arms", {})
    arm_configuration_match = all(
        arms.get("production_legacy", {}).get(key) == value
        for key, value in expected_legacy.items()
    ) and all(
        arms.get("production_strict", {}).get(key) == value
        for key, value in expected_strict.items()
    )
    delta = payload["paired_delta"]["production_strict"]
    same_features = all(
        fold["fit_diagnostics"]["production_legacy"]["selected_indices"]
        == fold["fit_diagnostics"]["production_strict"]["selected_indices"]
        for fold in payload["folds"]
    )
    below_max_iter = all(
        fold["fit_diagnostics"]["production_strict"]["ridge_n_iter"]
        < fold["fit_diagnostics"]["production_strict"]["ridge_max_iter"]
        for fold in payload["folds"]
    )
    details = {
        "pooled_delta": float(delta["pooled_delta"]),
        "mean_delta": float(delta["mean_delta"]),
        "same_selected_features": same_features,
        "below_max_iter": below_max_iter,
        "arm_configuration_match": arm_configuration_match,
    }
    return bool(
        details["pooled_delta"] >= -2.16e-5
        and same_features
        and below_max_iter
        and arm_configuration_match
    ), details


def main() -> None:
    base = load("ab_solver_strict_v2.json")
    shifted = load("ab_solver_strict_v2_offhalf.json")
    legacy_repro = load("ridge_repro_legacy.json")
    strict_repro = load("ridge_repro_strict.json")
    consistency = load("ridge_strict_consistency.json")
    sequential = load("ridge_strict_candidate_validation.json")
    production = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    candidate_hash = sha256_file(CANDIDATE)
    production_hash = sha256_file(PRODUCTION)

    base_passed, base_details = fold_gate(base)
    shifted_passed, shifted_details = fold_gate(shifted)
    legacy = legacy_repro["comparisons"]["1_vs_4"]
    strict = strict_repro["comparisons"]["1_vs_4"]
    reproducibility_passed = bool(
        legacy_repro["configuration"]["ridge_tol"] == 1e-4
        and legacy_repro["configuration"]["ridge_max_iter"] == 100
        and strict_repro["configuration"]["ridge_tol"] == 1e-8
        and strict_repro["configuration"]["ridge_max_iter"] == 2000
        and strict["same_selected"]
        and strict["max_abs_prediction_diff"] < 1e-6
        and strict["abs_score_diff"] < 1e-9
    )
    consistency_passed = bool(
        consistency["passed"]
        and consistency["max_abs_diff"] < 1e-6
        and consistency.get("model_sha256") == candidate_hash
    )
    prediction = sequential["prediction_comparison"]
    sequential_passed = bool(
        prediction["baseline_invalid_rows"] == 0
        and prediction["candidate_invalid_rows"] == 0
        and prediction["candidate_clipped_rows"] == 0
        and sequential["predict_calls"] > 0
        and sequential.get("candidate_model_sha256") == candidate_hash
        and sequential.get("baseline_model_sha256") == production_hash
    )

    comparable_keys = (
        "ridge_alpha", "design_basis", "market_alpha_ratio", "sample_modulo",
        "train_rows", "feature_count", "prediction_scale", "prediction_clip", "train_files",
    )
    configuration_match = all(production[key] == candidate[key] for key in comparable_keys)
    configuration_match &= production.get("cross_sectional_scaling", "none") == candidate.get(
        "cross_sectional_scaling", "none"
    )
    configuration_match &= candidate.get("ridge_tol") == 1e-8
    configuration_match &= candidate.get("ridge_max_iter") == 2000
    candidate_converged = bool(
        0 < int(candidate.get("ridge_n_iter", 0)) < int(candidate.get("ridge_max_iter", 0))
    )

    gates = {
        "base_grid": base_passed,
        "half_offset_grid": shifted_passed,
        "thread_reproducibility": reproducibility_passed,
        "train_inference_consistency": consistency_passed,
        "sequential_inference": sequential_passed,
        "configuration_match_except_solver": bool(configuration_match),
        "candidate_converged": candidate_converged,
    }
    payload = {
        "decision": "accept_strict_solver_candidate" if all(gates.values()) else "reject_or_investigate",
        "production_model_sha256": production_hash,
        "candidate_model_sha256": candidate_hash,
        "gates": gates,
        "base_grid": base_details,
        "half_offset_grid": shifted_details,
        "thread_reproducibility": {
            "legacy_max_abs_prediction_diff": legacy["max_abs_prediction_diff"],
            "strict_max_abs_prediction_diff": strict["max_abs_prediction_diff"],
            "improvement_factor": (
                legacy["max_abs_prediction_diff"] / strict["max_abs_prediction_diff"]
                if strict["max_abs_prediction_diff"] > 0 else float("inf")
            ),
            "strict_abs_score_diff": strict["abs_score_diff"],
        },
        "consistency": consistency,
        "sequential_validation": sequential,
        "candidate_solver": {
            "ridge_tol": candidate["ridge_tol"],
            "ridge_max_iter": candidate["ridge_max_iter"],
            "ridge_n_iter": candidate["ridge_n_iter"],
        },
    }
    json_path = OUTPUTS / "ridge_strict_acceptance.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Strict Ridge 验收结果", "",
        f"结论：`{payload['decision']}`。正式模型未覆盖。", "",
        "| 门禁 | 结果 |",
        "|---|---:|",
        *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gates.items()],
        "",
        f"- base-grid pooled Δ：{base_details['pooled_delta']:+.3e}",
        f"- half-offset pooled Δ：{shifted_details['pooled_delta']:+.3e}",
        f"- 线程预测漂移：{legacy['max_abs_prediction_diff']:.3e} → "
        f"{strict['max_abs_prediction_diff']:.3e}（改善 {payload['thread_reproducibility']['improvement_factor']:.1f}×）",
        f"- 候选训练/推理最大差：{consistency['max_abs_diff']:.3e}",
        f"- 完整顺序推理：{sequential['rows']:,} 行 / {sequential['predict_calls']:,} 次调用，"
        f"候选耗时 {sequential['timing']['candidate_predict_total_seconds']:.2f}s，"
        f"非法预测 {prediction['candidate_invalid_rows']}，clip {prediction['candidate_clipped_rows']} 行。",
    ]
    (OUTPUTS / "ridge_strict_acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "gates": gates}, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
