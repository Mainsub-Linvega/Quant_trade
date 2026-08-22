"""Stage C 补测的自检与判定逻辑。

三件事必须钉住，因为它们是「这轮结果能不能读」的前提：

1. **让步是常数平移** —— `mean_delta_vs_frozen_baseline` 对每个臂减的是同一个数。
   不钉住这条，后来的人会把「剥完 16 个里 14 个转正」读成 14 个发现。
2. **复现门** —— 自检臂对不上 08-18 锚点时必须判 FAIL，而不是静默通过。
3. **臂 → 维度族的映射** —— 多重比较纪律第 2 条要靠它读「过门槛的臂是否集中在某一族」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import responder_stage_c_fill as fill  # noqa: E402

RESULT_JSON = _REPO_ROOT / "outputs" / "experiments" / "responder_stage_c_fill.json"
ANCHOR_JSON = _REPO_ROOT / "outputs" / "experiments" / "horizon_auxiliary_cache_probe.json"


def arm_row(mean_delta: float, null_delta: float, *, peak: float = 1.75e-3,
            positive_folds: int = 2) -> dict:
    return {"mean_delta": mean_delta,
            "mean_delta_vs_frozen_baseline": mean_delta - null_delta,
            "baseline_peak_mean": peak, "positive_folds": positive_folds}


def synthetic_results() -> dict:
    null_delta = -6.7e-05
    return {"full": {
        "null_frozen_scale": arm_row(null_delta, null_delta, positive_folds=0),
        "responder_01": arm_row(-2.2e-05, null_delta, positive_folds=2),
        "responder_03": arm_row(-6.8e-05, null_delta, positive_folds=0),
    }}


# ------------------------------------------------------------------ 让步是常数

def test_concession_is_an_exact_constant_shift() -> None:
    summary = fill.summarize_concession(synthetic_results())
    assert summary["max_identity_deviation"] < 1e-18
    assert summary["n_checked"] == 2


def test_concession_ranks_arms_the_same_as_raw_delta() -> None:
    """常数平移不改排序 —— 这是「剥让步不能制造发现」的形式化说法。"""
    results = synthetic_results()
    summary = fill.summarize_concession(results)
    stripped = summary["per_baseline"]["full"]["stripped"]
    by_stripped = sorted(stripped, key=lambda a: -stripped[a])
    by_raw = sorted(stripped, key=lambda a: -results["full"][a]["mean_delta"])
    assert by_stripped == by_raw


def test_concession_flags_a_positive_arm_that_has_zero_positive_folds() -> None:
    """剥完为正但 0/4 折 —— 报告里就是拿这种臂说明「水平变了、证据没变」。"""
    results = synthetic_results()
    results["full"]["responder_06"] = arm_row(-4.6e-05, -6.7e-05, positive_folds=0)
    summary = fill.summarize_concession(results)
    assert summary["per_baseline"]["full"]["stripped"]["responder_06"] > 0
    assert results["full"]["responder_06"]["positive_folds"] == 0


def test_concession_null_arm_strips_to_zero() -> None:
    summary = fill.summarize_concession(synthetic_results())
    assert "null_frozen_scale" not in summary["per_baseline"]["full"]["stripped"]
    assert summary["per_baseline"]["full"]["n_arms"] == 2


# ------------------------------------------------------------------ 复现门

def test_reproduction_check_fails_when_a_point_estimate_drifts(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.json"
    good = {key: 0.5 for key in fill.ANCHOR_KEYS}
    anchor.write_text(json.dumps({"results": {
        baseline: {arm: dict(good) for arm in fill.ALREADY_PROBED}
        for baseline in ("full", "pure_e")}}), encoding="utf-8")

    results = {baseline: {arm: dict(good) for arm in fill.ALREADY_PROBED}
               for baseline in ("full", "pure_e")}
    assert fill.check_reproduction(results, anchor)["ok"]

    results["pure_e"][fill.ALREADY_PROBED[0]]["mean_delta"] = 0.5 + 1e-9
    drifted = fill.check_reproduction(results, anchor)
    assert not drifted["ok"]
    assert drifted["max_abs_delta"] == pytest.approx(1e-9)


def test_reproduction_check_requires_the_anchor_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="找不到锚点"):
        fill.check_reproduction({}, tmp_path / "missing.json")


def test_anchor_keys_exclude_bootstrap_quantities() -> None:
    """自检只比确定性的点估计；CI 因逐臂换随机流会不同，不能当自检项。"""
    assert "paired_bootstrap" not in fill.ANCHOR_KEYS
    assert set(fill.ANCHOR_KEYS) <= {"mean_delta", "relative", "mean_delta_drop_best",
                                     "positive_folds", "delta_A_relative", "delta_B_relative"}


# ------------------------------------------------------------------ 臂 → 族

def test_family_of_maps_every_responder_and_rejects_unknown() -> None:
    families = [{"family": "a", "members": ["responder_00", "responder_01"]},
                {"family": "b", "members": ["responder_02"]}]
    assert fill.family_of("responder_01", families) == "a"
    assert fill.family_of("responder_02", families) == "b"
    with pytest.raises(SystemExit, match="不在族群表里"):
        fill.family_of("responder_46", families)


def test_calibration_arms_are_the_08_18_set() -> None:
    """校准臂不得增删 —— harness_ok 的定义依赖它们。"""
    assert fill.CALIBRATION_ARMS == ("null_frozen_scale", "negctrl_shuffle",
                                     "known_negative_27")
    assert fill.ALREADY_PROBED == ("responder_00", "responder_02")


# ------------------------------------------------------------------ 落盘结果的回归

@pytest.mark.skipif(not RESULT_JSON.is_file(), reason="需要先跑 responder_stage_c_fill")
def test_landed_result_passed_both_validity_gates() -> None:
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    assert payload["verdict"]["reproduction_ok"], "自检臂没复现 08-18 锚点 ⟹ 结果不可解读"
    assert payload["verdict"]["harness_ok"], "harness 校准未过 ⟹ 结果不可解读"
    assert payload["reproduction"]["max_abs_delta"] == 0.0


@pytest.mark.skipif(not RESULT_JSON.is_file(), reason="需要先跑 responder_stage_c_fill")
def test_landed_result_covers_all_fourteen_untested_cells() -> None:
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    untested = payload["stage_c_gap"]["untested"]
    assert len(untested) == 14
    for baseline in payload["results"]:
        assert set(untested) <= set(payload["results"][baseline])


@pytest.mark.skipif(not (RESULT_JSON.is_file() and ANCHOR_JSON.is_file()),
                    reason="需要结果与锚点")
def test_landed_result_reproduces_the_anchor_arms_bit_for_bit() -> None:
    """08-18 那两个臂的点估计必须逐位复现 —— 这是环境与抽取重构的双重自检。"""
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    anchor = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))["results"]
    for baseline in ("full", "pure_e"):
        for arm in fill.ALREADY_PROBED:
            for key in fill.ANCHOR_KEYS:
                assert payload["results"][baseline][arm][key] == anchor[baseline][arm][key]


@pytest.mark.skipif(not RESULT_JSON.is_file(), reason="需要先跑 responder_stage_c_fill")
def test_landed_concession_identity_holds_on_real_numbers() -> None:
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    assert payload["concession"]["max_identity_deviation"] < 1e-18
    for row in payload["concession"]["per_baseline"].values():
        assert row["n_arms"] == len(payload["results"]["full"]) - 1


@pytest.mark.skipif(not RESULT_JSON.is_file(), reason="需要先跑 responder_stage_c_fill")
def test_landed_negative_controls_did_not_pass() -> None:
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    assert not any(payload["negctrl_passes"].values())
    assert all(value < 0 for value in payload["known_negative"].values())
    assert np.all([payload["results"][b]["negctrl_shuffle"]["pass"] is False
                   for b in payload["results"]])
