"""responder 族群表的分族规则与 Stage C 缺口派生。

两类用例分开：
- **规则用例**用合成的列统计，钉住 `sign_class` / 梯子重启 / 截断梯子三条规则的边界；
- **回归用例**读真实 parquet 的 row-group 统计（只读元数据，不加载数据），钉住
  「47 列切成 8 个族、大小 7/7/7/7/3/7/5/4」这个结果本身 —— 分族规则一旦漂移就会红。

⚠️ `sign_class` 的容差两端都撞过：`max > 0.5` 太松（路径类长窗成员被误判进 CDF 类），
`abs(max − 1.0) <= 1e-6` 太紧（`responder_00` 的 0.9999723 被踢出去），两次都切出 9 族。
下面有用例把这两个反面都钉住。
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

from responder_family_grid import (  # noqa: E402
    N_RESPONDERS, RESPONDER_COLUMNS, UNIT_INTERVAL_TOL, _ladder_restarts, build_families,
    read_column_stats, sign_class, truncation_rungs, untested_stage_c_cells,
)

DATA_PARQUET = _REPO_ROOT / "data" / "train" / "train_partition_000.parquet"
STAGE_B_JSON = (_REPO_ROOT / "outputs" / "experiments"
                / "responder_predictability_reaudit_phasebal_prodwindow.json")


def make_row(index: int, null_count: int, minimum: float, maximum: float) -> dict:
    return {"responder": f"responder_{index:02d}", "index": index, "null_count": null_count,
            "min": minimum, "max": maximum,
            "sign_class": sign_class(minimum, maximum)}


# ------------------------------------------------------------------ 规则

def test_sign_class_separates_the_four_signatures() -> None:
    assert sign_class(-0.2449, 0.0) == "nonpositive"
    assert sign_class(-0.0, 1.0) == "unit_interval"
    assert sign_class(0.0, 0.0102) == "nonnegative"
    assert sign_class(-4.0871, 4.2893) == "bidirectional"


def test_unit_interval_admits_responder_00_and_excludes_the_path_family() -> None:
    """两个实测的边界值 —— 容差改动一旦让任一条翻转，分族就会从 8 族变 9 族。"""
    assert sign_class(0.00030537531711161137, 0.9999722838401794) == "unit_interval"
    assert sign_class(0.00240972894243896, 0.8709947466850281) == "nonnegative"
    assert 1.0 - 0.9999722838401794 < UNIT_INTERVAL_TOL < 1.0 - 0.8709947466850281


def test_ladder_restart_only_fires_after_a_rise() -> None:
    """CDF 族开头的 279 → 9 → 1 → 0 是上升之前的下降，不算窗口梯子重启。"""
    assert _ladder_restarts([279, 9, 1, 0, 422, 934, 2397]) == []
    assert _ladder_restarts([0, 0, 0, 0, 422, 934, 2397, 0, 0, 0, 0, 422, 934, 2397]) == [7]
    assert _ladder_restarts([0, 0, 0, 0, 4, 9, 24, 0, 0, 4, 9, 24, 0, 526, 1035, 2490]) == [7, 12]


def test_truncation_rungs_take_the_increasing_suffix_only() -> None:
    """只取严格递增后缀 —— 否则 CDF 族的尾巴与另外三族相同这件事会被开头三个数掩盖。"""
    assert truncation_rungs([279, 9, 1, 0, 422, 934, 2397]) == [422, 934, 2397]
    assert truncation_rungs([0, 0, 0, 0, 422, 934, 2397]) == [422, 934, 2397]
    assert truncation_rungs([0, 526, 1035, 2490]) == [526, 1035, 2490]
    assert truncation_rungs([0, 0, 0, 0]) == []


def test_build_families_splits_a_run_at_the_ladder_restart() -> None:
    """同符号类的连续下标，遇到梯子重启要切开（真实数据里 14–20 与 21–27 就是这么分的）。"""
    rows = [make_row(i, n, -0.0, 0.13) for i, n in
            enumerate([0, 0, 0, 0, 422, 934, 2397, 0, 0, 0, 0, 422, 934, 2397])]
    families = build_families(rows)
    assert [len(f["members"]) for f in families] == [7, 7]
    assert families[0]["truncation_rungs"] == families[1]["truncation_rungs"] == [422, 934, 2397]


def test_family_reading_does_not_invent_a_label_for_nonnegative() -> None:
    """主办方没公布哪个非负族是上行 / 路径 / 摩擦 ⟹ 只报量级，不编标签。"""
    rows = [make_row(i, 0, 0.0, 0.0102) for i in range(3)]
    reading = build_families(rows)[0]["reading"]
    assert "判不出" in reading
    assert "非负" in reading


# ------------------------------------------------------------------ 回归（读真实元数据）

@pytest.mark.skipif(not DATA_PARQUET.is_file(), reason="需要 train 分区")
def test_real_partition_splits_into_eight_families_covering_all_47() -> None:
    families = build_families(read_column_stats(DATA_PARQUET))
    assert [len(f["members"]) for f in families] == [7, 7, 7, 7, 3, 7, 5, 4]
    covered = [m for f in families for m in f["members"]]
    assert sorted(covered) == sorted(RESPONDER_COLUMNS)
    assert len(covered) == N_RESPONDERS


@pytest.mark.skipif(not DATA_PARQUET.is_file(), reason="需要 train 分区")
def test_truncation_ladders_are_shared_across_families() -> None:
    """「同缺失数 = 同窗口，不同维度」—— 8 个族只用 3 条截断梯子。"""
    from responder_family_grid import shared_ladders

    families = build_families(read_column_stats(DATA_PARQUET))
    groups = shared_ladders(families)
    assert len(groups) == 3
    assert sorted(len(v) for v in groups.values()) == [2, 2, 4]


@pytest.mark.skipif(not DATA_PARQUET.is_file(), reason="需要 train 分区")
def test_read_column_stats_rejects_a_partition_without_responders() -> None:
    test_parquet = _REPO_ROOT / "data" / "test" / "test_partition_000.parquet"
    if not test_parquet.is_file():
        pytest.skip("需要 test 分区")
    with pytest.raises(SystemExit, match="缺少 responder 列"):
        read_column_stats(test_parquet)


# ------------------------------------------------------------------ Stage C 缺口

@pytest.mark.skipif(not STAGE_B_JSON.is_file(), reason="需要 Stage B 结果")
def test_untested_cells_are_derived_from_stage_b_not_hardcoded() -> None:
    """名单必须从 JSON 派生（CLAUDE.md §7）。这里核的是派生逻辑本身。"""
    cells = untested_stage_c_cells(STAGE_B_JSON, ("responder_00", "responder_02"))
    assert cells["clusters_total"] == 24
    assert cells["clusters_passed"] == 8
    # 16 个未通过的族**全部**只错 multi_member_family 这一条 —— 这就是「被启发式挡住」的定义
    assert cells["clusters_blocked_by_heuristic_only"] == 16
    assert cells["clusters_failed_on_evidence"] == []
    assert len(cells["untested"]) == 14
    assert set(cells["already_probed"]).isdisjoint(cells["untested"])


@pytest.mark.skipif(not STAGE_B_JSON.is_file(), reason="需要 Stage B 结果")
def test_untested_cells_disjoint_from_stage_c_frozen_representatives() -> None:
    """08-12 Stage C 冻结的 8 个代表与这 14 个必须无交集，否则「从未测过」的说法不成立。"""
    frozen = {"responder_27", "responder_24", "responder_37", "responder_15",
              "responder_08", "responder_40", "responder_18", "responder_11"}
    cells = untested_stage_c_cells(STAGE_B_JSON, ("responder_00", "responder_02"))
    assert frozen.isdisjoint(cells["untested"])


def test_untested_cells_reject_an_unknown_already_probed_name(tmp_path: Path) -> None:
    """已补测名单与 Stage B 对不上时必须当场失败，而不是静默给出错误的未测名单。"""
    stub = tmp_path / "stage_b.json"
    stub.write_text(json.dumps({"summary": {"clusters": [
        {"cluster": 1, "members": ["responder_00"], "pass": False,
         "checks": {"multi_member_family": False, "positive_folds": True}},
    ]}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="不在「只错 multi_member_family」的名单里"):
        untested_stage_c_cells(stub, ("responder_41",))


def test_untested_cells_separate_heuristic_blocks_from_evidence_failures(tmp_path: Path) -> None:
    stub = tmp_path / "stage_b.json"
    stub.write_text(json.dumps({"summary": {"clusters": [
        {"cluster": 1, "members": ["responder_00"], "pass": False,
         "checks": {"multi_member_family": False, "positive_folds": True}},
        {"cluster": 2, "members": ["responder_01"], "pass": False,
         "checks": {"multi_member_family": False, "positive_folds": False}},
        {"cluster": 3, "members": ["responder_02", "responder_03"], "pass": True,
         "checks": {"multi_member_family": True, "positive_folds": True}},
    ]}}), encoding="utf-8")
    cells = untested_stage_c_cells(stub, ())
    assert cells["untested"] == ["responder_00"]
    assert [row["cluster"] for row in cells["clusters_failed_on_evidence"]] == [2]


def test_build_families_is_order_stable_under_repeated_calls() -> None:
    rows = [make_row(i, n, -0.0, 1.0) for i, n in enumerate([279, 9, 1, 0, 422, 934, 2397])]
    first = build_families(rows)
    second = build_families(rows)
    assert [f["members"] for f in first] == [f["members"] for f in second]
    assert np.array_equal([f["family"] for f in first], [f["family"] for f in second])
