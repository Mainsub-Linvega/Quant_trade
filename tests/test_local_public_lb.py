"""本地公榜协议的门禁 —— 尤其是「确认段只准用一次」那一条。

无限次打同一个固定历史窗口必然刷出假赢家，所以窗口被切成搜索段与确认段。
**搜索段随便打，确认段每个候选一次。** 这条纪律靠账本强制，不靠记性 ——
本文件钉死它，以及分块几何与窗口切分。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import local_public_lb as L  # noqa: E402


def test_blocks_partition_the_window_without_gap_or_overlap() -> None:
    blks = L.blocks_for(np.array([0]), 888_480, 1_045_919)
    assert len(blks) == L.N_BLOCKS
    assert blks[0]["time_id_min"] == 888_480
    assert blks[-1]["time_id_max"] == 1_045_919
    for a, b in zip(blks, blks[1:]):
        assert b["time_id_min"] == a["time_id_max"] + 1, "块之间有缝或重叠"


def test_last_block_absorbs_the_remainder() -> None:
    """整除不尽时余数进最后一块 —— 否则末尾几个 time_id 会被静默丢掉。"""
    blks = L.blocks_for(np.array([0]), 0, 10)          # 11 个 id / 4 块
    assert blks[-1]["time_id_max"] == 10
    covered = sum(b["time_id_max"] - b["time_id_min"] + 1 for b in blks)
    assert covered == 11


def test_confirm_guard_rejects_a_second_visit(tmp_path: Path, monkeypatch) -> None:
    """⭐ 核心：同一个候选第二次打确认段必须被拒 —— 这是留出集唯一的保护。"""
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(L, "CONFIRM_LEDGER", ledger)
    L.confirm_guard(["arm_a", "base"], write=True)     # 第一次：放行并记账
    assert json.loads(ledger.read_text()).keys() >= {"arm_a", "base"}
    with pytest.raises(SystemExit, match="已经在确认段上打过分"):
        L.confirm_guard(["arm_a"], write=False)        # 第二次：拒绝


def test_confirm_guard_allows_a_new_candidate(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(L, "CONFIRM_LEDGER", ledger)
    L.confirm_guard(["arm_a"], write=True)
    L.confirm_guard(["arm_b"], write=False)            # 没打过的候选照常放行


def test_dry_run_does_not_consume_the_budget(tmp_path: Path, monkeypatch) -> None:
    """`write=False` 只做检查、不记账 —— 否则一次失败的运行会白白烧掉额度。"""
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(L, "CONFIRM_LEDGER", ledger)
    L.confirm_guard(["arm_a"], write=False)
    assert not ledger.exists()


def test_train_cut_constant_matches_the_public_window_start() -> None:
    """协议要求训练段止于 888,479，正好是公榜窗口起点前一格。"""
    from sealed_period_eval import seal_geometry
    geom = seal_geometry(888_480, 1_105_919)
    assert L.TRAIN_CUT == geom["test_time_id_min"] - 1
    # 搜索段与确认段合起来就是整个公榜窗口，且在密封段起点处切开
    assert geom["seal_time_id_min"] == 1_045_920
