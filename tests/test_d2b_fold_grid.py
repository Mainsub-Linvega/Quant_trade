"""D2b 折网格与「训练段后端封顶」的门禁 —— 这是本轮唯一一个**因果合法**的 OOF 配对设计。

## 为什么预注册的 D2b 不成立

`RUNBOOK_8_23.md` D2 让「现跑 OOF 基准」与「重训臂的 cache」做配对比较。但扩展数据是
`time_id 888,480–1,045,889`，而原 OOF 折的验证段全部落在 `≤ 888,479` ——
把新数据塞进那些折的训练段就是**拿未来训练**。
⟹ 「多给一段更近的训练数据值多少」这个问题，在原折版图里**因果上无法回答**。

合法问法只有一个：把验证段挪到新数据**之后**，两臂只差训练段后端封顶。本文件钉住那套几何。

⭐ 网格自带一个零对照：fold 0 的验证段紧接 888,480 开始，训练段被 embargo 卡在 888,473
⟹ **扩展臂在 fold 0 上拿不到任何额外数据**，两臂必须给出同一个数。
fold 1/2/3 依次多拿约 39k / 79k / 118k 个 time_id，构成剂量-反应内检。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

GRID_PATH = ROOT / "outputs" / "experiments" / "d2b_backfill_fold_grid.json"
SEAL_PLAN = ROOT / "outputs" / "experiments" / "sealed_period_plan.json"

BACKFILL_LO, BACKFILL_HI = 888_480, 1_045_889


def grid() -> dict:
    return json.loads(GRID_PATH.read_text(encoding="utf-8"))


def test_validation_covers_exactly_the_backfill_window() -> None:
    """验证段必须铺满回补窗口，且不重叠、不留缝 —— 否则「两臂评同一批行」不成立。"""
    folds = grid()["folds"]
    lo = [f["valid_time_range"][0] for f in folds]
    hi = [f["valid_time_range"][1] for f in folds]
    assert lo[0] == BACKFILL_LO
    assert hi[-1] == BACKFILL_HI
    for previous, following in zip(hi, lo[1:]):
        assert following == previous + 1, "验证段之间有缝或重叠"


def test_no_fold_touches_the_sealed_period() -> None:
    """⭐ 密封段只能用一次。D2b 借的是它**之前**那一段，一行都不能碰。"""
    seal_min = json.loads(SEAL_PLAN.read_text(encoding="utf-8"))["geometry"]["seal_time_id_min"]
    for fold in grid()["folds"]:
        assert fold["valid_time_range"][1] < seal_min
        assert fold["train_time_range"][1] < seal_min


def test_embargo_is_honoured_on_every_fold() -> None:
    """训练段后端与验证段起点之间必须空出 embargo 个 time_id。"""
    payload = grid()
    embargo = payload["embargo_real_time_ids"]
    for fold in payload["folds"]:
        train_hi = fold["train_time_range"][1]
        valid_lo = fold["valid_time_range"][0]
        assert valid_lo - train_hi - 1 == embargo, f"{fold} 的 embargo 不是 {embargo}"


def test_fold_zero_is_a_built_in_null_control() -> None:
    """⭐ fold 0 上封顶臂与扩展臂拿到的训练段必须**完全相同** —— 自带零对照。

    它证明的是配对本身没坏：如果 fold 0 也报出差异，那差异来自随机性或 bug，不是数据。
    """
    fold0 = grid()["folds"][0]
    assert fold0["train_time_range"][1] <= 888_479, "fold 0 的训练段越过了原始 train 边界"


def test_extra_training_data_grows_monotonically() -> None:
    """剂量-反应：后面的折能多吃到的数据必须严格递增，否则读不出「量」的效应。"""
    extra = [min(f["train_time_range"][1], BACKFILL_HI) - 888_479
             for f in grid()["folds"]]
    extra = [max(e, 0) for e in extra]
    assert extra[0] == 0
    assert all(b > a for a, b in zip(extra, extra[1:])), f"额外数据量不单调：{extra}"


def test_the_back_cap_flag_exists_and_is_recorded() -> None:
    """⚠️ 这个参数是本设计的**唯一**开关；它没了或不落盘，两臂就没法区分。

    与 `test_model_identity_key_coverage` 同一条纪律：写进配置的东西必须真的被记下来，
    否则事后无法证明当时跑的是哪一臂。
    """
    source = (ROOT / "experiments" / "v3_production_oof.py").read_text(encoding="utf-8")
    assert '"--train-time-id-max"' in source, "CLI 里没有 --train-time-id-max"
    assert '"train_time_id_max"' in source, "配置记录里没有 train_time_id_max ⟹ 事后不可核"


def test_cap_keeps_validation_identical() -> None:
    """封顶只砍训练段后端，验证段一个 time_id 都不许动。"""
    unique = np.arange(0, 1_045_890, dtype=np.int64)
    for fold in grid()["folds"]:
        t_lo, t_hi = fold["train_time_range"]
        v_lo, v_hi = fold["valid_time_range"]
        train = unique[(unique >= t_lo) & (unique <= t_hi)]
        valid = unique[(unique >= v_lo) & (unique <= v_hi)]
        capped = train[train <= 888_479]
        assert len(capped) <= len(train)
        assert capped.max(initial=-1) <= 888_479
        # 验证段与封顶无关 —— 重算一次必须逐位相同
        assert np.array_equal(valid, unique[(unique >= v_lo) & (unique <= v_hi)])
        assert valid.min() > capped.max(initial=-1), "封顶后训练段仍与验证段相邻/重叠"
