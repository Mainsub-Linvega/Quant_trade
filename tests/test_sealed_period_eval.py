from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from sealed_period_eval import (BOOTSTRAP_CHUNKS_PER_BLOCK, EMBARGO_REAL_TIME_IDS,
                                MIN_POSITIVE_BLOCKS, MIN_RELATIVE_GAIN, N_BLOCKS, SEAL_TIME_IDS,
                                _chunk_sums, _peak_from_sums, align_labels, assert_no_clip_hits,
                                block_metrics, judge, load_backfill_labels, resolve_model_dir,
                                seal_geometry)

# 仓库实测的 test 期边界（RUNBOOK §1）：3,217,458 行 / time_id 888,480–1,105,919
TEST_MIN, TEST_MAX = 888_480, 1_105_919


def test_geometry_matches_the_preregistered_split() -> None:
    """切分是 2026-08-20 用户定的；改它必须是有意的，不能被数据顺手改掉。"""
    geometry = seal_geometry(TEST_MIN, TEST_MAX)

    assert SEAL_TIME_IDS == 60_000 and N_BLOCKS == 4
    assert geometry["test_span_time_ids"] == 217_440
    assert geometry["seal_time_id_min"] == 1_045_920      # = 1,105,919 − 60,000 + 1
    assert geometry["seal_time_id_max"] == TEST_MAX
    assert geometry["block_time_ids"] == 15_000

    # 四块必须无缝、等长、闭区间首尾正好盖住密封段
    blocks = geometry["blocks"]
    assert [b["time_id_min"] for b in blocks] == [1_045_920, 1_060_920, 1_075_920, 1_090_920]
    assert blocks[-1]["time_id_max"] == TEST_MAX
    for earlier, later in zip(blocks, blocks[1:]):
        assert later["time_id_min"] == earlier["time_id_max"] + 1

    # embargo 必须真的隔开训练段与密封段 —— 少了它就是时序泄漏
    assert EMBARGO_REAL_TIME_IDS == 30
    assert geometry["decision_train_time_id_max"] == 1_045_920 - 30 - 1
    assert geometry["seal_time_id_min"] - geometry["decision_train_time_id_max"] - 1 == 30


def test_geometry_refuses_a_split_with_a_remainder() -> None:
    """余数落在最后一块会让「块均」被一个更大的块带偏 —— 预注册的分块必须精确。"""
    with pytest.raises(SystemExit, match="整除"):
        seal_geometry(TEST_MIN, TEST_MAX, seal_time_ids=50_001, n_blocks=4)
    with pytest.raises(SystemExit, match="超出"):
        seal_geometry(TEST_MIN, TEST_MAX, seal_time_ids=300_000)


def test_peak_is_invariant_to_a_global_rescale() -> None:
    """这条是「不反解 raw」的**全部依据**。

    runner 输出是已乘 scale 的值；`f → c·f` 时 A→cA、B→c²B ⟹ peak=A²/B 不变。
    ⚠️ slow/fast 下两个分量各有 scale，除以单一 prediction_scale 还原不出 raw ——
    所以正确做法是断言触限 0 行后直接算 peak，而不是做那个除法。
    """
    rng = np.random.default_rng(17)
    rows = 4000
    y = rng.normal(0.0, 0.02, rows)
    raw = 0.3 * y + rng.normal(0.0, 0.02, rows)
    w = rng.uniform(0.5, 1.5, rows)
    blocks = [{"block": 0, "time_id_min": 0, "time_id_max": rows - 1}]
    tid = np.arange(rows)

    base = block_metrics(tid, y, raw, w, blocks)[0]
    for scale in (1.16, 0.4496, 7.3):
        scaled = block_metrics(tid, y, scale * raw, w, blocks)[0]
        assert scaled["peak"] == pytest.approx(base["peak"], rel=1e-12)
        assert scaled["A"] == pytest.approx(scale * base["A"], rel=1e-12)
        assert scaled["B"] == pytest.approx(scale * scale * base["B"], rel=1e-12)
        # optimal_scale **不是**不变量 —— 报出来时必须记得它相对的是哪个 scale
        assert scaled["optimal_scale"] == pytest.approx(base["optimal_scale"] / scale, rel=1e-12)


def test_clip_hits_are_refused_instead_of_silently_compared() -> None:
    """限幅是唯一的非线性步骤；触限后 peak 不再与 raw 上的 peak 等价。"""
    clean = np.array([0.1, -0.2, 0.4204497])
    assert assert_no_clip_hits(clean, 0.5) == 0
    with pytest.raises(SystemExit, match="触到限幅"):
        assert_no_clip_hits(np.array([0.1, 0.5]), 0.5)


def test_partial_label_join_is_refused() -> None:
    """部分 join 会算出一个**看起来正常但错的**分数 —— 拿它做采纳决策就完了。"""
    pred_rows = np.array([10, 11, 12, 13], dtype=np.int64)
    take = align_labels(pred_rows, np.array([13, 11, 10, 12], dtype=np.int64))
    assert list(np.array([13, 11, 10, 12])[take]) == [10, 11, 12, 13]   # 乱序但完整 ⟹ 对齐

    with pytest.raises(SystemExit, match="不能用于裁决"):
        align_labels(pred_rows, np.array([10, 11, 12], dtype=np.int64))
    with pytest.raises(SystemExit, match="不能用于裁决"):
        align_labels(pred_rows, np.array([90, 91, 92, 93], dtype=np.int64))


def test_labels_without_weight_are_refused(tmp_path: Path) -> None:
    """公榜口径是 Σw(y−ŷ)²/Σw·y²，静默退化成无权会得到一个错的尺子。"""
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    good = pyarrow.table({"row_id": [1, 2], "target": [0.1, 0.2], "weight": [1.0, 2.0]})
    pq.write_table(good, tmp_path / "ok.parquet")
    row_id, target, weight = load_backfill_labels(tmp_path / "ok.parquet")
    assert list(row_id) == [1, 2] and weight.tolist() == [1.0, 2.0]

    bad_dir = tmp_path / "noweight"
    bad_dir.mkdir()
    pq.write_table(pyarrow.table({"row_id": [1], "target": [0.1]}), bad_dir / "bad.parquet")
    with pytest.raises(SystemExit, match="weight"):
        load_backfill_labels(bad_dir)


def _blocks(peaks: list[float]) -> list[dict]:
    return [{"block": i, "rows": 1000, "time_id_range": [0, 1], "peak": p,
             "A": 0.001, "B": 0.001, "optimal_scale": 1.0} for i, p in enumerate(peaks)]


def test_gate_mapping_is_three_of_four() -> None:
    """RUNBOOK D2「≥4/5 折」在 4 块下映射为 ≥3/4 块；去最好块后剩 3 块。"""
    assert MIN_POSITIVE_BLOCKS == 3 and N_BLOCKS == 4 and MIN_RELATIVE_GAIN == 0.03
    base = _blocks([1.0, 1.0, 1.0, 1.0])
    pooled_b = {"A": 1.0, "B": 1.0, "peak": 1.0}

    # 3/4 正、块均 +5%、去最好块仍为正 ⟹ 前五道全过
    arm = _blocks([1.10, 1.06, 1.04, 0.98])
    verdict = judge(base, arm, pooled_b, {"A": 1.06, "B": 1.0, "peak": 1.045},
                    {"ci_low": 0.01, "ci_high": 0.09, "median": 0.045, "samples": 10, "chunks": 4},
                    detection_floor=0.02)
    assert verdict["positive_blocks"] == 3
    assert verdict["gates"]["2_positive_blocks_at_least"] is True
    assert verdict["gates"]["3_survives_drop_best_block"] is True
    assert verdict["verdict"] == "PASS"

    # 只有 2/4 正 ⟹ 第 2 道挡下，即便块均为正
    weak = _blocks([1.30, 1.02, 0.97, 0.95])
    weak_verdict = judge(base, weak, pooled_b, {"A": 1.06, "B": 1.0, "peak": 1.06},
                         {"ci_low": 0.01, "ci_high": 0.2, "median": 0.06, "samples": 10, "chunks": 4})
    assert weak_verdict["block_mean_relative"] > 0
    assert weak_verdict["gates"]["2_positive_blocks_at_least"] is False
    assert weak_verdict["gates"]["3_survives_drop_best_block"] is False   # 去掉 +30% 那块就翻负
    assert weak_verdict["verdict"] == "FAIL"


def test_unknown_detection_floor_does_not_auto_pass() -> None:
    """第 7 道在 Tier 1 标定之前是**未知**，不是自动通过 —— 否则等于偷偷少一道门。"""
    base = _blocks([1.0, 1.0, 1.0, 1.0])
    arm = _blocks([1.10, 1.06, 1.04, 1.02])
    boot = {"ci_low": 0.02, "ci_high": 0.09, "median": 0.055, "samples": 10, "chunks": 4}
    pooled_b, pooled_a = {"A": 1.0, "B": 1.0, "peak": 1.0}, {"A": 1.06, "B": 1.0, "peak": 1.055}

    pending = judge(base, arm, pooled_b, pooled_a, boot, detection_floor=None)
    assert pending["gates"]["7_above_detection_floor"] is None
    assert pending["pending_gates"] == ["7_above_detection_floor"]
    assert pending["passes"] is False
    assert pending["verdict"] == "PENDING_CALIBRATION"

    # 标定出来之后才可能真过；标定值高于块均时照样挡下
    assert judge(base, arm, pooled_b, pooled_a, boot, 0.02)["verdict"] == "PASS"
    assert judge(base, arm, pooled_b, pooled_a, boot, 0.20)["verdict"] == "FAIL"


def test_mismatched_arms_are_refused() -> None:
    """块数或行数不同就不是配对比较 —— 这类错算出来的数看不出问题。"""
    base, pooled = _blocks([1.0, 1.0, 1.0, 1.0]), {"A": 1.0, "B": 1.0, "peak": 1.0}
    with pytest.raises(SystemExit, match="不是配对比较"):
        judge(base, _blocks([1.0, 1.0, 1.0]), pooled, pooled)
    short = _blocks([1.0, 1.0, 1.0, 1.0])
    short[2]["rows"] = 999
    with pytest.raises(SystemExit, match="不是配对比较"):
        judge(base, short, pooled, pooled)


def test_chunk_sums_reproduce_the_pooled_peak_exactly() -> None:
    """bootstrap 的加速靠「peak 的三个分量都是和」，这条必须是精确恒等式，不是近似。"""
    rng = np.random.default_rng(23)
    rows = 5000
    y = rng.normal(0.0, 0.02, rows)
    p = 0.25 * y + rng.normal(0.0, 0.02, rows)
    w = rng.uniform(0.5, 1.5, rows)
    n_chunks = N_BLOCKS * BOOTSTRAP_CHUNKS_PER_BLOCK
    chunk_id = rng.integers(0, n_chunks, rows)

    sums = _chunk_sums(y, p, w, chunk_id, n_chunks)
    assert sums.shape == (n_chunks, 3)
    direct = block_metrics(np.zeros(rows, np.int64), y, p, w,
                           [{"block": 0, "time_id_min": 0, "time_id_max": 0}])[0]["peak"]
    assert _peak_from_sums(sums.sum(axis=0)) == pytest.approx(direct, rel=1e-12)


def test_model_dir_accepts_both_layouts(tmp_path: Path) -> None:
    """生产的 meta 在 `model/` 下，候选的在本层 —— 给错一层原本只报裸 FileNotFoundError。

    这是干跑第一次就撞上的坑：`stage()` 要的是模型产物目录本身，不是策略目录。
    """
    candidate = tmp_path / "candidate"          # 候选布局：meta 在本层
    candidate.mkdir()
    (candidate / "hybrid_meta.json").write_text("{}", encoding="utf-8")
    assert resolve_model_dir(candidate) == candidate

    strategy = tmp_path / "strategy"            # 生产布局：meta 在 model/ 下
    (strategy / "model").mkdir(parents=True)
    (strategy / "model" / "hybrid_meta.json").write_text("{}", encoding="utf-8")
    assert resolve_model_dir(strategy) == strategy / "model"

    with pytest.raises(SystemExit, match="hybrid_meta.json"):
        resolve_model_dir(tmp_path / "nothing_here")
