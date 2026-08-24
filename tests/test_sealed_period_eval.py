from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from sealed_period_eval import (BOOTSTRAP_CHUNKS_PER_BLOCK, EMBARGO_REAL_TIME_IDS,
                                IDENTITY_KEYS, MIN_POSITIVE_BLOCKS, MIN_RELATIVE_GAIN, N_BLOCKS,
                                SEAL_TIME_IDS, _chunk_sums, _peak_from_sums, align_labels,
                                assert_no_clip_hits, baseline_overrides, block_metrics, clip_hits, judge,
                                load_backfill_labels, resolve_model_dir, seal_geometry)

sys.path.insert(0, str(ROOT / "scripts"))
from promote_v3_candidate import PUBLIC_BASELINE  # noqa: E402

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


# ---------------------------------------------------------------------------
# 2026-08-24：首轮 Tier 1 标定作废事故的回归 —— 见 `baseline_overrides` 的文档字符串
# ---------------------------------------------------------------------------


def _meta(tmp_path: Path, **fields) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "hybrid_meta.json").write_text(json.dumps(fields), encoding="utf-8")
    return candidate


def test_placeholder_candidate_meta_is_pulled_back_to_public_baseline(tmp_path: Path) -> None:
    """⭐ 核心回归：`train.py` 的占位 meta 必须被拨回公榜口径。

    候选目录落的是 `blend_weight=0.5 / prediction_scale=0.856`，而所有公榜真值都是在
    `1.0 / 1.16` 上打的。首轮标定因为传 `stage({}, ...)` 而让六个臂里的**四个**跑了
    另一个模型 —— `blend_weight` 不是缩放，peak 的尺度不变性救不了它。
    """
    candidate = _meta(tmp_path, blend_weight=0.5, prediction_scale=0.856)
    assert baseline_overrides(candidate) == {
        "blend_weight": PUBLIC_BASELINE["blend_weight"],
        "prediction_scale": PUBLIC_BASELINE["prediction_scale"],
    }


def test_already_baseline_meta_gets_no_overrides(tmp_path: Path) -> None:
    """生产目录（meta 本来就是 1.0/1.16）必须是无操作 —— 否则等于悄悄改生产口径。"""
    candidate = _meta(tmp_path, blend_weight=PUBLIC_BASELINE["blend_weight"],
                      prediction_scale=PUBLIC_BASELINE["prediction_scale"])
    assert baseline_overrides(candidate) == {}


def test_overrides_only_touch_the_two_placeholder_keys(tmp_path: Path) -> None:
    """不得顺手覆写 `long_window` 之类 —— 那会给 361 列的森林盖上「有长窗」的章。"""
    candidate = _meta(tmp_path, blend_weight=0.5, prediction_scale=0.856,
                      long_window=None, num_iteration=480, market_lambda=0.5)
    assert set(baseline_overrides(candidate)) == {"blend_weight", "prediction_scale"}


def test_missing_keys_are_treated_as_drift_not_as_ok(tmp_path: Path) -> None:
    """缺键 → NaN → 必须判为需要覆写，而不是「没写就算对」。"""
    assert set(baseline_overrides(_meta(tmp_path))) == {"blend_weight", "prediction_scale"}


def test_identity_print_covers_long_window(tmp_path: Path) -> None:
    """报告号称打印「模型身份」，漏印 `long_window` 就分不出 441 列和 361 列两种模型。"""
    assert "long_window" in IDENTITY_KEYS
    assert "history_window" in IDENTITY_KEYS
    for key in ("blend_weight", "prediction_scale", "market_lambda",
                "cross_section_weighted", "slow_fast_window"):
        assert key in IDENTITY_KEYS


def test_clip_check_is_scoped_to_the_rows_that_enter_the_peak() -> None:
    """⭐ 2026-08-24 回归：触限判据只看**密封段**，不看全量 test 期。

    旧作用域对全窗 3,217,458 行断言，而 peak 只由密封段那 856,319 行算出来
    （`arm_view` 紧接着每一步都是 `pred[seal]`）⟹ 一行落在评估窗口**之外**的触限
    就能毙掉整个臂。`r960`（两个负控制之一）实测正是这样：
    全窗触限 **1 行 / 3,217,458**，密封段内 **0 行**、段内 max|pred| = 0.4620392。
    脚本用 `set -e` 串跑时，它还会连带中断排在后面的臂。
    """
    seal = np.array([False, False, True, True])
    prediction = np.array([0.5, -0.1, 0.2, -0.3])      # 触限那行在密封段**之外**

    assert clip_hits(prediction, 0.5) == 1              # 全窗仍如实计数
    assert assert_no_clip_hits(prediction[seal], 0.5) == 0   # 段内判据放行

    inside = np.array([0.1, -0.1, 0.5, -0.3])          # 触限那行落在段内 ⟹ 必须拒绝
    with pytest.raises(SystemExit, match="触到限幅"):
        assert_no_clip_hits(inside[seal], 0.5)


def test_sealed_mask_matches_the_preregistered_row_count() -> None:
    """掩码必须落在预注册几何上 —— 856,319 行是 RUNBOOK / P10 记的实测值。"""
    import pandas as pd
    import pyarrow.parquet as pq
    from sealed_period_eval import sealed_rows

    test_dir = ROOT / "data" / "test"
    if not any(test_dir.glob("*.parquet")):
        pytest.skip("test 分区不在盘上")
    row_id = pd.concat([pq.read_table(p, columns=["row_id"]).to_pandas()
                        for p in sorted(test_dir.glob("*.parquet"))],
                       ignore_index=True)["row_id"].to_numpy(np.int64)
    assert int(sealed_rows(row_id, ROOT / "data").sum()) == 856_319


def test_slow_fast_keys_are_opt_in_and_keep_their_types(tmp_path: Path) -> None:
    """⭐ 两类臂的「按交付口径」不同 —— slow/fast 必须显式要，不能默认补。

    · Tier 1 那五个历史臂的公榜真值是在 slow/fast 转正**之前**打的，补上就不再是那个模型
      （实测佐证：不补时它们的 `max|pred|` 与留档公榜 CSV 逐位对上）；
    · 重训候选**必定**缺这三个键（`train.py` 的 CLI 里没有这个概念），而
      `main.py:222` 是 `PredictionTrail(...) if window else None` ⟹ **缺键静默关掉**。
      不补就等于拿「扩展数据 + 丢了 slow/fast」比「当前数据 + 有 slow/fast」，
      两个变量混在一起，而 slow/fast 公榜实测 +2.93%。
    """
    candidate = _meta(tmp_path, blend_weight=0.5,
                      prediction_scale=PUBLIC_BASELINE["prediction_scale"])
    assert set(baseline_overrides(candidate)) == {"blend_weight"}

    with_sf = baseline_overrides(candidate, slow_fast=True)
    assert set(with_sf) == {"blend_weight", "slow_fast_window",
                            "slow_fast_slow_relative", "slow_fast_fast_relative"}
    # `slow_fast_window` 是 int —— 写成 2000.0 会让 staged meta 与生产 meta 分型
    assert isinstance(with_sf["slow_fast_window"], int)
    assert with_sf["slow_fast_window"] == PUBLIC_BASELINE["slow_fast_window"]


def test_already_slow_fast_model_needs_no_override(tmp_path: Path) -> None:
    """生产目录本来就有这三个键 ⟹ 即使传 slow_fast=True 也必须是无操作。"""
    candidate = _meta(tmp_path, blend_weight=PUBLIC_BASELINE["blend_weight"],
                      prediction_scale=PUBLIC_BASELINE["prediction_scale"],
                      **{k: PUBLIC_BASELINE[k] for k in
                         ("slow_fast_window", "slow_fast_slow_relative",
                          "slow_fast_fast_relative")})
    assert baseline_overrides(candidate, slow_fast=True) == {}
