"""responder 监督的选列判据 —— 只测不依赖数据的部分。

最要紧的是 `test_correlation_profile_matches_select_features`：探针自己算相关（因为
`select_features` 只返回排序后的下标、拿不到分数），这个复制品与生产选择器的 top-k
必须逐位相同，否则「重合度」这个读数比的就不是同一个东西。

`assert_matches_select_features` 是探针在**每一折**都会调的守卫；这里额外钉住它
在口径真的漂移时确实会抛错，而不是静默通过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train import select_features  # noqa: E402

import responder_selection_probe as probe  # noqa: E402


def synthetic(n: int = 400, p: int = 37, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, p)).astype(np.float32)
    label = (matrix[:, 3] * 0.4 - matrix[:, 9] * 0.25
             + rng.normal(scale=2.0, size=n)).astype(np.float64)
    return matrix, label


# ------------------------------------------------------------------ 与生产选择器等价

@pytest.mark.parametrize("count", [1, 5, 20, 37])
def test_correlation_profile_matches_select_features(count: int) -> None:
    matrix, label = synthetic()
    scores = probe.correlation_profile(matrix, [label])[:, 0]
    assert np.array_equal(probe.top_k(scores, count),
                          np.asarray(select_features(matrix, label, np.ones_like(label), count),
                                     dtype=np.int64))


def test_correlation_profile_computes_all_labels_in_one_pass() -> None:
    """一次多标签与逐标签单算必须给出同一个答案（省的是内存，不是精度）。

    ⚠️ 不是逐位相同：`block.T @ label_matrix` 在 3 列和 1 列上走不同的 BLAS 路径，
    累加顺序不同。实测偏差 **1.4e-16**，比相关值本身（~2e-01）小 15 个数量级，
    且 top-k 完全不变 —— 而 top-k 才是 `assert_matches_select_features` 守卫依赖的性质。
    """
    matrix, label = synthetic()
    rng = np.random.default_rng(3)
    labels = [label, label * 2.0 - 1.0, rng.normal(size=len(label))]
    together = probe.correlation_profile(matrix, labels)
    for position, single in enumerate(labels):
        alone = probe.correlation_profile(matrix, [single])[:, 0]
        assert np.allclose(together[:, position], alone, rtol=0.0, atol=1e-14)
        for count in (5, 20):
            assert np.array_equal(probe.top_k(together[:, position], count),
                                  probe.top_k(alone, count))


def test_correlation_profile_is_invariant_to_label_affine_transform() -> None:
    """相关对标签的仿射变换不变 —— responder 带很大的非零均值，这条必须成立。"""
    matrix, label = synthetic()
    plain = probe.correlation_profile(matrix, [label])[:, 0]
    shifted = probe.correlation_profile(matrix, [label * 3.5 + 100.0])[:, 0]
    assert np.allclose(plain, shifted, rtol=1e-9, atol=1e-12)


def test_assert_matches_select_features_raises_when_scores_drift() -> None:
    matrix, label = synthetic()
    scores = probe.correlation_profile(matrix, [label])[:, 0]
    corrupted = scores.copy()
    corrupted[np.argsort(np.abs(corrupted))[:5]] = 1e3     # 把最弱的几列顶到最前
    with pytest.raises(AssertionError, match="口径已漂移"):
        probe.assert_matches_select_features(matrix, label, corrupted, 10, "unit")


def test_assert_matches_select_features_passes_on_the_real_scores() -> None:
    matrix, label = synthetic()
    scores = probe.correlation_profile(matrix, [label])[:, 0]
    probe.assert_matches_select_features(matrix, label, scores, 10, "unit")


def test_top_k_returns_sorted_original_positions() -> None:
    scores = np.array([0.1, -0.9, 0.5, -0.2, 0.7])
    assert np.array_equal(probe.top_k(scores, 3), np.array([1, 2, 4]))


# ------------------------------------------------------------------ 一致性判据

def test_consistency_score_averages_the_rungs() -> None:
    profile = np.array([[0.10, 0.08, 0.12], [-0.05, -0.05, -0.05]])
    assert np.allclose(probe.consistency_score(profile), [0.10, -0.05])


def test_consistency_score_cancels_sign_inconsistent_columns() -> None:
    """只与某一级相关、在别的级上翻号的列会被抵消 —— 这正是想要的降噪。"""
    consistent = np.array([0.06, 0.05, 0.055, 0.05])
    flipping = np.array([0.20, -0.19, 0.18, -0.19])
    profile = np.vstack([consistent, flipping])
    scores = probe.consistency_score(profile)
    assert abs(scores[0]) > abs(scores[1])


def test_consistency_score_equals_base_when_ladder_has_one_rung() -> None:
    """退化情形：梯子只有 target 一级时，新判据必须还原成现状判据。"""
    matrix, label = synthetic()
    profile = probe.correlation_profile(matrix, [label])
    assert np.array_equal(probe.top_k(probe.consistency_score(profile), 12),
                          probe.top_k(profile[:, 0], 12))


# ------------------------------------------------------------------ 梯子与决策规则

def test_ladder_admission_comes_from_the_atlas_flag(tmp_path: Path) -> None:
    """梯子成员由图谱自己的 `H_fit_is_equal_weight_MA` 决定，不是本脚本另设的阈值。"""
    import json

    atlas = tmp_path / "atlas.json"
    atlas.write_text(json.dumps({
        "criterion": "stub",
        "responders": [
            {"responder": f"responder_{i:02d}", "H_estimate": i + 1,
             "H_fit_rmse": 0.01 * (i + 1),
             "H_fit_is_equal_weight_MA": i not in (1, 6)}
            for i in range(7)
        ],
    }), encoding="utf-8")

    stats = [{"responder": f"responder_{i:02d}", "index": i, "null_count": 0,
              "min": -0.0, "max": 1.0, "sign_class": "unit_interval"} for i in range(7)]

    import responder_family_grid as grid
    original = grid.read_column_stats
    grid.read_column_stats = lambda _path: stats
    try:
        ladder = probe.build_ladder(atlas, tmp_path)
    finally:
        grid.read_column_stats = original

    assert [m["responder"] for m in ladder["members"]] == [
        "responder_00", "responder_02", "responder_03", "responder_04", "responder_05"]
    assert [m["responder"] for m in ladder["rejected"]] == ["responder_01", "responder_06"]
    assert ladder["admission"] == "responder_window_atlas.H_fit_is_equal_weight_MA"


def test_decision_line_is_below_the_feature_count() -> None:
    """决策线必须是「几乎不动」而不是「完全不动」，否则规则永远不会触发结案分支。"""
    assert 0 < probe.OVERLAP_DECISION_LINE < probe.FEATURE_COUNT
    assert probe.FEATURE_COUNT - probe.OVERLAP_DECISION_LINE == 10


def test_preregistered_constants_match_production_pipeline() -> None:
    """选列口径必须与生产逐项相同，否则量出来的重合度不是生产上的重合度。"""
    assert probe.FEATURE_COUNT == 200
    assert probe.HISTORY_COUNT == 40
    assert probe.TRAIN_WINDOW == 78_960
    assert probe.EMBARGO == 6
    assert probe.N_FOLDS == 5
    assert probe.SAMPLE_MODULO == 5
    assert probe.SAMPLING == "phase_balanced"
