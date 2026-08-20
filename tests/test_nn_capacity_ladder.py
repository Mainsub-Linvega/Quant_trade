from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from nn_capacity_ladder import (EXTENSION_RUNG, EXTENSION_TRIGGER, GATE_FRACTION,
                                REPRODUCTION_ANCHOR, REPRODUCTION_TOLERANCE,
                                assert_reproduces_anchor, assert_single_axis, judge, read_rung)


def rung_payload(max_iter: int, target_only: float, multitask: float,
                 **config_overrides) -> dict:
    """一档 `multitask_mlp.py` 输出的最小复制品。"""
    configuration = {
        "max_iter": max_iter, "label": f"multitask_mlp_e{max_iter}",
        "sample_modulo": 5, "train_window": 78_960, "current_feature_count": 200,
        "cross_hidden": [64, 32], "market_hidden": [32], "seed": 2026,
        "alpha": 1e-3, "learning_rate": 1e-3, "batch_size": 4096, "folds": [0],
    }
    configuration.update(config_overrides)
    return {
        "configuration": configuration,
        "elapsed_seconds": 60.0,
        "folds": [{
            "fold": 0,
            "baseline": {"peak": 0.001055954978483406},
            "arms": {
                "target_only": {"mlp_relative_to_baseline": target_only},
                "multitask": {"mlp_relative_to_baseline": multitask},
            },
        }],
    }


def as_rung(max_iter: int, target_only: float, multitask: float, **overrides) -> dict:
    """绕开磁盘，直接构造 read_rung 的输出形状。"""
    payload = rung_payload(max_iter, target_only, multitask, **overrides)
    fold = payload["folds"][0]
    arms = {arm: float(v["mlp_relative_to_baseline"]) for arm, v in fold["arms"].items()}
    return {
        "path": f"<{max_iter}>", "max_iter": max_iter,
        "configuration": payload["configuration"],
        "baseline_peak": fold["baseline"]["peak"], "arms": arms,
        "best_arm": max(arms, key=arms.get), "best_relative": max(arms.values()),
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def test_gate_reads_the_better_of_the_two_arms() -> None:
    """门槛问的是「NN 能不能到」⟹ 取两臂较好者。

    multitask（λ=0.3）是 08-19 预注册过的合法 NN 配置，不是事后挑出来的臂。
    """
    # target_only 不过线、multitask 过线 ⟹ 整体应判过
    rungs = [as_rung(12, 0.17, 0.20), as_rung(50, 0.30, 0.35),
             as_rung(150, 0.44, 0.52), as_rung(400, 0.45, 0.53)]
    verdict = judge(rungs)
    assert verdict["passes_gate"] is True
    assert verdict["best_arm"] == "multitask"
    assert verdict["best_relative"] == pytest.approx(0.53)

    # 两臂都不过线 ⟹ REJECTED，且措辞必须限定在「这个配方」
    low = [as_rung(12, 0.17, 0.20), as_rung(50, 0.28, 0.30),
           as_rung(150, 0.33, 0.34), as_rung(400, 0.33, 0.34)]
    rejected = judge(low)
    assert rejected["passes_gate"] is False
    assert rejected["verdict"] == "REJECTED"
    assert "不是" in rejected["action"]          # 「这不是『NN 不行』」这层限定不能丢


def test_extension_triggers_on_the_preregistered_rate() -> None:
    """条件延长按末档相对前一档 ≥ +5% 触发 —— 跑前写死，不是看到曲线才决定。"""
    climbing = [as_rung(150, 0.40, 0.44), as_rung(400, 0.44, 0.49)]   # +11.4%
    verdict = judge(climbing)
    assert verdict["still_climbing"] is True
    assert verdict["extension_due"] is True
    assert verdict["extension_rung"] == EXTENSION_RUNG
    assert verdict["climb_rate_last_step"] > EXTENSION_TRIGGER

    flat = [as_rung(150, 0.40, 0.44), as_rung(400, 0.40, 0.445)]      # +1.1%
    assert judge(flat)["still_climbing"] is False
    assert judge(flat)["extension_due"] is False

    # 掉头也必须被识别（过拟合），而不是当成「饱和」
    over = [as_rung(150, 0.40, 0.44), as_rung(400, 0.30, 0.33)]
    assert judge(over)["turned_over"] is True
    assert judge(over)["still_climbing"] is False


def test_config_drift_is_refused() -> None:
    """单轴保护：除 max_iter/label 外多动一项，曲线就无法归因给预算。"""
    clean = [as_rung(12, 0.17, 0.20), as_rung(50, 0.30, 0.33)]
    assert_single_axis(clean)                                        # 不应抛

    for key, bad in (("cross_hidden", [256, 128]), ("learning_rate", 3e-3),
                     ("current_feature_count", 323), ("seed", 7)):
        drifted = [as_rung(12, 0.17, 0.20), as_rung(50, 0.30, 0.33, **{key: bad})]
        with pytest.raises(SystemExit, match="单轴"):
            assert_single_axis(drifted)


def test_anchor_check_catches_a_changed_environment() -> None:
    """12 档必须复现 08-19 的结果 —— 阶梯的第一级就是它自己的回归测试。"""
    exact = [as_rung(12, REPRODUCTION_ANCHOR["target_only"],
                     REPRODUCTION_ANCHOR["multitask"])]
    assert assert_reproduces_anchor(exact)["checked"] is True

    # 容差内允许（浮点/线程差异）
    nudged = [as_rung(12, REPRODUCTION_ANCHOR["target_only"] + 0.5 * REPRODUCTION_TOLERANCE,
                      REPRODUCTION_ANCHOR["multitask"])]
    assert assert_reproduces_anchor(nudged)["max_abs_drift"] <= REPRODUCTION_TOLERANCE

    # 超出容差必须整条作废，而不是「差一点也算」
    with pytest.raises(SystemExit, match="整条阶梯作废"):
        assert_reproduces_anchor([as_rung(12, 0.30, 0.33)])


def test_read_rung_refuses_multi_fold(tmp_path: Path) -> None:
    """阶梯口径是固定 fold 0；多折 JSON 混进来会让「相对上一档」失去意义。"""
    payload = rung_payload(12, 0.17, 0.20)
    payload["folds"].append(dict(payload["folds"][0], fold=1))
    path = tmp_path / "multitask_mlp_e12.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="fold 0"):
        read_rung(path)


def test_gate_constant_matches_the_preregistered_value() -> None:
    """门槛是用户在 2026-08-20 定的，改它必须是有意的。"""
    assert GATE_FRACTION == 0.50
    assert EXTENSION_TRIGGER == 0.05
    assert EXTENSION_RUNG == 1200
