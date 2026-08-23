"""「模型身份」的键必须被每一个消费者覆盖 —— 漏一个就当场红。

## 为什么需要这一层

`promote_v3_candidate.PUBLIC_BASELINE` 是「榜上那份模型长什么样」的唯一真值源，
但真正**使用**它的地方有四处，而且各自维护自己的取值表 / 派生表 / 命令行参数：

| 消费者 | 作用 |
|---|---|
| `audit_submission_zip.public_baseline_drift` | 提交包的内容身份审计 |
| `retrain_extended.{production_structure, BASELINE_CHECKED_KEYS}` | 重训计划复现生产结构 |
| `verify_delivery_runtime.model_identity` | 交付报告打印的 meta 身份 |
| `make_submission` / `promote_v3_candidate` | 打包与转正 |

**同一类事故已经发生四次**，每次都是「往身份里加了一个键，但某个消费者没跟上」：

1. 2026-08-18 `slow_fast_*` 三键不在 `PUBLIC_BASELINE` 里 ⟹ 丢键会**静默**交出低 2.93% 的旧模型；
2. 2026-08-19 `retrain_extended` 的命令计划缺 `--weighted-cross-section` / `--market-model`
   ⟹ 跑出来的是低 21.99% 的 08-11 架构；
3. 2026-08-21 `long_window` 漏进 `PUBLIC_BASELINE` ⟹ 缺键静默关掉长窗，低 1.66%；
4. 2026-08-23（本文件的由来）`long_window` 漏进**重训计划** ——
   `train.py:335` 的 `--long-window` 默认 0 = 关闭，而 `retrain_extended` 从未传过它，
   `production_structure()` 也没派生它 ⟹ 8/23 跑 D1 会训出一个**没有长窗**的候选。
   转正门禁最终会拦下，但那是在**几小时训练之后**，而 8/23→8/31 只有 8 天。

逐次补洞已被证明不够。本文件把「记得同步」变成机械门禁：往 `PUBLIC_BASELINE` 加键时，
任何消费者没跟上都会让测试当场红，而不是等到训练完、打包完、甚至交出去之后。

## 豁免怎么写

确实无法覆盖的键必须进各自的显式豁免表**并写理由** —— 沿用
`make_submission.EXCLUDED_MODULES` 那套「偏离必须是按下去的，不是漏掉的」。
空着不写 = 红。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_submission_zip import public_baseline_drift  # noqa: E402
from promote_v3_candidate import PUBLIC_BASELINE  # noqa: E402
from retrain_extended import BASELINE_CHECKED_KEYS, production_structure  # noqa: E402
from verify_delivery_runtime import model_identity  # noqa: E402

# ---- `retrain_extended` 的正当豁免（每条都必须有理由）----
# 这些键**不由 train.py 产生**，而是 `promote_v3_candidate` 在 staging 阶段写进 meta 的，
# 所以「固定结构重训」的命令计划里本来就不该有它们。
RETRAIN_EXEMPT: dict[str, str] = {
    "slow_fast_window": "train.py 没有 slow/fast 概念，由 promote_v3_candidate staging 写入",
    "slow_fast_slow_relative": "同上",
    "slow_fast_fast_relative": "同上",
    "blend_weight": "候选 meta 落的是本地占位 0.5，由 promote_v3_candidate --blend-weight 覆写",
    "prediction_scale": "候选 meta 落的是本地占位，由 promote_v3_candidate --scale 覆写",
}

# ---- `verify_delivery_runtime.model_identity` 的键名别名 ----
# 它打印的是「几片森林 / 几个特征」这类计数，名字与 PUBLIC_BASELINE 不同但语义对应。
VERIFY_DELIVERY_ALIASES = {
    "history_positions_count": "n_history_positions",
    "n_seeds": "n_lgbm_models",
    "market_model_count": "n_market_models",
}

PRODUCTION_MODEL_DIR = ROOT / "strategies" / "v3_hybrid" / "model"


def baseline_meta() -> dict:
    """一份与 PUBLIC_BASELINE 完全一致的 meta，用来驱动 audit 的取值表。"""
    return {
        "blend_weight": PUBLIC_BASELINE["blend_weight"],
        "num_iteration": PUBLIC_BASELINE["num_iteration"],
        "history_window": PUBLIC_BASELINE["history_window"],
        "history_positions": list(range(PUBLIC_BASELINE["history_positions_count"])),
        "prediction_scale": PUBLIC_BASELINE["prediction_scale"],
        "lgbm_model_files": [f"s{i}.txt" for i in range(PUBLIC_BASELINE["n_seeds"])],
        "market_lambda": PUBLIC_BASELINE["market_lambda"],
        "market_model_files": [f"m{i}.txt"
                               for i in range(PUBLIC_BASELINE["market_model_count"])],
        "cross_section_weighted": PUBLIC_BASELINE["cross_section_weighted"],
        "slow_fast_window": PUBLIC_BASELINE["slow_fast_window"],
        "slow_fast_slow_relative": PUBLIC_BASELINE["slow_fast_slow_relative"],
        "slow_fast_fast_relative": PUBLIC_BASELINE["slow_fast_fast_relative"],
        "long_window": PUBLIC_BASELINE["long_window"],
    }


class AuditCoverageTest(unittest.TestCase):
    def test_audit_value_table_covers_every_baseline_key(self) -> None:
        # `public_baseline_drift` 自带硬失败：取值表缺 PUBLIC_BASELINE 的键就 SystemExit。
        # 这里只要能跑通且判无偏离，就说明 13 个键它都认识。
        self.assertEqual(public_baseline_drift(baseline_meta()), [])


class RetrainPlanCoverageTest(unittest.TestCase):
    def test_every_baseline_key_is_derived_or_explicitly_exempt(self) -> None:
        """⭐ 本文件的核心断言 —— 2026-08-23 打补丁前，它因 `long_window` 而红。"""
        derived = set(production_structure())
        missing = [key for key in PUBLIC_BASELINE
                   if key not in derived and key not in RETRAIN_EXEMPT]
        self.assertEqual(
            missing, [],
            f"这些身份键既没被 production_structure() 派生、也没写进 RETRAIN_EXEMPT："
            f"{missing}。要么让重训计划带上它（train.py 有对应参数时），"
            f"要么写进豁免表并说明理由 —— 不能空着。")

    def test_every_comparable_key_is_actually_compared(self) -> None:
        """派生了却不比对，等于白派生 —— 生产 meta 与 PUBLIC_BASELINE 分家时不会红。"""
        comparable = set(production_structure()) & set(PUBLIC_BASELINE)
        missing = sorted(comparable - set(BASELINE_CHECKED_KEYS))
        self.assertEqual(missing, [],
                         f"这些键被 production_structure() 派生了但不在 "
                         f"BASELINE_CHECKED_KEYS 里，生成计划时不会对拍：{missing}")

    def test_exemptions_all_carry_a_reason_and_are_still_real_keys(self) -> None:
        for key, reason in RETRAIN_EXEMPT.items():
            self.assertIn(key, PUBLIC_BASELINE,
                          f"{key} 已不在 PUBLIC_BASELINE 里，豁免表该清理")
            self.assertTrue(reason.strip(), f"{key} 的豁免没写理由")


class DeliveryReportCoverageTest(unittest.TestCase):
    def test_delivery_report_prints_every_baseline_key(self) -> None:
        if not PRODUCTION_MODEL_DIR.is_dir():
            self.skipTest("生产模型目录不在盘上")
        resolution = {"spec": "auto", "resolved": None, "scanned": [], "matched": []}
        identity = model_identity(PRODUCTION_MODEL_DIR, resolution)["meta_identity"]
        missing = [key for key in PUBLIC_BASELINE
                   if VERIFY_DELIVERY_ALIASES.get(key, key) not in identity]
        self.assertEqual(missing, [],
                         f"交付报告的 meta 身份漏印这些键：{missing}。"
                         f"报告号称打印「模型身份」，漏一个就等于报告在说谎。")

    def test_aliases_point_at_keys_that_exist_on_both_sides(self) -> None:
        if not PRODUCTION_MODEL_DIR.is_dir():
            self.skipTest("生产模型目录不在盘上")
        resolution = {"spec": "auto", "resolved": None, "scanned": [], "matched": []}
        identity = model_identity(PRODUCTION_MODEL_DIR, resolution)["meta_identity"]
        for baseline_key, identity_key in VERIFY_DELIVERY_ALIASES.items():
            self.assertIn(baseline_key, PUBLIC_BASELINE)
            self.assertIn(identity_key, identity)


if __name__ == "__main__":
    unittest.main()
