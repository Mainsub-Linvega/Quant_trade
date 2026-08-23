"""「模型身份」的键必须被每一个消费者覆盖 —— 漏一个就当场红。

## 为什么需要这一层

`promote_v3_candidate.PUBLIC_BASELINE` 是「榜上那份模型长什么样」的唯一真值源，
但真正**使用**它的地方有四处，而且各自维护自己的取值表 / 派生表 / 命令行参数：

| 消费者 | 作用 |
|---|---|
| `audit_submission_zip.public_baseline_drift` | 提交包的内容身份审计 |
| `retrain_extended.{production_structure, BASELINE_CHECKED_KEYS}` | 重训计划复现生产结构 |
| `verify_delivery_runtime.model_identity` | 交付报告打印的 meta 身份 |
| `promote_v3_candidate.validate_meta` | 转正前的 staging meta 身份校验 |
| `make_submission` | 打包 |

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
from promote_v3_candidate import PUBLIC_BASELINE, validate_meta  # noqa: E402
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

# ---- `promote_v3_candidate.validate_meta` 承载每个身份键的 meta 字段 ----
# ⚠️ 2026-08-23 加这张表的理由：本文件此前只覆盖 audit / retrain / verify_delivery
# **三家**，而对 promote 只 import 了 `PUBLIC_BASELINE` 常量本身 ——
# 等于「转正」这条最关键的路**从来没被这道门禁看过**。
# 实测后果：`long_window` 在 `PUBLIC_BASELINE` 里躺了两天，`validate_meta()` 却一条都没查它，
# 而脚本自己的默认候选 `outputs/candidates/v3_hybrid_mkt_shrunk`（long_window=None、
# 截面森林 361 列、公榜低 1.662%）**零参数就能三道门全过**。
#
# 表的方向是「PUBLIC_BASELINE 的键 → 它在 meta 里由哪些字段承载」。
# 有些键名不同（`n_seeds` 在 meta 里是 `lgbm_model_files` 的长度），所以不能靠键名相等来查。
PROMOTE_META_KEYS: dict[str, tuple[str, ...]] = {
    "blend_weight": ("blend_weight",),
    "num_iteration": ("num_iteration",),
    "history_window": ("history_window",),
    "history_positions_count": ("history_positions",),
    "prediction_scale": ("prediction_scale",),
    "n_seeds": ("lgbm_model_files",),
    "market_lambda": ("market_lambda",),
    "market_model_count": ("market_model_files",),
    "cross_section_weighted": ("cross_section_weighted",),
    "slow_fast_window": ("slow_fast_window",),
    "slow_fast_slow_relative": ("slow_fast_slow_relative",),
    "slow_fast_fast_relative": ("slow_fast_fast_relative",),
    "long_window": ("long_window",),
}

# 确实无法由 validate_meta 把关的键必须进这里**并写理由**（空着 = 红）。
PROMOTE_EXEMPT: dict[str, str] = {}

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


class PromoteGateCoverageTest(unittest.TestCase):
    """转正门禁（第四个消费者）—— 断言是**行为式**的：删键必须当场红。

    不用「名单比对」是有原因的：前四次事故里有三次，键名明明在某张表里，
    但那张表根本没被用来做判断。只有「把键拿掉、看门禁响不响」才证明它真的接上了。
    """

    def _baseline_kwargs(self) -> dict:
        return {"scale": PUBLIC_BASELINE["prediction_scale"],
                "n_seeds": PUBLIC_BASELINE["n_seeds"],
                "blend_weight": PUBLIC_BASELINE["blend_weight"]}

    def test_baseline_meta_passes(self) -> None:
        """先证明夹具本身是干净的，否则下面的「都红了」毫无意义。"""
        validate_meta(baseline_meta(), **self._baseline_kwargs())

    def test_every_baseline_key_is_mapped_or_exempt(self) -> None:
        """往 PUBLIC_BASELINE 加键而 promote 没跟上 —— 当场红。"""
        missing = [key for key in PUBLIC_BASELINE
                   if key not in PROMOTE_META_KEYS and key not in PROMOTE_EXEMPT]
        self.assertEqual(
            missing, [],
            f"这些身份键没被 PROMOTE_META_KEYS 映射、也没写进 PROMOTE_EXEMPT：{missing}。"
            f"转正是交付件的必经路 —— 要么让 validate_meta 查它，要么写明为什么查不了。")

    def test_dropping_any_identity_key_is_rejected(self) -> None:
        """⭐ 核心断言：**丢键必须失败**。

        四次同型事故全部是「键没了」而不是「值错了」，而 `main.py` 对缺键一律静默降级
        （`if long_window and ...` / `PredictionTrail(...) if window else None`）⟹
        缺键这条路上没有任何东西会报错，除非门禁自己拦。
        """
        for key, meta_keys in PROMOTE_META_KEYS.items():
            with self.subTest(key=key):
                meta = baseline_meta()
                for meta_key in meta_keys:
                    meta.pop(meta_key, None)
                with self.assertRaises(ValueError, msg=
                        f"meta 里丢掉 {meta_keys}（身份键 {key}）竟然通过了 validate_meta —— "
                        f"这正是 08-18 / 08-19 / 08-21 / 08-23 四次事故的形状"):
                    validate_meta(meta, **self._baseline_kwargs())

    def test_off_baseline_is_the_only_way_past(self) -> None:
        """偏离必须是**显式按下去**的，不能是漏掉的（沿用 EXCLUDED_MODULES 那套语义）。"""
        meta = baseline_meta()
        meta.pop("long_window")
        validate_meta(meta, **self._baseline_kwargs(), off_baseline=True)

    def test_exemptions_all_carry_a_reason_and_are_still_real_keys(self) -> None:
        for key, reason in PROMOTE_EXEMPT.items():
            self.assertIn(key, PUBLIC_BASELINE,
                          f"{key} 已不在 PUBLIC_BASELINE 里，豁免表该清理")
            self.assertTrue(reason.strip(), f"{key} 的豁免没写理由")


if __name__ == "__main__":
    unittest.main()
