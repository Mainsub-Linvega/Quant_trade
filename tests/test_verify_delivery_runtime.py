"""交付运行时验证的门禁回归。

三件事被钉住，每一件都对应一次真实的漏检：

1. **内存门禁**（2026-08-23 新增）。`docs/competition_description.md:158-159` 写明评测环境
   是 4 核 / 12 GB，「内存超限……严重情况下提交可能被判定为无效」，而此前这个脚本
   一个内存字段都没有。私榜截止后无法改代码、出错按填 0 处理 ⟹ 这是唯一一个能让整个
   提交归零而我们从未测量过的量。
2. **manifest 解析**。默认值曾写死 `v3_hybrid_slowfast`，08-21 转正 long512 后就过期了，
   于是 `model_matches_promotion_manifest` 长期红着 —— 红的原因是**比错了对象**，
   不是装错了模型。写死候选名必然随每次转正过期。
3. **公榜基线偏离**。取值表必须复用 `audit_submission_zip.public_baseline_drift`，
   不能另抄一份 —— 两张表分头维护正是 08-18（slow/fast 丢键）与 08-21（long_window 丢键）
   两次「静默降级」事故的形状。
4. **`--from-zip` 的归属锚点**（2026-08-25 新增）。此前本脚本永远指着 `strategies/v3_hybrid/`
   跑 —— 那是源目录，不是交出去的那件东西。现在解压真 zip 再跑，且 zip 的 sha256
   必须进落盘 JSON：没有它，「测了交付物」就只是一句自述。
5. **运行时间预算**（2026-08-29 新增）。主办方当日才补全 180 s / 50 ms / 那条总时长公式，
   而此前脚本把 `total_timeout_seconds` 写死成 `None` ⟹ runner 的 `aborted_after_timeout`
   分支从来没被走过，`not_aborted` 这道门禁**一直没有失败的机会**。
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from promote_v3_candidate import PUBLIC_BASELINE  # noqa: E402
from verify_delivery_runtime import (EVAL_CPU_CORES, EVAL_MEMORY_GB,  # noqa: E402
                                     EVAL_MEAN_PREDICT_BUDGET_S,
                                     EVAL_MODEL_INIT_TIMEOUT_S, EXPECTED_CALLS,
                                     ProgressProxy, extract_submission_zip, model_identity,
                                     peak_rss_bytes, public_baseline_drift,
                                     resolve_manifest, rss_verdict, total_timeout_budget)

GB = 1 << 30


def baseline_meta(**overrides) -> dict:
    """一份与 PUBLIC_BASELINE 完全一致的 meta；overrides 用来制造偏离。"""
    meta = {
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
        "prediction_clip": 0.5,
    }
    meta.update(overrides)
    return meta


class RssGateTest(unittest.TestCase):
    def test_limit_default_comes_from_organizer_doc(self) -> None:
        # 这两个常量不是随手挑的，改动它们等于改动交付判据。
        self.assertEqual(EVAL_CPU_CORES, 4)
        self.assertEqual(EVAL_MEMORY_GB, 12.0)

    def test_under_limit_but_without_headroom_is_reported_separately(self) -> None:
        v = rss_verdict(int(10.5 * GB), 12.0, 0.20)
        self.assertTrue(v["under_limit"])          # 能跑
        self.assertFalse(v["has_headroom"])        # 但没有余量吸收环境差异
        self.assertAlmostEqual(v["headroom_threshold_gb"], 9.6)
        self.assertAlmostEqual(v["utilization"], 10.5 / 12.0)

    def test_over_limit_fails_hard(self) -> None:
        v = rss_verdict(int(12.5 * GB), 12.0, 0.20)
        self.assertFalse(v["under_limit"])
        self.assertFalse(v["has_headroom"])

    def test_comfortable_run_passes_both(self) -> None:
        v = rss_verdict(int(5.0 * GB), 12.0, 0.20)
        self.assertTrue(v["under_limit"])
        self.assertTrue(v["has_headroom"])

    def test_boundary_is_strict(self) -> None:
        # 恰好等于上限**不算**通过 —— 边界上没有余量可言。
        self.assertFalse(rss_verdict(12 * GB, 12.0, 0.0)["under_limit"])

    def test_peak_rss_is_a_high_water_mark(self) -> None:
        before = peak_rss_bytes()
        self.assertGreater(before, 0)
        ballast = bytearray(200 * (1 << 20))      # 200 MB，确实触碰以免被优化掉
        ballast[::4096] = b"\x01" * len(ballast[::4096])
        after_alloc = peak_rss_bytes()
        del ballast
        after_free = peak_rss_bytes()
        self.assertGreaterEqual(after_alloc, before)
        # 高水位只涨不落：释放之后读到的值不得低于分配时的峰值。
        self.assertGreaterEqual(after_free, after_alloc)


class PublicBaselineDriftTest(unittest.TestCase):
    def test_identical_meta_has_no_drift(self) -> None:
        self.assertEqual(public_baseline_drift(baseline_meta()), [])

    def test_missing_long_window_is_caught(self) -> None:
        # 2026-08-21 那次事故的复刻：main.py 取不到 long_window 就**静默**关掉长窗，
        # 交出去的是低 1.66% 的旧模型。这一条必须被门禁抓到，而不是靠人眼。
        meta = baseline_meta()
        del meta["long_window"]
        drift = public_baseline_drift(meta)
        self.assertTrue(any("long_window" in d for d in drift), drift)

    def test_missing_slow_fast_is_caught(self) -> None:
        # 2026-08-18 同型事故：丢键 ⟹ 静默退回单一 scale，低 2.93%。
        meta = baseline_meta()
        del meta["slow_fast_window"]
        self.assertTrue(any("slow_fast_window" in d
                            for d in public_baseline_drift(meta)))

    def test_wrong_seed_count_is_caught(self) -> None:
        meta = baseline_meta(lgbm_model_files=["s0.txt", "s1.txt"])
        self.assertTrue(any("n_seeds" in d for d in public_baseline_drift(meta)))


class ResolveManifestTest(unittest.TestCase):
    def _fake_model_dir(self, root: Path) -> Path:
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "hybrid_meta.json").write_text(
            json.dumps(baseline_meta()), encoding="utf-8")
        (model_dir / "forest.txt").write_text("tree", encoding="utf-8")
        return model_dir

    def test_explicit_path_is_passed_through_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            model_dir = self._fake_model_dir(Path(d))
            res = resolve_manifest(model_dir, "/some/explicit/manifest.json")
            self.assertEqual(res["resolved"], "/some/explicit/manifest.json")
            self.assertEqual(res["scanned"], [])

    def test_auto_finds_exactly_the_matching_staging_in_the_real_repo(self) -> None:
        # 真仓库上的行为：生产目录必须唯一命中一份 staging。
        # 命中 0 份 ⟹ 生产目录来路不明；命中 >1 份 ⟹ 有重复 staging，两者都该看见。
        model_dir = ROOT / "strategies" / "v3_hybrid" / "model"
        if not model_dir.is_dir():
            self.skipTest("生产模型目录不在盘上")
        res = resolve_manifest(model_dir, "auto")
        self.assertEqual(len(res["matched"]), 1, res["scanned"])
        self.assertIsNotNone(res["resolved"])
        # 扫描过程本身要留证：每份 staging 都得有一行，供人核对挑的是哪一个。
        self.assertGreater(len(res["scanned"]), 1)
        self.assertEqual(sum(r["identical"] for r in res["scanned"]), 1)

    def test_auto_reports_false_not_none_when_nothing_matches(self) -> None:
        # 关键区分：找不到匹配是**失败**，不是「跳过比对」。
        # 旧代码在 manifest 不存在时写 identical=None，读报告的人容易当成「没测」。
        with tempfile.TemporaryDirectory() as d:
            model_dir = self._fake_model_dir(Path(d))
            res = {"spec": "auto", "resolved": None, "scanned": [], "matched": []}
            identity = model_identity(model_dir, res)
            self.assertIs(identity["manifest"]["identical"], False)
            self.assertIn("auto", identity["manifest"]["note"])

    def test_model_identity_carries_resolution_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            model_dir = self._fake_model_dir(Path(d))
            res = {"spec": "auto", "resolved": None, "scanned": [], "matched": []}
            identity = model_identity(model_dir, res)
            self.assertEqual(identity["manifest"]["resolution"], res)
            self.assertEqual(identity["public_baseline_drift"], [])


class ProgressProxyTest(unittest.TestCase):
    """代理必须**完全透明** —— 它夹在生产模型和官方 runner 之间，
    任何行为改变都会污染交付验证本身。"""

    class Spy:
        backend = "lightgbm"

        def __init__(self) -> None:
            self.seen: list[object] = []

        def predict(self, test: object) -> str:
            self.seen.append(test)
            return f"pred:{test}"

        def other_method(self) -> int:
            return 42

    def test_predict_return_value_and_argument_pass_through_unchanged(self) -> None:
        spy = self.Spy()
        proxy = ProgressProxy(spy, every=1000, expected=10)
        self.assertEqual(proxy.predict("frame-a"), "pred:frame-a")
        self.assertEqual(spy.seen, ["frame-a"])

    def test_non_predict_attributes_are_forwarded(self) -> None:
        # runner 只调 predict，但 backend 等字段要能被外部读到 ——
        # 读不到就会让身份检查看错对象。
        proxy = ProgressProxy(self.Spy(), every=1000, expected=10)
        self.assertEqual(proxy.backend, "lightgbm")
        self.assertEqual(proxy.other_method(), 42)

    def test_progress_prints_exactly_every_n_calls(self) -> None:
        import contextlib
        import io
        proxy = ProgressProxy(self.Spy(), every=3, expected=10)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            for _ in range(10):
                proxy.predict("f")
        # 10 次调用、每 3 次一行 ⟹ 第 3/6/9 次共 3 行
        self.assertEqual(buffer.getvalue().count("[进度]"), 3)

    def test_missing_attribute_still_raises_attribute_error(self) -> None:
        # __getattr__ 透传不能把打错的属性名吞成静默 None。
        proxy = ProgressProxy(self.Spy(), every=1000, expected=10)
        with self.assertRaises(AttributeError):
            proxy.definitely_not_there


class FromZipTest(unittest.TestCase):
    """`--from-zip`：跑的到底是不是那份交付物。

    ⚠️ 2026-08-25 新增。CLAUDE.md 伤疤规则 11 要的是「每个测量配一个**能失败**的归属检查」——
    对「我们验证了交付物的运行时」这句话，那个检查就是**落盘 JSON 里的 zip sha256**。
    """

    def _zip(self, directory: Path, name: str = "pkg.zip") -> Path:
        path = directory / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("main.py", "class Model: pass\n")
            archive.writestr("model/hybrid_meta.json", json.dumps(baseline_meta()))
        return path

    def test_zip_sha256_and_extraction_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._zip(root)
            target, evidence = extract_submission_zip(path, force=False, root=root / "out")
            self.assertEqual(evidence["sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(evidence["bytes"], path.stat().st_size)
            self.assertEqual(evidence["file_count"], 2)
            self.assertEqual(target.name, "pkg")
            self.assertEqual(evidence["extracted_to"], str(target))
            self.assertTrue((target / "main.py").exists())

    def test_audit_failure_is_recorded_not_raised(self) -> None:
        """审计不过要**落盘**是哪几项红了，不是抛异常把现场丢掉。

        拦截由 `checks["zip_audit_passed"]` 负责，与其它门禁同一层。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, evidence = extract_submission_zip(self._zip(root), force=False,
                                                 root=root / "out")
        self.assertFalse(evidence["audit_passed"])
        self.assertIn("required_files_present", evidence["audit_failing_checks"])
        self.assertIn("requirements_covers_dependencies", evidence["audit_failing_checks"])

    def test_existing_target_needs_force(self) -> None:
        """CLAUDE.md §5.10：产物不得静默覆盖。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._zip(root)
            extract_submission_zip(path, force=False, root=root / "out")
            with self.assertRaises(SystemExit) as caught:
                extract_submission_zip(path, force=False, root=root / "out")
            self.assertIn("--force", str(caught.exception))
            extract_submission_zip(path, force=True, root=root / "out")

    def test_missing_zip_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                extract_submission_zip(Path(directory) / "nope.zip", force=False,
                                       root=Path(directory) / "out")



if __name__ == "__main__":
    unittest.main()


class TimeBudgetTest(unittest.TestCase):
    """2026-08-29：官方补全的三条运行时间限制。"""

    def test_limits_come_from_organizer_doc(self) -> None:
        # 与 EVAL_MEMORY_GB 同性质：改动它们等于改动交付判据，不能顺手调。
        self.assertEqual(EVAL_MODEL_INIT_TIMEOUT_S, 180.0)
        self.assertEqual(EVAL_MEAN_PREDICT_BUDGET_S, 0.05)

    def test_default_budget_is_the_lower_bound(self) -> None:
        # a、b 都 ≥ 0，所以 a=b=0 是预算下限 ⟹ 按它判定永远比真实评测更严。
        self.assertAlmostEqual(total_timeout_budget(EXPECTED_CALLS), 0.05 * EXPECTED_CALLS)
        self.assertLess(total_timeout_budget(1000),
                        total_timeout_budget(1000, a=0.01, b=60.0))

    def test_budget_scales_linearly_with_n_time_id(self) -> None:
        # 关键性质：预算随 n_time_id 线性缩放 ⟹ 9 月实盘期变长不改变我们的占比。
        self.assertAlmostEqual(total_timeout_budget(2 * EXPECTED_CALLS),
                               2 * total_timeout_budget(EXPECTED_CALLS))

    def test_bias_term_is_a_constant_not_a_rate(self) -> None:
        self.assertAlmostEqual(total_timeout_budget(10, b=7.0) - total_timeout_budget(10), 7.0)
        self.assertAlmostEqual(total_timeout_budget(99, b=7.0) - total_timeout_budget(99), 7.0)
