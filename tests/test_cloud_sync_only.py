"""`cloud_sync.py push --only` 的回归。

为什么值得单独钉：`cmd_push` 原来上传完**无条件写整份本地清单**。全量 push 时这是对的，
但配上 `--only` 就会让云端 `.sync_manifest.sha256` **谎称全都同步了** —— 之后每次
`status` 都报「零差异」，而云端其实还差一百多个文件。那是一个**静默**失效：
没有报错、没有红灯，只有一份不再诚实的清单（CLAUDE.md §8.11）。

2026-08-29 引入 `--only` 的动机：云端落后 133 个文件，但只需要送一个探针过去；
既不想动云端的 `strategies/v3_hybrid/model/`，也不想让清单失真。
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cloud_sync import diff, parse_manifest, render_manifest, select_only  # noqa: E402


class SelectOnlyTest(unittest.TestCase):
    CHANGED = ["experiments/thread_default_probe.py",
               "experiments/lgbm_speed.py",
               "scripts/verify_delivery_runtime.py",
               "strategies/v3_hybrid/model/lgbm_seed2026.txt",
               "tests/test_verify_delivery_runtime.py"]

    def test_exact_path(self) -> None:
        self.assertEqual(select_only(self.CHANGED, ["experiments/thread_default_probe.py"]),
                         ["experiments/thread_default_probe.py"])

    def test_directory_prefix(self) -> None:
        self.assertEqual(select_only(self.CHANGED, ["experiments"]),
                         ["experiments/thread_default_probe.py", "experiments/lgbm_speed.py"])

    def test_trailing_slash_is_equivalent(self) -> None:
        self.assertEqual(select_only(self.CHANGED, ["experiments/"]),
                         select_only(self.CHANGED, ["experiments"]))

    def test_glob(self) -> None:
        self.assertEqual(select_only(self.CHANGED, ["*/*probe*.py"]),
                         ["experiments/thread_default_probe.py"])

    def test_repeatable_patterns_union_without_duplicates(self) -> None:
        picked = select_only(self.CHANGED, ["experiments", "experiments/lgbm_speed.py"])
        self.assertEqual(picked, ["experiments/thread_default_probe.py",
                                  "experiments/lgbm_speed.py"])

    def test_prefix_does_not_leak_into_sibling_names(self) -> None:
        # "experiment" 不该匹配 "experiments/..."：前缀比较必须带分隔符。
        self.assertEqual(select_only(self.CHANGED, ["experiment"]), [])

    def test_no_match_returns_empty_not_everything(self) -> None:
        # 空集必须是空集 —— 「没匹配到」绝不能退化成「传全部」。
        self.assertEqual(select_only(self.CHANGED, ["nope/"]), [])

    def test_model_dir_is_only_reachable_when_named(self) -> None:
        self.assertEqual(select_only(self.CHANGED, ["experiments"]), 
                         ["experiments/thread_default_probe.py", "experiments/lgbm_speed.py"])
        self.assertIn("strategies/v3_hybrid/model/lgbm_seed2026.txt",
                      select_only(self.CHANGED, ["strategies/v3_hybrid/model"]))


class ManifestHonestyTest(unittest.TestCase):
    """--only 之后，云端清单必须只记真的传上去的那几条。"""

    LOCAL = {"a.py": "aa", "b.py": "bb", "c.py": "cc"}
    REMOTE = {"a.py": "old", "b.py": "bb", "c.py": "old"}

    def _synced_after_only_push(self, pushed: list[str]) -> dict[str, str]:
        # 复刻 cmd_push 在 --only 分支下的清单构造。
        synced = dict(self.REMOTE)
        synced.update({relative: self.LOCAL[relative] for relative in pushed})
        return synced

    def test_untouched_files_keep_their_stale_remote_hash(self) -> None:
        synced = self._synced_after_only_push(["a.py"])
        self.assertEqual(synced["a.py"], "aa")        # 传了 ⟹ 更新
        self.assertEqual(synced["c.py"], "old")       # 没传 ⟹ 保持旧值，不许谎称已同步

    def test_status_still_reports_the_rest_as_pending(self) -> None:
        synced = self._synced_after_only_push(["a.py"])
        changed, vanished = diff(self.LOCAL, synced)
        self.assertEqual(changed, ["c.py"])           # 仍然待传，这正是诚实的证据
        self.assertEqual(vanished, [])

    def test_writing_the_full_local_manifest_would_hide_the_backlog(self) -> None:
        # 反例钉住：这是修复前的行为，必须能被区分出来。
        changed, _ = diff(self.LOCAL, self.LOCAL)
        self.assertEqual(changed, [], "写整份本地清单会让 status 报零差异 —— 那正是 bug")

    def test_manifest_round_trips(self) -> None:
        synced = self._synced_after_only_push(["a.py"])
        self.assertEqual(parse_manifest(render_manifest(synced)), synced)


class PushArgsTest(unittest.TestCase):
    def test_only_defaults_to_none_so_full_push_is_unchanged(self) -> None:
        # 默认必须是 None：--only 是可选加法，不能改变既有全量 push 的行为。
        from cloud_sync import build_parser  # noqa: PLC0415
        args = build_parser().parse_args(["push"])
        self.assertIsNone(args.only)
        self.assertFalse(args.dry_run)

    def test_only_is_repeatable(self) -> None:
        from cloud_sync import build_parser  # noqa: PLC0415
        args = build_parser().parse_args(["push", "--only", "x/", "--only", "y.py"])
        self.assertEqual(args.only, ["x/", "y.py"])


if __name__ == "__main__":
    unittest.main()
