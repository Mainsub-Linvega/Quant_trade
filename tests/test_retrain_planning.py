from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments")]

from joint_recalibration import matrix
from promote_v3_candidate import PUBLIC_BASELINE
from retrain_extended import (MARKET_MIN_DATA_SCALE, assert_matches_public_baseline,
                              command_plan, production_structure, validate_audit)


class RetrainPlanningTest(unittest.TestCase):
    def test_audit_gate_requires_train_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({"comparison": {"changed": True, "splits": {
                "train": {"added": ["new.parquet"], "removed": [], "modified": [], "row_delta": 5}}}}))
            self.assertTrue(validate_audit(path)["changed"])
            path.write_text(json.dumps({"comparison": {"changed": False, "splits": {"train": {}}}}))
            with self.assertRaises(SystemExit):
                validate_audit(path)

    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(data_root="data", candidate_dir="outputs/candidates/probe",
                                  ridge_alpha=2e6, ridge_feature_count=200,
                                  lgbm_feature_count=200, sample_modulo=5,
                                  num_iteration=480, n_seeds=3, prediction_scale=1.16)

    def test_command_plan_is_candidate_only(self) -> None:
        commands = command_plan(self._args())
        flattened = " ".join(part for command in commands for part in command)
        self.assertIn("outputs/candidates/probe", flattened)
        self.assertNotIn("strategies/v3_hybrid/model", flattened)

    def test_command_plan_reproduces_production_structure(self) -> None:
        """2026-08-19 的回归：计划此前不带任何结构开关。

        `--weighted-cross-section` 和 `--market-model` 都是 store_true，不传 = False
        ⟹ 跑出来的「固定结构」候选没有行级市场森林、截面块也不带权，
        等于退回 08-11 那版架构（公榜 0.0032523499，比生产低 21.99%）。
        转正门禁会拦住，但那是在几小时训练之后。
        """
        structure = production_structure()
        commands = command_plan(self._args(), structure)
        v3 = next(command for command in commands if "v3_hybrid" in " ".join(command))
        self.assertIn("--weighted-cross-section", v3)
        self.assertIn("--market-model", v3)
        self.assertEqual(v3[v3.index("--market-lambda") + 1], str(PUBLIC_BASELINE["market_lambda"]))
        self.assertEqual(v3[v3.index("--market-min-data-scale") + 1], str(MARKET_MIN_DATA_SCALE))
        spec = json.loads(v3[v3.index("--market-spec") + 1])
        self.assertEqual(spec, structure["market_spec"])
        # 市场块容量必须真的是收缩档（08-13 mkt_shrunk，公榜 +0.77%），不是 SPEC 默认
        self.assertEqual(spec["num_leaves"], 15)

    def test_command_plan_carries_long_window(self) -> None:
        """2026-08-23 的回归，与上一条同型 —— 只是这次漏的是长窗。

        `train.py:335` 的 `--long-window` **默认 0（＝关闭）**，而重训计划此前从未传过它，
        `production_structure()` 也没派生它 ⟹ D1 会训出一个**没有长窗**的候选，
        而长窗正是 08-21 转正、公榜实测 **+1.662%** 的那块结构。
        转正门禁最终会拦下（`PUBLIC_BASELINE` 含 `long_window: 512`），
        但那是在几小时训练之后，而 8/23→8/31 只有 8 天。
        """
        structure = production_structure()
        commands = command_plan(self._args(), structure)
        v3 = next(command for command in commands if "v3_hybrid" in " ".join(command))
        self.assertIn("--long-window", v3)
        self.assertEqual(v3[v3.index("--long-window") + 1],
                         str(PUBLIC_BASELINE["long_window"]))
        # 值必须从生产 meta 派生，不是常量
        self.assertEqual(int(v3[v3.index("--long-window") + 1]), structure["long_window"])


    def test_plan_rejects_structure_drift_vs_public_baseline(self) -> None:
        """生产 meta 与 PUBLIC_BASELINE 分家时，8/23 之前就要红。"""
        structure = production_structure()
        for key, bad in (("market_lambda", 0.0), ("cross_section_weighted", False),
                         ("num_iteration", 960), ("market_model_count", 0),
                         # 2026-08-23 补：缺键与 0 都等于「关掉长窗」，两种都要红。
                         # ⭐ 顺带说明一件事：`command_plan` 无条件调本守卫，所以
                         # 「长窗被关掉的结构」根本传不进计划生成 —— `command_plan` 里那句
                         # `if structure.get("long_window")` 在当前世界不可达，是与
                         # `market_model_count` 同型的防御性写法，留着以备长窗日后真被撤下。
                         ("long_window", None), ("long_window", 0)):
            with self.subTest(key=key):
                drifted = dict(structure, **{key: bad})
                with self.assertRaises(SystemExit):
                    assert_matches_public_baseline(drifted)

    def test_production_structure_matches_public_baseline(self) -> None:
        assert_matches_public_baseline(production_structure())

    def test_joint_matrix_is_finite_and_joint(self) -> None:
        payload = matrix()
        self.assertEqual(len(payload["ridge"]), 12)
        self.assertEqual(len(payload["lgbm"]), 9)
        self.assertTrue(all({"ridge_alpha", "feature_count", "sample_modulo"} <= row.keys()
                            for row in payload["ridge"]))


if __name__ == "__main__":
    unittest.main()
