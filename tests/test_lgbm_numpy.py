"""断言纯 numpy 树遍历与 lightgbm 逐位等价。

验收判据（`2026-08-09_计划.md` §2.2）：`max|Δ| / std(pred) < 1e-9`。
两条路径只该差 480 个 double 的求和顺序（~1e-19）；真翻了一个分裂，
输出会跳一个叶子值（~1e-3）—— 判据落在两者中间十几个数量级，怎么定都不会误判。

要求本机装了 lightgbm（离线校验才需要；提交包不需要）。没装就跳过。

用法：.venv/bin/python -m unittest tests.test_lgbm_numpy -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "strategies" / "v3_hybrid"
sys.path[:0] = [str(ROOT), str(STRATEGY)]

from lgbm_numpy import NumpyForest                      # noqa: E402


def load_strategy_main():
    """按**文件路径**加载 v3_hybrid/main.py。

    不能直接 `import main` —— 每个策略目录都有一个叫 `main` 的模块，
    `unittest discover` 一次跑完整个 tests/ 时，先跑的用例可能已经把
    v1_ridge 的 `main` 塞进 sys.modules 了，这里会拿到错的那个。
    """
    spec = importlib.util.spec_from_file_location(
        "v3_hybrid_main_under_test", STRATEGY / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    import lightgbm as lgb
except ImportError:                                     # pragma: no cover
    lgb = None

N_ASSETS = 15
TOLERANCE = 1e-9


def _load():
    meta = json.loads((STRATEGY / "model" / "hybrid_meta.json").read_text(encoding="utf-8"))
    paths = [STRATEGY / "model" / name for name in meta["lgbm_model_files"]]
    rounds = int(meta["num_iteration"])
    return meta, paths, rounds


def _reference(paths, design, rounds):
    """lightgbm 侧：与 main.py 一致，先按种子求和，最后再除以模型数。"""
    total = np.zeros(len(design), dtype=np.float64)
    for path in paths:
        total += lgb.Booster(model_file=str(path)).predict(design, num_iteration=rounds)
    return total


@unittest.skipIf(lgb is None, "本机没装 lightgbm，无法对拍")
class LgbmNumpyEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.meta, cls.paths, cls.rounds = _load()
        cls.forest = NumpyForest.from_files(cls.paths, cls.rounds)

    def _assert_matches(self, design, assets, label):
        reference = _reference(self.paths, design, self.rounds)
        mine = self.forest.predict_sum(design, assets)
        spread = float(np.std(reference)) or 1.0
        ratio = float(np.max(np.abs(reference - mine))) / spread
        self.assertLess(ratio, TOLERANCE, f"{label}: max|Δ|/std = {ratio:.3e}")
        return ratio

    def test_matches_on_synthetic_batches(self) -> None:
        """随机输入，2 万行 / 15 行一批 —— 比真实数据更能扫到分裂阈值附近。

        ⚠️ 列宽必须**从模型里读**（`forest.n_features`），不能写死。
        原来写死 201（= 200 dev + asset_id）—— 那是 08-10 那版无 history 的生产模型；
        08-13 生产换成 361 列（+ 4×40 history）后这条就炸了。
        """
        rng = np.random.default_rng(20260809)
        groups = 1400
        width = self.forest.n_features - 1                        # 最后一列是 asset_id
        features = rng.normal(0.0, 1.0, (groups * N_ASSETS, width)).astype(np.float32)
        features[rng.random(features.shape) < 0.02] *= 8.0        # 造点尾部
        assets = np.tile(np.arange(N_ASSETS), groups)
        design = np.column_stack([features, assets.astype(np.float32)])
        worst = 0.0
        for start in range(0, len(design), N_ASSETS):
            batch = slice(start, start + N_ASSETS)
            worst = max(worst, self._assert_matches(
                design[batch], assets[batch], f"batch@{start}"))
        self.assertLess(worst, TOLERANCE)

    def test_matches_on_real_test_partition(self) -> None:
        """真实测试数据，走生产的预处理管线（robust transform + 截面去均值）。"""
        import pyarrow.parquet as pq
        from features import apply_robust_transform, cross_sectional_deviation

        path = ROOT / "data" / "test" / "test_partition_000.parquet"
        if not path.exists():
            self.skipTest("没有 data/test/，跳过真实数据对拍")
        columns = ["time_id", "asset_id", *self.meta["lgbm_features"]]
        frame = next(pq.ParquetFile(path).iter_batches(
            batch_size=6000, columns=columns)).to_pandas()

        raw = frame.loc[:, self.meta["lgbm_features"]].to_numpy(dtype=np.float32, copy=True)
        apply_robust_transform(
            raw, *(np.asarray(self.meta[name], dtype=np.float32)
                   for name in ("lower", "upper", "center", "scale")))
        deviation = cross_sectional_deviation(raw, frame["time_id"].to_numpy(dtype=np.int64))
        assets = frame["asset_id"].to_numpy(dtype=np.int64)
        # ⚠️ 设计矩阵必须与生产 `main.Model.predict` 逐列对应：
        # `[xs_dev ‖ history 4 块 ‖ asset_id]`。原来这里只拼了 `[dev ‖ asset_id]`，
        # 那是无 history 时代的形状；生产一换成 361 列就炸。
        blocks = [deviation]
        if self.meta.get("history_positions"):
            from history import history_design_blocks
            hist, _ = history_design_blocks(raw, assets, self.meta["history_positions"],
                                            int(self.meta["history_window"]))
            blocks.extend(hist)
        # ⚠️ 2026-08-21：长窗块只进**截面**设计、且在 asset_id 之前（与 main.py 逐列对应）。
        long_blocks = []
        if self.meta.get("long_window") and self.meta.get("history_positions"):
            from history import AssetLongWindow
            positions = np.asarray(self.meta["history_positions"], dtype=np.int64)
            long_blocks = list(AssetLongWindow(feature_count=len(positions),
                                               window=int(self.meta["long_window"])).transform(
                raw[:, positions], assets))
        blocks.append(assets.astype(np.float32))
        design = np.column_stack(blocks[:-1] + long_blocks + blocks[-1:])
        self.assertEqual(design.shape[1], self.forest.n_features,
                         "设计矩阵列数与模型不符 —— 测试的口径落后于生产了")

        time_ids = frame["time_id"].to_numpy(dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(time_ids[1:] != time_ids[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(time_ids)])
        for start, count in zip(starts, counts):
            batch = slice(int(start), int(start) + int(count))
            self._assert_matches(design[batch], assets[batch], f"time_id@{start}")

    def test_rejects_repeated_and_out_of_range_assets(self) -> None:
        """按 asset 槽位摆行，重复会静默丢行 —— 必须抛，不能算错。"""
        design = np.zeros((3, self.forest.n_features), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.forest.predict_sum(design, np.array([0, 0, 1]))
        with self.assertRaises(ValueError):
            self.forest.predict_sum(design, np.array([0, 1, N_ASSETS]))
        with self.assertRaises(ValueError):
            self.forest.predict_sum(design, np.array([0, 1, -1]))

    def test_rejects_unsupported_model_text(self) -> None:
        """结构假设不成立时必须炸，绝不静默算错。"""
        from lgbm_numpy import _parse_model_text

        import re

        text = self.paths[0].read_text(encoding="utf-8")
        # ⚠️ 别写死 "decision_type=1 2" —— 那串取决于第一棵树长什么样，
        # 换一批模型就可能不存在，`str.replace` 会**静默不替换**，
        # 于是这条用例变成永远通过的空壳（08-13 换生产模型时正是这么暴露的）。
        match = re.search(r"^decision_type=(\S+)(.*)$", text, re.M)
        self.assertIsNotNone(match, "模型文本里没有 decision_type 行")
        bad_decision = text[:match.start()] + f"decision_type=9{match.group(2)}" + text[match.end():]
        self.assertNotEqual(bad_decision, text, "构造的坏样本与原文相同 —— 替换没生效")

        for broken, needle in (
            (text.replace("num_class=1", "num_class=2", 1), "num_class"),
            (text.replace("objective=regression", "objective=binary sigmoid:1", 1), "regression"),
            (text.replace("\nversion=v4", "\naverage_output=\nversion=v4", 1), "average_output"),
            (bad_decision, "decision_type"),
        ):
            self.assertNotEqual(broken, text, f"{needle} 的坏样本没造出来")
            with self.assertRaises(ValueError, msg=f"{needle} 没被拦下"):
                _parse_model_text(broken)


class BackendSelectionTest(unittest.TestCase):
    """main.Model 的后端选择与开机自检。"""

    def test_numpy_and_lightgbm_agree_end_to_end(self) -> None:
        strategy_main = load_strategy_main()

        path = ROOT / "data" / "test" / "test_partition_000.parquet"
        if not path.exists():
            self.skipTest("没有 data/test/，跳过端到端对拍")
        import pyarrow.parquet as pq
        frame = next(pq.ParquetFile(path).iter_batches(batch_size=3000)).to_pandas()
        time_ids = sorted(frame["time_id"].unique())[:100]
        frame = frame[frame["time_id"].isin(time_ids)]

        outputs = {}
        for backend in ("numpy", "lightgbm"):
            if backend == "lightgbm" and lgb is None:
                self.skipTest("本机没装 lightgbm")
            model = strategy_main.Model(backend=backend)
            self.assertEqual(model.backend, backend)
            outputs[backend] = np.concatenate(
                [model.predict(frame[frame["time_id"] == t]) for t in time_ids])
        spread = float(np.std(outputs["lightgbm"]))
        ratio = float(np.max(np.abs(outputs["lightgbm"] - outputs["numpy"]))) / spread
        self.assertLess(ratio, TOLERANCE, f"两后端最终预测 max|Δ|/std = {ratio:.3e}")

    def test_auto_backend_prefers_lightgbm_when_selfcheck_passes(self) -> None:
        strategy_main = load_strategy_main()

        if lgb is None:
            self.skipTest("本机没装 lightgbm")
        self.assertEqual(strategy_main.Model().backend, "lightgbm")


if __name__ == "__main__":
    unittest.main()
