"""扩展数据根 + 训练段边界门禁的回归用例。

## 为什么这一层必须有

`RUNBOOK_8_23.md` D1 写死「决策期重训必须止于 `time_id 1,045,889`，训进密封段等于把测试集
喂给模型，D2 之后的一切比较全部作废 —— **而且不会报错**」。原文自己点名了失效模式是
**静默**的，但直到 2026-08-24 之前，这条纪律**没有任何机械手段**：

- `strategies/{v1_ridge,v3_hybrid}/train.py` 都没有时间截断参数；
- `src/io.py:20` 按 manifest 顺序整分区读；
- 而密封段起点 1,045,920 **落在回补 p001 内部**（59.1% 在边界前）⟹ 分区级切分做不到。

现在边界由 `build_extended_data_root.py` 写进 `<root>/root_identity.json`，
`retrain_extended.load_root_identity` 读它并与密封期计划对拍。本文件钉死三件事：

1. 截断真的截在该截的地方，且**没有改动数据**（前缀逐 bit 相同）；
2. 回补分区被重命名成 009/010/011 —— 它们与本地 000/001/002 **同名内容全异**，
   不改名就会静默覆盖训练集前 1/3；
3. 门禁 **fail closed**：缺 identity、role 不符、边界不符，三种都必须拒绝生成计划。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_extended_data_root as B                                      # noqa: E402
from retrain_extended import ROLE_DECISION, ROLE_FULL, load_root_identity  # noqa: E402

N_ASSETS = 3
COLUMNS = ["row_id", "time_id", "asset_id", "weight", "target",
           "feature_000", "feature_001", "responder_00"]


def make_partition(path: Path, first_row: int, first_time: int, n_times: int) -> int:
    """写一个迷你 train 分区，形状与真数据同构（每个 time_id 恰好 N_ASSETS 行）。"""
    rng = np.random.default_rng(first_time)
    n = n_times * N_ASSETS
    time_id = np.repeat(np.arange(first_time, first_time + n_times, dtype=np.int64), N_ASSETS)
    table = pa.table({
        "row_id": pa.array(np.arange(first_row, first_row + n, dtype=np.int64)),
        "time_id": pa.array(time_id),
        "asset_id": pa.array(np.tile(np.arange(N_ASSETS, dtype=np.int8), n_times)),
        "weight": pa.array(rng.random(n).astype(np.float32)),
        "target": pa.array(rng.standard_normal(n).astype(np.float32)),
        "feature_000": pa.array(rng.standard_normal(n).astype(np.float32)),
        # 留一列带 NaN：截断重写若把 NaN 位模式改了，逐 bit 断言会抓到
        "feature_001": pa.array(np.where(rng.random(n) < 0.1, np.nan,
                                         rng.standard_normal(n)).astype(np.float32)),
        "responder_00": pa.array(rng.standard_normal(n).astype(np.float32)),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return n


class Fixture:
    """本地 2 个分区（time 0–199）+ 回补 2 个分区（time 200–399）。cutoff 落在回补第二个里。"""

    def __init__(self, base: Path) -> None:
        self.data_root = base / "data"
        self.backfill = base / "backfill"
        row = 0
        for index, first_time in enumerate((0, 100)):
            row += make_partition(
                self.data_root / "train" / f"train_partition_{index:03d}.parquet",
                row, first_time, 100)
        self.local_rows = row
        # ⚠️ 回补包的文件名故意与本地前两个**同名**——真包就是这样
        for index, first_time in enumerate((200, 300)):
            row += make_partition(
                self.backfill / "train" / f"train_partition_{index:03d}.parquet",
                row, first_time, 100)
        self.total_rows = row
        (self.data_root / "manifest.json").write_text(json.dumps(
            {"files": {"train": [f"train/train_partition_{i:03d}.parquet" for i in range(2)]}}))
        (self.data_root / "test").mkdir(parents=True, exist_ok=True)
        self.plan = base / "sealed_period_plan.json"
        # cutoff 349 落在回补第二个分区（300–399）**内部**——与真数据同型
        self.plan.write_text(json.dumps({"geometry": {
            "seal_time_id_min": 380, "embargo_real_time_ids": 30,
            "decision_train_time_id_max": 349}}))


class SealedCutoffTest(unittest.TestCase):
    def test_cutoff_comes_from_the_plan_and_is_arithmetically_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            self.assertEqual(B.decision_cutoff(fixture.plan), 349)

    def test_self_contradictory_plan_is_rejected(self) -> None:
        """embargo 是**空出来的 time_id 个数** ⟹ cutoff == seal_min - embargo - 1。

        这一条抓的是写这段代码时真犯过的差一错：起初断言写成 `seal_min - embargo`，
        对着真计划（1,045,920 − 30 vs 1,045,889）当场炸 —— 计划是对的，断言差一位。
        """
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "plan.json"
            bad.write_text(json.dumps({"geometry": {
                "seal_time_id_min": 380, "embargo_real_time_ids": 30,
                "decision_train_time_id_max": 350}}))          # 差一
            with self.assertRaises(SystemExit):
                B.decision_cutoff(bad)

    def test_missing_plan_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            B.decision_cutoff(Path("/nonexistent/sealed_period_plan.json"))


class PlanMembersTest(unittest.TestCase):
    def test_backfill_is_renamed_so_it_cannot_shadow_local_partitions(self) -> None:
        """⭐ 回补包与本地前几个分区**同名内容全异** —— 不改名就是静默覆盖训练集。"""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            members = B.plan_members(fixture.data_root, fixture.backfill, ROLE_FULL, 349)
            self.assertEqual([m["name"] for m in members],
                             [f"train_partition_{i:03d}.parquet" for i in range(4)])
            # 后两个必须指向回补包，且没有任何一个 name 被复用
            self.assertEqual([str(m["source"].parent.parent.name) for m in members],
                             ["data", "data", "backfill", "backfill"])
            self.assertEqual(len({m["name"] for m in members}), len(members))

    def test_decision_role_truncates_the_partition_containing_the_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            members = B.plan_members(fixture.data_root, fixture.backfill, ROLE_DECISION, 349)
            self.assertEqual(len(members), 4)
            self.assertEqual([m["truncate_at"] for m in members], [None, None, None, 349])

    def test_partitions_entirely_inside_the_seal_are_dropped(self) -> None:
        """cutoff 落在回补第一个分区里 ⟹ 第二个整块在密封段内，一行都不该收。"""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            members = B.plan_members(fixture.data_root, fixture.backfill, ROLE_DECISION, 249)
            self.assertEqual(len(members), 3)
            self.assertEqual(members[-1]["truncate_at"], 249)


class BuildTest(unittest.TestCase):
    def _build(self, base: Path, role: str, cutoff: int = 349) -> dict:
        fixture = Fixture(base)
        members = B.plan_members(fixture.data_root, fixture.backfill, role, cutoff)
        return B.build(base / "roots" / role, fixture.data_root, members, role, cutoff,
                       execute=True), fixture

    def test_decision_root_stops_exactly_at_the_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, _ = self._build(Path(tmp), ROLE_DECISION)
            self.assertEqual(identity["train_time_id_max"], 349)
            self.assertEqual(identity["role"], ROLE_DECISION)
            self.assertEqual(identity["truncated_member"], "train_partition_003.parquet")

    def test_full_root_keeps_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, fixture = self._build(Path(tmp), ROLE_FULL)
            self.assertEqual(identity["train_time_id_max"], 399)
            self.assertIsNone(identity["truncated_member"])
            self.assertEqual(identity["train_rows"], fixture.total_rows)

    def test_row_id_stays_globally_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._build(Path(tmp), ROLE_DECISION)
            root = Path(tmp) / "roots" / ROLE_DECISION
            names = json.loads((root / "manifest.json").read_text())["files"]["train"]
            previous = None
            for name in names:
                ids = pq.read_table(root / name, columns=["row_id"]).column(0).to_numpy()
                self.assertTrue(np.all(np.diff(ids) == 1))
                if previous is not None:
                    self.assertEqual(ids[0], previous + 1)
                previous = ids[-1]

    def test_truncation_does_not_alter_a_single_bit(self) -> None:
        """⭐ 截断分区是唯一被**重写**的文件 —— 它必须是源文件的逐 bit 前缀。

        含一列 10% NaN 的特征：只比数值会漏掉 NaN 位模式被改写的情况，
        而 LightGBM 差一个 ULP 就可能翻叶子（CLAUDE.md §8.3）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, fixture = self._build(Path(tmp), ROLE_DECISION)
            target = Path(tmp) / "roots" / ROLE_DECISION / "train" / "train_partition_003.parquet"
            source = fixture.backfill / "train" / "train_partition_001.parquet"
            n = pq.ParquetFile(target).metadata.num_rows
            self.assertEqual(n, 50 * N_ASSETS)                     # time 300–349
            self.assertTrue(pq.ParquetFile(target).schema_arrow.equals(
                pq.ParquetFile(source).schema_arrow))
            for column in COLUMNS:
                got = pq.read_table(target, columns=[column]).column(0).to_numpy(
                    zero_copy_only=False)
                want = pq.read_table(source, columns=[column]).column(0).to_numpy(
                    zero_copy_only=False)[:n]
                self.assertEqual(got.tobytes(), want.tobytes(), f"{column} 逐 bit 不同")

    def test_originals_are_symlinked_not_copied(self) -> None:
        """4.1 GB 不复制；同时也保证「派生根不会与 data/ 悄悄分家」。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._build(Path(tmp), ROLE_DECISION)
            train = Path(tmp) / "roots" / ROLE_DECISION / "train"
            links = [p for p in sorted(train.glob("*.parquet")) if p.is_symlink()]
            self.assertEqual(len(links), 3)                        # 只有截断那个是实体文件
            self.assertFalse((train / "train_partition_003.parquet").is_symlink())


class RootIdentityGateTest(unittest.TestCase):
    """⭐ 三种 fail-closed —— 门禁不响，D1 就是在没有护栏的情况下跑几个小时。"""

    def _root(self, base: Path, **overrides) -> Path:
        root = base / "root"
        root.mkdir(parents=True)
        identity = {"role": ROLE_DECISION, "train_time_id_max": 1_045_889,
                    "train_rows": 1, "train_partitions": 1} | overrides
        (root / "root_identity.json").write_text(json.dumps(identity))
        return root

    def test_missing_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bare = Path(tmp) / "bare"
            bare.mkdir()
            with self.assertRaises(SystemExit):
                load_root_identity(bare, ROLE_DECISION)

    def test_role_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp), role=ROLE_FULL)
            with self.assertRaises(SystemExit):
                load_root_identity(root, ROLE_DECISION)

    def test_training_into_the_sealed_period_is_rejected(self) -> None:
        """把边界改到密封段里 —— 这正是 RUNBOOK 说「不会报错」的那种失效。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp), train_time_id_max=1_105_919)
            with self.assertRaises(SystemExit):
                load_root_identity(root, ROLE_DECISION)

    def test_correct_decision_root_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            self.assertEqual(load_root_identity(root, ROLE_DECISION)["train_time_id_max"],
                             1_045_889)

    def test_full_role_still_needs_a_declared_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp), role=ROLE_FULL, train_time_id_max=None)
            with self.assertRaises(SystemExit):
                load_root_identity(root, ROLE_FULL)

    def test_real_decision_root_matches_the_sealed_plan(self) -> None:
        """盘上那个真的决策期根 —— 与密封期计划对得上才算数。"""
        root = ROOT / "outputs" / "data_roots" / ROLE_DECISION
        if not (root / "root_identity.json").is_file():
            self.skipTest("决策期数据根还没建")
        identity = load_root_identity(root, ROLE_DECISION)
        plan = json.loads((ROOT / "outputs" / "experiments"
                           / "sealed_period_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(identity["train_time_id_max"],
                         plan["geometry"]["decision_train_time_id_max"])


if __name__ == "__main__":
    unittest.main()
