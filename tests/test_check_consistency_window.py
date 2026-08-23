"""一致性门禁的**窗口宽度**本身也是门禁的一部分 —— 钉住它，别让它悄悄退回去。

## 为什么需要这一层

`scripts/check_consistency.py` 比的是「训练侧整块矩阵」与「推理侧逐 time_id 喂」两条路径。
但当前生产结构里有**三处跨 predict 的派生状态**，它们只有在窗口足够宽时才会被走到：

| 派生状态 | 需要多少个真实 time_id 才被走到 |
|---|---|
| `AssetLongWindow`（长窗 512，08-21 转正，公榜 +1.662%） | >512 才填满，>1024 才回绕 |
| `PredictionTrail`（slow/fast 2000 真实步，08-18 转正，公榜 +2.93%） | >2000 左端才开始移动 |

2026-08-23 之前默认是 `--n-time-ids 50`：每个 asset 只有 50 个观测 ⟹
长窗缓冲只填到 9.8%、**从未回绕**，slow/fast 窗只填到 2.5%、**左端从未移动**。
⟹ 那道门禁证明的是「一个还没热起来的模型两侧一致」，而榜上跑的是热的那个。
这与 CLAUDE.md §8.5「性能/正确性判断必须端到端」同一个毛病：**测了，但没测到点上**。

改成 2100 之后三条路径全部走到，实测代价只有 2.7s → 5.9s（lightgbm 后端）。

⚠️ 本文件**不跑**那个脚本（它要读 parquet、要加载六片森林，属于集成检查，
在 D4 门禁全链里跑）。这里只钉住「默认值必须宽到足以走到这些状态」这一条不变量 ——
它是纯常量断言，零成本，而且正是最容易被「顺手调小点让测试快些」偷走的东西。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_consistency  # noqa: E402

PRODUCTION_META = ROOT / "strategies" / "v3_hybrid" / "model" / "hybrid_meta.json"


def _defaults():
    """从 argparse 定义里取默认值，不手抄常量。"""
    original = sys.argv
    try:
        sys.argv = ["check_consistency.py"]
        return check_consistency.parse_args()
    finally:
        sys.argv = original


class ConsistencyWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        if not PRODUCTION_META.is_file():
            self.skipTest("生产 meta 不在盘上")
        self.meta = json.loads(PRODUCTION_META.read_text(encoding="utf-8"))
        self.n = int(_defaults().n_time_ids)

    def test_default_window_fills_and_wraps_the_long_window_ring_buffer(self) -> None:
        long_window = self.meta.get("long_window")
        if not long_window:
            self.skipTest("当前生产没有长窗块")
        self.assertGreater(
            self.n, 2 * int(long_window),
            f"默认 --n-time-ids={self.n} 不足以让长窗 {long_window} 的环形缓冲**回绕**。"
            f"不回绕就测不到 AssetLongWindow 的持久累积和相减 —— 而 hybrid_meta 的 "
            f"long_window_note 明确写着「不得用分块重起的 cumsum」，那正是回绕处才暴露的 bug。")

    def test_default_window_moves_the_slow_fast_left_edge(self) -> None:
        window = self.meta.get("slow_fast_window")
        if not window:
            self.skipTest("当前生产没有 slow/fast")
        self.assertGreater(
            self.n, int(window),
            f"默认 --n-time-ids={self.n} 不足以让 slow/fast 的 {window} 真实步窗填满并"
            f"开始丢弃左端。窗没满时 trailing mean 退化成「从头到now的均值」，"
            f"与生产期的滚动均值不是同一个函数。")

    def test_atol_still_has_headroom_at_the_default_window(self) -> None:
        """max|Δ| 随窗口单调增长（实测 50→4.0e-09 … 3000→1.6e-08）⟹ 放宽窗口会吃掉余量。

        这条不是断言实测值（那要跑集成），而是钉住 atol 没被顺手调松。
        """
        self.assertEqual(_defaults().atol, 1e-6)


if __name__ == "__main__":
    unittest.main()
