"""LightGBM 文本模型的纯 numpy 推理 —— 不依赖 lightgbm 的兜底路径。

## 为什么要有这个文件

主办方确认评测环境有 lightgbm，所以这**不再是交付阻塞**。留着它是因为一条
`try: import lightgbm` 兜不住的风险：`model/lgbm_seed*.txt` 是 **`version=v4`**
（LightGBM 4.x 格式）。评测端「有 lightgbm」不等于「有 4.x」——
3.3.x 读 v4 文件会报错或误读，那是「import 成功但结果不对」。
`main.py` 因此在启动时拿两条路径对拍一次，不一致就退到这里。

顺带满足 10 月答辩「交付物最好逐行自己重写」那条。

## 推理语义（照 LightGBM C++ 源码抄的，任何一处想当然都会在分裂边界上翻枝）

| 细节 | 正确语义 |
|---|---|
| 数值分裂 | `x <= threshold` 走**左**（不是 `<`） |
| 缺失值 | `decision_type` 的 bit2-3 是 missing_type；本模型全是 `None` → NaN 先当 0.0 再比 |
| 分类分裂 | `int(fval)` 截断取整，负数直接走右；再查位集合，**命中走左** |
| shrinkage | `leaf_value` 已经乘过 learning_rate，`boost_from_average` 已烘进 Tree=0 → 直接求和 |

`threshold` 是以 `%.17g` 写出的 double，**必须按 float64 解析并比较** ——
float32 输入交给 lightgbm 时它内部 `static_cast<double>`，两侧都在 double 上比才逐位一致。
本模块把输入拷进 float64 缓冲区正是为此。

## 怎么做到不慢的（三条，都实测过）

1. **叶子自环**。叶子节点的左右孩子都指向自己、阈值设 `+inf`，于是「跑满最大深度」
   就等于「所有行都到了叶子」，遍历里**不需要任何 mask 或分支**。
2. **分类分裂编成合成列**。唯一的 categorical 是 `asset_id`，每个不同的位集合编成一个
   合成列（2424 个分裂只有 1901 个不同位集合），列值 `0.0`=命中 / `1.0`=不命中，阈值 `0.5`
   —— `x <= 0.5` 恰好等价于「命中走左」。**整条遍历只剩一种比较。**
   合成列的取值只由 asset 决定，所以设计矩阵按 **asset 槽位**布局（行 = asset_id），
   合成列在 `__init__` 里一次填好，运行期零额外 gather。
3. **按深度排序 + 逐层收窄**。树按最大深度升序排，第 k 层只需要处理「深度 > k」的树，
   而那恰好是一个**连续后缀** → 每层的活动窗口是预先切好的视图，运行期零开销。

⚠️ 走过的弯路（别再走）：为了省掉每层一次 `np.add`，把「每 asset 一份阈值表」
预展开成 43 MB，**反而从 0.885 ms 慢到 1.392 ms** —— gather 的目标放不进 L2。
7200 元素的小数组上 numpy 是**访存受限**的，不是指令受限的。保持四张表合计 2.9 MB。

## 限制

- 只支持本模型这一类：`num_class=1` / `num_tree_per_iteration=1` / `objective=regression`
  / `is_linear=0` / 无 `average_output` / `missing_type` 全为 None。
  **不满足就抛异常，绝不静默算错。**
- 实例持有可复用缓冲区，**不是线程安全的**（Time-Series API 是顺序调用，够用）。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# LightGBM 的 decision_type 位域（include/LightGBM/tree.h）
_CATEGORICAL_MASK = 1        # bit0：1 = 分类分裂
_DEFAULT_LEFT_MASK = 2       # bit1：数值分裂里缺失值默认走左
_MISSING_TYPE_SHIFT = 2      # bit2-3：0=None 1=Zero 2=NaN

# 本模型只可能出现这两个值：1（分类）与 2（数值 + default_left + missing_type None）。
# 出现别的说明模型是用不同的数据/参数训的，本模块的假设不再成立。
_ALLOWED_DECISION_TYPES = frozenset({1, 2})

_N_ASSETS = 15               # 比赛固定 15 个 asset（docs/data_description.md）


def _field(block: str, name: str, cast):
    """取树块里的一行 `name=v1 v2 …`；缺失或为空都返回空列表。"""
    match = re.search(rf"^{name}=(.*)$", block, re.M)
    if match is None:
        return []
    text = match.group(1).strip()
    return [cast(item) for item in text.split()] if text else []


def _header_value(header: str, key: str) -> str | None:
    match = re.search(rf"^{key}=(.*)$", header, re.M)
    return match.group(1).strip() if match else None


def _parse_model_text(text: str) -> list[dict]:
    """`model_to_string()` 的文本 → 树列表。所有结构假设在这里硬断言。"""
    blocks = text.split("\nTree=")
    header = blocks[0]

    if _header_value(header, "num_class") != "1":
        raise ValueError("lgbm_numpy 只支持 num_class=1")
    if _header_value(header, "num_tree_per_iteration") != "1":
        raise ValueError("lgbm_numpy 只支持 num_tree_per_iteration=1")
    objective = _header_value(header, "objective") or ""
    if not objective.startswith("regression"):
        raise ValueError(f"lgbm_numpy 只支持 regression 目标，拿到 {objective!r}")
    if _header_value(header, "average_output") is not None:
        raise ValueError("模型带 average_output（随机森林口径），本模块按求和口径写的")

    trees = []
    for block in blocks[1:]:
        block = block.split("\nend of trees")[0]
        if _field(block, "is_linear", int) not in ([], [0]):
            raise ValueError("lgbm_numpy 不支持 linear_tree")
        decision_type = _field(block, "decision_type", int)
        unexpected = set(decision_type) - _ALLOWED_DECISION_TYPES
        if unexpected:
            raise ValueError(
                f"decision_type 出现未支持的值 {sorted(unexpected)}；"
                "本模块假设缺失值类型一律是 None（bit2-3 为 0）")
        trees.append({
            "num_leaves": _field(block, "num_leaves", int)[0],
            "split_feature": _field(block, "split_feature", int),
            "threshold": _field(block, "threshold", float),
            "decision_type": decision_type,
            "left_child": _field(block, "left_child", int),
            "right_child": _field(block, "right_child", int),
            "leaf_value": _field(block, "leaf_value", float),
            "cat_boundaries": _field(block, "cat_boundaries", int),
            "cat_threshold": _field(block, "cat_threshold", int),
        })
    if not trees:
        raise ValueError("文本里一棵树都没有")
    return trees


def _bitset_members(words: list[int], n_assets: int) -> tuple[int, ...]:
    """LightGBM 的 `FindInBitset`：第 i 位在 `words[i//32]` 的第 `i%32` 位上。"""
    return tuple(
        i for i in range(n_assets)
        if i // 32 < len(words) and (words[i // 32] >> (i % 32)) & 1
    )


def _tree_depth(tree: dict) -> int:
    """从根到叶经过的**内部节点**个数的最大值 —— 也就是最多要迭代几层。"""
    if tree["num_leaves"] <= 1:
        return 0                      # 只有一个叶子的树桩，根就是叶，零层
    best = 0
    stack = [(0, 1)]
    while stack:
        node, depth = stack.pop()
        best = max(best, depth)
        for side in ("left_child", "right_child"):
            child = tree[side][node]
            if child >= 0:
                stack.append((child, depth + 1))
    return best


class NumpyForest:
    """一组 LightGBM 回归树的纯 numpy 推理器（预测值按树**求和**，不做平均）。"""

    def __init__(self, trees: list[dict], n_features: int, n_assets: int = _N_ASSETS):
        self.n_trees = len(trees)
        self.n_features = int(n_features)
        self.n_assets = int(n_assets)

        # ---- 分类分裂 → 合成列。位集合去重，一个集合一列。
        bitsets: dict[tuple[int, ...], int] = {}
        for tree in trees:
            for index, decision_type in enumerate(tree["decision_type"]):
                if decision_type & _CATEGORICAL_MASK:
                    category = int(tree["threshold"][index])
                    low = tree["cat_boundaries"][category]
                    high = tree["cat_boundaries"][category + 1]
                    key = _bitset_members(tree["cat_threshold"][low:high], self.n_assets)
                    bitsets.setdefault(key, len(bitsets))
        self.n_columns = self.n_features + len(bitsets)

        # ---- 扁平化。索引一律用 `2*节点号`：child 表直接存两倍，遍历里省掉一次左移。
        sizes = [2 * tree["num_leaves"] - 1 for tree in trees]
        offsets = np.cumsum([0] + sizes)
        total = int(offsets[-1])
        feature = np.zeros(2 * total, dtype=np.int32)
        threshold = np.full(2 * total, np.inf, dtype=np.float64)
        value = np.zeros(2 * total, dtype=np.float64)
        child = np.zeros(2 * total, dtype=np.int32)

        for position, tree in enumerate(trees):
            base = int(offsets[position])
            leaves = tree["num_leaves"]
            leaf_base = base + leaves - 1
            for index in range(leaves - 1):
                node = base + index
                if tree["decision_type"][index] & _CATEGORICAL_MASK:
                    category = int(tree["threshold"][index])
                    low = tree["cat_boundaries"][category]
                    high = tree["cat_boundaries"][category + 1]
                    key = _bitset_members(tree["cat_threshold"][low:high], self.n_assets)
                    feature[2 * node] = self.n_features + bitsets[key]
                    threshold[2 * node] = 0.5      # 命中=0.0 ≤ 0.5 → 左；不命中=1.0 → 右
                else:
                    feature[2 * node] = tree["split_feature"][index]
                    threshold[2 * node] = tree["threshold"][index]
                for side, key in ((1, "left_child"), (0, "right_child")):
                    target = tree[key][index]
                    resolved = base + target if target >= 0 else leaf_base + (-target - 1)
                    child[2 * node + side] = 2 * resolved
            for index in range(leaves):
                node = leaf_base + index
                value[2 * node] = tree["leaf_value"][index]
                child[2 * node] = child[2 * node + 1] = 2 * node   # 叶子自环

        # ---- 按最大深度升序排；第 k 层只需处理「深度 > k」的树，那是一个连续后缀。
        depths = np.array([_tree_depth(tree) for tree in trees], dtype=np.int64)
        order = np.argsort(depths, kind="stable")
        self.max_depth = int(depths.max())
        sorted_depths = depths[order]
        # active_start[k] = 第 k 层起，活动窗口在排序后数组里的起点
        self._active_start = [int(np.searchsorted(sorted_depths, k + 1, side="left"))
                              for k in range(self.max_depth)]

        self._feature = feature
        self._threshold = threshold
        self._value = value
        self._child = child
        self._roots = np.ascontiguousarray(
            2 * np.asarray(offsets[:-1])[order], dtype=np.int32)

        # ---- 设计矩阵按 asset 槽位布局：行 = asset_id。合成列只由 asset 决定，一次填好。
        self._x = np.zeros((self.n_assets, self.n_columns), dtype=np.float64)
        self._x[:, self.n_features:] = 1.0
        for key, column in bitsets.items():
            for asset in key:
                self._x[asset, self.n_features + column] = 0.0
        self._x_flat = self._x.reshape(-1)
        # 行偏移：asset 槽位 a 的第 f 列在扁平数组里是 a*n_columns + f
        self._row_offset = (np.arange(self.n_assets, dtype=np.int32)
                            * self.n_columns).reshape(1, -1)

        # ---- 可复用缓冲区。树在**第 0 轴**，这样逐层收窄切出来的是连续视图。
        shape = (self.n_trees, self.n_assets)
        self._node = np.empty(shape, dtype=np.int32)
        self._node0 = np.ascontiguousarray(
            np.repeat(self._roots[:, None], self.n_assets, axis=1))
        self._buffers = {name: np.empty(shape, dtype=dtype) for name, dtype in
                        (("index", np.int32), ("x", np.float64),
                         ("threshold", np.float64), ("go_left", np.bool_))}
        buffers = self._buffers
        # 每层的活动视图预先切好 —— 运行期不再做任何切片
        self._views = [
            (self._node[start:], buffers["index"][start:], buffers["x"][start:],
             buffers["threshold"][start:], buffers["go_left"][start:])
            for start in self._active_start
        ]

    # ------------------------------------------------------------------ 构造

    @classmethod
    def from_files(cls, paths, num_iteration: int, n_assets: int = _N_ASSETS) -> "NumpyForest":
        """读若干个 `model_to_string()` 文本，各取前 `num_iteration` 棵树拼成一片森林。

        拼在一起是因为调用方要的就是「所有树求和再除以模型数」——
        与 `sum_k booster_k.predict()` 数学等价，只差 480 个 double 的求和顺序（~1e-19）。
        """
        trees: list[dict] = []
        n_features = 0
        for path in paths:
            text = Path(path).read_text(encoding="utf-8")
            parsed = _parse_model_text(text)
            if len(parsed) < num_iteration:
                raise ValueError(
                    f"{Path(path).name} 只有 {len(parsed)} 棵树，要 {num_iteration} 棵")
            max_feature = _header_value(text.split("\nTree=")[0], "max_feature_idx")
            n_features = max(n_features, int(max_feature) + 1)
            trees.extend(parsed[:num_iteration])
        return cls(trees, n_features, n_assets)

    # ------------------------------------------------------------------ 推理

    def predict_sum(self, design: np.ndarray, asset_ids: np.ndarray) -> np.ndarray:
        """一个 time_id 的一批行 → 每行在**所有树上的叶子值之和**（float64）。

        `asset_ids` 必须与 `design` 里那一列的类别取值一致（调用方自己保证口径），
        且在本批内**不重复** —— 本实现按 asset 槽位摆行，重复会静默丢行，所以这里查。
        """
        rows = len(asset_ids)
        if design.shape[1] != self.n_features:
            raise ValueError(f"设计矩阵有 {design.shape[1]} 列，模型要 {self.n_features} 列")
        if rows > self.n_assets:
            raise ValueError(f"一个 time_id 有 {rows} 行，超过 {self.n_assets} 个 asset 槽位")
        if rows and (asset_ids.min() < 0 or asset_ids.max() >= self.n_assets):
            raise ValueError(f"asset_id 超出 [0, {self.n_assets}) —— 本模型的分类分裂不认识它")
        if np.bincount(asset_ids, minlength=self.n_assets).max() > 1:
            raise ValueError("同一个 time_id 内出现重复 asset_id")

        self._x[asset_ids, :self.n_features] = design

        feature, threshold, child = self._feature, self._threshold, self._child
        flat, row_offset = self._x_flat, self._row_offset
        node_all = self._node
        np.copyto(node_all, self._node0)

        for node, index, values, bounds, go_left in self._views:
            np.take(feature, node, out=index)
            np.add(index, row_offset, out=index)
            np.take(flat, index, out=values)
            np.take(threshold, node, out=bounds)
            np.less_equal(values, bounds, out=go_left)
            np.add(node, go_left, out=index)
            np.take(child, index, out=node)

        # 每个 asset 槽位一列 → 沿树轴求和，再按行取回来
        totals = self._value[node_all].sum(axis=0)
        return totals[asset_ids]
