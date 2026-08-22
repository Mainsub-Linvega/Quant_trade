# responder 族群表（`responder_family_grid`）

> 由 parquet 的 row-group 统计信息派生，**未加载任何数据**。
> 「像什么」一列是**解读**，不是主办方公布的语义，不得当事实引用（CLAUDE.md §3）。

来源：`/home/mainsub/Documents/Quant_trade/data/train/train_partition_000.parquet`（47 个 responder，target 取值域 [-2.2349, +2.2335]）

## 维度族

| 族 | 成员 | 维度类别 | 取值域 | 缺失数梯子 | 像什么 |
|---|---|---|---|---|---|
| a | `responder_00`–`responder_06`（7） | `unit_interval` | `[-0.0000, +1.0000]` | 279, 9, 1, 0, 422, 934, 2397 | 上界饱和到 1 ⟹ 概率 / CDF 类 |
| b | `responder_07`–`responder_13`（7） | `nonpositive` | `[-0.2449, +0.0000]` | 0, 0, 0, 0, 422, 934, 2397 | 非正 ⟹ 下行 / 回撤类 |
| c | `responder_14`–`responder_20`（7） | `nonnegative` | `[-0.0000, +0.1340]` | 0, 0, 0, 0, 422, 934, 2397 | 非负，量级 ~1e-01（上行 / 路径 / 摩擦之一，**本表判不出是哪个**） |
| d | `responder_21`–`responder_27`（7） | `nonnegative` | `[-0.0000, +0.8710]` | 0, 0, 0, 0, 422, 934, 2397 | 非负，量级 ~9e-01（上行 / 路径 / 摩擦之一，**本表判不出是哪个**） |
| e | `responder_28`–`responder_30`（3） | `bidirectional` | `[-4.0871, +4.2893]` | 526, 1035, 2490 | 双向、与 target 同量级（±2.23）⟹ 收益类 |
| f | `responder_31`–`responder_37`（7） | `nonnegative` | `[+0.0000, +0.0102]` | 0, 0, 0, 0, 4, 9, 24 | 非负，量级 ~1e-02（上行 / 路径 / 摩擦之一，**本表判不出是哪个**） |
| g | `responder_38`–`responder_42`（5） | `nonnegative` | `[+0.0000, +0.0102]` | 0, 0, 4, 9, 24 | 非负，量级 ~1e-02（上行 / 路径 / 摩擦之一，**本表判不出是哪个**） |
| h | `responder_43`–`responder_46`（4） | `nonnegative` | `[+0.0000, +0.1265]` | 0, 526, 1035, 2490 | 非负，量级 ~1e-01（上行 / 路径 / 摩擦之一，**本表判不出是哪个**） |

合计 7 + 7 + 7 + 7 + 3 + 7 + 5 + 4 = **47** ✅

## 共用窗口梯子 —— 「同缺失数 = 同窗口，不同维度」的直接证据

| 缺失数梯子（截断档） | 共用它的族 |
|---|---|
| `422,934,2397` | a, b, c, d |
| `526,1035,2490` | e, h |
| `4,9,24` | f, g |

responder 构造自「未来不可见区间」（`docs/data_description.md:169`）⟹ 窗口越长，
分区末端越多行算不出来。缺失数因此是**窗口的精确指纹**，且跨维度族逐位对齐。

⭐ 这与 `responder_analysis.py` 的 24 族聚类是**正交的两把刀**：那把按 `1 − |corr|` 聚，
切出来的是**窗口组**（cluster 13 = {25,26,27,44,45,46} 横跨两个量纲不同的维度但窗口相同）；
本表切的是**维度组**。

## Stage B 的启发式缺口

来源：`/home/mainsub/Documents/Quant_trade/outputs/experiments/responder_predictability_reaudit_phasebal_prodwindow.json`

```text
24 族   通过 8   只被 multi_member_family 挡住 16   因证据不过 0
```

被挡住的 16 个单成员族里，08-18 的 `horizon_auxiliary_cache_probe` 只补测了 `responder_00`, `responder_02` ⟹ **剩 14 个从未进过 Stage C**：

```text
responder_01, responder_03, responder_04, responder_05, responder_06, responder_10, responder_13, responder_17, responder_20, responder_28, responder_29, responder_30, responder_42, responder_43
```

⚠️ **这个缺口不构成「有收益」的理由。** 两条必须一起读：

1. 其中最显眼的 `responder_03`/`responder_28`/`responder_29`/`responder_04` 在 A0 阶段 1 **已被逐列量过**（当训练目标），且**同期相关最高的 `responder_03` 是全场最差**（−15.47%、0/7 阶梯）—— 见 `responder_targets_stage1.md:14-22`、`CLAUDE.md:119`。
2. 「把 responder 的预测值叠进 blend」属于 `responder_reaudit_20260814.md:93-100` 母条件**明令排除**的「线性叠加 / 对预测值做二层校准」机制族 ⟹ 补测它的价值是**结案**，不是找收益。
