# XS LightGBM 残差全特征纯化扫描与独立 P0 结果

日期：2026-08-20

## 结论

已放弃 Ridge 二阶交互寻找。XS 残差的相同路线结果为 `failed_p0`，也不进入 pair 特征训练。
不生成 candidate，不生成 CSV，不放宽门槛。

## XS 残差定义

```text
target_cross_unweighted - e_lgbm
```

它检验的是现有截面 LightGBM 预测后仍未解释的非线性二阶结构。

## Proposal 粗筛

- folds：0、1、2；gate folds 未参与排序。
- 323 个源特征精确枚举 52,003 对，全部有限。
- 276 对通过固定廉价资格筛选。
- 冻结 192 core + 64 diversity，共 256 对。
- pair manifest SHA-256：`8edba49d1fa68a77644c06f6b4102ff87a36eb3ffdfbfe25fbf4dc75c9ee0c64`。
- 40,000 行/块 benchmark：10.61 秒；全量外推 538.62 秒；峰值 RSS 2.68 GB。

## 独立 P0

- 只使用 folds 3、4，共 583,881 行；proposal folds 不在 gate 输入中。
- 8 x 8 bins，4 个 null seeds，256 个冻结 pair，3,072 个 null 样本。
- 95% null 阈值：-0.0000345484。
- 最终通过：0 / 256。
- 完整运行：226.65 秒；峰值 RSS 1.06 GB。

| 检查项 | 通过数 |
|---|---:|
| 至少两个正 gain 块 | 1 / 256 |
| drop-best mean gain > 0 | 0 / 256 |
| coverage >= 0.80 | 256 / 256 |
| tail share <= 0.50 | 255 / 256 |
| median gain > null | 5 / 256 |

最强 pair `(29, 126)` 的三个时间块 gain 为：

```text
-0.0008075218
+0.0001211761
+0.0000429150
```

它在两个块为正且超过 null，但第一块明显反向，drop-best mean 为 `-0.0003823034`。
失败原因是跨时间稳定性，而不是覆盖率、稀疏 cell 或资源问题。

## 决策边界

- Ridge pair 路线封存，不继续寻找或调门槛。
- XS pair 路线不进入配对 outer OOF 训练。
- 冻结的 `market_lambda=0.7`、`blend_weight=1.17`、`prediction_scale=1.16` 不变。
- 下一步应评估不同的信号族，而不是继续扩大静态二阶 pair 搜索。
