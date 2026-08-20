# Ridge 全特征低采样纯化扫描与独立 P0 结果

日期：2026-08-20

## 结论

Ridge 交互路线在冻结门槛下为 `failed_p0`。不训练交互 candidate，不生成 CSV，不放宽门槛。

## Proposal 粗筛

- folds：0、1、2；gate folds 未参与排序。
- 全部 323 个源特征精确枚举 52,003 对。
- 473 对通过固定廉价资格筛选。
- 冻结 192 core + 64 diversity，共 256 对。
- pair manifest SHA-256：`7863707650465c6a29b7fe615d63684c0ee7eb7b30df74712a58618cf8b95511`。
- 40,000 行/块 benchmark：11.79 秒；全量外推 598.86 秒；峰值 RSS 2.68 GB。

## 独立 P0

- 只使用 folds 3、4，共 583,881 行；proposal folds 不在 gate 输入中。
- 8 x 8 bins，4 个 null seeds，256 个冻结 pair，3,072 个 null 样本。
- 95% null 阈值：0.0033289865。
- 最终通过：0 / 256。
- 完整运行：228.78 秒；峰值 RSS 1.04 GB。

| 检查项 | 通过数 |
|---|---:|
| 至少两个正 gain 块 | 5 / 256 |
| drop-best mean gain > 0 | 1 / 256 |
| coverage >= 0.80 | 256 / 256 |
| tail share <= 0.50 | 253 / 256 |
| median gain > null | 0 / 256 |

最接近的 pair 是 `(240, 268)`：median gain 0.0001445801，drop-best 0.0000213614，
但 median gain 仅为 null 阈值的 4.34%。负结果不是轻微门槛偏差，不能通过小幅放宽合理挽救。

## 边界

- `market_lambda=0.7`、`blend_weight=1.17`、`prediction_scale=1.16` 未改变。
- 没有修改 production、candidate 或 submission。
- 没有使用 gate folds 选择 pair。
- Ridge 未通过完整 P0，因此不进入 P1 配对 outer OOF 训练。
- 下一项若执行，应是独立预注册的 XS proposal，而不是继续调 Ridge 门槛。
