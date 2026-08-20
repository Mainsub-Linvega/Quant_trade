# 全特征低采样纯化候选生成设计

> 日期：2026-08-20
> 分支：`exp/adaptive-feature-search`
> 状态：设计冻结，等待用户复核；尚未实施

## 1. 决策与目标

采用低采样、全特征对纯化扫描作为新的候选生成方法：

```text
323 个源特征
-> 早期 OOF 时间段的完整 time_id 确定性抽样
-> 全部 52,003 个二阶组合的 4 x 4 纯化粗筛
-> 固定最多 256 对的候选 manifest
-> 后期未参与筛选的 OOF 时间段
-> 现有完整 P0 null 与稳定性门禁
```

候选生成器只负责降低完整 P0 的 pair 数量，不负责接受交互。它不得修改生产模型、候选模型、融合参数
或提交 CSV。

冻结研究基线保持不变：

```text
market_lambda     = 0.7
blend_weight      = 1.17
prediction_scale  = 1.16
asset adapter     = disabled
history           = frozen
```

第一轮只执行 Ridge。Ridge 通过后再复用同一基础设施执行 XS，最后才考虑 Market。

## 2. 设计依据

已完成的 64 对覆盖性 P0 结果为：

| Task | Positive median | Positive drop-best | Above null | Accepted |
|---|---:|---:|---:|---:|
| Ridge | 2 / 64 | 1 / 64 | 0 / 64 | 0 / 64 |
| XS | 1 / 64 | 0 / 64 | 1 / 64 | 0 / 64 |
| Market | 7 / 64 | 1 / 64 | 0 / 64 | 0 / 64 |

这说明随机扩大完整 P0 预算的命中率较低，但不能推出剩余 51,939 对全部无效。新方法必须同时满足：

- 覆盖全部特征对，而不是 lexical prefix 或少量树路径；
- 直接近似最终纯化统计量，而不是用路径频率代替预测增量；
- 能发现一个特征处于特定范围时，另一个特征作用方向或强度改变；
- 候选选择和正式 P0 使用不同时间段，避免 winner's curse；
- 在当前本地时间预算内完成。

## 3. 明确排除

本阶段不做以下事情：

- 不删除或替换原有 Top200、History40；
- 不枚举三阶及以上组合；
- 不使用公榜分数、测试集或最终 outer validation 选择 pair；
- 不在粗筛阶段计算四个 seed 的完整经验 null；
- 不根据粗筛结果调整 bins、行数、门槛或候选预算；
- 不重新调整 `market_lambda`、`blend_weight` 或 scale；
- 不把候选生成器作为第四个模型；
- 不因没有候选通过而放宽正式 P0 门槛。

## 4. 时间隔离

输入继续使用严格 OOF 缓存和已构建的任务残差。按照 OOF `fold` 做一次固定时间切分：

```text
proposal folds = 0, 1, 2
gate folds     = 3, 4
```

proposal folds 可以生成、排序和截断候选。gate folds 在候选 manifest 写入并计算 SHA-256 之前完全不可用于
pair 排序、参数选择或运行时降采样决策。

当前任务输入 NPZ 只有 `features/residual/weight/time_id/feature_indices`。实施时必须从同一 OOF source
加入逐行 `fold`；候选生成 CLI 缺少 `fold` 时 fail closed，不能用行数比例猜测边界。Market 输入的
`fold` 取每个 time_id 内唯一的 OOF fold，并验证组内完全一致。

候选通过 gate 后，仍必须在后续 P1/P2/P3 中执行嵌套、配对 outer OOF。gate 通过不等于可以生成公榜
候选。

## 5. 确定性低采样

proposal folds 合并后按时间切成四个连续块。每块只选择完整 `time_id`，不拆散资产截面。

Ridge/XS 主规格：

```text
proposal blocks       = 4
row cap per block     = 40,000
sampling              = evenly spaced complete time_id groups
proposal bins         = 4 x 4
minimum cell weight   = 32.0
```

Market 若后续启用：

```text
proposal blocks       = 4
time cap per block    = 20,000
proposal bins         = 4 x 4
minimum cell weight   = 8.0
```

每块 time_id 的选择只由块边界、row cap 和固定等距算法决定。禁止随机行抽样。输入、选中 time_id、行索引
和算法版本全部写入 manifest。

## 6. 全 pair 粗筛统计量

对每个 split 使用 expanding-train / next-block：

```text
block 0       -> block 1
block 0..1    -> block 2
block 0..2    -> block 3
```

每个特征的四分位边界只用该 split 的训练行拟合一次。训练和验证特征随后转换为 `uint8` bin 矩阵；缺失值
使用独立 sentinel，并在 pair surface 中映射为零贡献。

对全部 `(j, k), j < k`：

1. 使用预计算 bin 构造最多 16 个 joint cells；
2. 在训练段计算加权 residual cell mean 和 support；
3. 对二维面做与正式 P0 相同的加权 functional-ANOVA 纯化；
4. 将低支持和验证未见 cell 映射到零；
5. 在下一时间块计算相对零残差预测的 normalized SSE gain；
6. 记录 coverage、dominant-cell gain share、finite 和 surface checksum。

粗筛不生成经验 null。它的结果只能用于排序，不能被解释为显著性或模型收益。

## 7. 候选排序与预算

每个 pair 汇总三个验证块：

```text
median_gain
mean_gain
drop_best_mean_gain
positive_blocks
minimum_coverage
maximum_dominant_cell_gain_share
```

先应用固定的粗筛资格：

```text
all finite                         = true
positive blocks                    >= 2 / 3
drop-best mean gain                > 0
minimum coverage                   >= 0.80
maximum dominant-cell gain share   <= 0.50
```

合格 pair 按以下固定顺序排序：

```text
drop_best_mean_gain descending
median_gain descending
mean_gain descending
pair lexical ascending
```

最多输出 256 对。该数字是完整 P0 的 pair 计算预算，不是原始特征数或最终派生特征数限制。

为避免单个父特征垄断，同时不压掉真正强信号，候选分成两部分：

- core 192：完全按上述排序，不限制父特征重复；
- diversity 64：从剩余合格 pair 中选择，每个父特征在 diversity 部分最多出现 4 次，并优先至少一个
  父特征位于当前基线 Top200 外的 pair。

“Top200 外”以冻结 `0.7/1.17` 候选中对应任务的最终直接特征索引清单为参照；清单路径和 SHA-256 写入
proposal manifest。它只影响 diversity 64 的顺序，不影响 core 192。

若合格 pair 少于预算，全部保留，不补入失败 pair。core 和 diversity 合并后按 pair lexical 排序并冻结
为 manifest。

## 8. 运行时门禁

正式全扫描前先用同一采样数据运行固定 lexical 前 1,024 对，只测运行时间和内存，不读取或汇报 pair
排名。根据 wall time 线性外推：

- 预计 Ridge 全扫描不超过 30 分钟：使用每块 40,000 行；
- 预计超过 30 分钟：固定回退到每块 20,000 行后重新 benchmark；
- 20,000 行仍预计超过 30 分钟：停止并报告资源阻塞，不继续缩小样本。

回退只能由运行时间触发，不能由 gain、候选数量或候选身份触发。峰值 RSS 预算为 4 GiB，超过即停止。

## 9. 独立完整 P0

候选 manifest 冻结后，正式 P0 只读取 gate folds 3、4，并沿用已冻结协议：

```text
Ridge/XS bins          = 8 x 8
Market bins            = 4 x 4
inner blocks           = 4
null seeds             = 2026, 2027, 2028, 2029
null quantile          = 0.95
minimum positive       = 2 blocks
minimum coverage       = 0.80
maximum tail share     = 0.50
```

正式 P0 的 null、稳定性和 drop-best 门禁不因粗筛结果而改变。只有完整 P0 通过的 pair 才允许进入对应模型
的配对 OOF 重训。

## 10. 实施边界

新增独立候选生成 CLI，不扩展生产训练入口：

```text
experiments/v3_low_sample_purified_proposal.py
tests/test_v3_low_sample_purified_proposal.py
outputs/experiments/v3_low_sample_purified_protocol_v1.json
outputs/experiments/<label>_pair_scores.npz
outputs/experiments/<label>_manifest.json
outputs/experiments/<label>.md
```

CLI 只能写实验 JSON、NPZ 和 Markdown。代码中不提供 candidate 或 CSV 参数。

## 11. 测试与验收

实现前必须先添加以下失败测试：

- 精确生成 `323 choose 2 = 52,003` 对；
- 完整 time_id 抽样，不拆散资产组；
- proposal 和 gate fold 不重叠；
- bin edges 只使用训练块；
- 快速预分箱结果与现有 `score_pair_split` 在同一数据上数值一致；
- 纯加法信号不会获得高排名；
- 零边际 XOR/范围切换信号能够进入前列；
- 相同输入重复运行得到相同 pair 顺序和 SHA-256；
- 未通过粗筛资格的 pair 不用于补足 256；
- core/diversity 配额不改变 core 排名；
- CLI 不生成 candidate 或 CSV；
- benchmark 回退只依赖运行时间，不读取 gain。

候选生成阶段完成的验收条件：

```text
full suite passes
synthetic XOR enters shortlist
additive controls fail eligibility
52,003 pairs scanned exactly once per split
peak RSS <= 4 GiB
estimated/full Ridge runtime <= 30 minutes
manifest deterministic
no production/candidate/submission changes
```

## 12. 决策规则

第一轮执行 Ridge：

- 没有粗筛合格 pair：记录有效负结果，停止，不运行 gate；
- 有粗筛候选但完整 P0 为 0：记录有效负结果，停止，不放宽门槛；
- 至少一个 pair 通过完整 P0：为 Ridge 制定单独的 P1 配对 outer OOF 计划；
- Ridge 基础设施和资源门禁通过后，才以相同冻结协议执行 XS；
- Market 保持最低优先级，不能因为 Ridge/XS 失败而自动启用。

本设计取代“直接扩大完整 P0 的随机 pair 数量”，但保留已完成的 64 对负结果作为基准证据。
