# 生产模型演进：Ridge → 截面森林 → History → 双森林

本文件按时间整理模型架构如何形成。分数来自 `experiments/ledger.csv`；同一个模型的低分点可能是
为求解 scale 抛物线而故意提交的第二点，不代表版本回退。

## 1. 2026-07-23：v1 Ridge 截面基线

首个可工作的生产模型使用：

- 200 个匿名特征；
- 稳健裁剪和缩放；
- 原始特征与逐 time_id 截面去均值特征拼接；
- 加权 Ridge；
- prediction scale 与 clip。

公榜起点为 **0.00119088**。这个模型建立了后续一直保留的两个分量视角：

```text
prediction = market mean + cross-sectional deviation
```

## 2. 2026-08-07/08：训练窗口、稳健统计与严格求解器

训练扩展到最近四个分区、修复 NaN 对统计量的污染，并用更严格的 LSQR 停止条件提高可复现性。
公榜依次达到约 0.00151886、0.00186805 和 **0.00187232**。

严格求解器的收益很小，但线程漂移下降约两个数量级。它说明生产价值不只来自分数：对后续
两点法、公榜差分和模型身份校验而言，稳定系数本身就是基础设施。

## 3. 2026-08-09：v3_hybrid 引入 LightGBM 截面分量

分析显示 target 的大部分方差属于市场共同分量，而 Ridge 对市场分量相对更稳；LightGBM 对
截面残差更有优势。初始 v3 采用：

```text
market = Ridge 的逐 time_id 无权均值
cross  = Ridge deviation 与 LGBM deviation 的混合
```

初始 blend50 两点法峰值约 0.00229575，比严格 Ridge 峰值高约 22.6%。随后利用 blend 的线性关系
验证 replace 更优，`blend_weight=1.0` 在 scale 1.16 的公榜为 **0.0024872338**。

重要修正：早期文档曾长期把 v3 描述为“五五混合”，但生产模型从 replace 上位后已不是该结构。

## 4. 2026-08-10：轮数 480 与 phase-balanced Ridge

嵌套模型让 80/160/320/480 轮可从同一组模型文件中比较。公榜方向是轮数增加到 480 更好，
与本地判断相反。480 轮后，phase-balanced 重训 Ridge 市场分量使公榜达到 **0.0026330806**。

这时架构仍是：

```text
phase-balanced Ridge market
+
480-round LGBM cross-sectional replacement
```

## 5. 2026-08-11：每资产历史特征进入截面森林

`AssetHistory` 为选中的 40 个特征构造四类严格历史块：

- previous；
- difference；
- rolling mean；
- rolling deviation；
- window=5。

共新增 160 列历史特征，在线状态与离线训练逐位对拍。该改动接入 Ridge 时为负，接入 LGBM
截面块时通过 5 折门禁。公榜从 0.0026330806 跳到 **0.0032523499**，是当时最大跃迁。

补交 scale 0.90 后得到该结构的两点真值：峰值约 0.00325527，scale 1.16 已达到峰值的 99.91%，
因此不再为微调 scale 消耗额度。

History80 本地看起来更好，公榜却与 history40 完全打平且更慢，最终保留 history40。

## 6. 2026-08-13：第二市场森林与 weighted XS

### 动机

Ridge 市场分量虽稳定，但仍可能遗漏非线性共同状态；同时截面模型的训练损失应与加权比赛指标
更一致。实验将两项拆开并组合：

1. 行级 LGBM 预测 `y`，最终只取逐 time_id 无权截面均值作为第二市场分量；
2. 截面残差森林使用 sample weight 训练。

市场模型设计矩阵为：

```text
raw200 | xs_dev200 | history160 | asset_id = 561 columns
```

截面模型设计矩阵为：

```text
xs_dev200 | history160 | asset_id = 361 columns
```

市场模型本身保持无权训练；实验发现给它加权反而削弱最终无权市场均值。

### 结果

组合候选 `mkt_we` 在同 scale 1.16 下达到 **0.0039673997**，比 history-only 基线高 21.99%。
这次结构由本地预注册臂选出，第一次公榜提交即兑现方向，未使用公榜反馈选型。

生产组合变为：

```text
market = 0.5 * ridge_market + 0.5 * lgbm_market
cross  = weighted_lgbm_cross
prediction = clip(1.16 * (market + cross), ±0.5)
```

## 7. 2026-08-13：两片森林容量分化与最终转正

两片森林此前沿用同一套 SPEC，但任务不同：市场森林预测行级 `y` 后只保留截面均值；截面森林
直接预测最终保留的残差。容量扫描结果：

| 候选 | 变化 | 公榜 |
|---|---|---:|
| `mkt_we` | 基线 | 0.0039673997 |
| `mkt_moderate` | 市场块中度收缩 | 0.0039867107 |
| `mkt_shrunk` | 市场块进一步收缩 | **0.0039977510** |
| `xs_shrunk` | 截面块收缩 | 0.0035771492 |
| `r960` | 两片统一增至 960 轮 | 0.0037609312 |

结论：市场模型需要更少容量，截面模型需要更多容量；480 轮对当前统一轮数是内部极值。
`mkt_shrunk` 于 2026-08-13 转正到 `strategies/v3_hybrid/model/`。

## 8. 当前模型身份

当前生产配置应从 `strategies/v3_hybrid/model/hybrid_meta.json` 读取。关键身份包括：

- `blend_weight=1.0`；
- `num_iteration=480`；
- 3 个截面森林和 3 个市场森林；
- `market_lambda=0.5`；
- weighted cross-section；
- history40/window5；
- scale 1.16；
- phase-balanced Ridge。

任何一个字段或模型文件发生变化，都应被视为新模型，而不是“同一模型的小配置差异”。

## 9. 未使用与关闭的模型线

- v2 LightGBM baseline：保留为历史和参考实现，不是当前生产线。
- v4 target-only MLP：推理导出基础设施可复用，但模型太弱，关闭搜索。
- Responder 多任务 NN：前置残差门禁失败，未进入训练。
- history80：公榜打平、成本更高，关闭。
- 2 种子：速度收益仅约 5.45%，保留为明确超时情形下的退路，不是默认生产。
