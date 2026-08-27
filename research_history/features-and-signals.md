# 特征与信号探索史

本文件关注“新增了什么信息”以及为什么一些看似强相关的方向没有转化为可部署收益。

## 1. 市场共同分量与截面分量

早期数据诊断发现逐 time_id 市场共同分量占 target 方差的大头。九分区复验给出 0.626–0.732，
均值约 68.4%。长期分区均值方向不稳定，但逐 time_id 市场分量存在弱样本外可预测性。

这两个事实不能混为一谈：

- “长期漂移符号随机”说明不能押分区级方向；
- “逐时刻共同状态弱可预测”说明横截面共享信号仍可积累分数。

模型最终因此长期采用 market/cross 分解，而不是对所有行使用一个不分结构的单模型解释。

## 2. 原始截面特征与 Ridge

Ridge 基线使用原始特征和逐 time_id deviation。对特征数、截面缩放、分开正则、截距和采样相位
做过多轮消融。主要结论：

- 更多匿名特征不稳定地改善本地结果，不能简单认为 323 全量优于 200；
- 截面标准差归一化没有稳定增益；
- 对 market/cross 两块分开加罚未产生可采用收益；
- 严格求解器改善可复现性，但不是主要信号来源。

这些实验促使项目从“继续调线性旋钮”转向寻找新结构。

## 3. LightGBM 截面残差

LightGBM 对截面残差的非线性拟合明显优于 Ridge，因此 v3 首先用树替换 cross component。
关键工程约束是推理端没有 weight，所有截面均值和投影必须无权；训练端可对残差损失使用 weight，
但输出仍需投影成无权零均值，避免污染市场分量。

`ê_raw -= mean(ê_raw)` 不是可选后处理，而是结构约束：任何非零截面均值都会把树的伪市场量混入
已经由专门模型负责的 market component。

## 4. 每资产历史：第一个成功的时间信号轴

在 history 之前，主要特征来自同期横截面。A1 探索把每资产过去状态引入模型。

### 失败路径：接入 Ridge

相同 history 特征接入 Ridge 后，按 peak 重算为负。早期固定 scale 下的正结果是口径假象。
这说明历史信号需要非线性模型或与截面残差的交互，不能直接当作通用线性增量。

### 成功路径：接入 LGBM cross

选择 40 个已入模特征，构造 window5 的 previous、difference、rolling mean 和 rolling deviation。
在严格在线/离线一致性约束下，本地 5 折通过，公榜提升到 0.0032523499。

### Lag 的含义与当前实现

`lag` 是同一资产在当前观测之前的历史值，不等同于“采样矩阵中的上一行”。由于资产面板可能缺行，
且训练使用 phase-balanced 采样，历史状态必须在**每一条原始行**上推进，采样只决定该行是否保留。
直接在采样后的矩阵上 `groupby(asset_id).shift(1)` 会跳过未采样观测，与线上时序状态不一致。

当前生产 history40/window5 从 40 个已选 LGBM 特征构造四个块：

```text
previous          = 最近一次该资产观测
difference        = current - previous
rolling_mean      = 最近最多 5 次历史观测均值，不含 current
rolling_deviation = current - rolling_mean
```

线上一次只处理一个完整 time_id，使用 `AssetHistory.transform_online()`；离线使用等价的整块路径。
冷启动没有历史时按 0 填充。两条路径必须保持相同窗口、浮点累加顺序和状态更新时间，否则树模型会
把 ulp 级输入差异放大成叶节点跳转。

### 宽度与窗口

本地联合梯子曾偏好更多 history 列，但 history80 公榜与 history40 打平且更慢。该反例成为
“本地结构内宽度也可能不迁移”的证据，最终保留 history40。V4-T 对更长窗口、趋势和状态的已有扩张
也没有提供独立增量；未来若重开 lag，只允许测试预注册的因果变化族，而不是重新做窗口网格。

证据：`outputs/experiments/history_peak_lgbm_scoped.md`、`history_peak*.md`、ledger。

## 5. Responder：同期关系强，但可部署增量失败

Responder 探索分成多个阶段，避免从同期相关性直接跳到昂贵多任务模型。

### A：结构与同期关系

47 个 responder 聚为 24 个族群，PCA 约 10 个成分解释 90%。部分 responder 与 target 同期相关
很高，但这些值在推理时不可见，只能作为训练辅助目标候选。

### B：样本外可预测性

使用严格 rolling folds、训练折内选代表和共享 Gram 多目标 Ridge，24 族中 8 族可由 feature
样本外预测，且多折同号。因此“responder 完全不可预测”被否定。

### C：Target 残差增量

把预测到的 responder 信息加入 target Ridge 后，五折全部为负，折均约 −20.64%。
结论是：**可预测的 responder 部分不是当前 target 模型缺少的稳定残差信息**。

因此停止多任务 NN。只有扩展数据显著改变 B/C 关系时，才允许从门禁重新开始，不能直接训练大模型。

## 6. 多尺度时间特征

V4-T 试图扩展单一 window5 history，加入更长窗口、趋势和状态。三臂都未通过确认门槛。
这不是否定所有时间信息——history40 已经成功——而是说明当前多尺度扩张没有提供独立增量。

关闭条件：不继续搜索窗口组合，不因单折正向重开。

## 7. 压缩市场 Regime

V4-R 将冻结 history 的截面状态压缩为少量市场统计，并加入当前 regime、lag 和 difference。
结果约 +1.34%、4/5 折，ΔA 正且 ΔB 基本不涨，机制干净；但未达到预注册 +3% 门槛且存在负折。

决策：当前数据不采用，不支付完整训练与公榜成本。若 8/23 后训练数据确实扩展，只允许按原来的
20 维压缩规格复验一次，不扩张搜索空间。

## 8. Target-only MLP

旧实验 `target_mlp_screen`：为避免 responder 泄漏，MLP 只使用可部署输入，分成 market head 和
cross head，并实现 sklearn 权重到纯 NumPy 推理的导出对拍。它只在原始训练集滚动 OOF 口径下评估，
与 cached v3 OOF 做 blend；模型与 v3 预测相关较低，但单模只有 v3 peak 的约 24.3%，50/50 集成
下降约 54.49%。结论是：**旧训练分布上的 NN 没有稳定信号**，低相关来自模型太弱而不是独立 alpha。

8/23 公榜回填标签后新增 `backfill_nn_train.py`，这不是延续旧 `target_mlp_screen` 的结论，也不是调旧
NN 参数；它是**训练实验重新打开**：训练集改为原始 train + 公榜回填非保留段，评分只看公榜回填尾段
（默认最后 60,000 个真实 `time_id`，本次 cutoff `time_id >= 1045920`）。因此它的价值在于检验
“新增公榜期标签进入训练后，NN 是否重新出现可用信号”。

当前结果：target-only 在回填 holdout 上有可读数值；responder 辅助不稳定，`shortlist_s20` −1.13%、
`ladder_s20` +1.12%、加密到 `ladder_s10` 又变成 −1.04%。所以当前只保留 target-only NN 训练主线，
不把 responder 辅助视为稳定增益。

下一步 target-only 复验已做三 seed（s10 / 100 特征 / 8 iter）：seed2026 `0.00313382`、seed2027
`0.00210080`、seed2028 `0.00015554`，均值 `0.00179672`、std `0.00151224`、CV `0.84`。
结论：**单 seed target-only NN 不稳定，当前不能导 CSV**；后续若继续，应先做 seed averaging 或训练稳定化，
而不是扩大容量或恢复 responder 辅助。

稳定化第一步已执行：同一 s10 数据、固定预处理/选列，一次读取后训练 5 个 target-only seed
（2026..2030），对 holdout 预测逐行平均。单 seed peak 为 `0.00313382 / 0.00210080 / 0.00015554 /
0.00274635 / 0.00095470`，均值 `0.00181824`；5-seed ensemble peak **`0.00357272`**，高于
单 seed 均值约 `+96.49%`，也高于最好单 seed 约 `+14.00%`。结论更新为：**seed averaging 是当前
NN 稳定化主线**；进入 CSV 前还需 s5 或 10-seed 复验。

10-seed 复验已完成：同一 s10 holdout、seeds 2026..2035，单 seed 均值 `0.00145242`、std `0.00101934`，
10-seed ensemble peak **`0.00394484`**、unit scale 分数 `0.00392462`。这确认 5-seed 不是偶然；
下一步转为 s5 加密采样检查，仍不直接导 CSV。

s5 加密采样复验完成：5-seed ensemble（2026..2030）在 `3,117,772` train / `171,258` holdout 抽样行上
达到 peak **`0.00455755`**，unit scale 分数 `0.00454666`，optimal scale `1.0514`；单 seed 均值
`0.00216202`、std `0.00094707`。这比 s10 / 10-seed ensemble 的 `0.00394484` 更高，说明
seed averaging + 更密采样方向成立。下一阶段才是候选工程化：保存每 seed MLP 权重与预处理、构建 CPU 推理/CSV
路径，并在公榜回填 holdout 上复验落盘预测一致性。

冷启动工程化已完成：target-only 每 seed 的 market / cross `.npz` 现在会同时保存 selected 特征索引和
`robust_transform_fit` 的 `lower/upper/center/scale`，新增的冷启动入口可直接从原始特征、`time_id` 和
`asset_id` 重建 holdout 设计矩阵并加载磁盘模型。最新 `backfill_nn_target_only_s5_5seed_coldstart`
实验在 holdout 上的 `cold_start_replay_max_abs` 也是 `1.029e-07`，和训练态 replay 完全对齐；这意味着
下一步可以把同一条冷启动路径接到 CSV/提交件生成，而不是再依赖训练进程里的内存对象。

## 9. 第二市场森林

行级 LGBM 使用 raw、xs deviation、history 和 asset id 预测 `y`，最终只取同一 time_id 的无权均值。
它与 Ridge market 以 λ=0.5 组合。

关键消融：

- 市场模型带权训练更差；最终聚合是无权均值，行级 sample weight 与最终目标错位；
- 只用显式市场统计或缩减列集合的多个变体未通过；
- 尝试捡回被截面均值投影丢弃的偏差部分没有收益；
- 市场模型容量收缩小幅有益，说明松模型把部分容量用于最终会被投影掉的截面变化。

## 10. 当前信号地图

| 方向 | 状态 | 结论 |
|---|---|---|
| 同期 raw + cross deviation | 生产 | Ridge 与 LGBM 的基础输入 |
| 每资产 history40 | 生产 | 最大结构增益之一 |
| 行级非线性 market | 生产 | 与 weighted XS 组合后大幅提升 |
| weighted XS loss | 生产 | 与比赛指标更对齐 |
| responder 线性/投影/多任务前置 | 关闭 | 不补 target 残差 |
| 多尺度 history 扩张 | 关闭 | 无独立增量 |
| 压缩 market regime | 条件复验 | 机制正、量级未过门槛 |
| target-only MLP | 关闭 | 低相关但过弱 |
| peer lead-lag / lagged market | 关闭或无增量 | 完整 peer matrix 不支持；只允许未来测试低维严格滞后摘要 |
| 基础 per-asset XS scale | 本地候选 | 3s×480 严格 OOF +1.99%，当前唯一通过的新残差适配 |
| 条件化 asset/regime adapter | 关闭 | 部分时期正向但跨 meta fold 不稳定 |
| PCA / sparse asset×feature / raw dispersion | 关闭 | 严格 OOF 明显负增益 |

## 11. 生产等效残差信号重审（2026-08-14）

在 `modulo5 / phase_balanced / train_window78960 / embargo6` 下生成完整生产架构严格 OOF，随后只允许
二层参数在最早 meta fold 拟合并冻结到后续折。主要结论：

- 基础 per-asset XS scale 在 3 seeds × 480 下仍有约 +1.99% Peak、3/4 正折、drop-best +1.34%，通过；
- asset×预测幅度与 asset×regime 在部分 meta fold 有收益，但后期 fold 失效，不能上线；
- prediction-only linear/HGB market expert、soft gate 均显著为负；
- PCA factor×asset、sparse asset×feature Ridge、raw-feature dispersion gate 均显著为负。

当前最佳本地研究候选是 `outputs/candidates/v3_asset_cross_3s480_shrink500/`。它复用生产森林，只对
XS 分量做 OOF 拟合的资产缩放并重新投影为逐 time_id 零均值。候选目录不作为长期模型副本入库；
代码、参数报告和一致性报告足以恢复。

## 12. Lag 下一次允许重开的形式

静态残差校准已基本搜索完毕。若继续时间信号，只允许测试与当前 window5 明确不同的因果变化：

```text
lag1 / lag3 / lag5 / lag10
current - lag_k
一阶变化的再次变化（acceleration）
rolling_std / rolling volatility
过去截面变化的低维摘要
```

每个 family 必须单独接到 current cross residual 上，先 1 seed screening，再 3-seed confirmation。
禁止先全量计算未来可见的滚动量再切 fold，也禁止在采样后才推进历史。完整 peer matrix、静态 PCA、
普通 asset×feature interaction 和 scalar regime gate 已被现有证据关闭。

## 13. 回填训练重加权筛选（2026-08-27）

按新增公榜数据的严格时间切分，对当前 v3 身份固定、160 轮、1 seed 做训练损失重加权筛选。训练使用原始
数据加回填 `time_id < 948480`，中间 `948480..948485` 为 embargo，评分只用回填
`948486..1008479`；采样为 `phase_balanced / modulo5`，训练 2,825,517 行，评分 179,943 行。
评分权重保持原始非负 `weight`；重加权只进入 Ridge 与 weighted XS 的拟合损失，market forest 继续未加权。

结果（生产尺度 `1.16`，五个连续评分块）：

| 策略 | score | peak | 相对 none 的 peak | 正块 | 去最好块 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| `none` | 0.00318351 | 0.00319100 | — | — | — | 参考 |
| `backfill_x2` | 0.00327586 | 0.00328293 | +3.10e-05 | 4/5 | -1.52e-05 | 关闭 |
| `half_life_39480` | 0.00317717 | 0.00318472 | -3.44e-05 | 3/5 | -6.73e-05 | 关闭 |
| `recent_window_78960` | 0.00318351 | 0.00319100 | 0 | 0/5 | 0 | 关闭 |

`backfill_x2` 的总体提升由最好块贡献，未通过“均值为正、至少 4/5 正块、去最好仍为正、
2ΔA > ΔB”的预注册门槛；因此不进入 frozen validation，也不改变生产模型。完整原始结果保存在
`outputs/experiments/recency_adaptation_calibration.{json,md}`。
