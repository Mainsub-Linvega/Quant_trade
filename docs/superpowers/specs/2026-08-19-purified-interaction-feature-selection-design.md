# 纯化残差交互特征选择方案

> 日期：2026-08-19
> 分支：`exp/adaptive-feature-search`
> 状态：设计冻结，等待用户审阅；尚未实施

## 1. 决策

停止继续放宽完整路径或稳定子路径门槛，采用：

```text
时序基准 OOF 残差
-> 全特征池二阶非加法扫描
-> 加权交互纯化
-> 任务保持型经验 null 门禁
-> 时间稳定门禁
-> RuleFit 式预测选择
-> 派生列加入原模型
-> 严格配对 outer OOF
```

目标不是寻找“树经常使用的路径”，而是证明某个联合效应能够解释父特征各自主效应无法解释、并能迁移
到后续时间块的残差。树频率、split gain、SHAP 和 FAST 只用于提出候选，不决定入选。

研究基线固定为尚未转正的 fork 候选：

```text
market_lambda     = 0.7
blend_weight      = 1.17
prediction_scale  = 1.16
最终确认           = 480 轮 x 3 种子
asset adapter     = 关闭
```

本设计不修改生产目录、不生成提交 CSV，也不转正该候选。

## 2. 修订依据

完整路径筛选没有找到在两个内层时间块重复的路径。放松到 16 个支持区间后只得到二阶候选，而且两个已
评估外折都下降：

| Fold | Baseline Peak | Interaction Peak | Delta |
|---:|---:|---:|---:|
| 0 | 0.00086059 | 0.00081983 | -0.00004076 |
| 1 | 0.00166379 | 0.00163368 | -0.00003011 |

候选集中存在大量 `q0/q15` 极端阈值、互补 `gt/le` 规则和显著折间差异。这说明重复路径几何不是
增量预测证据，许多候选只是尾部主效应。

## 3. 冻结项与隔离项

交互实验保持以下内容不变：

- Ridge 的 current 与 cross-sectional deviation 输入；
- XS Top200、Market 现有输入和 History40；
- 预处理、采样、clip、模型超参数和轮数；
- `market_lambda=0.7`、`blend_weight=1.17`、scale 1.16；
- 现有严格 rolling OOF 和 promotion 门禁。

不删除基线特征。Top200 外特征只能通过已接受的派生交互列进入。任务对齐的直接特征重选保持为独立
实验，不能与交互实验同时进行；每组交互也不能单独重调融合和 scale。

明确排除：继续放宽路径门槛、枚举全部高阶组合、使用公榜选型、增加第四个模型、预设必须保留的交互
数量，以及 strict OOF 未通过却生成候选或 CSV。

## 4. 时序验证结构

外层沿用生产等效五折：

```text
n_folds       = 5
train_window  = 78960 sampled time_ids
embargo       = 6
sample_modulo = 5
sampling      = phase_balanced
```

每个 outer validation 只用于最终比较。基准训练外预测、分箱、纯化、正则参数和停止决策全部只能使用该
折训练窗口。每个外层训练窗口再切成四个连续内层块；每次只用较早数据拟合，在较晚块预测，禁止随机
IID 切分。

## 5. 三种任务残差

- Ridge：`r_ridge = y - ridge_oof`，使用官方训练权重。
- XS：令 `e_it = y_it - m_t`，使用 `r_xs = e - xs_oof`。
- Market：按冻结口径聚合每个 `time_id`，使用
  `r_market = market_target - market_oof`。

XS 优先检查至少一个父特征位于 Top200 外的组合；两个父特征都在 Top200 内的组合作为诊断对照，因为
现有 LightGBM 通常已经能够表达它们。Market 因样本只有时间点数量，使用更少分箱、更强 cell
shrinkage 和更高最小支持。

第一轮冻结 History40。只有 current-current 交互通过完整 outer OOF，才检查 current-history 或
history-history。

## 6. 二阶候选与纯化

首轮在全部 323 个源特征上扫描二阶组合。对每个特征对 `(j,k)`，在内层训练段比较：

```text
父特征加法面：g_j(x_j) + g_k(x_k)
二维联合面：  g_jk(x_j, x_k)
```

主规格在实施计划中预注册为 Ridge/XS `8 x 8`、Market `4 x 4`。最多保留一个固定细规格作为
敏感性检查，不能事后挑选更高分的规格。

对联合残差面进行加权纯化：

```text
h_jk = g_jk
       - E[g_jk | x_j]
       - E[g_jk | x_k]
       + E[g_jk]
```

所有统计量只来自训练段。迭代进行加权行列中心化；低支持或验证期未见 cell 收缩到零。每个“任务 +
特征对”默认只产生一个连续纯化列，不展开互补 `gt/le` 列。不同分辨率属于同一候选家族，只获得一次
试验身份。

## 7. 经验噪声门槛

每个外层训练窗口构造保持任务结构的负对照：

- XS：在同一 `time_id` 内打乱资产残差；
- Market：将残差时间块循环移位，距离大于 embargo；
- Ridge：结合同时间资产打乱与满足 embargo 的时间块移位。

对照数量、seed、移位距离和 null quantile 在首次正式得分前冻结。候选增量必须超过对应任务经验 null
门槛；不使用假设面板行 IID 的 pooled p-value。

## 8. 稳定门禁

候选进入预测选择器前必须满足：

- 在预注册多数内层时间块中取得正的非加法增量；
- 中位增量超过经验 null 门槛；
- 每个支持块有足够覆盖率与非空 cell；
- 收益不依赖单个极端尾部 cell；
- 纯化面方向或秩结构跨时间块兼容；
- 没有重复或互补家族；
- 训练和验证变换全部有限。

具体数值由实施计划一次冻结，观察 outer OOF 后不得放宽。

## 9. RuleFit 式预测选择

使用训练外纯化候选列拟合任务残差：

```text
residual ~ purified interaction families
```

采用 Elastic Net 或 group regularization，实施计划中二选一并固定；正则强度通过内层时序 OOF 和
one-standard-error rule 选择。同一特征对的不同规格属于一组。

保留者必须具有跨时间块非零选择频率、贡献方向一致、held-out residual gain 为正和
drop-best-block gain 为正。选择器只是训练期过滤器，不是第四个模型。通过者作为派生列加入原 Ridge、
XS 或 Market 训练矩阵，再按冻结超参数重训原模型。

## 10. 外层验收与早停

交互模型和同折冻结基线配对比较，必须同时满足：

```text
positive folds       >= 4 / 5
mean delta Peak       > 0
drop-best delta Peak  > 0
2 * delta_A           > delta_B
all folds finite      = true
```

若固定顺序的前两个外折都为负，最终 `4/5` 已不可能，立即早停。报告必须包含分量残差增量、候选
家族、父特征是否位于 Top200、null 门槛、覆盖率、系数稳定性、内存和推理成本。

## 11. 实验顺序与预算

1. `P0-diagnostic`：基准残差、null、覆盖率和 pair-score 分布，不添加列；
2. `P1-ridge`：只给 Ridge 添加纯化二阶列；
3. `P2-xs`：只给 XS 添加纯化二阶列；
4. `P3-market`：只给 Market 添加纯化二阶列；
5. `P4-combined`：只组合已经独立通过的分量；
6. `P5-higher-order`：仅在二阶通过完整 OOF 后启用。

每一臂只有通过前置门禁才进入完整重训，失败即结案，不放宽阈值续命。1-seed screening 通过后最多进行
一次 `3 seeds x 480 rounds` 确认，并继续固定 `0.7/1.17/1.16`。确认通过后才允许单独预注册
联合重标定。

## 12. 高阶扩展

高阶搜索默认关闭。只有已接受二阶家族 `(j,k)` 才能加入第三个特征：

```text
h_jkl = pure_joint_residual(r | h_jk, x_l)
```

三阶必须证明相对二阶仍有训练外增量，并重新通过纯化、null、稳定性和预测选择门禁。四阶只能从已接受
三阶继续扩展。固定预算浅树可以报警无二阶前驱的潜在高阶结构，但第一版不能据此直接生成模型列。

## 13. 产物与有效负结果

实施应产出冻结协议 JSON、每折候选与 null manifest、纯化面 manifest、选择器稳定性报告、配对 OOF
JSON/Markdown 和资源报告。未通过门禁时不生成 candidate 或 CSV。

以下均为有效终点，不允许据此放宽方案：没有 pair 超过噪声地板；稳定 pair 被稀疏选择器压零；选择器
通过但加入原模型后 OOF 失败；二阶有效而三阶无增量；只有一个模型分量通过。

无交互通过意味着现有模型已吸收可用联合结构，或剩余交互低于当前数据能够可靠识别的水平。

本设计取代继续放宽 stable-subpath 的主动路线。旧设计和失败产物保留为研究证据，不删除、不回写。
