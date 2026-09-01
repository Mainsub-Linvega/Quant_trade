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

为避免 responder 泄漏，MLP 只使用可部署输入，分成 market head 和 cross head，并实现 sklearn
权重到纯 NumPy 推理的导出对拍。模型与 v3 预测相关较低，但单模只有 v3 peak 的约 24.3%；
50/50 集成下降约 54.49%。

这个案例形成一条重要规则：低相关可能来自模型太弱，而不是独立 alpha。集成必须直接验证组合指标。

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

---

## 从 ROADMAP §4「行动面板」归档（2026-09-01）

收官后整理。ROADMAP §4 曾积累 **1,169 行**，其中绝大多数是**已结案课题的完整证据卡片** ——
一个叫「行动面板」的章节里 80% 是历史。下列条目按原文**逐字**迁入本文件，
ROADMAP §4 只保留一行结论与指回这里的链接。
课题编号（`P0`…`P12`、`P-XXX`）的含义见 [`../GLOSSARY.md`](../GLOSSARY.md) §1。

### P-V4R — V4-R 压缩 market regime 扩展数据复验 —— `REJECTED`（2026-08-24，轴关闭）

ROADMAP §3.7 里「V4 中唯一保留扩展数据复验资格」的那一项，按原规格
（`temporal_multiscale.py --arms baseline t4_regime`，1s160 筛选档）在
`outputs/data_roots/decision` 上复验：

| | 2026-08-12（原始数据） | 2026-08-24（扩展数据） |
|---|---:|---:|
| 相对增益 | +1.34% | **+1.378%** |
| 正折 | 4/5 | **3/5** |
| 去最好折 | — | **−8.47e-07（翻负）** |

五道检查只过两道（`mean_delta_positive`、`mechanism_2dA_gt_dB`），
`positive_at_least_4of5`、`survives_drop_best`、`relative_gain_at_least_3pct` 全不过。
增益几乎全集中在 fold 4（+1.042e-04，是次大折的 **5 倍**）。
脚本裁决 `passed_arms: []`、`next: "stop temporal expansion"`。

⟹ **多给 17.8% 更近的数据没有救活它，反而更差**（正折 4/5→3/5、去最好折翻负）。
这与 8/14 `asset × observable regime` 和 8/23 `xs_market_state_probe` 两次独立否决一致：
**「市场态」这条信息通道在当前数据上就是稀薄不稳的**。V4 轴现已全部关闭。
证据：`outputs/experiments/temporal_regime_recheck_0824.{json,md}`。


### P-RESP — responder 轴重开条件已消化 —— `REJECTED`（2026-08-24，这次是真的关严了）

D0.2b 判 `backfill_has_responders` ⟹ 触发 2026-08-22 那四项 `REJECTED` 的统一重开条件。
按原规格复验（不换臂、不调阈值、不加梯子成员）：

- **Stage C 补测**（`responder_stage_c_fill_0824`）：**REJECT**，14 个未测格子无一过门禁。
  ⚠️ **但它回答不了触发它的那个问题** —— 复现自检 `max|Δ| = 0.000e+00`，
  因为它读的 `responder_oof_*_f323.npz` 覆盖 `time_id 394,982–888,478`，
  **止于回补窗口开始前一格**，回补的 3,217,458 行一行都没进去。
  真做扩展数据版需重建两份缓存（多目标岭回归 ~25 min + v3 基准 **3s480** OOF ~1.5–2 h），
  **未做**：28 个格子一致为负，且负控制 `negctrl_shuffle` 的 −4.47% 与真 responder 的
  −1.6%~−4.9% 落在同一区间 ⟹ responder 辅助与「灌噪声」在当前判据下不可区分。
- ⭐ **选列判据**（`responder_selection_probe_0824`）：**这一半真的吃到了扩展数据**
  （3,117,682 行 / 209,143 采样 time_id，`--data-root outputs/data_roots/decision`）。
  结论与 08-22 一致：五折最小重合 **169/200** < 决策线 190；
  换掉的是原判据排第 **19** 名的列、换进来的排第 **300** 名（共 323 列），
  全局 Spearman 仅 0.76–0.80 ⟹ **假设 A（边缘搅动）被证伪、假设 B（实质分歧）成立**，
  预注册写的「降方差」机制**未兑现**。事后逐级标准化后 175→182/200，**仍在 190 之下**
  ⟹ 裁决对规格选择稳健。
- ⟹ **重开条件已消化：能看到新数据的那一半看了，答案没变。** responder 轴维持关闭，
  8/31 前不再碰。证据：`responder_stage_c_fill_0824.md`、`responder_selection_probe_0824.md`。


### P8 — 多任务辅助监督 —— `CLOSED_FAIL`（2026-08-19）

- **状态**：`CLOSED_FAIL`。Stage 1 fold-0 符号筛不过，按预注册停止，**未跑 Stage 2**。
- **结果**：⭐ **机制是真的，增量不存在**。辅助损失确实让 MLP 自身 peak 从基准的 17.4% 提到
  **20.3%**（相对 +16.7%），残差信号由 −7.17e−05 翻正到 +7.14e−05；但即便在 **oracle 上界**
  口径下，与生产 3s480 的两分量最优配比增益也只有 **+0.026%**（比 3% 门槛低约 115 倍），
  最优配比给 MLP 的幅度占比仅 **1.8%**。
- **为什么与立项时的 oracle +6.97% 差 270 倍 —— 基准强度**：
  `target_mlp_screen` 的基准是 1s160 / modulo 10 / 窗 39,480 / 100 特征（fold0 peak 0.00069987），
  生产 3s480 强 **1.51×**（0.00105595）；MLP 相对强度从 40.2% 掉到 17~20%，与基准的 corr
  从 0.24 升到 0.41~0.42 ⟹ **MLP 那点信息生产基准里已经有了**。
  正是 CLAUDE.md §8.6/§8.7 点名的形状。⚠️ 两次 screen 折版图不同，不是配对比较。
- **⚠️ 预注册里一道门槛写错并已订正**：原第二道 `2ΔA>ΔB` 在「配比被重解」时不成立
  （`A→cA`、`B→c²B`，peak 不变但 ΔA/ΔB 只反映缩放）。已换成尺度不变的残差分解
  `Δpeak = (A_m − A_b·C/B_b)²/(B_m − C²/B_b)`（恒等式独立验算，相对差 2.9e−13）
  加上本就该有的 **3% 幅度门槛** —— **收紧，不是放松**，重跑后判定不变。
- **重新开放条件**：8/23 回补数据后基准本身变化，按**原规格**复验一次；
  或出现能让 MLP 自身 peak 达到基准 **70%** 以上的模型族（当前 20.3%）。
- **证据**：`outputs/experiments/multitask_mlp_stage1.{json,md}`、
  `target_mlp_oracle_blend.{json,md}`；单测 `tests/test_multitask_mlp.py`。
- 以下为立项时的预注册记录，保留备查：
- **为什么重开**（08-12 那条否决**不覆盖**这个机制）：被否的是把 responder 的**预测值**
  当**输入特征**（两阶段误差累积，且 runner 剥列 ⟹ 线上不成立）；本条是**辅助损失**
  （共享 trunk，responder 只在训练时提供梯度，推理只留 target 头）。08-18
  `horizon_auxiliary_cache_probe` 的重开条件是「不是换目标 / 线性叠加 / 对预测值做二层校准」
  —— 三条都不是，字面满足。
- **⭐ 新证据**（`outputs/experiments/target_mlp_oracle_blend.{json,md}`，08-19，不训练）：
  从 `target_mlp_screen` 逐折 A/B 反解交叉项，算出 **oracle 最优配比折均 +6.97%、5/5 折、
  去最好折 +4.86%** ⟹ 当年「等权集成 −54.49%」否掉的是**等权掺弱模型**，
  不是「MLP 没有独立信息」。⚠️ 但那是 **oracle 上界**；按仓库量过的冻结系数让步
  （−2.54%~−3.84%）折算只剩 **+3.1%~+4.4%**，**恰好卡在③类 +3% 门槛上**。
- **预注册**（`experiments/multitask_mlp.py` 的模块常量 + docstring + 单测三处钉死）：

  ```text
  λ            0.3（唯一超参，不搜索）
  辅助目标集   梯子 5 个 responder_00/02/03/04/05（H=1/2/4/7/10，夹着 target 的 H=5）
  实现         多输出 MLPRegressor，辅助列乘 √λ ⟺ 该头损失权重 λ（alpha=0 时严格）
  对照臂       target_only（同架构/同种子/同迭代）—— 没有它，正结果无法归因给辅助损失
  Stage 1      fold 0，oracle 配比，只看符号：Δpeak>0 且 2ΔA>ΔB 且 multitask > target_only
  Stage 2      五折冻结系数：折均 ≥ +3%、≥3/5 正折、去最好折 > 0、2ΔA>ΔB
  ```

- **停止条件**：Stage 1 任一门槛不过即停 —— **不调 λ、不调 hidden、不换激活、不换辅助目标集**。
  ⚠️ 最大的风险不是「NN 学不出来」，是**超参搜索会让 OOF 尺子当场失效**。
- **验收条件**：过 Stage 2 才谈公榜验迁移率（1 次，且必须在 8/23 之前 —— 公榜停更后没有外部裁判）。


### P9 — NN 独立能力阶梯 —— `REJECTED`（2026-08-20，天花板 28.8%）

- **状态**：`REJECTED`。曲线在 **<50% 门槛处掉头**，按预注册停止，**未触发**条件延长（1200 档）。
- **问的问题**：`target_mlp_screen`(08-12) 与 `multitask_mlp_stage1`(08-19) 报的 MLP 独立
  peak（24.3% / 17.4~20.3% of 基准）都是 `max_iter=12` 下测的，而早停是关掉的
  （`tol=0.0`、`n_iter_no_change=max_iter+1`），JSON 里 `iterations` 全等于 12
  ⟹ **那是跑完预算被掐断，不是收敛**。所以「NN 只有树的 20%」当时是**预算事实**，不是能力事实。
- **曲线**（fold 0，独立 MLP peak / 生产 3s480 基准 peak）：

  | max_iter | target_only | multitask | 较好者 | 相对上一档 |
  |---:|---:|---:|---:|---:|
  | 12 | 17.4% | 20.3% | 20.3% | — |
  | **50** | 26.2% | **28.8%** | **28.8%** | **+42.0%** |
  | 150 | 6.5% | 5.7% | 6.5% | −77.3% |
  | 400 | 1.4% | 1.2% | 1.4% | −78.3% |

- **两个结论，方向相反，都成立**：
  1. ⭐ **12 档确实被预算掐住了** —— 给到 50 档相对涨 **+42%**。
     「NN 没被公平测过」这个判断是对的。
  2. ❌ **但天花板是 28.8%，不是 50%**，而且 50 档之后**崩溃**（−77% / −78%）。
     ⟹ **绑定约束不是预算，是正则化** —— 这个配方没有任何阻止过拟合的机制
     （早停关闭、无学习率调度、无 dropout/LayerNorm，只有 `alpha=1e-3`）。
- ⭐ **辅助损失的符号随过拟合翻转**：欠拟合区（12/50 档）`multitask > target_only`，
  过拟合区（150/400 档）反过来。机制说得通 —— 辅助损失在容量饥饿时是**正则**，
  在开始记忆时变成**容量竞争**。这是 08-19 P8「机制成立但增量不存在」的补充证据。
- ⚠️ **不得据此重开 P8（B 线）**：50 档的混合增益 **+6.46%** 会过 08-19 Stage 1 的门槛，
  但那次预注册在 `max_iter=12`；**挑一个让混合增益最大的 epoch 数就是看结果选参**。
  何况这个读数本身不可信 —— 见下条。
- ⚠️ **方法学发现：单折 oracle 混合增益不可信。** 独立强度单调崩溃
  （28.8% → 6.5% → 1.4%），混合增益却**非单调**（+6.46% → +1.02% → +3.26%）；
  400 档 MLP 独立只剩基准的 **1.4%**，oracle 混合却仍报 **+3.26%**。
  ⟹ **追认 08-19 要求「冻结系数 + 5 折」做终审的决定是必要的**，不是形式主义。
- **对 v5 的范围结论**（8/31 之后）：不是「给更多算力」，而是
  ① 能防过拟合的训练配方（早停 / LR 调度 / dropout / LayerNorm）；
  ② `asset_id` 用 embedding 而非 15 维 one-hot；
  ③ ~~特征选择不再沿用为线性/树挑的 `|corr(feature, e)|` top-200~~ ——
  **2026-08-22 已探过并结案**：换成 responder 窗口一致性选出的 200 列（与原判据重合 175/200）
  按原规格复跑整条阶梯，天花板 28.8% → **27.4%**，且**倒 U 形状一模一样**（50 档仍是峰值、
  150/400 档仍然崩溃）⟹ **换特征集换不动天花板**，独立印证「绑定约束是正则化不是特征」。
  ⟹ **v5 的可改项从 3 条收缩到 2 条**。证据：`nn_capacity_ladder_respsel.md`。
  曲线已经把「预算」这条排除掉了 —— v5 若还只是加算力，可以直接不做。
- **适用范围**：本阶梯否掉的是 **sklearn `MLPRegressor` + 生产特征表示 + 这套预算**这个配方，
  **不是**「NN 这个模型族」。
- **重新开放条件**：上面三条范围项**至少改掉一条**后按原规格复验；或 8/23 回补数据后基准变化。
  **不得**只加 epoch / 只加宽网络重跑。
- **证据**：`outputs/experiments/nn_capacity_ladder.{json,md}`（预注册
  `nn_capacity_ladder_plan.json` 的 sha256 记在里面）、`multitask_mlp_e{12,50,150,400}.{json,md}`。
  ⭐ 12 档对 08-19 锚点的偏差 **0.00e+00**（逐位复现）⟹ 环境与数据一致，曲线可解读。


