# ROADMAP.md — 当前状态与行动面板

> **状态日期：2026-08-13。** 本文件只描述当前有效状态和未来动作。完整探索过程见
> [`NOTES.md`](NOTES.md) 与 [`research_history/`](research_history/README.md)。生产真值以模型产物、
> promotion manifest 和 [`experiments/ledger.csv`](experiments/ledger.csv) 为准。

## 1. 当前目标与节点

- **8/23**：公榜停止更新并等待主办方标签/数据回补；收到更新包后先审计，不先训练。
- **8/31**：私榜策略文件提交截止；私榜共 10 次机会，至少保留 3 次余量。
- **当前主目标**：守住已转正的 `v3_hybrid_mkt_shrunk`，完成交付风险闭环；若数据更新真实存在，
  再按预注册矩阵重训和重标定。
- **研究原则**：当前结构轴已经带来主要收益，普通容量和轮数旋钮接近耗尽；不再用无边界网格搜索
  追逐小波动。

## 2. 当前生产基线

| 项目 | 当前值 | 真值来源 |
|---|---|---|
| 生产目录 | `strategies/v3_hybrid/model/` | 当前文件系统 |
| 来源候选 | `outputs/candidates/v3_hybrid_mkt_shrunk/` | promotion manifest / Git 记录 |
| 公榜分数（**生产目录本身**） | **0.0039977510**（2026-08-13） | `experiments/ledger.csv` |
| 公榜最好成绩 | **0.0041150085**（2026-08-17，slow/fast 纯 CSV 变换，**+2.93%**） | `experiments/ledger.csv` |
| ⚠️ 二者的区别 | 最好成绩是对生产 CSV 做的**后处理**，**不在任何模型产物里**。私榜要用它必须在 `main.py` 实现逐 asset 的自身预测滚动均值（跨 `predict` 状态 ⟹ 模型身份变更 + promotion 全套门禁）。**当前若直接打包私榜，拿到的是 0.0039977510 那一档。** | 本表 |
| 公榜第一 | **0.0060**（2026-08-17，**用户报告**，非本地测量） | 用户 |
| 与第一的差距 | **+50.1%**（IC +22.5%）；旧记录 0.00520002 标 `SUPERSEDED` | 本表两行相除 |
| 截面块 | weighted LGBM，480 轮 × 3 种子，history40 | `hybrid_meta.json` |
| 市场块 | unweighted row-level LGBM，λ=0.5，480 轮 × 3 种子 | `hybrid_meta.json` |
| 截面混合 | `blend_weight=1.0`，即 LGBM 截面分量全替换 | `hybrid_meta.json` |
| 后处理 | `prediction_scale=1.16`，clip=0.5 | `hybrid_meta.json` |
| 训练采样 | `sample_modulo=5`，`phase_balanced` | `hybrid_meta.json` |
| promotion 校验 | 双后端最大差 `2.498e-16`，结构敏感性门禁通过 | `outputs/promotions/v3_hybrid_s1.16_w1_3seed/promotion_manifest.json` |
| train/inference 一致性 | 最近候选记录约 `1.913e-08` | `experiments/ledger.csv` / NOTES 历史 |

⚠️ `experiments/ledger.csv` 里 08-11、08-13 两行注释中的 `0.00520002` 是**当时**的榜首真值，
按 CLAUDE.md §7 不回写历史；以本表为当前值。0.0050 / 0.0055 现在分别是榜首的 83% / 92%，
不再等价于「追平第一」。

当前生产文件 hash 与 promotion staging 一致；详见
[`research_history/delivery-and-incidents.md`](research_history/delivery-and-incidents.md)。

### 性能风险

- 最近同结构前身 `mkt_we` 的完整 runner 实测 `predict_total=6.23` 分钟；当前 `mkt_shrunk` 的精确
  全量 wall-clock 尚未在 manifest 中单独记录，不能把 6.23 分钟冒充当前模型精确值。
- NumPy 双森林兜底按已有测量约 15 分钟，是私榜环境的主要剩余风险。
- 2 种子旧候选只节省 5.45% 全量推理时间且掉分未知，不作为默认生产方案。

## 3. 当前有效的研究判断

1. **结构收益仍是主要来源。** history、第二市场分量和带权截面训练贡献远大于后续容量微调。
2. **市场块和截面块的容量方向相反。** 市场块收缩小幅有益，截面块收缩显著有害。
3. **480 轮已形成内部极值。** 旧结构 320 轮下降，新结构 960 轮下降；统一轮数轴结案。
4. **普通②类调参余量很小。** 2026-08-13 容量扫描只得到 +0.77%，低于 2% 关注门槛。
5. **本地尺子不能稳定判断拟合紧密度。** alpha、轮数、history 宽度都出现过本地与公榜量反。
6. **Responder 线性变换和多任务前置门禁已关闭。** 可预测不等于能补 target 残差。
7. **V4 中只有压缩 market regime 值得在扩展数据上原规格复验一次。** 多尺度 history 与 target-only
   MLP 均已否决。
8. **Scale 1.16 已足够接近峰值。** 当前没有为第二个 scale 点消耗额度的经济性。
9. **占分比已翻转，旧的 60.1%/39.9% 作废。** 当前 OOF 上 market : cross = **41.2% : 58.8%**
   （bootstrap CI [27.5%, 53.1%]）。08-10 那组 `60.1%/39.9%` 标记
   `SUPERSEDED`：它是**公榜**三点解方程的产物，且模型还在 history40 与行级市场模型**之前**
   （原文见 [`research_history/source_snapshots/NOTES.pre-doc-refactor.md`](research_history/source_snapshots/NOTES.pre-doc-refactor.md)
   第 753 行附近，保留不删）。⚠️ 新旧两个数分属 OOF 与公榜两把尺子，不能宣称推翻了公榜测量。
   证据：`outputs/experiments/v3_block_ceiling_3s480.md`。
10. **两块的边际价值接近平手。** cross 块的收割率是 market 块的 2.62×，但 market 块每提升
    一点 R² 值 2.460× 的分（`w_m/w_e`）。乘积接近 1 ⟹ OOF 没有给出明确的精力偏向。
11. ~~**纯①类后处理旋钮已扫干净。**~~ `SUPERSEDED`（同日被自己的测量证伪）。时间平滑、
    分 phase scale/混合比、解开 `market_lambda` 四项确实都被否，但**同一天在同一条线索上
    找到了更大的一个**：见下一条。教训是「这一轴已耗尽」不能由几次否决推出来。
12. **`slow/fast` 分离是当前唯一活着的①类杠杆。** 逐 asset 因果 trailing mean 把预测拆成
    慢/快两块后，两块需要差 2.8 倍的 scale（slow 0.2828 / fast 0.7881，对照单一 scale 0.7296）。
    严格 OOF 扩展窗口 **+5.77%**、3/4 折、6/6 预注册门槛全过，第二份 cache 同 K 复现 +5.87%；
    全分辨率口径核对（fold 2/3/4 各 20,000 连续真实 time_id、系数冻结）合并后 **+5.93%**，
    与 OOF 吻合 ⟹ 采样格口径没有骗人。**未建候选、未改生产**，是否推进由用户决定。
    机制是 `2ΔA > ΔB`（ΔA +8.07% / ΔB +11.21%），不是纯减方差。
    证据：`v3_slow_variance_3s480.md`、`v3_fullres_slow_probe_summary.md`。

## 4. 行动面板

### P0 — 私榜交付闭环

- **状态**：`IN_PROGRESS`
- **目标**：确认生产目录可被完整、可审计、在时限内打包和运行。
- **动作**：
  1. 在最终打包前重新运行完整 unittest、双后端一致性和全量 runner。
  2. 记录当前 `mkt_shrunk` 的 model init、predict total、wall clock、最大单步时间和非有限值数量。
  3. 在接近私榜环境的 4 核设置下验证 LightGBM 主路径；单独记录 NumPy 兜底风险。
  4. 由用户执行 `scripts/make_submission.py`，并运行 zip 审计；至少留 3 次私榜机会。
- **验收条件**：生产 meta 与公榜模型一致；两后端对拍通过；全量行数正确；0 非有限值；耗时在
  主办方限制内；包内无训练代码和多余产物。
- **证据**：promotion manifest、`scripts/audit_submission_zip.py`、最终 runner JSON。

### P1 — 8/23 数据更新审计

- **状态**：`BLOCKED_UNTIL_DATA_REFRESH`
- **目标**：先确认数据和主办方原文是否真的变化，再决定是否重训。
- **动作顺序**：
  1. 对 `docs/`、`examples/`、`timeseries_api/` 做 Git diff，只读核验主办方变化。
  2. 用 `scripts/audit_data_release.py` 对比 `outputs/data_audits/data_release_20260812.json`。
  3. 若 train split 未变化，停止固定结构重训；不要为了日期变化重跑模型。
  4. 若 train split 变化，先在回补标签上复算历史提交和真实指标，修正本地尺子。
- **验收条件**：审计 JSON 明确 added/removed/modified；数据 hash 可追溯；任何训练动作都有审计结果
  作为输入。

### P2 — 扩展数据固定结构重训与联合重标定

- **状态**：`BLOCKED_BY_P1`
- **目标**：量化纯数据收益，再比较预注册的相关参数组合。
- **动作**：
  1. 使用 `scripts/retrain_extended.py` 先 dry-run；只写候选目录，不写生产目录。
  2. 固定当前结构重训，作为“只增加数据”的基线。
  3. 执行 `outputs/experiments/joint_recalibration_plan.json` 中冻结的 Ridge 12 格和 LGBM 9 格。
  4. 原规格复验 V4-R regime；不重开 T1/T2/T3 和 MLP 搜索。
- **验收条件**：训练/验证严格时序；不读结果扩格；报告包含配对增量、同号折数、A/B 和推理成本。

### P3 — 市场森林独立轮数 —— `CLOSED_FAIL`（2026-08-17 同步；实验其实早已跑完）

- **状态**：`CLOSED_FAIL`。本条此前一直挂着 `BACKLOG_NOT_SCHEDULED`，但
  `outputs/experiments/v3_market_round_scan_phasebal_prodwindow.md` 早就把它跑完了：
  160 / 240 / 320 / 400 / 480 **全部 FAIL**，报告里写着 `Selected: None`。
  ROADMAP 当时没同步 —— 这类「实验做了但面板没更新」本身就是要防的坑。
- **结果**：480 轮是这一族里最好的（相对 160 轮 +2.02%），但只有 3/5 折、去最好折 +0.81%，
  不过门槛。⟹ **截短市场森林没有收益，维持 480。**
- **重新开放条件**：市场块结构本身改变（不是轮数），或回补数据后原规格复验。

### P4 — recency / 训练窗阶梯 —— `CLOSED`（两侧都已结案）

- **状态**：缩短方向 `CLOSED_FAIL`（2026-08-18）。60,000 档 −9.50%（1/5 折）、
  40,000 档 −24.54%（0/5 折），远超检出下限 8.7% ⟹ 是**测得出来的负结果**。
  机制是 ΔB 大幅抬升（+6.5%~+23.2%）而 ΔA 只轻微下降 —— 数据少 ⟹ 系数噪 ⟹ 方差涨。
  `frozen` 臂确认结论对 `min_data_in_leaf` 混淆稳健。**维持 78,960。**
- **不需要公榜裁决**：②类的量反风险只在**采纳**改动时才要公榜背书；这里的空动作
  就是保持现状，拒绝改动不需要花名额。
- **加长方向也已测完（2026-08-18）**：扩展窗五折（+25%~+100% 数据）折均 +1.08%、
  **正折 2/5**、去最好折 −3.84%、配对 CI [−6.71e−05, +1.03e−04] 跨 0 ⟹ **测不出效应**。
  ⟹ **两侧合读：78,960 已过收益递减点** —— 拿走数据明确掉分，加数据什么也换不来。
  **P4 整条结案，训练窗轴关闭。**
- **证据**：`v3_recency_ladder_3s480.md`。⚠️ `w60000_frozen` 臂被中止未产出，
  但 40,000 档两臂一致、60,000 与 40,000 同向，缺它不影响结论。

## 5. 已结案项目

| 日期 | 项目 | 结论 | 证据入口 |
|---|---|---|---|
| 8/17 | **slow/fast 分离** | OOF +5.77%（6/6 门槛、3/4 折）、复现 +5.87%、全分辨率合并 +5.93%；**未建候选**；可纯改 CSV 验证 | `v3_slow_variance_3s480.md` / `v3_fullres_slow_probe_summary.md` |
| 8/17 | z-score（temporal 最后一项） | `REJECTED`：+0.70%、3/5 折、去最好折为负 ⟹ **temporal 族全族关闭** | `temporal_zscore_screen.md` |
| 8/18 | **P4 训练窗（收官）** | `CLOSED`：减数据明确有害（−24.5%/0-of-5），加数据测不出（+1.08%/2-of-5、CI 跨 0）⟹ **数据量已饱和**，78,960 维持 | `v3_recency_expanding_ladder_1s160.md` |
| 8/18 | per-asset **叠加**（blend） | `REJECTED`：全部为负，per-asset 臂（−3.91%、0/4）比 `shared` 对照（−1.19%）**还差**；corr 0.57~0.63 非低相关 ⟹ 替换与叠加两种用法都关闭 | `asset_blend_check.md` |
| 8/18 | per-asset 完整载荷 | `RESULT`→关闭：线性内异质性大（+95.8%、5/5 折、系数相关仅 +0.419），但 per-asset ridge 仍比生产 LGBM 截面块**低 50.5%** —— 树的 `asset_id` categorical 早已吃掉这块 | `asset_loading_diagnostic.md` |
| 8/18 | responder 窗口图谱 | `RESULT`：测出窗口梯子 H=1/2/4/**5(target)**/7/10；但重建 R² 只 0.883、单步 u 不存在 ⟹ horizon 分解缺前提，不推进 | `responder_window_atlas.md` |
| 8/18 | P4 训练窗缩短 | `REJECTED`：60k −9.50%（1/5）、40k −24.54%（0/5），机制是 ΔB 抬升而非 ΔA 丢失；维持 78,960。阶梯单调 ⟹ **扩展窗（+50% 数据）未测** | `v3_recency_ladder_3s480.md` |
| 8/17 | **slow/fast 公榜裁决** | ⭐ **+2.93%（0.0041150085），新最好成绩**，08-13 以来第一次上涨，且未重训未改模型。⚠️ 迁移率 **0.51×**（本地 +5.77%），**项目首次本地高估** | ledger / `v3_slow_variance_3s480.md` |
| 8/17 | asset adapter 公榜裁决 | `REJECTED`：Δ=−6.9e-06，按预注册 \|Δ\|<1e-5 判为**不可辨别** ⟹ asset scale 轴关闭，不再调 | ledger |
| 8/17 | **market 同口径复测** | `REJECTED`：直接回归 `m_t` 在**每一个** α 上都输，最好 −22.55%、1/5 折 ⟹ market 侧六条路全关 | `market_direct_recheck_3s480.md` |
| 8/17 | 市场森林独立轮数 | `CLOSED_FAIL`：160~480 全 FAIL、`Selected: None`（实验早已完成，本次只是同步面板） | `v3_market_round_scan_phasebal_prodwindow.md` |
| 8/17 | 解开 `market_lambda` | `REJECTED`：OOS −1.62%、2/4 折、去最好折 −6.17%，λ=0.5 保持不动 | `v3_slow_variance_3s480.md` |
| 8/17 | market/cross 分块天花板 | 占分翻转为 41.2%:58.8%；收割率之比 2.62× 与兑换率 2.46× 接近抵消 | `v3_block_ceiling_3s480.md` |
| 8/17 | 时间平滑（lag 平滑器） | `REJECTED`：预测比信号平滑得多，最小可测 lag 的 OOS 增益为负 | `v3_temporal_smoothing_3s480.md` |
| 8/17 | 分 phase scale / 混合比 | `REJECTED`：两臂 6 门槛均未过；`A_p` 离散度测不出超过抽样噪声 | `v3_phase_scale_3s480.md` |
| 8/13 | 市场块容量 | shrunk 赢 +0.77%，方向真实但接近饱和 | `experiments/ledger.csv` |
| 8/13 | 截面块容量 | shrunk −9.84%，保留 loose | `experiments/ledger.csv` |
| 8/13 | 轮数 960 | −5.20%，480 内部极值 | `experiments/ledger.csv` |
| 8/13 | 市场模型 + weighted XS | 公榜 +21.99%，进入生产架构 | `combo_market_weight.md` / ledger |
| 8/12 | Responder A/B/C | 可预测但不补 target 残差，停止多任务 NN | `responder_*.md` |
| 8/12 | V4 temporal / MLP | 仅 regime 保留扩展数据复验资格 | `temporal_multiscale_screen.md` 等 |
| 8/11 | 每资产 history40 | 公榜大幅提升，已进入生产 | `history_peak_lgbm_scoped.md` / ledger |
| 8/10 | phase-balanced Ridge | 公榜 +1.97%，已进入生产 Ridge | ledger |
| 8/8–10 | 验证框架、严格求解器、A/B 分解 | 已形成当前研究判定规则 | `research_history/validation-and-calibration.md` |

完整失败路径和结论翻转见 [`research_history/`](research_history/README.md)，不要从本表反推实验细节。

## 6. 更新规则

- 当前生产模型变化时，同时更新本节、promotion manifest 引用和 ledger；不要只改文字。
- 新任务进入行动面板时必须写状态、目标、动作、验收条件和证据。
- 结案任务移入“已结案项目”，长推导进入主题历史，ROADMAP 不保留整段实验日志。
- 每次更新在下表追加一行，不回写成没有日期的“现在”。

| 日期 | 更新 |
|---|---|
| 2026-08-13 | 文档体系重构；以已转正的 `mkt_shrunk` 和公榜 0.0039977510 重建当前状态。 |
| 2026-08-17 | 三项纯 OOF 后处理诊断结案（分块天花板 / 时间平滑 / 分 phase A·B）；占分比 60.1%/39.9% 标 `SUPERSEDED`。生产目录与模型身份未改动。 |
| 2026-08-17 | 顺着时间平滑的否定结论找到 `slow/fast` 分离（OOF +5.77%、全分辨率核对 +5.93%）；本节第 11 条被同日测量证伪并改写。`market_lambda` 结案。仍未建候选、未改生产。 |
| 2026-08-17 | 公榜第一更新为 0.0060（用户报告）；market 侧同口径复测 `REJECTED`，六条路全关，我上一轮「下一个方向是 market」的判断标 `SUPERSEDED`；P3 同步为 `CLOSED_FAIL`，新增 P4 recency 预注册。 |
