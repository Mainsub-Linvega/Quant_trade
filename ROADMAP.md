# ROADMAP.md — 当前状态与行动面板

> **状态日期：2026-08-17。** 本文件只描述当前有效状态和未来动作。完整探索过程见
> [`NOTES.md`](NOTES.md) 与 [`research_history/`](research_history/README.md)。生产真值以模型产物、
> promotion manifest 和 [`experiments/ledger.csv`](experiments/ledger.csv) 为准。

## 1. 当前目标与节点

- **8/23**：公榜停止更新并等待主办方标签/数据回补；收到更新包后先审计，不先训练。
- **8/31**：私榜策略文件提交截止；私榜共 10 次机会，至少保留 3 次余量。
- **当前主目标**：守住已转正的 `v3_hybrid_mkt_shrunk`，完成交付风险闭环；fork 本地实验继续保留
  `market_lambda=0.7 / blend_weight=1.17` 作为未转正候选。若数据更新真实存在，再按预注册矩阵
  重训和重标定。
- **研究原则**：当前结构轴已经带来主要收益，普通容量和轮数旋钮接近耗尽；不再用无边界网格搜索
  追逐小波动。

## 2. 当前生产基线

| 项目 | 当前值 | 真值来源 |
|---|---|---|
| 生产目录 | `strategies/v3_hybrid/model/` | 当前文件系统 |
| 来源候选 | `outputs/candidates/v3_hybrid_mkt_shrunk/` | promotion manifest / Git 记录 |
| 公榜分数 | **0.0039977510**（2026-08-13） | `experiments/ledger.csv` |
| 截面块 | weighted LGBM，480 轮 × 3 种子，history40 | `hybrid_meta.json` |
| 市场块 | unweighted row-level LGBM，λ=0.5，480 轮 × 3 种子 | `hybrid_meta.json` |
| 截面混合 | `blend_weight=1.0`，即 LGBM 截面分量全替换 | `hybrid_meta.json` |
| 后处理 | `prediction_scale=1.16`，clip=0.5 | `hybrid_meta.json` |
| 训练采样 | `sample_modulo=5`，`phase_balanced` | `hybrid_meta.json` |
| promotion 校验 | 双后端最大差 `2.498e-16`，结构敏感性门禁通过 | `outputs/promotions/v3_hybrid_s1.16_w1_3seed/promotion_manifest.json` |
| train/inference 一致性 | 最近候选记录约 `1.913e-08` | `experiments/ledger.csv` / NOTES 历史 |

当前生产文件 hash 与 promotion staging 一致；详见
[`research_history/delivery-and-incidents.md`](research_history/delivery-and-incidents.md)。

### Fork 本地最佳候选（未转正）

| 项目 | 当前值 | 说明 |
|---|---|---|
| 候选身份 | `market_lambda=0.7 / blend_weight=1.17` | 只改融合元数据，不复制或修改六片森林 |
| 公榜分数 | **0.00407075**（2026-08-17） | 当前 fork 最佳；高于上游生产 0.0039977510 |
| 市场融合 | `market_lambda=0.7` | 0.8 已回落，保留较稳健的 0.7 |
| 截面融合 | `blend_weight=1.17` | 0.6 下的 1.1/1.17/1.2 梯子显示平台区，取 1.17 |
| 后处理 | `prediction_scale=1.16`，clip=0.5 | 沿用已验证公榜口径 |
| 资产缩放 | **关闭** | strict OOF 通过但公榜降至 0.0039613753，已否决 |

该候选尚未 promotion，不得把它写成协作者主仓库的生产真值。完整记录见
[`research_history/local-calibration-and-feature-reselection-2026-08-17.md`](research_history/local-calibration-and-feature-reselection-2026-08-17.md)。

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
9. **融合参数存在小幅公榜余量。** `0.5/1.0` 到 `0.7/1.17` 提升约 1.83%，但属于后处理平台区，
   不再继续细搜小数点。
10. **每资产截面缩放不迁移。** fold 0 拟合、folds 1--4 冻结的 OOF 门禁通过，但公榜相对
    `0.7/1.17` 下降 2.69%；同类资产条件化参数停止公榜尝试。
11. **当前选列是单变量相关性过滤。** 市场森林仍复用按截面目标选择的 200 列，任务目标不对齐；
    下一结构实验应检查选列方法，而不是扩大特征数或继续调树参数。

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

### P3 — 市场森林独立轮数

- **状态**：`COMPLETED_KEEP_480`
- **结果**：1-seed screening 的 240/320/400/480 均未通过预注册门槛；生产容量 3-seed 确认中，
  480 相对 160 的 mean Peak `+2.91%`、4/5 折为正、drop-best `+2.05%`。
- **决策**：市场和截面森林都保留 480 轮。该轴已经结案，不再测试 240/320/400，也不新增
  `market_num_iteration` 覆盖。
- **证据**：`outputs/experiments/v3_market_round_scan_phasebal_prodwindow.*`、
  `outputs/experiments/v3_confirm_3s480_decision.*`。

### P4 — 任务对齐的特征重新筛选

- **状态**：`READY_FOR_LOCAL_SCREEN`
- **目标**：保持模型容量、融合参数和特征数量不变，只检查当前相关性选列是否错过稳定信号。
- **冻结候选**：
  1. `baseline_corr`：现行 pooled absolute correlation，XS/market 共用 200 列，history40 从中二选。
  2. `market_task_aligned`：XS 200 与 history40 不动；市场森林独立按逐 `time_id` 无权市场目标选择
     200 列。
  3. `xs_time_stable`：保持 200 列；按连续时间块的相关性中位数和方向一致率排序，市场与 history
     暂不改变。
  4. `history_lag_aligned`：主 200 列不动；history40 按 previous/difference/rolling 特征对未来截面
     目标的训练期时序相关性重选。
- **固定项**：`market_lambda=0.7`、`blend_weight=1.17`、scale=1.16、XS/market 480 轮、3 种子、
  feature count 200、history count 40；每次只改一个筛选环节。
- **门槛**：strict rolling OOF mean Peak > 0、至少 4/5 同号、drop-best > 0、`2ΔA>ΔB`；screening
  通过后才做 3-seed 确认，不因结果扩充候选矩阵，不自动生成公榜 CSV。
- **预注册文件**：`outputs/experiments/feature_reselection_plan_20260817.json`。

## 5. 已结案项目

| 日期 | 项目 | 结论 | 证据入口 |
|---|---|---|---|
| 8/13 | 市场块容量 | shrunk 赢 +0.77%，方向真实但接近饱和 | `experiments/ledger.csv` |
| 8/13 | 截面块容量 | shrunk −9.84%，保留 loose | `experiments/ledger.csv` |
| 8/13 | 轮数 960 | −5.20%，480 内部极值 | `experiments/ledger.csv` |
| 8/13 | 市场模型 + weighted XS | 公榜 +21.99%，进入生产架构 | `combo_market_weight.md` / ledger |
| 8/12 | Responder A/B/C | 可预测但不补 target 残差，停止多任务 NN | `responder_*.md` |
| 8/12 | V4 temporal / MLP | 仅 regime 保留扩展数据复验资格 | `temporal_multiscale_screen.md` 等 |
| 8/11 | 每资产 history40 | 公榜大幅提升，已进入生产 | `history_peak_lgbm_scoped.md` / ledger |
| 8/10 | phase-balanced Ridge | 公榜 +1.97%，已进入生产 Ridge | ledger |
| 8/8–10 | 验证框架、严格求解器、A/B 分解 | 已形成当前研究判定规则 | `research_history/validation-and-calibration.md` |
| 8/14 | 市场独立轮数 | 3-seed 确认保留 480；较少轮数不采用 | `v3_confirm_3s480_decision.md` |
| 8/17 | 融合参数平台 | fork 本地最佳为 0.7/1.17，公榜 0.00407075；停止细搜 | `experiments/ledger.csv` |
| 8/17 | 每资产截面缩放 | OOF 通过但公榜 -2.69%，否决并恢复无 adapter 基线 | `v3_asset_adapter_public_verdict_local_07_117.md` |

完整失败路径和结论翻转见 [`research_history/`](research_history/README.md)，不要从本表反推实验细节。

## 6. 更新规则

- 当前生产模型变化时，同时更新本节、promotion manifest 引用和 ledger；不要只改文字。
- 新任务进入行动面板时必须写状态、目标、动作、验收条件和证据。
- 结案任务移入“已结案项目”，长推导进入主题历史，ROADMAP 不保留整段实验日志。
- 每次更新在下表追加一行，不回写成没有日期的“现在”。

| 日期 | 更新 |
|---|---|
| 2026-08-13 | 文档体系重构；以已转正的 `mkt_shrunk` 和公榜 0.0039977510 重建当前状态。 |
| 2026-08-17 | 记录 fork 融合参数公榜梯子、否决 per-asset adapter、结案市场独立轮数，并预注册特征重筛。 |
