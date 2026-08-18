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
| 来源候选 | `outputs/candidates/v3_hybrid_slowfast/` | promotion manifest / Git 记录 |
| 公榜分数（**生产目录本身**） | **0.0041150085**（2026-08-18 转正后） | `experiments/ledger.csv` |
| slow/fast | **已转正**：window=2000 真实步，两个 scale 0.4496 / 1.2530 | `hybrid_meta.json` |
| 转正前基线 | 0.0039977510（2026-08-13 `mkt_shrunk`），差 **+2.93%** | `experiments/ledger.csv` |
| 公榜第一 | **0.0060**（2026-08-17，**用户报告**，非本地测量） | 用户 |
| 与第一的差距 | **+45.8%**（**IC 只差 +20.8%**）；旧记录 0.00520002 标 `SUPERSEDED` | 本表两行相除 |
| 截面块 | weighted LGBM，480 轮 × 3 种子，history40 | `hybrid_meta.json` |
| 市场块 | unweighted row-level LGBM，λ=0.5，480 轮 × 3 种子 | `hybrid_meta.json` |
| 截面混合 | `blend_weight=1.0`，即 LGBM 截面分量全替换 | `hybrid_meta.json` |
| 后处理 | `prediction_scale=1.16`，clip=0.5；**slow/fast 分离**（逐 asset 自身预测的因果滚动均值，K=2000 真实 time_id 步）| `hybrid_meta.json` |
| 训练采样 | `sample_modulo=5`，`phase_balanced` | `hybrid_meta.json` |
| promotion 校验 | 双后端最大差 `2.082e-16`，结构敏感性门禁通过 | `outputs/promotions/v3_hybrid_slowfast/promotion_manifest.json` |
| train/inference 一致性 | **`4.019e-09`**（两后端同值，2026-08-18 P0 重测，默认 `--partition-index 8 --n-time-ids 50`）。⚠️ 本表此前记的是 `8.111e-09`，同参数复测对不上 —— 两个数都远低于 1e-6 门槛，未去追因，以本次实测为准。⚠️ `check_consistency.py` 已改为 **slow/fast-aware** —— 训练端没有该后处理的概念，不补上会永久报红 9.4e-02 | `scripts/check_consistency.py` |

⚠️ `experiments/ledger.csv` 里 08-11、08-13 两行注释中的 `0.00520002` 是**当时**的榜首真值，
按 CLAUDE.md §7 不回写历史；以本表为当前值。

⚠️ **2026-08-18 订正**：本表此前写的「+50.1%（IC +22.5%）」是对**转正前**基线
0.0039977510 算的，slow/fast 转正后没重算。正确值是 0.0060/0.0041150085 = **+45.8%**。
**差距要看 IC 不看 score** —— 峰值处 `peak = IC²`（本项目自己记的 IC 0.06299 = √0.0039674
就是这个关系），所以 IC 差距只有 √1.458 − 1 = **+20.8%**。
若该差距来自一个**独立**新信号，其 IC = √(0.0060−0.0041150085) = **0.0434**，
单独打分 0.0018850 = 我们当前总分的 45.8% ⟹ 是「多一个同量级独立信号源」或
「同样信息提取效率高 20%」，**不是数量级差距**。
⚠️ 另一半：公榜每天 5 次 × 7/1–8/23 共 54 天 ⟹ 最多约 270 次评分，而本项目一共用了 31 次
（ledger 全部行）。在 R²≈0.005 的信号上做几百次公榜选型是在硬拟合公榜时期 ⟹
**0.0060 是其真实边际的上界**；私榜是 9/1–9/30 实盘前向，排名会重排。

**2026-08-18 转正记录**：候选 `v3_hybrid_slowfast` 与转正前生产的**唯一差别**是 meta 里 4 个
`slow_fast_*` 键 —— 6 片森林 + 冻结岭回归 hash 逐字节相同、**未重训**。
公榜两次独立确认：纯 CSV 后处理版 `0.0041150085`、走官方 runner 版 `0.0041150085`（逐位同分，
预注册预测「差异 ≤2e-08」兑现）。备份在 `outputs/promotions/backups/model_before_20260818_104355`。
⚠️ 迁移率 **0.51×**（本地 OOF +5.77% → 公榜 +2.93%），项目**首次本地高估**；私榜是另一时段，
但该改动只是把单一 1.16 重新分配为 0.4496/1.2530、未引入新信号源，下行接近打平。

当前生产文件 hash 与 promotion staging 一致；详见
[`research_history/delivery-and-incidents.md`](research_history/delivery-and-incidents.md)。

### 性能风险

- **⭐ 2026-08-18 P0 复测：两条路径都在钉死的 4 线程下重跑并落盘**（此前记的 5.15 / 10.44
  分钟没有产物、也没记线程数，而开发机是 32 核 ⟹ 那两个数不能替私榜的 4 核环境背书）。
  完整数字见 §4 的 P0 表；要点：

  | | LightGBM 主路径 | NumPy 兜底 |
  |---|---:|---:|
  | `predict_total` @ 4 线程 | **5.26 分钟** | **10.94 分钟**（2.08×）|
  | 此前无线程记录的旧值 | 5.15 分钟 | 10.44 分钟 |

  证据：`outputs/experiments/delivery_runtime_{lightgbm,numpy_fallback}_4t.{json,md}`。
- **兜底是单核 100%**（纯 numpy 树遍历不并行，实测 RSS 4.56 GB）⟹ **4 核评测机不会比
  32 核开发机更慢**，这一条以前没验过，是本轮把兜底风险从「未知」降到「已量化」的关键。
  两条路径全量 321 万行的 `max|pred|` 完全相同（0.4204497），与 staging 对拍 2.082e-16 一致。
  测法：用 shim 让 `import lightgbm` 抛错，走官方 runner —— 这就是评测机上 lightgbm 不可用时
  的真实路径，不是拿 `--backend` 假装（那个参数根本传不进官方 runner）。
  ⟹ 兜底仍是主要剩余风险（比主路径慢 2.08×），但比原估算宽裕且不随核数恶化。
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

- **状态**：`AWAITING_USER`（2026-08-18：动作 1–3 全部完成并落盘；只剩动作 4，只能由用户执行）
- **目标**：确认生产目录可被完整、可审计、在时限内打包和运行。
- **动作**：
  1. ✅ 完整 unittest（**64 passed / 18 subtests**）、双后端一致性、全量 runner 都已重跑。
  2. ✅ 记录当前 **`slowfast`**（原文写的 `mkt_shrunk` 已过期）的 model init / predict total /
     wall clock / 最大单步 / 非有限值 —— 见下表，已落盘 JSON。
  3. ✅ **4 核下两条路径都实测完**（此前 ROADMAP 记的 5.15 / 10.44 分钟没有落盘产物、
     也没记线程数，而开发机是 32 核 ⟹ 那两个数不能替 4 核背书）。
  4. ⏳ **由用户执行**：`scripts/make_submission.py` + zip 审计；至少留 3 次私榜机会。
- **4 核实测**（`scripts/verify_delivery_runtime.py`，走官方 runner 的 `run_loaded_model`，
  全程不写任何 CSV）：

  | | LightGBM 主路径 | NumPy 兜底 |
  |---|---:|---:|
  | `predict_total` | **5.26 分钟** | **10.94 分钟**（2.08×）|
  | wall clock | 6.20 分钟 | 11.81 分钟 |
  | model init | 0.36 s | 0.37 s |
  | 单步最大 / 平均 | 0.682 s / 1.47 ms | 0.658 s / 3.06 ms |
  | 行数 / 调用 | 3,217,458 / 214,538 | 同 |
  | 超时 / 非有限值 / 触 clip | 0 / 0 / 0 | 0 / 0 / 0 |
  | `max\|pred\|` | 0.4204497 | **0.4204497（与主路径相同）** |

  兜底是**单核 100%**（纯 numpy 树遍历不并行）⟹ 4 核评测机不会比 32 核开发机更慢，
  这一条以前没验过。两条路径的 8 个模型文件 sha256 均与 promotion manifest 逐字节一致。
- **⚠️ 本轮修掉的交付缺口**（都属于「审计通过但交出去的不是榜上那个模型」）：
  1. `PUBLIC_BASELINE` 里**没有 slow/fast 三个键** —— 而它们是公榜 0.0041150085 与
     0.0039977510 的**全部差别**。`main.py:222` 是 `PredictionTrail(...) if window else None`，
     缺键时 slow/fast 被**静默关掉、不报错** ⟹ 直接交出低 2.93% 的旧模型。已补进
     `PUBLIC_BASELINE` + `make_submission` + `promote_v3_candidate` + `audit_submission_zip` 四处。
  2. `audit_submission_zip.py` 只核 `lgbm_model_files` 在不在包里，**`market_model_files` 一个都不核**
     —— 市场森林是架构的一半（公榜 +21.99% 的来源）。已补。
  3. 审计不核 `main.py` 无条件 import 的 `features/lgbm_numpy/history` 三个模块。已补进 `REQUIRED`。
  - 4 个新回归用例钉住上述门禁；`--off-baseline` 是有意偏离的出口（留给 8/23 回补数据后重训）。
- **⚠️ 现存 `outputs/v3_hybrid_submission_20260813.zip` 是 slow/fast 转正前的旧模型**：
  旧审计八项全 PASS，加 `--expect-public-baseline` 才被拦下（三个 slow/fast 键 drift）。
  **8/31 不要拿它提交。**
- **验收条件**：生产 meta 与公榜模型一致；两后端对拍通过；全量行数正确；0 非有限值；耗时在
  主办方限制内；包内无训练代码和多余产物。
- **证据**：`outputs/experiments/delivery_runtime_lightgbm_4t.{json,md}`、
  `delivery_runtime_numpy_fallback_4t.{json,md}`、promotion manifest、
  `scripts/audit_submission_zip.py --expect-public-baseline`。
- **用户执行**：

  ```bash
  .venv/bin/python scripts/make_submission.py --strategy v3_hybrid
  .venv/bin/python scripts/audit_submission_zip.py \
      outputs/v3_hybrid_submission_<YYYYMMDD>.zip --expect-public-baseline
  ```

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
| 8/18 | **slow/fast 转正** | ⭐ 公榜两次独立确认 0.0041150085（CSV 后处理版 / 官方 runner 版逐位同分）；未重训、仅 4 个 meta 键；耗时 5.15 分钟反比前身更快 | ledger / `promotion_manifest.json` |
| 8/18 | per-asset **叠加**（blend） | `REJECTED`：全部为负，per-asset 臂（−3.91%、0/4）比 `shared` 对照（−1.19%）**还差**；corr 0.57~0.63 非低相关 ⟹ 替换与叠加两种用法都关闭 | `asset_blend_check.md` |
| 8/18 | per-asset 完整载荷 | `RESULT`→关闭：线性内异质性大（+95.8%、5/5 折、系数相关仅 +0.419），但 per-asset ridge 仍比生产 LGBM 截面块**低 50.5%** —— 树的 `asset_id` categorical 早已吃掉这块 | `asset_loading_diagnostic.md` |
| 8/18 | **选列宽度（树前面的线性筛子）** | `REJECTED`：323→200 的 \|corr\| 单变量筛子装在 LGBM 前面、123 列从未进过模型，且截断处落差仅 1.33% 无断崖。拆开测三个单变量臂：`xs323` **−1.00%**（2/5）、`mkt323` **+1.09%**（3/5，但只有检出下限的 0.70×、CI 跨 0）、`both323` +0.10%。⭐ 两效应精确可加（−1.00+1.09≈+0.10 实测 +0.10）⟹ 测的是真效应，只是太小。**回补数据后按原规格复验一次** | `feature_screen_1s160.md` |
| 8/18 | **`*_exact` OOF cache 失效** | `INCIDENT`：该 cache（08-14 11:12）早于脚本首次提交（08-15 11:18）⟹ 出自已不存在的代码版本，与当前输出差 `max\|Δ(market_ridge)\|=3.37e-05`（折均 peak 的 2.4%）。已用当前代码现跑替代基准并落盘。⚠️ P4 扩展窗臂曾以它配对，效应同量级，重开训练窗轴前必须现跑基准重测 | NOTES / `v3_production_oof_1s160_prodwindow_20260818.json` |
| 8/18 | **responder_00/02 的 Stage-C 空白格** | `REJECTED`：这两个短窗口候选此前被 `multi_member_family` **启发式**（而非证据）挡在 Stage C 外。缓存探针补测（不训练、2.2 秒）：最好一格 `pure_e/responder_00` +1.38%、3/4 折、CI 下界为正，但**去最好折为负、只有检出下限的 0.43×**、机制是 ΔB −2.01% 减方差 ⟹ 不过门禁。负控制与已测族 `responder_27`（−3.7%）校准通过 | `horizon_auxiliary_cache_probe.md` |
| 8/18 | **重建测试补测落盘** | `RESULT`：NOTES 的 0.207/0.818/0.835/0.732/**0.883** 在全量 1,322 万行上五格全部复现（±0.006）⟹ 下一行的关闭理由现在有产物支撑。⚠️ 口径是**带截距中心化 R²**（此前未记）；换成项目指标口径（无截距、分母 Σw·y²）同一设计从 0.84 掉到 0.16 | `responder_reconstruction.md` |
| 8/18 | responder 窗口图谱 | `RESULT`：测出窗口梯子 H=1/2/4/**5(target)**/7/10；但重建 R² 只 0.883、单步 u 不存在 ⟹ horizon 分解缺前提，不推进。⚠️ 重建那张表当时只写在 NOTES、无产物，已于同日补测确认 | `responder_window_atlas.md` / `responder_reconstruction.md` |
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
| 2026-08-18 | **`slow/fast` 转正为生产基线**（公榜 0.0041150085，+2.93%）；`check_consistency.py` 改为 slow/fast-aware 以免永久报红；P4 与 per-asset 两条轴结案。 |
| 2026-08-17 | 顺着时间平滑的否定结论找到 `slow/fast` 分离（OOF +5.77%、全分辨率核对 +5.93%）；本节第 11 条被同日测量证伪并改写。`market_lambda` 结案。仍未建候选、未改生产。 |
| 2026-08-17 | 公榜第一更新为 0.0060（用户报告）；market 侧同口径复测 `REJECTED`，六条路全关，我上一轮「下一个方向是 market」的判断标 `SUPERSEDED`；P3 同步为 `CLOSED_FAIL`，新增 P4 recency 预注册。 |
| 2026-08-18 | 核对 `NEXT_STEPS_horizon_auxiliary_oof_validation.md`：引用数字全对，但立项论证漏引 08-14 的同机制否决；发现 `responder_00/02` 从未进 Stage C（被单成员族启发式挡住）。缓存探针补测 ⟹ `REJECTED`；重建测试补测落盘 ⟹ NOTES 数字确认、口径澄清为中心化 R²。生产目录与模型身份未改动。 |
| 2026-08-18 | 选列宽度轴 `REJECTED`（三个单变量臂全不过，两效应精确可加）；发现 `*_exact` cache 出自已不存在的代码版本并落盘替代基准；`RUNBOOK_8_23.md` 与 `public_replay.py`（21 份 CSV 全归属）就位，8/23 当天无需再做设计决策；与公榜第一的差距订正为 +45.8%（IC +20.8%）。 |
| 2026-08-18 | **P0 推进到 `AWAITING_USER`**：4 核下 LightGBM（5.26 分钟）与 NumPy 兜底（10.94 分钟）全量实测并落盘，兜底确认为单核绑定 ⟹ 不随核数恶化。修掉三道交付门禁都不认识 slow/fast 的缺口（丢键会静默交出低 2.93% 的旧模型），补 4 个回归用例；现存 `v3_hybrid_submission_20260813.zip` 已被新审计判为旧模型。只剩用户执行打包 + zip 审计。 |
