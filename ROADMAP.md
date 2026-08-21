# ROADMAP.md — 当前状态与行动面板

> **状态日期：2026-08-19。** 本文件只描述当前有效状态和未来动作。完整探索过程见
> [`NOTES.md`](NOTES.md) 与 [`research_history/`](research_history/README.md)。生产真值以模型产物、
> promotion manifest 和 [`experiments/ledger.csv`](experiments/ledger.csv) 为准。

## 1. 当前目标与节点

- **8/23**：公榜停止更新并等待主办方标签/数据回补；收到更新包后先审计，不先训练。
- **8/31**：私榜策略文件提交截止。⚠️ 主办方原文是「共可以进行最多 `10` 次策略文件提交，
  **最终采用最新提交版本**」（`docs/competition_description.md:201`）——
  **不是 best-of-10**。⟹ 最后一次上传的那份就是最终答案；10 次是上传失败的重试余量，
  **不是**「交一组分散候选、由主办方挑最好」的额度。高方差候选在这里**没有期权价值**。
- **当前主目标**：守住已转正的 `v3_hybrid_slowfast`，完成交付与数据版本验证；8/23 若数据更新
  真实存在，再按预注册矩阵重训和重标定。
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
    与 OOF 吻合 ⟹ 采样格口径没有骗人。该状态已经 `SUPERSEDED`：08-17 公榜 +2.93%，
    08-18 官方推理路径同分确认并转正，现为生产结构。机制是 `2ΔA > ΔB`
    （ΔA +8.07% / ΔB +11.21%），不是纯减方差。
    证据：`v3_slow_variance_3s480.md`、`v3_fullres_slow_probe_summary.md`。
13. **当前数据上的新静态/时序表示没有找到榜首级信号。** 一层 rank/tail、change-rank、lag3/10、
    multi-horizon change、volatility、trend、market set summary 和 asset panel 均未过门禁；唯一均值略正
    的 `lag3+lag10` 只有 +0.38%、3/5 折且 drop-best 为负，不升级。
14. **`phase_id` 只有弱筛选信号，不是候选。** 1s×160 pooled Peak 约 +1.1%，逐折仅 3/5 正；
    未达到 +3%/4-of-5/drop-best 门槛，不跑 3s×480。`periodic` 与 `phase_balanced` 的现有结果因
    validation 行组成不同，不能作严格配对裁决；生产继续保持 `phase_balanced`。
15. **full-resolution 本地资源路径已打通，但正式同跨度 OOF 尚不可运行。** 固定生产 200 特征、
    1,182,292 train rows、300,000 valid rows 的 160 轮双森林顺序 smoke 完成，max RSS≈11.5GB；
    它明确标记 `oof_valid=false`。保持生产真实跨度需约 5.92m train rows，现有 30GB/无 swap 机器
    连续触发 24.8–26.1GB cgroup OOM；正式实验需 chunked design writer 或 64GB+ CPU 服务器。

## 4. 行动面板

### P0 — 私榜交付闭环 —— `CLOSED`（2026-08-19）

- **状态**：`CLOSED`。动作 1–3 于 08-18 完成并落盘，动作 4 于 08-19 由用户执行完毕。
- **结案证据**：`outputs/experiments/submission_audit_v3_hybrid_20260819.json` ——
  `outputs/v3_hybrid_submission_20260819.zip`（sha256 `3f1da29cad36d89b7…`，5,819,904 B）
  审计 **`passed: true`**、`public_baseline_drift: []`、`unexpected_modules: []`、
  `missing: []`，包内恰好 4 个执行模块 + 8 个模型文件。
  ⟹ 「模型身份」与「包内容身份」两道门同时通过，且**有落盘证据**。
- ⚠️ 盘上现有三个 v3 提交包，只有 **20260819** 那份是当前生产 + 通过全部门禁的；
  20260813 是 slow/fast 转正**前**的旧模型，20260818 多带一个研究模块 `temporal.py`。
  8/31 上传前用文件名日期核对，并重跑一次带 `--expect-public-baseline --output` 的审计。
- 以下为 08-18 的原始记录，保留备查：
- **目标**：确认生产目录可被完整、可审计、在时限内打包和运行。
- **动作**：
  1. ✅ 完整 unittest（**73 passed / 18 subtests**）、双后端一致性、全量 runner 都已重跑。
  2. ✅ 记录当前 **`slowfast`**（原文写的 `mkt_shrunk` 已过期）的 model init / predict total /
     wall clock / 最大单步 / 非有限值 —— 见下表，已落盘 JSON。
  3. ✅ **4 核下两条路径都实测完**（此前 ROADMAP 记的 5.15 / 10.44 分钟没有落盘产物、
     也没记线程数，而开发机是 32 核 ⟹ 那两个数不能替 4 核背书）。
  4. ⏳ **由用户执行**：`scripts/make_submission.py` + zip 审计；至少留 3 次私榜机会。
     ⚠️ **2026-08-19 复查**：`outputs/v3_hybrid_submission_20260818.zip` 已经存在（08-18 16:32），
     8 个模型文件与 `main/features/history/lgbm_numpy` 四个执行模块与生产**逐字节相同**
     （sha256 实测），4 个 slow_fast 键齐全 ⟹ **模型身份是对的**。两个缺口：
     (a) 盘上**没有任何审计记录** —— `audit_submission_zip.py` 默认只打印，不带 `--output` 就不落盘；
     (b) 它多装了一个 `temporal.py`（研究模块，不在 `main.py` 的 import 闭包里）。
     新审计对它的判定是 `passed=false`，**且失败项只有 `no_unexpected_modules`**、
     `public_baseline_drift` 为空 —— 正好印证上面两句。
     ⟹ 重新打包一次并带 `--output` 落盘审计即可结案，**不需要重测耗时**
     （执行路径上的四个模块一字节未动）。
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
- **⚠️ slow/fast 转正前的旧模型包已于 08-19 改名封存**为
  `outputs/v3_hybrid_submission_20260813.PRE-SLOWFAST.zip`（旧审计八项全 PASS，
  加 `--expect-public-baseline` 才被拦下：三个 slow/fast 键 drift）。
  改名就是防呆措施本身——**不要改回去**。8/31 交的是 `..._20260819.zip`。
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
  2. 用 `scripts/audit_data_release.py` 对比 `outputs/data_audits/data_release_20260818.json`。
  3. 若 train split 未变化，停止固定结构重训；不要为了日期变化重跑模型。
  4. 若 train split 变化，先在回补标签上复算历史提交和真实指标，修正本地尺子。
- **验收条件**：审计 JSON 明确 added/removed/modified；数据 hash 可追溯；任何训练动作都有审计结果
  作为输入。
- **⭐ 2026-08-19 已端到端演练**（用当前数据，全 file hash，未用 `--no-file-hash`）：
  `outputs/data_audits/data_release_20260819_rehearsal.json` 报 `comparison.changed = false`、
  `train.row_delta = 0`，`retrain_extended.py` 随即以
  「audit does not prove a changed training split; fixed retraining is blocked」**明确拒绝**。
  ⟹ 工具链与第一道闸门都验过，8/23 当天只需换 `--output` 日期。

### P2 — 扩展数据固定结构重训与联合重标定

- **状态**：`BLOCKED_BY_P1`
- **目标**：量化纯数据收益，再比较预注册的相关参数组合。
- **动作**（顺序不可颠倒）：
  1. 使用 `scripts/retrain_extended.py` 先 dry-run；只写候选目录，不写生产目录。
     ⚠️ dry-run 输出里现在有一段 `production_structure`，**先核它再 `--execute`**。
  2. **用当前代码现跑配对基准**。⚠️ 早于 `v3_production_oof.py` 首次提交（08-15 11:18）的
     缓存都出自已不存在的代码版本。**两份已隔离**：`..._phasebal_prodwindow_exact.npz`
     （08-18 `INCIDENT`，差 3.37e-05，与被测效应同量级）与 `..._phasebal_prodwindow.npz`
     （08-20 复查新增，签名完全一致但当时漏点名）。隔离由
     `src/oof_cache.assert_reproducible_cache` 强制，指过去会当场报错。
     唯一确认由当前代码产出的是 `..._1s160_prodwindow_20260818.npz`。
  3. **recency 阶梯**（⭐ 2026-08-19 新增预注册臂，见下）。
  4. 固定当前结构重训，作为“只增加数据”的基线。
  5. 执行 `outputs/experiments/joint_recalibration_plan.json` 中冻结的 Ridge 12 格和 LGBM 9 格。
  6. 原规格复验 V4-R regime；不重开 T1/T2/T3 和 MLP 搜索。
- **验收条件**：训练/验证严格时序；不读结果扩格；报告包含配对增量、同号折数、A/B 和推理成本。

#### P2-R — recency 阶梯（预注册，`BLOCKED_BY_P1`）

⚠️ **P4 的「数据已饱和」不覆盖这一条。** 两者是不同的轴：

```text
P4 已测（CLOSED）  滑动窗 78,960 → 扩展窗，往训练段**前端**加旧数据   = volume 轴
                   结果：+1.08%、2/5 折、CI 跨 0 ⟹ 测不出效应
8/23 回补          紧邻私榜期的新标签，往训练段**后端**加新数据       = recency 轴
                   此前**根本没法测**（没有更近的标签），不是被否决
```

回补后训练期延长约 **+24.5% 的 time_id**，而且是**最靠近 9 月实盘期**的那一段；
私榜是 9/1–9/30 前向实盘 ⟹ recency 与 volume 的先验完全不同。

- **设计（跑前钉死）**：fold 版图固定、只往后端延训练段（沿用 `v3_production_oof.py`
  的 `--train-truncate` 那套「固定 fold 版图、只动训练段边界」的做法，不改验证段）；
  与动作 2 的现跑基准天然配对。
- **门槛**：沿用 RUNBOOK D2 的六道 + 检出下限（折均 > 0、≥4/5 折、去最好折 > 0、
  相对 ≥ 3%、`2ΔA>ΔB`、配对 CI 下界 > 0）。
  ⚠️ 1s160/5 折的检出下限实测是基准 peak 的 **6.1%**，3s480 是 **8.7%** —— 3% 那档没有牙，
  所以「不过门槛」在这里的正确读法是**测不出来**，不是「没有效果」。
- **⚠️ 这一条不需要公榜裁决**：8/23 后公榜已停更，且本地尺子要先经 D0.3 校准；
  采纳与否只能靠 OOF + 机制，因此门槛只能守严不能放松。
- **顺序**：审计 → 现跑基准 → recency 阶梯 → **才轮到**②类本地网格（alpha/轮数/history 宽度）。
  ②类是回补标签的首要用途（`NOTES.md §2` 第 3 条），但要在 recency 定了之后做，
  否则两个轴混在一起无法归因。

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

### P5 — 当前数据验证与 full-resolution 路线

- **状态**：`RESOURCE_SMOKE_PASS / FORMAL_OOF_DEFERRED`（2026-08-19）。
- **当前数据验证顺序**：
  1. 完整 unittest（当前基线 **73 passed / 18 subtests**）；
  2. 用 2026-08-18 audit 做 metadata 对比；8/23 到包后再做完整 hash；
  3. 重跑 v3 LightGBM/NumPy train-inference consistency；
  4. 必要时重跑 4 核官方 runner，不因实验代码变化自动替换生产。
- **full-resolution 已验证**：`v3_fullres_resource_smoke_160.json`，固定生产 200 特征/statistics，
  跳过 Ridge 和 OOF score，XS/market 顺序训练，160 轮成功、max RSS≈11.5GB。
- **边界**：该 smoke 的真实训练窗只有 78,960 time_ids，不能用于判断 full-resolution 是否提分；
  同生产跨度约 394,800 real time_ids / 5.92m rows，本地正式 OOF 暂停。
- **重新启动条件**：完成真正的 chunked design writer，或把正式 1s×160 OOF 放到 64GB+ CPU
  服务器；任何 fixed-production smoke 报告必须保留 `oof_valid=false`，不得进入候选排名。
- **生产决策**：`v3_hybrid_slowfast` 原样保持，不根据 phase/periodic/fullres smoke 改 meta。

### P7 — slow/fast 抛物线顶点标定 —— `CLOSED`（2026-08-19，顶点已测出）

- **状态**：`CLOSED`。第三点已交（`S2 = 0.0039374211`），顶点闭式解出，**判定不改交付**。
- **结果**：

  ```text
  完整性检查   a = −1.474225e−04 < 0                          PASS
  三点联立     t* = 0.897692     Score(t*) = 0.0041165516
  ⭐ 当前生产点 t=1 已处在这条线峰值的 99.9625%
  ⭐ slow/fast 捕获了线上总可得增益的 98.70%
  挪到真顶点   +0.0375%（绝对 +1.543e−06）
  半步收缩后   +1.157e−06  <  预注册的 1e−05「不可辨别」线   ⟹ 不改交付
  ```

- ⭐ **这条轴现在是「测出了顶点位置」，不是「没测」。** 私榜维持 `t=1`
  （即当前生产的 `slow_fast_*_relative = 0.387610 / 1.080181`），**生产目录不动**。
- ⚠️ `S2 < S0`（0.0039374 < 0.0039978）⟹ t=2 比**完全不做 slow/fast** 还差，与 `t*≈0.9` 自洽。
- ⭐ **顺带验证了一个方法**：08-17 那次凭 OOF **相对模式**搬来的 `(0.4496, 1.2530)`
  （「只搬相对模式、保留公榜标定的绝对水平」），事后证明落在最优点的 **0.04% 以内**。
  这个做法本身被独立确认了一次。
- **重新开放条件**：模型本身改变（森林重训 ⟹ slow/fast 的最优配比要重解），
  或 8/23 回补数据后按原规格复验。**不得**在同一模型上继续搜第二个收缩系数或换第三点位置。
- **证据**：`outputs/experiments/slow_fast_vertex_solution.{json,md}`（预注册
  `slow_fast_line_geometry.json` 的 sha256 记在里面 ⟹ 判据先于结果可核验）、
  `experiments/ledger.csv` 2026-08-19 行。
- 以下为立项时的预注册记录，保留备查：
- **机制**：沿 `c(t) = (1−t)·(1.16,1.16) + t·(0.4496,1.2530)`，`pred(t)` 逐行线性 ⟹
  `Score(t)` 是 t 的**精确二次式**，二次项系数 `a = −⟨d,d⟩_w/D` **恒为负**（构造决定）。
  已有 t=0（0.0039977510）与 t=1（0.0041150085）两点，**再取一点即闭式解出顶点**。
- **第三点**：`t=2`，`c=(−0.2608, 1.3460)`，meta relative `(−0.224828, 1.160345)`。
  ⭐ **限幅已实测**（不是估算）：`max|pred| = 0.445934`、**触限 0 行**；
  二分出的 clip 边界是 **t ≈ 2.6968**。
- **预注册**（`outputs/experiments/slow_fast_line_geometry.{json,md}`，**先于提交落盘**）：
  - 完整性：`S2 < 2·S1 − S0 = 0.0042322660`，否则 `a ≥ 0` ⟹ **停下来查触限/查模型身份**；
  - 增益闭式 `gain = (S1−S0)(t*−1)²/(2t*−1)`，只取决于 t*；
  - 私榜**半步收缩** `c_used = c1 + 0.5(c*−c1)`（恰好拿到理论增益的 75%）；
  - 采纳线沿用 08-17 asset adapter 的「|Δ| < 1e-5 视为不可辨别」⟹
    **t\* < 1.470（约 S2 < 0.0041113）就不改交付**。
- **⚠️ 期望值要诚实**：典型情形只有 **+0.0%~+0.9%**；+2% 以上只在 S2 贴近 `a→0` 边界时出现，
  而那时 t\* 已越过 clip 边界、拿不到全部。花它的理由是**公榜名额 8/23 作废、不用即归零**，
  且三点同源（ρ≈0.99）⟹ 顶点估计很精确 —— **不是**因为它能翻盘。
- **用户执行**：

  ```bash
  .venv/bin/python experiments/slow_fast_csv.py \
      --scale-slow -0.2608 --scale-fast 1.3460 \
      --output outputs/submission_slowfast_t2.csv
  # 提交 → 回填 ledger → 解顶点
  .venv/bin/python experiments/slow_fast_vertex.py --s2 <公榜分>
  ```

- **采纳路径**：**不是**改 CSV 交私榜。系数必须走
  `scripts/promote_v3_candidate.py --slow-fast-slow-relative/--slow-fast-fast-relative`
  写进候选 meta，再过完整套转正门禁（CLAUDE.md §6）。
- **不做**：5 点平面顶点。先看 S2 再议（当前 2 点 + t=2 已是 3 个方程，平面 5 参数还差 2 个
  非共线点 ⟹ 线优先不浪费额度）。

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
  ③ 特征选择不再沿用为线性/树挑的 `|corr(feature, e)|` top-200。
  曲线已经把「预算」这条排除掉了 —— v5 若还只是加算力，可以直接不做。
- **适用范围**：本阶梯否掉的是 **sklearn `MLPRegressor` + 生产特征表示 + 这套预算**这个配方，
  **不是**「NN 这个模型族」。
- **重新开放条件**：上面三条范围项**至少改掉一条**后按原规格复验；或 8/23 回补数据后基准变化。
  **不得**只加 epoch / 只加宽网络重跑。
- **证据**：`outputs/experiments/nn_capacity_ladder.{json,md}`（预注册
  `nn_capacity_ladder_plan.json` 的 sha256 记在里面）、`multitask_mlp_e{12,50,150,400}.{json,md}`。
  ⭐ 12 档对 08-19 锚点的偏差 **0.00e+00**（逐位复现）⟹ 环境与数据一致，曲线可解读。

### P10 — 密封期尺子 —— `PREREGISTERED`（2026-08-20，判据已落盘，等 8/23）

- **状态**：`PREREGISTERED / BLOCKED_UNTIL_DATA_REFRESH`。代码与判据全部就位并干跑验证过；
  8/23 标签一到就能用。
- **问的问题**：8/23 回补的**就是公榜期的标签**（`docs/data_description.md:172`
  「公榜截止后会发布标签回补数据，该部分数据将作为扩展训练数据使用」；实测 `data/test/*.parquet`
  326 列**无 weight/target/responder**，`data/train/*.parquet` 375 列带全部三样）。
  ⟹ 那段数据**只能用一次**：当训练数据，或当干净测试集。
- **为什么这是个真取舍**：

  | | 行数 | 性质 |
  |---|---:|---|
  | 公榜期 `888,480–1,105,919` | **3,217,458** | 真出样本、最靠近 9 月实盘 |
  | 现有 OOF 5 折验证合计 | 约 150 万 | 检出下限 6.1%（1s160）/ 8.7%（3s480）|

  ⟹ 公榜期评估行数约是现有 OOF 的 **2.1×**。而正是那个 6.1~8.7% 的下限把 `mkt323` (+1.09%)、
  `phase_id` (+1.1%)、`responder_00` (+1.38%)、`lag3+lag10` (+0.38%)、扩展窗 (+1.08%) 全判成
  「测不出来」。**若把回补数据全部拿去训练，8/23–8/31 最可能的结局是「重训了但测不出有没有用，
  于是按 D6 维持现状」** —— 那是一个没有信息量的结局。
- **切分**（2026-08-20 由用户定，已落盘 `sealed_period_plan.json`，sha256 记进每份裁决产物）：

  ```text
  密封测试集   1,045,920 – 1,105,919   60,000 real time_id   实测 856,319 行
    4 块 × 15,000                       每块实测 203,176 / 224,970 / 203,264 / 224,909 行
  embargo      30 real time_id（= OOF 的 6 采样步 × sample_modulo 5）
  决策期训练   ≤ 1,045,889（比现在多 157,410 个 time_id，+17.7%）
  最终交付     决定拍完后用 0–1,105,919 全量（+24.5%）重训一次
  ```

  ⚠️ 每块行数不等（203k vs 225k，摆动约 10%，且交替出现）—— 逐块 peak 是各自的比值不受影响，
  但读 pooled 数时要记得块权重不均。
- **门禁**：RUNBOOK D2 六道在 4 块上的映射（`≥4/5 折` → **`≥3/4 块`**），加配对 block bootstrap
  （每块 25 chunk、重抽 2000 次、seed 2026、95%）。⚠️ 第七道「超过检出下限」在标定前判 `None`
  **而不是自动通过** —— 整体判 `PENDING_CALIBRATION`，回归用例钉住这一点。
- **读数口径订正**：设计初稿写的是「`raw = pred / prediction_scale` 反解」，**在 slow/fast 下是错的**
  （最终值是 `clip(s_slow·slow + s_fast·fast)`，两个分量各有 scale）。正确做法更简单：
  `peak = A²/B` 对全局缩放严格不变 ⟹ **断言触限 0 行后直接算 peak** 就与拿 raw 算逐位等价，
  不需要那步除法。有回归用例把这条不变性钉死。
- **Tier 1（零重训成本，盘上现成，每个约 6 分钟）**：`production_slowfast` / `mkt_shrunk` /
  `mktwe` / `asset_adapter` / `r960` / `xs_shrunk` —— 六个**已知公榜真值**的候选。
  这一层的产出**不是找收益，是标定这把尺子的检出下限**；其中 `asset_adapter` 还顺带问一句
  「OOF 说 +1.99%、公榜说 −0.17%，密封期站哪边」。
- **Tier 2（每个一次 3s480 重训，排在 recency 之后）**：`mkt323`、V4-R regime、`phase_id`、
  `lag3+lag10`、`responder_00`。⚠️ **只有 Tier 1 标定出的下限低于各自点估计时才开跑。**
- **干跑证据**（2026-08-20，用当前数据、无标签）：
  - 推理链路：官方 runner 全量 test **3,217,458 行**、`status=ok`、0 超时、
    `max|pred|=0.420450`、**触限 0 行** —— 与 ledger 08-18 走官方 runner 那次
    （`max|pred|=0.4204`、0 行触 clip）和 P0 的 `0.4204497` 对上 ⟹ 推理路径确认无误。
  - 判据链路：`--synthetic-labels` 走通 join / 分块 / bootstrap / 判据，产物强制
    `adjudication_valid=false`（沿用 full-resolution smoke 的 `oof_valid=false` 先例），
    self-vs-self 得 +0.00% / 0-of-4 / FAIL —— 空改动**不过**门禁，符合预期。
- **⚠️ 不落提交格式 CSV**：官方 runner 必须给 `output_path`，脚本把它指向临时目录，读完随目录
  销毁；预测落 `outputs/cache/sealed_pred_<label>.npz`。两个理由：CLAUDE.md §1.4，以及盘上不留
  看起来像提交文件、8/31 可能被误传的东西（P0 已经因为盘上三个 zip 写过一次警告）。
- **诚实预期**：这套东西**不会**补上离榜首 IC +20.8% 的差距 —— Tier 2 全部点估计 ≤ +1.4%。
  它买的是让 8/23–8/31 那个最重要的决定变成一个**有数**的决定。⚠️ 最大不确定性是**这把尺子
  自己的检出下限仍是未知数**；Tier 1 那六枪就是去测它，若测出来也在 6% 以上，Tier 2 直接不跑
  —— 那也是一个干净的结论，只花 40 分钟。
- **证据**：`outputs/experiments/sealed_period_plan.{json,md}`、`seal_dryrun_gates.{json,md}`、
  `experiments/sealed_period_eval.py`、`tests/test_sealed_period_eval.py`（11 个用例）。
- **RUNBOOK 接口**：新增 D0.4（Tier 1 标定）与 D4.5（最终 100% 重训 + 三层回退）。

### P6 — 磁盘清理（**清单由 AI 出，删除/改名由用户执行**，CLAUDE.md §1.1）

- **状态**：`CLOSED`（2026-08-20，用户已执行；本条为执行后复查）。
- **复查实测**（2026-08-20）：`outputs` **21G → 3.5G**、`/` 空闲 **64G → 81G**。三项可回收项
  （`fullres_rows_mod1` / `mt_aggregates.npz` / `nvim.log`）已不在盘上；两项改名封存已生效
  （`..._exact.STALE-DO-NOT-USE.npz` 67M、`v3_hybrid_submission_20260813.PRE-SLOWFAST.zip` 5.6M）。
  ⭐ **「绝对不要删」三项全部完好**：`outputs/submission_*.csv` **22 份 / 1.4G**（D0.3 的全部原料）、
  `outputs/promotions/` 57M（含 `backups/`）、`data_release_20260818.json`；
  8/31 要交的 `v3_hybrid_submission_20260819.zip` 仍是 **5,819,904 B**，与 P0 结案记录逐字节一致。
  ⟹ 8/23 回补（+24.5%，约 +5G）加 D1/D2 新缓存的空间已备好，**这一条不再是 8/23 的前置阻塞**。
- **为什么当时提**：`/` 剩 **64G**，其中 `data` 20G、`outputs` 21G。8/23 回补数据约 +24.5% time_id，
  加上 D1 重训与 D2 现跑基准的新缓存，先腾空间比事到临头腾安全。
- **可回收，约 18.2G**：

  | 路径 | 体积 | 依据 |
  |---|---:|---|
  | `outputs/cache/fullres_rows_mod1/` | **17G** | 08-19 full-resolution 的 disk-backed memmap。P5 已判 `FORMAL_OOF_DEFERRED`，不在关键路径；可由 `experiments/v3_fullres_resource_smoke.py --cache` 重建 |
  | `outputs/cache/mt_aggregates.npz` | 1.1G | `.gitignore` 注明可由 `experiments/mt_lagged.py` 重新生成 |
  | `nvim.log` | 0 B | 空文件 |

- **建议改名而不是删除**（留证 + 消除误取风险）：

  | 路径 | 体积 | 依据 |
  |---|---:|---|
  | `outputs/cache/v3_production_oof_phasebal_prodwindow_exact.npz` | 67M | **地雷**：出自已不存在的代码版本（08-18 `INCIDENT`），与当前代码差 `max\|Δ(market_ridge)\|=3.37e-05`（折均 peak 的 2.4%）。RUNBOOK D2 明令不得用作配对基准 ⟹ 加 `.STALE-DO-NOT-USE` 后缀比留着原名安全 |
  | `outputs/v3_hybrid_submission_20260813.zip` + 同名目录 | 19M | slow/fast 转正**前**的旧模型，RUNBOOK 已两次警告 8/31 别拿它提交。加 `.PRE-SLOWFAST` 后缀即可 |

- **⚠️ 绝对不要删**：
  - `outputs/submission_*.csv`（**21 份，1.3G**）—— 它们是 RUNBOOK **D0.3「修尺子」的全部原料**
    （逐行预测值，覆盖公榜 0.0015→0.0041 的整个 v3 时代）。删掉 8/23 就没有本地尺子可校准，
    而 8/23 之后公榜停更、没有第二把尺子。**这一条是本清单里最重要的一句。**
  - `outputs/candidates/v3_hybrid_slowfast/`、`outputs/promotions/`（含 `backups/`）—— 转正与回滚路径。
  - `outputs/data_audits/data_release_20260818.json` —— D0.2 的比较基线。

## 5. 已结案项目

| 日期 | 项目 | 结论 | 证据入口 |
|---|---|---|---|
| 8/21 | ⭐ **长窗 w512 确认档（3s480）** | **`PASS_BUT_BELOW_DETECTION_FLOOR`**：pooled **+7.77%**、**5/5 折**、去最好折 +6.49%、配对 CI 下界 **+4.18%**，五道门槛全过；但只有 3s480 检出下限的 **0.89×** ⟹ **方向可信、幅度测不出**。筛选→确认迁移率 1.14×（未衰减）；基准更强而增益仍在；线性对拍 10/10 逐位相同。⚠️ 只测截面块（占分 58.8%），全模型粗估 +2%~+5%；探针的 cumsum 口径不得进生产。⟹ 只够作为花一次公榜额度的理由，**不构成晋级依据**；生产未动 | `long_window_confirm.md` / `..._plan.json` |
| 8/21 | ⭐ **长窗列数阶梯** | **筛选档 PASS**（1s160，只截面块）：`w512` pooled **+6.80%**、**5/5 折**、去最好折 +5.84%、bootstrap CI 下界 +3.87%、**超检出下限 1.12×**，五道全过。⭐⭐ 预注册机制预测兑现 —— **同一评价器(线性)上 240 列 +0.69% → 80 列 +5.75%** ⟹ 信号一直在，是被 240 列的估计代价淹掉的。`w512` 是唯一两个评价器都认的臂。⚠️ 仅筛选档、迁移率历史区间 0.51×~2.3×、只超下限 1.12×、三臂多重比较 ⟹ **必须先过 3s480 确认**，生产未动 | `long_window_ladder.md` / `..._plan.json` |
| 8/21 | **选列准则探针** | `REJECTED`：三臂五门槛全败。`lasso200` −1.29%（2/5）、`hist_lag1` **−4.38%（0/5）**、`hist_roll5` **−10.39%（0/5）**。⭐ 与 base 分歧越大掉得越多（单调）⟹ 不是碰巧，是**当期准则挑的那 40 列确实更好**。⭐⭐ 选列轴至此**三向封死**：更宽（323 全给）打平、history 更宽（c80 超集）公榜 0.00%、换准则全负 ⟹ 对 `feature_fraction=0.7` 的树，前置筛子的选择质量**不是绑定约束** | `selection_criterion_probe.md` / `..._plan.json` |
| 8/21 | **长历史窗探针** | `REJECTED`，但**给出了定价**：窗口 {64,512,4096} 的长窗块 **ΔA=+3.36%（信号是真的）而 ΔB=+15.40%**，`2ΔA<ΔB` ⟹ 估计方差是所拿信号的 4.6 倍。pooled +0.69%、4/5 正折、去最好折 −0.49%、bootstrap CI 下界 −5.96%。⚠️ **判据 4 无效并已收回**：逐折 `A比/√(B比) ≡ IC比` ⟹ ΔA/ΔB 混着两臂解的共同尺度（fold 0 的 A 比 0.9555，A 其实是降的），且 `2ΔA>ΔB` 是两分量判别式、不适用于嵌套模型 —— **P8 栽过的坑本次重犯**。据它推出的「完美提取器上界 +3.36% IC」作废；本实验**未**给出长窗信号的有效定价。裁决不受影响（判据 1/3/5 独立失败，IC/peak 尺度不变）。⚠️ fold 4 内层 alpha 选到梯底致 −19.49%，但去掉它 pooled 仍只有 +2.77%，结论稳健 | `long_history_probe.md` / `..._plan.json` |
| 8/21 | **函数类探针** | `REJECTED`：换函数类换不出增量。RFF-岭回归在**与生产截面块逐列相同**的 361 列上拿到 `r=0.798`（P9 的 sklearn MLP 只有 0.54 ⟹ 「NN 天花板 28.8%」是配方产物不是能力事实），但 `ρ` 同步涨到 0.702 ⟹ oracle 集成只有 **+0.91%**，门槛 +3%。线性对照 0.611、核 0.798、树 1.000 三者互相 ρ≈0.6~0.7 ⟹ **三种函数类在读同一个东西，这 361 列已榨干**。⭐ 剩下的唯一方向是**换输入**（更长历史窗），不是换模型。⚠️ 预注册缺陷：alpha 梯子上界 1e-1 太低，10 次拟合 9 次选到边界 | `function_class_probe.md` / `..._plan.json` |
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
| 2026-08-18 | 核对当时的本地工作笔记 `NEXT_STEPS_horizon_auxiliary_oof_validation.md`（未入库、现已不在盘上；结论证据见 `outputs/experiments/horizon_auxiliary_cache_probe.{json,md}`）：引用数字全对，但立项论证漏引 08-14 的同机制否决；发现 `responder_00/02` 从未进 Stage C（被单成员族启发式挡住）。缓存探针补测 ⟹ `REJECTED`；重建测试补测落盘 ⟹ NOTES 数字确认、口径澄清为中心化 R²。生产目录与模型身份未改动。 |
| 2026-08-18 | 选列宽度轴 `REJECTED`（三个单变量臂全不过，两效应精确可加）；发现 `*_exact` cache 出自已不存在的代码版本并落盘替代基准；`RUNBOOK_8_23.md` 与 `public_replay.py`（21 份 CSV 全归属）就位，8/23 当天无需再做设计决策；与公榜第一的差距订正为 +45.8%（IC +20.8%）。 |
| 2026-08-18 | **P0 推进到 `AWAITING_USER`**：4 核下 LightGBM（5.26 分钟）与 NumPy 兜底（10.94 分钟）全量实测并落盘，兜底确认为单核绑定 ⟹ 不随核数恶化。修掉三道交付门禁都不认识 slow/fast 的缺口（丢键会静默交出低 2.93% 的旧模型），补 4 个回归用例；现存 `v3_hybrid_submission_20260813.zip` 已被新审计判为旧模型。只剩用户执行打包 + zip 审计。 |
| 2026-08-19 | **当前数据剩余结构搜索与 full-resolution 资源验证收官**：rank/change/lag/volatility/trend、market set/panel 全未过门禁；phase_id 仅弱 +1.1%、3/5，不升级；periodic 比较因 validation 组成不同不作裁决。修复 disk-backed loader、fixed history 映射和后台 systemd 监控；短跨度 fixed-200 双森林 160 轮 smoke 成功（max RSS≈11.5GB，`oof_valid=false`），同跨度正式 OOF 因 5.92m rows 暂缓。全量测试 73 passed / 18 subtests。 |
| 2026-08-19 | 补三道交付/重训缺口（本轮由仓库结构复查发现，均**不在**原行动面板上）：① `retrain_extended.py` 的「固定结构重训」计划缺 `--weighted-cross-section`/`--market-model`/`--market-spec`/`--market-min-data-scale`，跑出来的是 08-11 架构（比生产低 21.99%），现改为从生产 `hybrid_meta.json` 派生并与 `PUBLIC_BASELINE` 对拍；② `train.py` 没有 slow/fast 概念 ⟹ 重训候选必缺三键且会被 `main.py` 静默降级，现由 `promote_v3_candidate` 在 staging 写入（`--slow-fast-*`，默认即公榜值）；③ 提交包此前「除 train.py 外全收 `*.py`」，把研究模块 `temporal.py` 也装了进去，审计只查缺文件不查多文件 —— 现由 `make_submission.SUBMISSION_MODULES` 唯一声明 + `main.py` 的 AST import 闭包双向对拍。全量测试 **84 passed / 22 subtests**。 |
| 2026-08-19 | 按外部 `HANDOFF.md` 推进 A/B/C 三线，并核出它三处与仓库不符（P0 的 wall-clock 复测 08-18 已完成；私榜是「**最新提交版本生效**」不是 best-of-10 —— 此前 ROADMAP/RUNBOOK 都漏了这半句；t=2 的 max|pred| 数对但仓库无证据）。**P0 结案**（用户 20260819 包审计 `passed:true`、零漂移、无多余模块）。**P7 预注册落盘**：slow/fast 顶点闭式解 + 限幅几何实测（clip 边界 t≈2.6968）+ 锚点交叉验证（max|Δ|=5.0e-09）；增益闭式 `(S1−S0)(t*−1)²/(2t*−1)`，诚实期望只有 +0.0%~+0.9%。**P8 `CLOSED_FAIL`**：辅助损失机制成立（MLP 自身 peak +16.7%）但对生产基准增量仅 +0.026%；顺带订正一道写错的机制门槛（收紧）。**P2-R** recency 预注册臂立项（P4 测的是 volume 轴，不覆盖）。**P1 端到端演练通过**（全 hash 审计 + 闸门正确拒绝）。全量测试 **92 passed / 22 subtests**。 |
| 2026-08-19 | **P7 结案**：slow/fast 顶点第三点已交（`S2 = 0.0039374211`）。完整性检查 `a = −1.474e−04 < 0` 通过；`t* = 0.897692`、`Score(t*) = 0.0041165516` ⟹ **当前生产点已处在这条线峰值的 99.9625%**，slow/fast 捕获了线上总可得增益的 **98.70%**；半步收缩后增益 1.157e−06 < 预注册 1e−05 线 ⟹ **不改交付、生产不动**。⟹ 该轴从「没测」变成「顶点已测出」。顺带确认了 08-17「只搬 OOF 相对模式、保留公榜标定绝对水平」这个做法 —— 它落在最优点的 0.04% 以内。 |
| 2026-08-20 | **P6 清理后的收尾审计**，查出两件事：① 08-18 判毒的 OOF 缓存**不止一份** —— `..._phasebal_prodwindow.npz`（08-14 10:56、13 数组、无 checkpoint）与被判毒那份**签名完全一致**，当时漏点名，已一并隔离；② **改名挡不住** —— 四个实验脚本把毒缓存写死成 `--oof` 默认值，改名后只报裸 `FileNotFoundError`，而旁边就是 `.STALE-DO-NOT-USE` 文件，赶工时最省事的「修法」就是指回去。⟹ 隔离改成代码强制：`src/oof_cache.assert_reproducible_cache`（含改名后路径，`load_oof_bundle` 内也调用），四个脚本改 `required=True` 并接上守卫，3 个回归用例钉住。逐条核过对已有结论无影响（多任务 Stage 1 用的是未隔离的 `confirm_3s480`，且 ±2.4% 基准误差下增益仍在 +0.025%~+0.026%）。顺带修掉 P6 清理造成的文档漂移（旧 zip 名 ×3、`lgbm_mt` docstring）。全量测试 **95 passed / 22 subtests**。 |
| 2026-08-20 | **P9 NN 独立能力阶梯 `REJECTED`**：单轴 epoch 阶梯（12/50/150/400，测量路径复用 `multitask_mlp.py` 一行未改，12 档对 08-19 锚点偏差 0.00e+00）。曲线倒 U：**峰值 28.8% @ 50 档**（相对 12 档 +42% ⟹ 此前确实被预算掐住），但随后崩溃到 6.5% / 1.4% ⟹ **绑定约束是正则化不是预算**，天花板 28.8% < 50% 门槛。辅助损失符号随过拟合翻转（欠拟合区 multitask 更好，过拟合区反之）。⚠️ 方法学发现：**单折 oracle 混合增益不可信**（独立强度单调崩溃而混合增益非单调；400 档独立仅 1.4% 却报 +3.26%）⟹ 追认 08-19「冻结系数 + 5 折终审」的必要性；**不得**据 50 档的 +6.46% 重开 P8。v5 范围因此重定为「训练配方 / embedding / 特征选择」三条，**加算力不在其中**。全量测试 **101 passed / 22 subtests**。 |
| 2026-08-20 | **P10 密封期尺子 `PREREGISTERED`**：核出 RUNBOOK 漏掉的一个取舍 —— 8/23 回补的**就是公榜期的标签**（实测 test 分区 326 列无 weight/target/responder，train 375 列全有），那段 3,217,458 行**只能用一次**：当训练数据或当干净测试集。而它的评估行数约是现有 OOF 5 折合计的 **2.1×**，正是 OOF 那个 6.1~8.7% 的检出下限把 `mkt323`/`phase_id`/`responder_00`/`lag3+lag10`/扩展窗全判成「测不出来」⟹ 全部拿去训练的话，8/23–8/31 最可能的结局是「重训了但测不出，按 D6 维持现状」。⟹ 预注册封存最后 **60,000 real time_id**（4 块 × 15,000、embargo 30、实测 856,319 行），六道门禁按 `≥3/4 块` 映射，第七道「检出下限」标定前判 `None` **而非自动通过**。⭐ 读数口径订正：初稿的「`raw = pred/prediction_scale` 反解」在 slow/fast 下是错的（两个分量各有 scale），改为「断言触限 0 行后直接算 peak」—— peak 对全局缩放严格不变，有用例钉死。干跑双通过：官方 runner 全量 test 3,217,458 行、0 超时、`max|pred|=0.420450` 触限 0 行（与 ledger 08-18 那次对上）；合成标签走通判据链路且强制 `adjudication_valid=false`、self-vs-self 判 FAIL。**不落任何提交格式 CSV**（runner 输出只进临时目录）。全量测试 **112 passed / 22 subtests**。 |
| 2026-08-21 | **函数类探针 `REJECTED`，但把两个假设分开了**：诊断出树对时间/截面维扩容的三次失败都是「ΔB 涨幅是 ΔA 三倍」的**函数类**指纹后，用 RFF-岭回归在**一列不改**的生产截面设计上做预注册对照（含线性阴性对照、生产强度 3s480 基准、五折行对齐以 target/weight 逐位验证）。结果：`r=0.798`、`ρ=0.702`、oracle 集成 **+0.91%**、5/5 折 `r>ρ`（**符号全对、幅度差 3.3 倍**），判据 2/4 不过 ⟹ `REJECTED`。⭐ 两个方向相反的结论都成立：① **P9 的「NN 天花板 28.8%」被否** —— 一个没有任何训练配方的核方法就到 79.8%；② **但到 80% 恰恰说明没用** —— ρ 随 r 同步涨，线性/核/树互相 ρ≈0.6~0.7，在读同一个东西。⟹ 「换提取器」这条线关闭，只剩「**换输入**」（`history_window=5` 与 slow/fast 的 K=2000 之间是空的）。顺带查出 `INCIDENT`（未爆）：所有 OOF 缓存的 `e_target` 列是**全 NaN**（`v3_production_oof.py:512` 显式 continue 跳过赋值），`src/oof_cache.py:19` 仍把它列在 COMPONENT_COLUMNS —— 当前无脚本消费，未污染任何结论。另订正我自己写错的集成增益表：恒等式是 `1+(r−ρ)²/(1−ρ²)`，oracle 恒 ≥ 0 且关于 r=ρ 对称（单测在跑数据前抓到）。⚠️⚠️ **还查出一个测试门禁漏洞**：`NOTES §4` 文档的 `unittest discover -s tests` **静默少跑 36 个用例**并照样报 `OK` —— 7 个模块是 pytest 风格（裸 `def test_x()` 无 TestCase 子类），unittest 只收 TestCase 子类。被跳过的头两个正是 **`test_sealed_period_eval`(11) 与 `test_oof_cache`(6)** —— P10 密封期尺子和缓存出处隔离的把关用例。pytest 实测 **122 passed / 22 subtests**（含本轮新增 10），unittest 只有 86。`NOTES §4` 已改为 pytest 并写明该陷阱。 |
| 2026-08-21 | **长历史窗探针 `REJECTED` —— 两条路都定价完毕**：核出 temporal 全族的 `MAX_LAG = 20`（`experiments/temporal_multiscale.py:49`）⟹ 20 到 slow/fast 的 K=2000 之间是 100 倍未测跨度，本枪补上。40 个 history 特征 × 窗口 {64,512,4096} × {滚动均值, 偏离} = 240 列，与生产设计配对。结果 pooled **+0.69%**、4/5 正折、去最好折 −0.49% ⟹ `REJECTED`。⚠️⚠️ **本轮最重要的自查**：我最初把 ΔA=+3.36%/ΔB=+15.40% 读成「有信号但付不起方差」，**那是错的并已收回** —— 逐折 `A比/√(B比) ≡ IC比`，ΔA/ΔB 混着共同尺度（fold 0 的 A 比 0.9555，A 是降的）；且 `2ΔA>ΔB` 是**两分量配比**判别式，不适用于**嵌套模型**比较，从一开始就不该写进预注册。**这正是 P8 已记录过的同一个错误**（见 2026-08-19 行）。⟹ 「序列模型值不值得做」**仍未定价**；要定价得做列数阶梯/单窗口消融，本轮没做。⚠️ 两枪各暴露一个镜像的预注册缺陷：`function_class_probe` 的 RFF 全钉在 alpha 梯**顶**，本枪线性 fold 4 钉在梯**底**（−19.49%）⟹ `ALPHA_LADDER` 两端都太窄，复验前必须加宽；但去掉 fold 4 后 pooled 仍只有 +2.77%，**结论稳健**。⟹ 合读：「换提取器」已定价（oracle +0.91%）；「换输入」只证明**按 240 列这个加法**净效果不过门槛。8/31 前追平榜首仍不现实，但「长窗有没有可用信号」这个问题**尚未被回答**。 |
| 2026-08-21 | **选列准则探针 `REJECTED`** —— 用户指出 `select_features` 是单变量筛子、忽略特征间相关结构；另发现 `history_positions` 按**当期**相关选列却把它们当**滞后量**用。跑前实测确认分歧真实（top-40 重合：当期vs lag1 = 24/40，而 lag1 vs rollmean5 = 35/40 ⟹ 两个滞后准则彼此一致、都与当期不一致）。四臂预注册（评价器必须是**树** —— 对 LASSO 臂用线性评价是循环论证；四臂 history 列取并集只扫一次，等价性有用例钉住）。结果：**换掉 45% 的 200 列（LASSO）或 40~70% 的 history 40 列（滞后准则），树的 peak 一个都没变好**，且分歧越大掉得越多。⟹ **诊断成立、代价不存在**。选列轴三向封死。⚠️ 事后解释（非本实验发现）：history 的 `difference` 与 `rolling_deviation` **都含当前值**，该块并非纯滞后，当期准则对它们反而匹配 —— 要证实需四子块消融，本轮未做。顺带补 `tests/test_select_features.py`（9 用例）：该函数有 **77 个调用点、此前 0 个测试**。 |
| 2026-08-21 | ⭐ **长窗列数阶梯：今天第一个 PASS（筛选档）**。把 `long_history_probe` 那 240 列拆成单窗口各 80 列，`w512` 在树上 pooled **+6.80% / 5-of-5 / 去最好折 +5.84% / CI 下界 +3.87%**，五道门槛全过且**超检出下限 1.12×**。⭐⭐ 最有说服力的一条：**同一个评价器（线性）上，240 列给 +0.69%、80 列给 +5.75%** —— 预注册时写下的机制（『信号被估计代价淹掉』）**跑之前就写死、跑之后兑现**。两处跑前声明的变更都生效：主评价器改树 ⟹ 线性 alpha 选参那类脆弱性消失（40 次拟合无一撞端，`long_history_probe` 的 −19.49% 坏折消失，追认为选参失效）；`WIDE_ALPHA_LADDER` 另起常量 ⟹ 旧 `ALPHA_LADDER` 一字未动（有单测断言），两个已结案实验仍可复现。⚠️ 克制读法：筛选档（1s160/单块）、迁移率历史 0.51×~2.3×、只超下限 1.12×、三臂多重比较、且本脚本的 cumsum 口径**不能**照搬进生产（`history.py` 刻意不用 cumsum 以保离线/在线逐位一致）。⟹ 下一步是 **3s480 确认**（只留 base 与 w512 两臂），生产目录未动。 |
| 2026-08-21 | **长窗 w512 走完确认档**：3 种子 × 480 轮、只截面块、fold 版图与筛选档相同、base 现跑。结果 pooled **+7.77% / 5-of-5 / 去最好折 +6.49% / 配对 CI 下界 +4.18%**，五道门槛全过，但 **0.89× 检出下限** ⟹ 预注册三档裁决里的中间那档 `PASS_BUT_BELOW_DETECTION_FLOOR`（跑前就预判最可能）。⭐ 三条正面旁证：筛选→确认迁移 **1.14× 未衰减**；**基准更强（+2.8%~21%）而增益仍在**（P8 栽跟头的反面）；**线性对拍 10/10 逐位相同**（线性不依赖树超参 ⟹ 排除『换了个跑法』）。⚠️ 三条克制：只测截面块（占分 58.8%，全模型粗估 +2%~+5%，与 slow/fast 同量级）；slow/fast 那次迁移率是 0.51× 本地高估 ⟹ 不能拿 1.14× 当规律；探针的 float64 cumsum **不得**进生产（`history.py` 刻意不用 cumsum 以保离线/在线逐位一致）。⟹ **是否花 8/23 前的公榜额度由用户定**；生产目录仍一字节未动。⚠️ 工具可靠性：本轮收到 3 次与落盘日志不符的后台通知（2 次数字错、1 次提前报完成），所有数字均已从产物文件核对。 |
