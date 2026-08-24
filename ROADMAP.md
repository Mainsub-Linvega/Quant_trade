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
| 来源候选 | **`outputs/candidates/v3_hybrid_extended_full/`**（2026-08-24 转正）| `outputs/promotions/v3_hybrid_extended_full_20260824/promotion_manifest.json` |
| ⭐ 训练数据 | **3,289,030 行**（`time_id 0–1,105,919`，含 8/23 回补的全部数据，对前一版 **+24.3%**）| `hybrid_meta.json` |
| 转正前那版 | `v3_hybrid_long512`（公榜 0.0041833953，2,645,530 行）；备份在 `outputs/promotions/backups/model_before_20260824_150921` | 备份目录 |
| 采纳依据 | 密封期 **+6.03% / 4 块全正 / 配对 CI [+2.15%, +10.49%]**（7 门过 6）+ 独立 OOF 第二读数 **+4.23% 同号**。见 §P2-E / §P2-D2B | 两份实验报告 |
| ⚠️ 公榜分数 | **无** —— 公榜 8/23 已停更，本版从未上过榜。上面 0.0041833953 是**前一版**的 | — |
| 公榜分数（**生产目录本身**） | **0.0041833953**（2026-08-21 转正后） | `experiments/ledger.csv` |
| slow/fast | **已转正**：window=2000 真实步，两个 scale 0.4496 / 1.2530 | `hybrid_meta.json` |
| 转正前基线 | 0.0041150085（2026-08-18 `slowfast`），差 **+1.66%** | `experiments/ledger.csv` |
| 公榜第一 | **0.0060**（2026-08-17，**用户报告**，非本地测量） | 用户 |
| 与第一的差距 | **+43.4%**（**IC 只差 +19.7%**）；旧记录 0.00520002 标 `SUPERSEDED` | 本表两行相除 |
| 截面块 | weighted LGBM，480 轮 × 3 种子，history40 + **长窗 w512**（441 列） | `hybrid_meta.json` |
| 市场块 | unweighted row-level LGBM，λ=0.5，480 轮 × 3 种子 | `hybrid_meta.json` |
| 截面混合 | `blend_weight=1.0`，即 LGBM 截面分量全替换 | `hybrid_meta.json` |
| 后处理 | `prediction_scale=1.16`，clip=0.5；**slow/fast 分离**（逐 asset 自身预测的因果滚动均值，K=2000 真实 time_id 步）| `hybrid_meta.json` |
| 训练采样 | `sample_modulo=5`，`phase_balanced` | `hybrid_meta.json` |
| promotion 校验 | 双后端最大差 `1.388e-16`，结构敏感性门禁通过 | `outputs/promotions/v3_hybrid_long512/promotion_manifest.json` |
| train/inference 一致性 | **`1.098e-08`**（两后端同值，2026-08-23 实测，**新默认** `--partition-index 8 --n-time-ids 2100`）。⭐ 旧默认 50 太窄：每 asset 只有 50 个观测 ⟹ 长窗 512 的环形缓冲只填到 **9.8%、从未回绕**，slow/fast 的 2000 步窗只填到 **2.5%、左端从未移动** ⟹ 测的是「还没热起来的模型」，而榜上跑的是热的那个。见 §4 P12。⚠️ 本表此前那条「记 `8.111e-09`、同参数复测对不上、**未去追因**」**已追因**：`max\|Δ\|` 随 `--n-time-ids` **单调增长**（50→4.019e-09、200→7.603e-09、600→**8.117e-09**、1000→8.770e-09、1500/2100→1.098e-08、3000→1.630e-08），而同参数重复跑**逐位相同**（n=50 / n=600 各验两次）⟹ 那两个数的差别是**当时用了不同的窗口**（8.111e-09 对应约 600），**不是不确定性**。⚠️ `check_consistency.py` 是 **slow/fast-aware** 的 —— 训练端没有该后处理的概念，不补上会永久报红 9.4e-02 | `scripts/check_consistency.py` |

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
- **⭐ 2026-08-23 首次实测内存**（此前交付链路完全没有内存口径）：4 核 / 12 GB cgroup 下
  LightGBM 主路径峰值 RSS **11.47 GB = 12 GB 上限的 95.6%**，`predict_total` **5.40 分钟**
  （比 32 核钉 4 线程那次只慢 2.7%）。⭐ 但其中 **11.09 GB 是主办方 harness 自己的**
  （零预测桩模型对照臂），**我们的模型只占 +0.38 GB** ⟹ 不为内存改模型。详见 §4 P11。
- 2 种子旧候选只节省 5.45% 全量推理时间。⭐ **2026-08-24：掉分不再是「未知」** —— D0.3 离线复算 `submission_slowfast_t2.csv` = **0.0039374211**，对 slowfast 基准 **−4.32%**（换 5.45% 耗时代价太高）。不作为默认生产方案。

## 3. 当前有效的研究判断

1. **结构收益仍是主要来源。** history、第二市场分量和带权截面训练贡献远大于后续容量微调。
2. **市场块和截面块的容量方向相反。** 市场块收缩小幅有益，截面块收缩显著有害。
3. **480 轮已形成内部极值。** 旧结构 320 轮下降，新结构 960 轮下降；统一轮数轴结案。
4. **普通②类调参余量很小。** 2026-08-13 容量扫描只得到 +0.77%，低于 2% 关注门槛。
5. **本地尺子不能稳定判断拟合紧密度。** alpha、轮数、history 宽度都出现过本地与公榜量反。
6. **Responder 三种用法全部走完，这条轴现在由证据关闭**（2026-08-22 收口）。可预测不等于能补 target 残差；此外 Stage C 此前有 **14 / 47 列**是被 `multi_member_family` 这条**启发式**挡着、从未测过的，现已补测，**28 格无一过门禁**。用 responder 打分选列（唯一没做过、也唯一不在母条件排除项里的用法）在前置测量处否掉：重合 170~180/200 低于决策线，且换掉的是原排名 #16 的列 ⟹ 预注册的降方差机制**未兑现**。证据：`responder_stage_c_fill.md` / `responder_selection_probe.md`。
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

### P12 — 转正门禁补 `long_window` + 一致性窗口扩宽 —— `INCIDENT`（未爆）/ `RESULT`（2026-08-23）

- **状态**：`CLOSED`。生产预测**一位未变**，生产目录与 long512 manifest 8 文件逐字节相同。
- **⭐ 这是同型事故的第五个现场，也是第一个「零参数可达」的。** 前四次（08-18 slow/fast →
  08-19 结构开关 → 08-21 `PUBLIC_BASELINE` → 08-23 重训计划）都需要某个特定动作才触发；
  这一次不需要：`promote_v3_candidate.py` 的**默认** `--candidate`
  就是 `outputs/candidates/v3_hybrid_mkt_shrunk`（`long_window=None`、截面森林 361 列、
  公榜低 **1.662%**），实测 `check_against_public_baseline` / `validate_meta` /
  双后端烟测**三道门全过**并写出 staging。
- **根因（值得单独记，因为它反直觉）**：`long_window` 在 `PUBLIC_BASELINE` 里躺了两天
  （08-21 加入），但 `validate_meta()` 的 `checks` 字典**一条都没查它**。
  而推理侧**确实**有一道硬校验（`lgbm_numpy.py:283` 的列宽 ValueError）——
  它只抓「meta 与森林**打架**」（meta 说有长窗、森林是 361 列），
  **抓不到「两边一致地错」**。盘上 13 个旧候选全是后者 ⟹ 一致性校验对它们完全无感。
  ⟹ **一致性校验不能替代身份校验**，两者抓的是正交的两类错误。
- **⚠️ 修法与 slow/fast 相反，不能照抄**：`slow_fast_*` 是纯后处理，`train.py` 不产出，
  由 staging 补写；而 `long_window` 决定**截面设计矩阵的宽度**（441 = 361 + 80，
  多出的 80 列 = 40 长窗均值 + 40 偏离），是**训练进森林里的**（`train.py:593` 会写它，
  默认 `--long-window 0` ⟹ 写 `None`）。给 361 列的森林盖上「有长窗」的章，
  好的情况是推理期撞列宽错，坏的情况是交出一个错模型。
  ⟹ 最终实现是**只校验、绝不覆写**，也不加 CLI 出口；有意偏离走已有的 `--off-baseline`。
- **⭐ 一致性门禁此前几乎没测到当前生产结构**：默认 `--n-time-ids 50` 下每 asset 只有 50 个
  观测 ⟹ 长窗 512 缓冲填到 9.8%、**从未回绕**；slow/fast 的 2000 真实步窗填到 2.5%、
  **左端从未移动**。而这两块正是 08-18 与 08-21 转正、公榜合计 **+4.6%** 的结构。
  默认已提到 **2100**（长窗满且回绕 4.1 次、slow/fast 窗满且左端开始移动），
  实测代价只有 **2.7s → 5.9s**（lightgbm），不值得做成可选档。
- **一致性报告此前不记模型身份**：只记 `baseline_model.json`（冻结岭回归）一个文件的 hash，
  六片森林和 `hybrid_meta.json` 一个都不记 ⟹ 能证明「两条口径一致」，
  却不能证明「一致的是哪个模型」。现复用 `verify_delivery_runtime.model_identity`
  （**不另造第二份取值表**，CLAUDE.md §7），报告里现有 8 个文件 hash +
  13 个身份键 + `public_baseline_drift` + manifest 逐字节对拍。
- **四道新门禁都做了变异测试**（证明会咬，不是摆设）：退回改动前的 promote ⟹
  `test_dropping_any_identity_key_is_rejected` 当场红；往 `PUBLIC_BASELINE` 塞第 14 个键 ⟹
  `test_every_baseline_key_is_mapped_or_exempt` 当场红；把默认窗口调回 50 ⟹
  长窗回绕与 slow/fast 左端两条断言当场红。
- **⚠️ 两处副作用**：
  1. 盘上 **13 个长窗转正前的旧候选现在需要 `--off-baseline` 才能 staging**。这是有意的
     （它们确实是旧架构），且**不影响 RUNBOOK D0.4** —— 那步走 `sealed_period_eval.py`，
     不经过 promote。
  2. `v1_ridge` 在新窗口下 `max|Δ|` = **1.192e-07**，仍低于 `atol=1e-6` 但余量收到 **8.4×**
     （v3_hybrid 是 1.098e-08，91×）。v1_ridge 非生产策略，不改；记在这里以免下次被当成回归。
- **证据**：`scripts/promote_v3_candidate.py`（`validate_meta` 的 `long_window_matches`）、
  `scripts/check_consistency.py`、`tests/test_model_identity_key_coverage.py`
  （新增 `PromoteGateCoverageTest`，把**转正**接进消费者表 —— 此前只有 audit / retrain /
  verify_delivery 三家）、`tests/test_promote_v3_candidate.py`（新增 `LongWindowIdentityTest`）、
  `tests/test_check_consistency_window.py`（新文件）。
  全量 **273 passed / 41 subtests**（本次 +12 用例）。生产目录**一字节未动**。

### P11 — 评测环境资源门禁 —— `RESULT`（2026-08-23，首次实测，结论是「别改模型」）

- **状态**：`RESULT / NO_MODEL_CHANGE_REQUIRED`。**四条环境 × 后端组合全部实测完毕**，
  两条执行路径在**真实评测机**上都被验过 —— 这是 8/31 前最后一个交付未知数，已关闭。
- **为什么立项**：`docs/competition_description.md:158-159` 写明评测环境
  **4 核 / 12 GB / 无 GPU / 无外网**，:166「内存超限……严重情况下提交可能被判定为无效」，
  :199「截止后无法修改代码，实盘出错按填 0 处理」。而交付验证脚本此前
  **一个内存字段都没有** —— 唯一那个 RSS 数字只写在 NOTES 正文、不是产物、量自 30 GB 机器。
  ⟹ 这是唯一一个能让整个提交归零、而我们从未测量过的量。
- **⭐ 首次在真实约束下实测**（`systemd-run -p MemoryMax=12G -p MemorySwapMax=0
  -p AllowedCPUs=0-3`，走官方 runner 全量 3,217,458 行）：

  | 臂 | 峰值 RSS | 占 12 GB | predict_total |
  |---|---:|---:|---:|
  | 主办方 harness 单独（零预测桩，跑 3 次） | **11.09 / 11.47 / 11.35 GB** | 92–96% | 0.06 分钟 |
  | 生产模型 LightGBM 主路径 | **11.47 GB** | 95.6% | **5.40 分钟** |
  | 生产模型 NumPy 兜底 | **11.55 GB** | **96.2%** | **10.90 分钟** |

- **⭐⭐ 结论：内存基本全是主办方 harness 的，我们的模型量不出来。**
  ⚠️ **诚实读法**：harness 臂**自己**跑三次就摆动 **11.09–11.47 GB**（0.38 GB），与「模型净增」
  同量级 ⟹ 只能说**模型贡献 ≤ 约 0.5 GB、与跑间波动不可分辨**，不能报成精确的 +0.38 GB。
  结论方向不变且更强：**不要为内存改模型** —— 能省的那部分连测都测不出来。
- **峰值发生在哪个阶段（1 秒间隔追踪，`--trace-interval`）**：峰值在 **18.0s / 36.8s = 49% 处**
  达到，正是 `iter_test_slices` 逐分区 `pd.read_parquet` 那段（`test_partition_000.parquet`
  1.68 GB）；后半程（遍历 214,538 个 time_id + 最后那次 `pd.concat`）`VmHWM` 一点没涨、
  `VmRSS` 反落到 3~4 GB。⟹ **峰值由分区大小决定，不由 time_id 数量决定** ⟹ 9 月更长的
  实盘期不会推高它。⚠️ 本条初稿写「第一个分区加载时就到顶」是**单次中途采样的误读**，已订正。
  `timeseries_api/` 是主办方原文、只读（CLAUDE.md §1.3），这块我们改不了。
- **跨 predict 状态增长已定价，安全**：`AssetLongWindow` 是固定环形缓冲 2.46 MB；
  `PredictionTrail` 是唯一无界的，实测 **314.6 B/time_id**（40 万步上完全线性），
  公榜期 214,538 步只占 **64 MB**，剩余 0.53 GB 余量还能吃公榜期的 **8.4 倍**。
- **⚠️ 剩余风险（不在我们控制内）**：12 GB 这条线本身很紧 —— cgroup `memory.events`
  记到 `max 990`（990 次顶到上限被迫回收，`oom_kill 0`）。若正式环境的分区更大或
  harness 版本变化，余量会先于我们的模型耗尽。
- **两条执行路径本地均已在 4 核 / 12 GB 下走完全量**，除 `peak_rss_has_headroom` 外门禁全过；
  两后端预测差在第 15 位有效数字（`max|pred|` 0.40209875023052816 vs 0.4020987502305287，
  相对 5.4e-16），即 `main.py` 已记的求和顺序差异，非模型身份问题。
- **⭐ 2026-08-23 云端实测完成（用户执行）—— 真实评测机上两条路径都过**。
  `FACT`（用户实测）：**该机器超过 12 GB 即 OOM**，即官方那条 12 GB 是硬限、不是名义值。

  | 环境 | 后端 | 峰值 RSS | predict_total | wall | 单步最大 |
  |---|---|---:|---:|---:|---:|
  | 本地 4 核 / 12 GB cgroup | LightGBM | 11.47 GB | 5.40 分钟 | 6.35 分钟 | 0.688 s |
  | 本地 4 核 / 12 GB cgroup | NumPy 兜底 | 11.55 GB | 10.90 分钟 | 11.76 分钟 | 0.684 s |
  | **云端 JupyterHub**（128 核钉 4 线程）| LightGBM | 未记录（旧版脚本）| **12.13 分钟** | **14.00 分钟** | **2.802 s** |
  | **云端 JupyterHub** | **NumPy 兜底** | **10.93 GB**（91.1%）| **33.78 分钟** | **36.28 分钟** | **0.050 s** |

- **⭐ 兜底在真实机器上跑通了**：`peak_rss_under_limit` ✅、行数 3,217,458 ✅、
  0 超时 / 0 非有限值 / 0 触 clip ✅、模型身份两道 ✅，只差 20% 余量线。
  ⟹ **lightgbm 万一不可用，兜底不会 OOM，只会慢**。
  ⭐ 云端峰值 **10.93 GB 反而比本地的 11.55 GB 低 0.62 GB**（numpy 1.24.3 vs 2.5.1、
  pyarrow/pandas 版本不同 ⟹ parquet 加载路径的内存不同）—— 真实评测机比开发机宽裕。
- **⚠️ 我的外推错了 35%，机制值得记**：跑前按「本地兜底/主路径 = 2.02×」外推云端约 **25 分钟**，
  实测 **33.78 分钟**。原因是**兜底是单核绑定**（纯 numpy 树遍历不并行，已记录），
  而主路径吃 4 线程 —— 两者对「云端单核更慢」的敏感性根本不同：
  云端/本地在主路径上是 2.25×，在兜底上是 **3.10×**；兜底/主路径在本地是 2.02×、云端是 **2.78×**。
  ⟹ CLAUDE.md §5.7「代理量不可跨结构搬用」在**跨环境**上同样成立，这次是实证。
- **⭐ 单步耗时把风险排序反过来了**：兜底总耗时是主路径的 2.78×，但**单步最大只有 0.050 s，
  是主路径 2.802 s 的 1/56**。⟹ 「按单步超时」这条风险**在主路径上，不在兜底**。
  ⚠️ 主路径那 2.802 s 是四次跑里最高的（本地两条都是约 0.68 s），像是首调用或环境抖动的
  单点离群，不是模型的内在成本（平均只有 3.39 ms）—— 但**没有第二次云端主路径观测来证实**，
  当前只能标为「已知的单点异常」。
- **⭐ 浮点差异的分解（两条独立轴，量级差 7 个数量级）**：
  同机器换后端 `max|pred|` 相对差 **1.4e-15**（求和顺序）；同后端换机器 **2.3e-8**
  （numpy 1.24.3 vs 2.5.1）。⟹ 跨机器差异**主导**，但传到 Score 仍约 1e-8，可忽略。
  ⟹ 兜底在真实机器上产出的是**同一组预测**，这一点现在有直接证据。
- **⚠️ 剩余交付风险，按严重度排**：
  1. **兜底 36.28 分钟 wall**（主路径 14.00 分钟的 2.6×）。官方总时限「以正式评测环境设置为准」
     （`docs/competition_description.md:164`），本地 runner 三个 timeout 参数默认都是 `None`
     ⟹ **阈值未知**。若存在 30 分钟量级的总时限，兜底会撞线 ⟹ 填 0。
     ⚠️ 但兜底只在 lightgbm 缺失或开机对拍失败时才走，而云端实测 **lightgbm 4.3.0 在**
     ⟹ 这是「备胎的备胎」，不改设计。
  2. 12 GB 余量：云端 10.93 GB = 91.1%，比本地宽 0.6 GB，但仍无 20% 余量，且不在我们控制内。
- **云端产物已 `cloud_sync.py pull` 回本地**：`outputs/cloud/delivery_cloud_{numpy_4t,py311_4t}.{json,md}`。
  ⚠️ `outputs/cloud/` 被 `.gitignore:96` 忽略 ⟹ **这两份原始 JSON 不进版本控制**，
  本节表格里的数字才是受控记录。8/31 前若要清盘，这两份与 `outputs/submission_*.csv`
  同属「别删」清单（P6 §「绝对不要删」的同类）。
- **两道旧红门禁已查明并修好**（`INCIDENT`，未爆）：`--manifest` 默认值写死
  `v3_hybrid_slowfast`，08-21 转正 long512 后过期 ⟹ 两份交付报告长期判 FAIL，
  而红的原因是**比错了对象**（生产与 long512 manifest 8 文件逐字节全中）。已改
  `--manifest auto`（扫描 `outputs/promotions/*` 挑逐字节相同那份，扫描过程写进 JSON 留证），
  并新增一道**非循环**的 `model_matches_public_baseline`（复用
  `audit_submission_zip.public_baseline_drift`，不另抄取值表）。两道现均通过。
- **证据**：`outputs/experiments/delivery_4c12g_{lightgbm,numpy_fallback}.{json,md}`、
  `harness_memory_harness_4t.{json,md}`、`harness_memory_trace_4t.{json,md}`（含逐秒序列）、
  `scripts/measure_harness_memory.py`、`tests/test_verify_delivery_runtime.py`（14 用例）。
  全量 **241 passed / 26 subtests**（改动前 227）。生产目录与模型身份**一字节未动**。

### P0-B — 私榜包重打 —— `CLOSED`（2026-08-22，用户已执行并落盘审计）

- **结案证据**：`outputs/experiments/submission_audit_20260822.json` ——
  `outputs/v3_hybrid_submission_20260822.zip` 审计 **`passed: true`**、
  `public_baseline_drift: []`、`unexpected_modules: []`、`missing: []`
  ⟹ 模型身份与包内容身份两道门同时通过，且有落盘证据。
- ⚠️⚠️ **盘上现在有五个 v3 zip**（`20260813.PRE-SLOWFAST` / `20260818` / `20260819` /
  `20260822`，加上 v1 的两个不算）。**8/31 只能交 `20260822` 那份** ——
  它是唯一带长窗 w512（`long_window: 512`）且审计全过的。上传前用文件名日期核对，
  并按 RUNBOOK D6 的收尾顺序执行（**上传完不要再上传任何东西**）。
- 以下为立项时的原始记录，保留备查：

- **`outputs/v3_hybrid_submission_20260819.zip` 现在装的是旧模型。** 2026-08-21 长窗 w512
  转正（公榜 0.0041833953，+1.66%）后，`PUBLIC_BASELINE["long_window"]` 由 `None` 改为 `512`；
  该 zip 缺这个键，审计已实测判 **`passed: false`**、`public_baseline_drift:
  ["long_window: None != 公榜基线 512"]`。
- ⚠️ **缺键不会报错，只会静默关掉长窗** —— `main.py` 里 `long_window` 取不到就是 `None`，
  交出去的是低 **1.66%** 的旧模型。与 08-18 slow/fast 丢键完全同型。
- **用户执行**：

  ```bash
  .venv/bin/python scripts/make_submission.py --strategy v3_hybrid
  .venv/bin/python scripts/audit_submission_zip.py \
      outputs/v3_hybrid_submission_<YYYYMMDD>.zip --expect-public-baseline \
      --output outputs/experiments/submission_audit_<YYYYMMDD>.json
  ```
  判据：`passed: true` / `public_baseline_drift: []` / `unexpected_modules: []`。
- ⚠️ 盘上现在有**五个** v3 zip，只有 `20260822` 那份是当前生产。8/31 上传前用文件名日期核对。

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
  改名就是防呆措施本身——**不要改回去**。~~8/31 交的是 `..._20260819.zip`。~~
  ⚠️ **2026-08-24 订正**：这句写于 08-19，08-21 长窗转正后 `..._20260819.zip`
  自己就过不了 `--expect-public-baseline`（`long_window: None != 512`）。
  **8/31 的兜底是 `..._20260822.zip`**（08-24 复跑十一项全过）。
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

### P1-R — ⭐ D0.3 修尺子 —— `RESULT`（2026-08-24）：**21/21 复现，且公榜排名多半是 regime**

- **尺子已可信**：21 份有公布分数的历史 CSV **全部离线复现**，最大偏差 **1.916e-09**
  （阈值 1e-7；该量级正好是公榜 8 位小数的舍入底噪），join 覆盖率全 1.0、非有限值 0。
  先修了复算器一个**从未执行过**的 bug（预测列与标签列同名 `target`，merge 后被改成
  `target_x`/`target_y`；08-18 的干跑是 inventory 模式，评分块整块没跑到）。
  证据：`outputs/experiments/public_replay_scored_20260824.{json,md}`。
- **顺带定住两份无归属 CSV**：`submission_long512.csv` = **0.0041833953**（当前生产），
  `submission_slowfast_t2.csv` = **0.0039374211** ⟹ 2 种子那份的掉分**不再是「未知」**：
  对 slowfast 基准 **−4.32%**（本表 §性能风险 最后一行据此订正）。
- **⭐ 公榜期内部漂移 +49%，所有模型同步漂**。按 time_id 五等分（每段约 43k）：
  当前生产从段0 的 0.0039403 涨到段4 的 **0.0058618**；`slowfast` 0.0036923 → 0.0059232；
  连负控制 `r960` 段4 也有 0.0059992。
- **模型间离散度随段递减**：段0 极差/中位 149.7% → 段4 **56.9%**；
  而**前 10 名在段4 内部的相对差全部落在 +1.28% ~ −4.18%**。
- **全窗排名到段4 基本重排**：全窗前三（`long512` +1.66% / `slowfast` 0 / `slowfast_runner`）
  在段4 排 8/5/6；全窗第 10 的 `r960`（−8.60%）在段4 排 **第 1**。
- ⚠️ **正确读法**：段4 不是说「该换成 r960」，而是**在最靠近私榜期的那段上前 10 名互相
  测不出来**（±1.3% 远低于任何检出下限）。被证伪的是「全窗 Δ% 可当作前向增量估计」本身。
  具体地，08-21 转正的 `long512` 那 **+1.662%** 几乎全部来自段0（+6.71%）与段1（+2.84%），
  段2/段4 是 −0.40%/−1.04% ⟹ **1-of-5 段为强正**，按 D2 的「≥4/5 折」口径根本不该过。
  ⚠️ 这**不构成回滚建议** —— 段4 同样测不出它变差；见下一条。
- **⟹ 交付结论（「本地增量该打几折」）**：在当前证据下，**任何单一窗口的 Δ%（本地 OOF 或
  公榜全窗）都不构成前向增量的证据，除非它在分段上同号**。这比原先设想的「乘一个迁移率
  折扣」更严 —— 迁移率假设符号稳定，而分段数据显示前 10 名连符号都不稳定。
  实践含义：**8/28 之前不要为 <3% 的全窗增量做任何结构改动**；空动作不需要额外证据。
- ⚠️ **纪律**：段3/段4 与密封段（1,045,920–1,105,919）重叠。本项只**报告**观察，
  **不据此改动 P10 的候选清单、块数或门槛**（`sealed_period_plan.json` `not_doing` 第 5 条）。

### P-D45 — 最终交付件（全量重训）—— `RESULT`（2026-08-24）

`scripts/retrain_extended.py --role extended_full`，数据根 `outputs/data_roots/extended_full`
（12 分区 / 16,445,150 行 / `time_id 0–1,105,919`，**含密封段**）。

- **身份**：与当前生产、决策期件**结构逐项相同**（`long_window=512` / `history_window=5` /
  `num_iteration=480` / `market_lambda=0.5` / `cross_section_weighted` / 3+3 森林 /
  `sample_modulo=5` / `phase_balanced`）；岭回归仍是**逐字节冻结拷贝**（`54dc6afb…`）。
  训练行数 **3,289,030**，对生产 **+24.3%**、对决策期件 **+5.5%**。
- **staging**：`outputs/promotions/v3_hybrid_extended_full_20260824`，
  双后端 `numpy_vs_lightgbm_max_abs = **8.33e-17**`（机器精度）。
- **⭐ 4 核耗时：五个测量互相不可区分，全量件峰值 RSS 反而最低**

  | | predict_total | 单步均值 | 峰值 RSS |
  |---|---:|---:|---:|
  | 生产对照臂（今日现跑） | 5.97 分 | 1.670 ms | 11.51 GB |
  | 决策期件 | 5.98 分 | 1.673 ms | 11.53 GB |
  | **全量件** | **6.03 分** | **1.686 ms** | **11.32 GB** |
  | 决策期件 numpy 兜底 | 10.75 分 | 3.006 ms | 11.42 GB |
  | **全量件 numpy 兜底** | **10.80 分** | **3.022 ms** | **11.34 GB** |

  主路径对 **50 ms** 预算 **30×** 余量，兜底 **16.5×**；兜底/主路径 = 1.79×。
  五份报告的未过项**逐项相同**，都只有 `peak_rss_has_headroom`
  （根因是主办方 harness 自占 11.09 GB，见 §P11）⟹ 与候选无关。
- ⚠️ **必须记住的风险（RUNBOOK 原文的口径）**：这一份训练在**密封段**上，
  **没有任何评估覆盖过它**。缓解是它与刚被密封期验过 `+6.03% / 4/4 块 / 配对 CI 排除 0`
  的那份**是同一结构**，且 D4 覆盖机械正确性。
  三层回退仍在：全量件 → 决策期件（被密封期评过分）→ `v3_hybrid_submission_20260822.zip`。
- 证据：`outputs/experiments/delivery_runtime_{lightgbm,numpy}_4t_full_20260824.{json,md}`。

### P-D4 — 扩展候选的转正门禁 —— `RESULT`（2026-08-24，机械正确性全过）

**未转正**：`--activate` 需要用户明确授权（CLAUDE.md §1.5），我只做到 staging + 校验。
staging 落在 `outputs/promotions/v3_hybrid_extended_fixed_20260824`
（**没有覆盖**已有的 `v3_hybrid_s1.16_w1_3seed` —— 那道归属检查正确拦下了我第一次尝试）。

| 门 | 结果 |
|---|---|
| 全量单测 | **313 passed / 41 subtests**（基线 273，本轮新增 40） |
| `check_consistency` lightgbm | `max\|train − infer\| = 1.098e-08` ✅ |
| `check_consistency` numpy | `1.098e-08`（两后端同值）✅ |
| staging 双后端对拍 | `numpy_vs_lightgbm_max_abs = **7.63e-17**`（机器精度）✅ |
| 冻结岭回归 | staging 后 sha 仍是 `54dc6afb…` ✅ |
| `model_matches_public_baseline` | ✅ 偏离为空 |
| `model_matches_promotion_manifest` | ✅ 解析到本次 staging |

**⭐ 4 核耗时：同机同条件 A/B（这是本项的关键，不是拿今天的数去比 08-18 的数）**

| | predict_total | wall | 单步均值 |
|---|---:|---:|---:|
| 当前生产（今日对照臂） | 5.97 分 | 6.96 分 | 1.670 ms |
| 扩展候选 | 5.98 分 | 6.97 分 | 1.673 ms |

⟹ 差 **+0.17%**，纯噪声。单步均值对 **50 ms** 预算有 **30×** 余量；
`model_init` 0.37s（限 180s）；`zero_timeouts / zero_clip_rows / zero_non_finite` 全过。
⚠️ 若拿候选的 5.98 分去比 ROADMAP 记的 5.26 分会得出「慢 11%」的**错误结论** ——
那是今天这台机器比 08-18 慢，不是候选变慢。**跨日期比耗时必须现跑对照臂。**

⚠️ **唯一的红是 `peak_rss_has_headroom`（11.53 GB vs 余量线 9.60 GB），
而当天的生产对照臂红的是同一道门**（未过项列表逐项相同）⟹ 与候选无关。
根因见 §P11：那 11.09 GB 是主办方 harness 自己的（零预测桩对照臂实测），
我们的模型只占 +0.38 GB。

**兜底路径（NumPy）**：`predict_total` **10.75 分**（wall 11.53）、单步均值 **3.006 ms**、
峰值 RSS 11.42 GB，未过项同样只有 `peak_rss_has_headroom`。
兜底/主路径 = **1.80×**（ROADMAP §性能风险 记的基线是 2.08×，本次更宽裕）。
⟹ **即使评测机上 lightgbm 不可用、走纯 numpy 兜底，单步 3.006 ms 对 50 ms 预算仍有 16.6× 余量。**

⚠️ 顺带解释一个表面矛盾：主路径今天比 08-18 记的 5.26 分慢 13%，兜底却比 10.94 分**快** 1.7%。
不是数据有问题 —— **兜底是单核 100%**（纯 numpy 树遍历不并行），对机器上的其他负载不敏感；
主路径吃 4 线程，会被同机其他任务挤。这进一步说明跨日期比耗时不可靠，
**同 session 对照臂才是唯一可信的口径**。

证据：`outputs/experiments/delivery_runtime_{lightgbm,numpy}_4t_extended_20260824.{json,md}`、
`delivery_runtime_lightgbm_4t_production_control_20260824.{json,md}`。

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

### P2-E — 扩展数据固定结构重训 —— `RESULT`（2026-08-24，密封期 6/7 道门通过）

- **候选**：`outputs/candidates/v3_hybrid_extended_fixed`（`scripts/retrain_extended.py --role decision`，
  数据根 `outputs/data_roots/decision`，训练段止于 `time_id 1,045,889`）。
  与当前生产**结构逐项相同**（`long_window=512` / `history_window=5` / `num_iteration=480` /
  `market_lambda=0.5` / `cross_section_weighted` / 3+3 片森林 / `sample_modulo=5` / `phase_balanced`），
  岭回归是**逐字节拷贝**（sha256 与 `PRODUCTION_RIDGE_SHA256` 相同，未重训）。
  唯一差别：训练行数 **3,117,682 vs 2,645,530（+17.8%）**。
- **密封期裁决**（基准 = 当前生产，检出下限按 D0.4b 标定的 9.68%）：

  | 块 | 基准 peak | 候选 peak | 相对 |
  |---|---:|---:|---:|
  | 0 | 0.0088569 | 0.0092998 | **+5.00%** |
  | 1 | 0.0061782 | 0.0065264 | **+5.64%** |
  | 2 | 0.0069486 | 0.0076385 | **+9.93%** |
  | 3 | 0.0047286 | 0.0048968 | **+3.56%** |

  块均 **+6.03%**、pooled +6.06%、**正块 4/4**、去最好块 **+4.73%**、
  ΔA +7.45% / ΔB +8.85%（`2ΔA>ΔB` 通过）、**配对 CI [+2.15%, +10.49%]（排除 0）**。
- **七道门：6 过 1 不过。** 唯一不过的是第七道（+6.03% < 登记的检出下限 9.68%）。
  ⭐ **它是 Tier 1 + 本臂共六个臂里唯一通过第 6 道（配对 CI 下界 > 0）的** ——
  连公榜低 14.49% 的 `xs_shrunk` 都没做到。
- ⚠️ **关于第七道门，不回头改那个数**：9.68% 是拿 Tier 1 五个**结构不同**的臂的 CI 半宽中位数
  登记的；本臂与基准**结构完全相同、只差训练数据**，配对差方差小得多，
  自己的 CI 半宽只有 **4.17%**（不到 Tier 1 中位的一半）。
  ⟹ 检出下限不是尺子的单一属性，是「(基准, 臂) 这一对」的属性；第 6 道门本来就是它的臂内版本。
  这是**观察**，不是改判据的理由 —— 按 8/24 预注册的决策规则，第七道门本来就不是否决项
  （三个否决项全是负向证据，此处一个都没触发）。
- ⚠️ **必须一起读的反向读数**：P1-R 测出 `sealed ≈ public + 5.18pp`，照此外推
  +6.03% 的「公榜窗口等价值」约 **+0.85%**。但那条回归拟合在五个都比基准差的臂上
  （公榜 −4.4%~−14.5%），外推到 +6% 属**样本外**，只能当警示不能当结论；
  且私榜窗口在密封段**之后**，密封期读数在时序上更相关。
- ⚠️ **口径事故（已修）**：候选 meta 缺 slow/fast 三键（`train.py` 的 CLI 里没这概念），
  首次评分时被 `main.py:222` **静默关掉** ⟹ 等于拿「扩展数据 + 丢 slow/fast」比
  「当前数据 + 有 slow/fast」。已加 `sealed_period_eval --slow-fast`（沿用当前标定，
  即 RUNBOOK D1 坑 1 的 (b) 路），重跑后候选身份与生产逐项一致。
- 证据：`outputs/experiments/sealed_extended_vs_production.{json,md}`。

### P2-D2B — D2 的 OOF 交叉验证 —— **预注册写法因果不成立，已改设计**（2026-08-24）

- ⚠️ **原 D2 回答不了它自己那个问题。** 扩展数据是 `time_id 888,480–1,045,889`，
  而原 OOF 折的验证段**全部落在 ≤ 888,479** ⟹ 把新数据放进那些折的训练段就是**拿未来训练**。
  「多给一段更近的训练数据值多少」在原折版图里**因果上无法回答**。
  （本轮第三个「预注册程序回答不了自己那个问题」的例子，前两个见 NOTES 的 §六 与 `INCIDENT`。）
- **合法替代设计**：验证段整体挪到新数据**之后**（`outputs/experiments/d2b_backfill_fold_grid.json`，
  4 折铺满 888,480–1,045,889，**密封段一行不碰**），两臂只差
  新加的 `v3_production_oof.py --train-time-id-max`（只封训练段**后端**，验证段逐位不动）：

  ```text
  臂 A 封顶 888,479   只用原始 train
  臂 B 不封顶          吃到扩展数据
  ```

  ⭐ 网格自带零对照与剂量-反应：fold 0 的训练段被 embargo 卡在 888,473
  ⟹ 扩展臂拿不到额外数据、两臂必须同值；fold 1/2/3 依次多拿约 39k/79k/118k 个 time_id。
  门禁：`tests/test_d2b_fold_grid.py`（7 个用例）。
- ⚠️ **首次执行 OOM**（30 GB、**无 swap**）：折网格用的是**全历史**训练段，
  fold 3 训练约 299 万行 ⟹ 截面设计 441 列 ≈ 5.3 GB + 市场设计 561 列 ≈ 6.7 GB，
  比参考配置那个滚动 78,960 窗（约 2.1 GB）放大 2.5 倍还叠两块。
  改用 `--sample-modulo 10` 筛选档 + `--disk-cache` 重跑（实测 RSS 7.6 GB）。
  ⟹ **本项是筛选档的独立第二读数，不是对 D1 候选的重新测量**（后者是 modulo 5 全量）。

- **结果**（`d2b_extended_vs_capped`）：块均 **+4.23%**、Δ折均 1.189e-04（= 检出下限 1.442e-04 的
  **0.82×**）、配对 CI [−2.55e-05, +2.63e-04] 跨 0 ⟹ 按门禁 **REJECT**。
  **但与 D2a 的 +6.03% 同号。**

  | 折 | 额外 time_id | 基准 peak | Δ | 相对 |
  |---:|---:|---:|---:|---:|
  | 0 | 0 | 0.0017200 | **0.000e+00** | **0.00%** |
  | 1 | 39,346 | 0.0037990 | 1.426e-04 | +3.75% |
  | 2 | 78,698 | 0.0039702 | −1.930e-05 | −0.49% |
  | 3 | 118,050 | 0.0017590 | 3.525e-04 | **+20.04%** |

- ⭐ **零对照逐位命中**：fold 0 的 Δ **恰好等于 0.0** —— 扩展臂在该折拿不到任何额外数据时
  两臂逐位相同 ⟹ **配对本身没坏**，+4.23% 不是对齐误差造出来的。
- ⭐ **剂量-反应 corr = +0.751**（3 个有效折），最大增益 +20.04% 落在额外数据最多的 fold 3。
- ⚠️ **口径订正**：脚本报的 `positive_folds: 2/4` **低估了** —— fold 0 **构造上不可能为正**，
  诚实分母是 3，即 **2 正 1 负**。这类「分母里混进不可能为正的格子」是判据读数的常见陷阱。
- ⟹ **两把尺子、两个窗口、两套折，符号一致（+6.03% / +4.23%）。**
  按 2026-08-24 预注册的决策规则：「停下来问用户」的条件（两尺符号相反且各自超下限）**未触发**，
  三个否决项**一个都没触发** ⟹ 继续以扩展重训件为交付件。
- 证据：`outputs/experiments/d2b_extended_vs_capped.{json,md}`、`d2b_backfill_fold_grid.json`。

### P1 — 8/23 数据更新审计 —— `CLOSED`（2026-08-24，审计完成，四项差异已修）

- **结论**：回补包是**公榜窗口的标签回填**，不是新特征数据。3,217,458 行与 `data/test`
  **逐行 row_id 相同、323 个特征逐 bit 相同**，新增 `weight`/`target`/`responder_00..46`。
  时间上与 train 无缝衔接（888,479 → 888,480）。
  D0.2 判 `added=[009,010,011] / row_delta=+3,217,458`；D0.2b 判
  **`backfill_has_responders`** ⟹ 触发 D3.5 重开条件。
- **⚠️ 审计同时查出四个「预注册 RUNBOOK 与实际交付形态对不上」**，其中两个会直接掐死 D1
  （回补包与本地分区**同名内容全异**；密封段边界**无任何机械手段**；D1 命令计划
  `--train-partitions 999` 撞 `v1_ridge/train.py:261` 一跑就崩；**岭回归身份不在任何门禁表里**
  —— 同型事故第六次，且是第一次发生在「已冻结组件被计划重训」这个方向）。
  五道门禁已装并有回归用例，全量 **297 passed / 41 subtests**（基线 273）。
  详见 `NOTES.md` 2026-08-24 条目。
- **数据根**：`outputs/data_roots/{extended_full,decision}`，由
  `scripts/build_extended_data_root.py` 生成；`data/` 全程只读未改。

  | role | 分区 | 行数 | time_id 上界 | 用途 |
  |---|---:|---:|---:|---|
  | `extended_full` | 12 | 16,445,150 | 1,105,919 | D0.2 审计、D4.5 最终交付件 |
  | `decision` | 11 | 15,588,381 | **1,045,889** | D1 决策期重训 |

- **原状态**：`BLOCKED_UNTIL_DATA_REFRESH`
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
  1. 完整 unittest（当前基线 **273 passed / 41 subtests**，2026-08-23 P12 后）；
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

### P10 — 密封期尺子 —— `RESULT`（2026-08-24 标定完成）：**排序满分，区间无力**

**Tier 1 标定读数**（基准 = 当前生产 `long512`，密封段 856,319 行 / 60,000 time_id，
基准 pooled peak = 0.0065590）：

| 臂 | 密封块均Δ% | 正块 | 去最好块 | 配对 CI | 公榜Δ% | 符号 |
|---|---:|---:|---:|---|---:|---|
| `mkt_shrunk` | **+1.62** | 3/4 | +0.59 | [−3.65, +7.18] | −4.44 | **反号** |
| `asset_adapter` | **+1.58** | 3/4 | +1.01 | [−3.57, +7.26] | −4.60 | **反号** |
| `mktwe` | −2.81 | 2/4 | −5.48 | [−13.54, +5.83] | −5.16 | 同号 |
| `r960`（负控制） | −5.43 | 1/4 | −9.32 | [−18.47, +5.10] | −10.10 | 同号 |
| `xs_shrunk`（负控制） | −9.95 | 0/4 | −12.84 | [−23.66, +0.62] | −14.49 | 同号 |

五个臂**全部判 FAIL**（都没到 +3%，且配对 CI 下界都不 > 0）—— 这是对的，它们在公榜上本来就
全都比当前生产差。

#### ⭐ 结论一：排序**满分**

**Spearman ρ = 1.0000（5/5 名次完全一致）**，Pearson r = 0.9505（p=0.013）。
密封期把公榜的名次一个不差地复现了：

```text
密封期  mkt_shrunk > asset_adapter > mktwe > r960 > xs_shrunk
公榜    mkt_shrunk > asset_adapter > mktwe > r960 > xs_shrunk
```

回归 `sealed = 1.054 × public + 5.18pp` —— **斜率≈1**（几乎无偏），
但有 **+5.18pp 的加性偏移**：非生产臂在最近这段上被系统性地抬高约 5 个百分点。
⟹ 这不是测量偏差，是 P1-R 那个「离散度 149.7%→56.9%」的定量版：
**当前生产的领先幅度在最近窗口里缩水约 5pp**。两个读数各自都对，只是量的是不同时段。

#### ⚠️ 结论二：区间判据**几乎没有牙**

**没有任何一个臂的配对 CI 排除 0** —— 连公榜低 **14.49%** 的 `xs_shrunk`，
CI 都是 [−23.66%, +0.62%]，**跨 0**。配对 bootstrap CI 半宽：

```text
xs_shrunk 12.14%   r960 11.79%   mktwe 9.68%   asset_adapter 5.41%   mkt_shrunk 5.41%
中位 9.68%   均值 8.89%   最大 12.14%        （对照：OOF 1s160/5折 6.1%，3s480 8.7%）
```

⟹ **`--detection-floor` 登记为 0.0968（中位半宽）**，并记明：本次标定连 −14.5% 都没测出来
⟹ 预注册的「≥3%」那道门在这把尺子上**不可达**，第 6 道（CI 下界 > 0）实际上也不可达。

#### ⟹ 这把尺子该怎么用

**用序不用值。** 有真实鉴别力的是**块级符号/名次**（ρ=1.0、去最好块、正块数），
不是幅度和显著性。这恰好印证了 8/24 预注册的决策规则：
其第 3 条否决项「4 块中 ≥3 块为负」走的正是序，而依赖检出下限的第 2 条基本不会触发。

⭐ 顺带结掉 P10 立项时点名的悬案：`asset_adapter`（OOF +1.99% / 公榜 −0.17%）
密封期给 **+1.58%、3/4 块正**、名次第 2 —— 与公榜名次一致，站 OOF 那边；
但仍未过门禁，**不采纳**。

证据：`outputs/experiments/sealed_tier1_calibration.{json,md}`。

⚠️ **本次标定曾整轮作废并重跑** —— 首轮六个臂里四个用的是 `train.py` 的占位
`blend_weight=0.5 / scale=0.856`，不是那个有公榜真值的模型。
重跑后六个臂的 `max|pred|` 与留档公榜 CSV **逐位对上**（比值 1.0000）才认。
详见 `NOTES.md` 2026-08-24 的 `INCIDENT` 条。

- 以下为立项时的原始记录：

#### P10 原文 —— `PREREGISTERED`（2026-08-20，判据已落盘，等 8/23）

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
| 8/23 | ⭐ **截面块窄 peer 对**（`xs_peer_pair_probe` / `xs_peer_pair_confirm_3s480`） | **`PASS_BUT_BELOW_DETECTION_FLOOR`，但当前设计不可部署**（关键限制见下）。诊断先行：`asset_grouping_diagnostic.py` 用 OOF cache 算 15 资产两两 `e` 相关，逐 time_id 零和约束把均值机械压到 −1/14≈−0.071，**真正的信号是偏离这条基线的对子**——`(0,6)+0.18`、`(2,14)+0.13`、`(1,13)+0.12`，层次聚类在 k=3~5 稳定成群，且"模型解释前的 e"与"解释后残差"两个相关矩阵几乎逐位相同（生产模型完全没碰这部分结构，因为 `asset_id` categorical 分裂看不到"另一个资产这一刻在干什么"）。特征＝partner 资产上一采样 time_id 的 `e`，只加 3 对、1 列。1s160 筛选档 `REJECTED`（pooled +2.39%、4/5 折、去最好折 +1.31%、CI 下界 −0.49%、0.39× 检出下限）；3s480 确认档翻盘：**pooled +3.29%、5/5 折单调递增（+1.71%→+5.04%）、去最好折 +2.93%、`2ΔA(+4.46%)>ΔB(+2.18%)`、bootstrap CI 下界 +2.30%（清楚为正）**，六道门禁过五道，只差检出下限（0.38×）——与长窗 w512 confirm 档当年同一个桶（`PASS_BUT_BELOW_DETECTION_FLOOR`，那次 0.89× 后来公榜验证涨了 +1.66%）。⚠️⚠️ **部署路径当场查出走不通，未动 `train.py`**：特征 `peer_e_lag1` 由**真实 target** 反推（`e = y − 截面均值`），而 `main.py:22` 的 `forbidden={"weight","target",...}` 在 `timeseries_api/runner.py` 交给 `predict()` 前就把 target 剥掉——推理时这个量**不存在**，不是工程不便，是信息本身拿不到。诊断/两次探针全部用的是 OOF cache 里的**真实** `e`/`e_lgbm`（训练期标签已知），这是只在离线分析里成立的 oracle 量，不能原样进候选，否则训练用真值、推理被迫换成别的东西（比如自身历史预测），是 CLAUDE.md §8.4 那类"训练推理不同口径"的坑。**本轮因此未生成候选、未碰公榜、未改生产**。⟹ **重新开放条件**：把特征换成"模型自身对 partner 的历史预测值"（沿用 `PredictionTrail` 的因果状态模式，与 slow/fast 同构）。⭐⭐ **该条件已于同日定价（`xs_peer_deployable_probe`）**：前置只读测量显示把搭档量从真实 `e_j` 换成模型自身 `ê_j` 后，驱动相关 `\|均值\|` 0.01664 → 0.00401（**存活 24.1%**）／当期口径 0.00318（19.1%），**同号数从 6/6 掉到 4/6 和 2/6**；根因是 `corr(e_j, ê_j)` 逐资产只有 **0.023~0.098**，`ê_j` 是 `e_j` 的极弱代理。缓存探针（零训练、验证段 ê 覆盖 100%、逐有向对拆 6 列给 `evaluate_arm`）四臂结果：`oracle_lag1` **+0.69%**（3/4 折、CI 下界为正，但去最好折翻负、0.44× 检出下限 ⟹ **未过门禁**）、`deployable_lag1` **−3.21%（0/4）**、`deployable_now` **−2.41%（1/4）**、阴性对照 `shuffled_lag1` −1.76%。⚠️ 按**预注册原文**（"oracle 过"）判 **`INCONCLUSIVE_NO_DETECTION_POWER`**：线性尺子对该机制检出力不足，可部署臂的阴性结果**不得**升级为"没效果"。⭐ 事后旁证（不在预注册里）：相对阴性对照，oracle **+2.45pp**、`deployable_lag1` **−1.45pp**、`deployable_now` −0.65pp ⟹ 尺子能把 oracle 与噪声列分开，而两个可部署臂落在噪声列同侧或更低。⟹ **实操结论：不推进**；树版另需在训练段生成 `ê`（训练段覆盖实测 0%/25%/50%/75%/100%，直接换列会引入伪时间信号），代价是扩展 fold 版图的 OOF（小时级），只有可部署臂过门禁才值得 | `asset_grouping_diagnostic.{json,md}`（2026-08-23 补落盘）、`xs_peer_pair_probe.{json,md}`、`xs_peer_pair_confirm_3s480.{json,md}`、`xs_peer_deployable_{plan,probe}.json` |
| 8/23 | **截面块市场态交互探针**（`xs_market_state_probe`） | `REJECTED`：给截面块（XS LGBM）加一列 `market_pred_t`（训练折内拟合的行级 LGBM 打 y 的截面均值），让树自己学 `asset_id × market` 交互，六道门禁只过 1 道。pooled **+0.77%**（门槛 3%）、**3/5 折**、去最好折 **−0.23%**（翻负）、bootstrap CI 下界 **−1.43%**（跨 0）、仅 6.1% 检出下限的 0.13×；`2ΔA>ΔB` 通过（ΔA +2.46%/ΔB +3.02%），说明有一丝真信号但方差撑不住。⭐ **与 8/14 `asset × observable regime`（同样 3/5 折、跨期不稳）落在同一个坑**——本轮换了机制（训练时输入列 vs 训练后 2-bin adapter；market_pred_t vs 预测截面 RMS；树自己学交互 vs 手工分箱），结果仍然不稳，说明这不是那次实现细节的问题，是"市场态"这条信息通道本身在当前数据上就稀薄不稳（与市场块样本外 R² 仅 0.0018 的已知事实吻合）。⟹ **"截面块对市场态完全瞎"这条架构缺口由代码事实转为实验证据关闭**，不建候选，生产未动 | `xs_market_state_probe.md` / `xs_market_state_interaction_plan.json` |
| 8/22 | **P9 范围项 ③：把新选列喂给 NN** | `REJECTED`，且**这一条结案本身就是产出**：按 P9 **原规格**（`max_iter ∈ {12,50,150,400}`，不新选点）复跑，天花板 28.8% → **27.4%**，门槛 50%，条件延长未触发。⭐ 真正的信息不在那 ±1.4pp，而在**倒 U 形状一模一样** —— 换了 25 列输入后 50 档仍是峰值、150/400 档仍然崩溃 ⟹ **独立印证 P9 的机制结论：绑定约束是正则化不是特征集**（P9 当时是从形状*推断*，本轮是变更另一个轴而曲线不动）。⟹ **v5 可改项由 3 条收缩到 2 条**。环境自检：另跑一次默认选列的 12 档当锚点，对 17.4287%/20.2833% 偏差 **0.000e+00**（该臂自己的 12 档按设计不该复现锚点）。⚠️ 只测 fold 0、只换 cross 块、只换一种新选列 | `nn_capacity_ladder_respsel.md` |
| 8/22 | ⭐ **Stage C 补测（14 个空白格）** | `REJECTED`：Stage B 七道 check 里未通过的 **16 个族全部只错 `multi_member_family`**（单成员族），因证据不过 **0 个** ⟹ 此前是启发式在挡路。08-18 只补了 r00/r02，剩 14 个从未测过且与 Stage C 冻结的 8 个代表交集为空。本轮 14 臂 × 2 基准 = **28 格无一过门禁**，折均无一为正（最好 `responder_01` −1.25%/−0.94%；同期相关最高的 `responder_03` −3.86%/−4.53%）。两道自检全过：08-18 锚点复现 **0.000e+00**（逐位）、`harness_ok=True`。⭐ 方法学：**「剥掉冻结系数让步」是精确的常数平移**（36 臂恒等式偏差 5.4e−20）⟹ 只改水平不改排序，剥完「转正」的臂里有 0/4 折的 ⟹ 不可能制造发现。⟹ **Stage C 现覆盖全部 24 族 / 47 列**。⚠️ 本机制属母条件明令排除的「线性叠加」族，价值是关严不是收益 | `responder_stage_c_fill.md` |
| 8/22 | **responder 监督的选列判据** | `REJECTED`（前置测量即结案，**未花那次 OOF**）：梯子由 `responder_window_atlas` 自己的 `H_fit_is_equal_weight_MA` 派生（r00/02/03/04/05 + target），**不看与 target 的相关**。五折重合 **170/175/180**/200 低于预注册决策线 190；对照臂（全行 vs complete-case）199~200/200 ⟹ 隔离干净。⭐⭐ 真正做决定的是 churn 诊断：**换掉的列原排名最高 #16、换进来的低到 #314**（共 323 列）、全局 Spearman 0.715~0.838 ⟹ 不是截断线附近的搅动，是全局实质分歧 —— **而预注册写下的机制恰恰是「边缘搅动/降方差」⟹ 机制未兑现**，按规则不跑 OOF。事后敏感性：逐级标准化后 175→182，仍低于 190 ⟹ 裁决稳健。自检：自算相关与 `select_features` 的 top-k 10 次比较全部逐位相同（**未改动那个 97 处调用点的生产函数**）| `responder_selection_probe.md` |
| 8/22 | ⭐ **responder 族群表**（`RESULT`） | 只读 parquet row-group 统计（不加载数据）：缺失数是**窗口指纹**、取值域是**维度指纹** ⟹ 47 列切成 **8 个维度族，7/7/7/7/3/7/5/4 = 47**，且 **8 族只用 3 条截断梯子**（422/934/2397 → 4 族；526/1035/2490 → 2 族；4/9/24 → 2 族）⟹「同缺失数 = 同窗口，不同维度」有了直接证据。⭐ 与 `responder_analysis.py` 的 24 族聚类是**正交的两把刀**：那把按 `1−\|corr\|` 聚出来的其实是**窗口组**（cluster 13 横跨两个量纲不同的维度但窗口相同），本表是**维度组** —— 这解释了为什么现有聚类会把 a 族与 e 族的每个成员切成单成员族，进而被 `multi_member_family` 系统性筛掉。⚠️「像什么」是解读不是主办方语义 | `responder_family_grid.md` |
| 8/21 | ⭐ **长窗 w512 公榜裁决** | **0.0041833953（+1.662%，新最好）**，同 scale 1.16、两份 CSV 均 0 行触限 ⟹ 不依赖任何近似。峰值口径 +1.72~1.74%（B_old 拨 ±8% 不翻），Σp²(新)/Σp²(旧)=1.0128 ⟹ **无隐藏 scale 效应**。⟹ 分数 +1.66% ≈ **IC +0.83%**，与榜首 IC 差距 +20.8% → **+19.7%**，填掉约二十分之一。⚠️⚠️ **迁移率 0.22×**（确认档截面块 +7.77% → 公榜全模型 +1.72%），按占分 58.8% 打折仍只 0.37× —— **连续第二次本地高估**（slow/fast 0.51×），此前③类全是低估（1.20×/1.6×/2.3×）| ledger / `long_window_confirm.md` |
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
| 2026-08-23 | ⚠️ **`INCIDENT`（未爆，8/23 前拦下）：重训计划漏掉 `long_window`，并给这类漏键装了机械门禁**。核证据时发现 `strategies/v3_hybrid/train.py:335` 的 `--long-window` **默认 0（＝关闭）**，而 `scripts/retrain_extended.py` **从未传过它**，`production_structure()` 派生的 8 个键与 `BASELINE_CHECKED_KEYS` 的 7 个键里**也都没有它** ⟹ 8/23 跑 D1 会训出一个**没有长窗**的候选，而长窗正是 08-21 转正、公榜实测 **+1.662%** 的那块结构。转正门禁最终会拦下（`PUBLIC_BASELINE` 含 `long_window: 512`），但那是在**几小时训练之后**，而 `BASELINE_CHECKED_KEYS` 上面那行注释写的恰恰是「8/23 之前就要红，而不是训练几小时之后才红」—— 该守卫在 08-21 长窗转正后没有同步。⚠️⚠️ **这是同一类事故的第四次**（08-18 slow/fast 丢键 → 08-19 `--weighted-cross-section`/`--market-model` 漏传 → 08-21 `long_window` 漏进 `PUBLIC_BASELINE` → 本次漏进重训计划）⟹ 逐次补洞已被证明不够，新增 `tests/test_model_identity_key_coverage.py`：遍历 `PUBLIC_BASELINE` 全部 13 个键，断言四个消费者（audit / retrain / delivery / 打包）都覆盖，覆盖不了的必须写进显式豁免表**并附理由**（沿用 `make_submission.EXCLUDED_MODULES` 那套「偏离必须是按下去的」）。⭐ **验收方式是先让门禁红**：打补丁前实测 `AssertionError: ['long_window'] != []`，补丁后转绿 —— 证明它真会抓，不是恒真断言。顺带查明交付报告那一处 08-21 已补过（该臂本就是绿的）。dry-run 走真实 CLI 复核：v3 命令现含 `--long-window 512`，值从生产 meta 派生。⚠️ 过程中踩到一个小坑并写进 RUNBOOK：挑 v3 命令不能用「字符串含 `v3_hybrid`」，候选目录名 `v3_hybrid_extended_fixed` 里也含它。全量 **261 passed / 28 subtests**（原 254/26）。生产目录与模型身份未动，未执行任何重训。 |
| 2026-08-23 | ⭐ **peer 对轴收口：重新开放条件已定价，判 `INCONCLUSIVE` 而非 `REJECTED`**。8/23 的 `xs_peer_pair_confirm_3s480` 在 3s480 上过五道门禁（+3.29%/5-of-5/CI 下界 +2.30%），但特征是 oracle（由真实 target 反推的 `e`，推理端被 `forbidden` 剥掉）。ROADMAP 留的"换成模型自身预测"至今没有数字，本轮补上：只读前置测量 —— 驱动相关 `\|均值\|` 0.01664 → 0.00401（**存活 24.1%**）、同号 **6/6→4/6**；根因 `corr(e_j, ê_j)` 只有 0.023~0.098。⚠️⚠️ **"换列重跑 6 分钟"这条捷径经查不成立**：`e_lgbm` 只在 `fold>=0` 行有值，训练段覆盖是 **0%/25%/50%/75%/100%**，fold 0 两臂等价、fold 1-3 覆盖率与时间强相关 ⟹ 树会学到伪时间信号。改用缓存探针（零训练 2.7 秒、验证段 ê 覆盖 100%、逐有向对拆 6 列喂 `evaluate_arm`，既有函数一行未改）：oracle +0.69%（未过门禁）、deployable −3.21%/−2.41%、阴性对照 −1.76%。⭐ **自查一处预注册与实现不一致**：预注册写"oracle **过门禁**"、初版代码写成"oracle **为正**"，而实测恰好落在两者之间 —— 按严格的那条判 `INCONCLUSIVE`，代码已对齐预注册而不是反过来。⭐ 顺带补上 `asset_grouping_diagnostic` 的产物缺口（此前只 print，ROADMAP 自己标着"无产物文件"），并加一道与 ROADMAP 记录的三对相关值的对拍断言；同时订正一处弱论证："两个相关矩阵几乎逐位相同"是算术必然（模型只解释约 0.4% 方差），该结论真正的支撑是 `asset_id` categorical 分裂看不到别的资产当刻值这个代码事实。全量 **254 passed / 26 subtests**（新增 9）。生产目录与模型身份未动。 |
| 2026-08-23 | ⭐ **P11 评测环境资源门禁 `RESULT`**：核出交付验证从来没有内存口径，而官方环境是 4 核 / **12 GB**（`docs/competition_description.md:158-159`）、超限可判提交无效、且私榜截止后无法修改。首次在 `MemoryMax=12G` + `AllowedCPUs=0-3` 下走官方 runner 全量：峰值 RSS **11.47 GB = 上限的 95.6%**、cgroup `memory.events` 记到 `max 990`（990 次顶到上限被迫回收、`oom_kill 0`）。⭐⭐ **决定性的一步是做归属**：新写 `measure_harness_memory.py`，用零预测桩模型走同一条 `run_loaded_model`，测得 harness 单独就要 **11.09 GB**。⚠️ 但 harness 臂自己跑三次就摆动 11.09–11.47 GB，与「模型净增」同量级 ⟹ 只能说**模型贡献 ≤ 约 0.5 GB、与跑间波动不可分辨**。1 秒间隔追踪定位到峰值在 **18.0s / 36.8s = 49% 处**（分区加载段），后半程遍历 214,538 个 time_id 与最后的 `pd.concat` 一点没涨 ⟹ **峰值由分区大小决定、不随运行长度增长**，9 月更长的实盘期不会推高它。⟹ **不要为内存改模型**。NumPy 兜底同口径 **11.55 GB / 10.90 分钟**，两条路径除 `peak_rss_has_headroom` 外门禁全过。⭐ **同日用户在真实评测机（JupyterHub）上把兜底也跑完了**：峰值 **10.93 GB（91.1%）比本地还低 0.62 GB**、**33.78 分钟 / wall 36.28 分钟**、3,217,458 行 / 0 超时 / 0 非有限 / 0 触 clip ⟹ **lightgbm 万一不可用，兜底不会 OOM，只会慢**。用户实测该机超 12 GB 即 OOM ⟹ 官方那条 12 GB 是硬限（`FACT`）。⚠️ 我跑前按本地比例外推 25 分钟、实测 33.78，**低估 35%** —— 兜底是单核绑定而主路径吃 4 线程，两者对「云端单核更慢」的敏感性不同（云端/本地：主路径 2.25× vs 兜底 3.10×）⟹ CLAUDE.md §5.7「代理量不可跨结构搬用」在**跨环境**上同样成立。⭐ 单步耗时把风险排序反了：兜底单步最大仅 **0.050 s**，是主路径 **2.802 s** 的 1/56 ⟹ 单步超时风险在主路径不在兜底。⭐ 浮点差异分两轴：同机换后端 1.4e-15、同后端换机 2.3e-8，跨机器主导但传到 Score 仍约 1e-8 可忽略。跨 predict 状态也定价了：`AssetLongWindow` 固定 2.46 MB，`PredictionTrail` 唯一无界但实测 **314.6 B/time_id**（40 万步线性），公榜期仅 64 MB、余量还能吃 8.4 倍 ⟹ 安全。顺带查明两份交付报告长期判 FAIL 是 `--manifest` 默认值写死 `slowfast` **比错了对象**（生产与 `long512` manifest 8 文件逐字节全中），改为 `auto` 解析并新增非循环的 `model_matches_public_baseline`（复用 `audit_submission_zip.public_baseline_drift`，不另抄表）。另订正 `VmHWM` 在本机内核上非严格单调（221.49→220.94 MB），`peak_rss_bytes()` 叠模块级高水位保证不低报。全量 **241 passed / 26 subtests**（新增 14）。生产目录与模型身份未动。 |
| 2026-08-23 | ⭐ **截面块窄 peer 对：确认档 `PASS_BUT_BELOW_DETECTION_FLOOR`，但部署路径查出走不通**。用户提出"15 个资产能不能分组估计"，诊断出 3 对残差共动明显偏离零和基线的资产（`(0,6)(2,14)(1,13)`），1s160 筛选档 `REJECTED`（+2.39%、4/5 折）但机制干净（`2ΔA>ΔB`）；升级到 3s480 确认档翻盘：pooled +3.29%、5/5 折单调递增、bootstrap CI 下界 +2.30%，六道门禁过五道，只差检出下限——与长窗 w512 当年 confirm 档同一个桶。用户提议趁 8/23 公榜停更前花一次配额验证，动手前查 `main.py`/`timeseries_api/runner.py` 的 `forbidden` 集合，发现特征依赖的 `e`（真实 target 反推）在推理时不存在（target 被剥掉）——诊断和探针全部用的是 OOF cache 里训练期已知标签的 oracle 量，不能直接进候选。**未生成候选、未碰公榜、未改生产**，原样记录，重新开放条件是换成模型自身历史预测值（causal，同 slow/fast 的 PredictionTrail 模式）后从头验证。 |
| 2026-08-23 | **截面块市场态交互探针 `REJECTED`**：核出代码事实——截面块设计矩阵对市场态完全不可见（无市场块预测值/聚合量），新写 `experiments/xs_market_state_probe.py`（评价器用树，不用线性，因为假设的机制是 `asset_id×market_pred_t` 的非线性分裂）跑满 5 折 1s160 筛选档。pooled +0.77%、3/5 折、去最好折翻负、CI 下界跨 0，六道门禁仅过 1 道 ⟹ `REJECTED`。与 8/14 `asset × observable regime`（同样 3/5 折不稳）互相印证：两套不同机制（训练时输入列 vs 训练后 adapter）在同一条信息通道上得到同一个不稳定结论，判定该架构缺口已由证据关闭。生产目录与模型身份未改动。 |
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
| 2026-08-21 | **长窗 w512 候选已就绪，等公榜裁决**（生产未动）。新增 `history.AssetLongWindow`：`AssetHistory` 是 O(window)、512 窗在线跑不动，改用**持久累积和相减**（同 `PredictionTrail`）；逐位一致是构造上的 —— 离线 `np.cumsum(float64)` 本身定序累加、在线持久 running total，实测 max|Δ|=0，而**分块重起**的 cumsum 不同（正是 `history.py` 警告的写法）。11 个用例钉住「离线整块 ≡ 离线分批 ≡ 在线逐 time_id」。长窗块**只进截面设计**（训练日志自证：截面 441 列、市场仍 561 = raw200 + 截面块 **361**）；只重训截面森林，市场森林与冻结岭回归 hash 与生产**逐字节相同**；只多一个 meta 键。门禁：缺键时生产仍 4.019e-09、交付配置 4.019e-09（两后端）、双后端 1.39e-16、4 核全量 LightGBM **5.33 分钟**（+1.3%）/ NumPy 兜底 **10.55 分钟**（−3.6%）、0 超时 / 0 非有限值 / **0 触 clip**（max|pred| 0.402099）⟹ 与公榜 0.0041150085 同 scale 可直接比大小、二次式精确。⭐ 一致性数值与生产相同（4.019e-09）经直接对拍排除「加载错模型」（森林 361 vs 441 特征、预测 max|Δ| 5.11e-02）—— 相同是良性的，说明**长窗块对训练/推理差异贡献为 0**。顺带补三个登记缺口（`PUBLIC_BASELINE`/两张取值表/`verify_delivery_runtime` 的身份摘要），其中 `make_submission.py:170` 的防呆当场抓到我漏改第二张表。全量 **174 passed / 27 subtests**。 |
| 2026-08-21 | **长窗 w512 公榜裁决：0.0041833953，+1.662%，新最好成绩**（生产**仍未转正**）。峰值口径 +1.72%，且对 `B_old` 的 ±8% 扰动稳健；`Σp²` 实测**上升** 1.28% ⟹ 排除「最优 scale 上移导致 1.16 处低估」。⟹ 折成 **IC +0.83%**，与榜首差距 +20.8% → +19.7%。⚠️⚠️ **本项目连续第二次③类本地高估**：迁移率 0.22×（slow/fast 那次 0.51×），而 08-13 之前③类全是**低估**（1.20× / 1.6× / 2.3×）。**符号翻转本身是信号** —— 两次高估的共同点是「在已有块上加派生量」（预测的慢/快拆分、特征的长窗摘要），而三次低估都是「加新的信息通道或新分量」（相位采样、history40、行级市场森林）。⟹ 本地尺子对**派生量**类改动系统性乐观，这一条应写进③类的先验。⟹ 是否转正由用户定：+1.66% 是真涨且门禁全过，但代价是新增一个跨 predict 状态。 |
| 2026-08-22 | **responder 轴收口：三种用法全部走完，由证据关闭**。① 族群表 —— 缺失数/取值域两个指纹把 47 列切成 8 个维度族（7/7/7/7/3/7/5/4），8 族只用 3 条截断梯子，且与现有 24 族聚类**正交**（那把切的是窗口组）。② Stage C 的 **14 个空白格**填满 —— 未通过 Stage B 的 16 个族**全部只错 `multi_member_family`**、因证据不过 0 个 ⟹ 此前是启发式在挡路；补测 28 格**无一过门禁**，08-18 锚点逐位复现 0.000e+00。③ 用 responder 打分选列（唯一没做过、也唯一不在母条件排除项里的用法）在**前置测量**处结案、**省下那次 OOF** —— churn 诊断显示换掉的是原排名 #16 的列、Spearman 0.72~0.84 ⟹ **预注册的降方差机制未兑现**。⭐ 方法学产出一条：**「剥掉冻结系数让步」是精确的常数平移**（36 臂恒等式偏差 5.4e−20），只改水平不改排序，剥完「转正」的臂里有 0/4 折的 ⟹ 不能当发现读；08-18 那个被反复引用的 `pure_e/responder_00` +3.92% 也是同一回事。工程：`horizon_auxiliary_cache_probe` 的门禁+bootstrap 抽成 `evaluate_arm()` 供复用，**抽取前后 JSON 逐字段相同**（仅两处 `nan!=nan`）；`multitask_mlp` / `nn_capacity_ladder` 各加一个**默认不改变行为**的入口（后者复跑默认路径，JSON 只多一个新键、数值逐字节不变）；**未改动 `select_features`**（97 处调用点、在生产训练路径上），改为逐折硬断言自算相关与它的 top-k 逐位相同。全量测试 **217 passed / 26 subtests**（原 174/26，新增 43）。⚠️ 顺带订正：多处文档记的「27 subtests」实测是 **26**，本轮之前就已差一个。 |
| 2026-08-22 | **P9 范围项 ③ 结案**：把发现 3 的新选列（与原判据重合 175/200）喂给 `multitask_mlp` 按原规格复跑整条阶梯 —— 天花板 **28.8% → 27.4%**（门槛 50%），`REJECTED`、条件延长未触发。⭐ 关键读数是**曲线形状不变**（50 档峰值、150/400 崩溃照旧）⟹ 给「绑定约束是正则化不是特征集」补上一条**正交**证据。⟹ v5 可改项 3 → 2 条。工程：`--cross-selection-override` 默认不改变行为；因该臂的 12 档按设计不复现 08-19 锚点，另跑一次默认选列的 12 档做环境自检（偏差 0.000e+00），`nn_capacity_ladder` 相应新增 `--anchor-label` / `--summary-label`（默认行为不变，已复跑验证）。 |
| 2026-08-22 | **P0-B 结案**（用户执行）：重打私榜包 `v3_hybrid_submission_20260822.zip` 并落盘审计 —— `passed: true` / `public_baseline_drift: []` / `unexpected_modules: []` / `missing: []` ⟹ 长窗 w512 转正后的模型身份已装进包里（此前 `20260819` 那份缺 `long_window` 键，会**静默**交出低 1.66% 的旧模型）。⚠️ 盘上现有**五个** v3 zip，8/31 只能交 `20260822`。 |
