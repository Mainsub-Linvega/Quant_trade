# NOTES.md — 研究上下文、方法与近期日志

> 本文件服务于人与 AI 的日常研究交接：保留**当前问题、实验方法、常用命令和近期推理过程**。
> 当前行动见 [`ROADMAP.md`](ROADMAP.md)；长期协作规则见 [`CLAUDE.md`](CLAUDE.md)；完整旧笔记和
> 主题化历史见 [`research_history/`](research_history/README.md)。

## 1. 如何记录一项探索

新条目使用以下模板，避免只记结果不记推理：

```text
日期：
标签：FACT / RESULT / HYPOTHESIS / REJECTED / SUPERSEDED / INCIDENT
问题：
动机与机制：
实验设计与固定项：
结果：
解释与限制：
决策：
证据：
后续问题：
```

约束：

- `FACT` 只能由用户新增；AI 只能引用。
- `RESULT` 必须说明数据、切分、比较基线和指标口径。
- 负结果保留重新开放条件，避免后续重复试验。
- “当前最好”“生产”“待办”只写在 ROADMAP；本文件用日期和模型名描述当时状态。

## 2. 当前研究上下文（2026-08-19）

### 生产模型

当前生产为 `strategies/v3_hybrid/model/`，来源候选 `v3_hybrid_slowfast`，公榜
**0.0041150085**（2026-08-18 转正）。核心结构：

```text
ridge market component
  blended with λ=0.5
row-level unweighted LGBM market component
+
weighted LGBM cross-sectional component (replace, blend_weight=1.0)
+
asset history40 features
→ prediction_scale 1.16
→ slow/fast 分离（逐 asset 自身预测的因果滚动均值，K=2000 真实 time_id 步；
   slow 0.4496 / fast 1.2530，即 1.16 × relative 0.3876 / 1.0802）
→ clip 0.5
```

精确配置以 `strategies/v3_hybrid/model/hybrid_meta.json` 为准；不要从本节复制参数生成模型。
⚠️ 本节 08-13 那版写的是 `v3_hybrid_mkt_shrunk` / 0.0039977510，已按 slow/fast 转正更新；
「当前最好」的唯一真值在 ROADMAP §2 与 `experiments/ledger.csv`。

### 当前问题

1. ~~**交付风险**：当前模型的精确全量 wall-clock 尚需在最终环境复测；NumPy 双森林兜底约 15 分钟。~~
   `RESOLVED`（2026-08-18）：4 核下两条路径全量实测并落盘 —— LightGBM `predict_total` 5.26 分钟、
   NumPy 兜底 10.94 分钟（2.08×，且**单核绑定** ⟹ 不随核数恶化）。
   证据：`outputs/experiments/delivery_runtime_{lightgbm,numpy_fallback}_4t.{json,md}`。
   剩余交付事项只有用户执行打包 + 落盘 zip 审计（ROADMAP P0 动作 4）。
2. **数据更新**：8/23 若收到回补数据，必须先审计 train split 是否改变，再决定重训。
   执行细节全部写死在 `RUNBOOK_8_23.md`，当天不需要再做设计决策。
3. **本地尺子**：alpha、轮数和 history 宽度曾与公榜量反；回补标签优先用于重建评估可信度。
   ⟹ 这是**唯一仍然开着**的老问题；回答它的动作是 RUNBOOK D0.3（`experiments/public_replay.py`
   离线复算 21 份历史公榜 CSV）。⚠️ 那 21 份 `outputs/submission_*.csv` 是它的全部原料，不能清理。
4. ~~**剩余研究轴**：市场森林独立截短有机制依据且无需重训，但未排期。~~
   `CLOSED_FAIL`（实验其实早已跑完，08-17 才同步到面板）：160/240/320/400/480 全部 FAIL、
   `Selected: None`。证据：`outputs/experiments/v3_market_round_scan_phasebal_prodwindow.md`。
5. **扩展数据复验**：V4-R regime 是唯一保留的原规格结构复验；其他 V4 和 responder 路线关闭。
6. **交付链路的「装错东西」风险**（2026-08-19 新增）：模型身份有 `PUBLIC_BASELINE` 把关，
   但**包内容身份**此前无人把关（「除 `train.py` 外全收 `*.py`」），重训计划也不复现生产结构。
   两处现在都有门禁，见本文件同日日志。

## 3. 研究判定方法

### 比赛指标

统一使用 `src/metric.py`：

```text
Score = 1 − Σ w(y−ŷ)² / Σ wy²
      = 2aA − a²B          （当预测只乘 scale=a 且未触 clip）
peak  = A² / B
IC    = sqrt(peak)
```

- 调 scale 只改变抛物线位置，不改变模型 IC。
- 比模型优先比较 peak/IC；或在相同 scale、相同 clip 条件下比较原始公榜分。
- 触发 clip 后二次式不再严格成立，必须单独报告触限行数。

### 时序验证

- 使用 walk-forward；训练段与验证段之间保留 embargo。
- 特征选择、稳健变换、超参内层选择全部在训练折内拟合。
- 结论看配对增量、正向折数、去最好折和机制拆解，不看单个绝对 validation score。
- offset 对照用于估噪声地板；效应没有明显超过边界漂移时写“测不出来”，不写“没有效果”。

### 参数分类

1. **后处理参数**：scale 等；可利用精确代数或已验证线性关系，通常无需重训。
2. **拟合紧密度参数**：alpha、轮数、容量、样本密度；本项目多次本地/公榜量反，需真实测试期裁决。
3. **结构/信息参数**：新历史状态、新分量、新损失；可看本地点估计和机制，但仍需独立确认。

该分类的历史来源见
[`research_history/validation-and-calibration.md`](research_history/validation-and-calibration.md)。

## 4. 常用只读与验证命令

### 单元测试与语法检查

```bash
.venv/bin/python -m pytest tests -q          # ⚠️ 必须用 pytest，见下
.venv/bin/python -m compileall -q src strategies scripts tests timeseries_api experiments examples
```

⚠️⚠️ **不要用 `python -m unittest discover -s tests`**（2026-08-21 查出）。
仓库里有 7 个测试模块是 **pytest 风格**（裸 `def test_x()`，没有 `unittest.TestCase` 子类），
而 `unittest discover` **只收 TestCase 子类** —— 对这些文件既不报错也不报 skip，直接当不存在，
然后打印一个绿色的 `OK`。实测差 **36 个用例**：

```text
pytest              122 passed / 22 subtests      ← 全部
unittest discover     86 passed                   ← 静默少跑 36 个，仍报 OK
```

被静默跳过的是：`test_sealed_period_eval`(11)、`test_oof_cache`(6)、
`test_nn_capacity_ladder`(6)、`test_structural_signal_screen`(5)、`test_v3_asset_adapter`(3)、
`test_v3_residual_signal_search`(3)、`test_v3_sparse_residual`(2)。
⚠️ **头两个正是给 P10 密封期尺子和 OOF 缓存出处隔离把关的那些用例** ——
ROADMAP 里「11 个用例」「3 个回归用例钉住」引用的就是它们。
ROADMAP 历史记录的 `112 passed / 22 subtests` 是 pytest 的数；本文件此前文档的却是 unittest 命令
⟹ 按文档执行的人拿到的是 86，且不会察觉。

### 训练/推理一致性

```bash
.venv/bin/python scripts/check_consistency.py \
  --strategy v3_hybrid --n-time-ids 500

.venv/bin/python scripts/check_consistency.py \
  --strategy v3_hybrid --backend numpy --n-time-ids 500
```

### 数据更新审计

```bash
.venv/bin/python scripts/audit_data_release.py \
  --data-root data \
  --baseline outputs/data_audits/data_release_20260812.json \
  --output outputs/data_audits/data_release_<YYYYMMDD>.json
```

最终审计不得使用 `--no-file-hash`。固定结构重训必须消费比较后的 audit JSON，先 dry-run：

```bash
.venv/bin/python scripts/retrain_extended.py \
  --audit outputs/data_audits/data_release_<YYYYMMDD>.json
```

### 预注册联合矩阵

```bash
.venv/bin/python experiments/joint_recalibration.py
```

输出计划位于 `outputs/experiments/joint_recalibration_plan.json`。不得在看结果后扩大网格。

### 人工专属交付动作

以下命令仅记录入口，AI 不执行：

```bash
# 用户确认候选后才可转正
.venv/bin/python scripts/promote_v3_candidate.py ... --activate --allow-production-overwrite

# 私榜打包由用户执行
.venv/bin/python scripts/make_submission.py --strategy v3_hybrid
```

## 5. 近期研究日志

> ⚠️ 本节多条旧记录引用 `NEXT_STEPS_*.md`（如 `NEXT_STEPS_horizon_auxiliary_oof_validation.md`
> 的 §3 / §5 / P2）。那是当时的**本地工作笔记，从未入库**（`git log --all -- '*NEXT_STEPS*'` 为空）。
> ⭐ **2026-08-20 订正**：此前这里写「现已不在盘上」是错的 —— 它们在
> `~/Documents/Quant_trade_notes/` 下（`NEXT_STEPS_horizon_auxiliary_oof_validation.md` 32KB、
> `NEXT_STEPS.md`、`PLAN.md`、`ARCHITECTURE.md` 等），章节号与引用对得上，**可核**。
> 那是一个独立的笔记目录、不是 git 仓库，所以 `git log` 查不到 ≠ 文件不存在。
> 按 CLAUDE.md §7「旧结论不删除」，引用原样保留以说明当时的推理来源；
> 可核验的**实验证据**一律仍以 `outputs/experiments/` 与 `experiments/ledger.csv` 为准。

### 2026-08-22 — `REJECTED`：responder 的三种用法全部走完 —— 这条轴现在由**证据**关闭

**日期**：2026-08-22
**标签**：`REJECTED`（Stage C 补测 + 选列判据）／`RESULT`（族群表）

**问题**（用户提的三问）：「responder 之间的信息我们之前有考虑过吗？能否构造一些交叉特征？
或者用 responder 来筛 features？」

先纠正一个前提：**主办方从来没有说过「responder 不可以在 NN 里用」。** `docs/` 全量 grep
无此限制。真实约束是物理的 —— test/私榜 parquet 只有 326 列、根本没有这 47 列
（`docs/data_description.md:173,175`；实测 train 375 列），且 `timeseries_api/runner.py:74-84`
会主动剥掉 `responder_` 前缀。⟹ **推理端用**不可能；**训练端用**完全合法（P8 就用过）。

逐条核完：前两问**做过且做得很透**（8 次实验、约 55 份产物、`ledger.csv` 零条公榜提交）；
第三问**零次**，而且它是唯一不落在 `responder_reaudit_20260814.md:93-100` 母条件
「不是换目标 / 线性叠加 / 对预测值做二层校准」排除项里的用法。本轮把这三件事各推进一步。

---

#### ⭐ 发现 1（`RESULT`）：47 列是一张「维度 × 窗口」的网格，与现有的 24 族聚类正交

只读 parquet 的 row-group 统计信息（**不加载数据**）就能读出两个指纹：

```text
缺失数 null_count   ⟹ 窗口指纹。responder 构造自「未来不可见区间」，窗口越长，分区末端
                       越多行算不出来。全 47 列只取 11 个离散值。
取值域 (min, max)    ⟹ 维度指纹。非正 / 上界饱和到 1 / 非负 / 双向四类。
```

分族规则（跑前写死）：先按 `sign_class` 切成极长游程，再在**窗口梯子重启点**切开
（`null_count` 在已经上升过之后又下降的位置）。实测切出 **8 个族，大小 7/7/7/7/3/7/5/4 = 47**：

| 族 | 成员 | 取值域 | 截断梯子 |
|---|---|---|---|
| a | 00–06 | `[0, 1]`（饱和到 1）| 422, 934, 2397 |
| b | 07–13 | `[−0.24, 0]` | 422, 934, 2397 |
| c | 14–20 | `[0, 0.13]` | 422, 934, 2397 |
| d | 21–27 | `[0, 0.87]` | 422, 934, 2397 |
| **e** | **28–30** | **`[−4.09, +4.29]`**（与 target 的 ±2.23 同量级）| 526, 1035, 2490 |
| f | 31–37 | `[0, 0.0102]` | 4, 9, 24 |
| g | 38–42 | `[0, 0.0102]` | 4, 9, 24 |
| h | 43–46 | `[0, 0.13]` | 526, 1035, 2490 |

⭐ **8 个族只用 3 条截断梯子**（a/b/c/d 共用一条、e/h 一条、f/g 一条）⟹「同缺失数 = 同窗口，
不同维度」有了直接证据，与主办方原文「覆盖多个预测窗口和多个市场响应维度」逐字对应。

⭐ **它与 `responder_analysis.py` 的 24 族聚类是正交的两把刀**：那把按 `1−|corr|` 聚，
切出来的是**窗口组**（cluster 13 = {25,26,27,44,45,46} 横跨 d/h 两个量纲不同的维度但窗口相同）；
本表切的是**维度组**。这解释了为什么现有聚类会把 a 族和 e 族的每个成员都切成单成员族。

⚠️ **限制**：「像什么」是解读不是主办方语义；四个非负族里哪个是上行/路径/摩擦，本表判不出来，
只报量级不编标签。

---

#### 发现 2 + 裁决（`REJECTED`）：Stage C 的 14 个空白格全部填完，**28 格无一过门禁**

逐族核 `responder_predictability_reaudit_phasebal_prodwindow.json`：

```text
24 族   通过 8   未通过 16
未通过的 16 个族，failed checks **全部恰好** == ['multi_member_family']
因证据不过的：0 个
```

即其余六道门 16 个单成员族全过，是被一条**稳健性启发式**挡在 Stage C 之外的。08-18 的
`horizon_auxiliary_cache_probe` 只补测了 r00/r02 ⟹ **剩 14 个从未测过**，且与 Stage C 实际
冻结的 8 个代表**交集为空**。

**设计**：直接调 `horizon_auxiliary_cache_probe.evaluate_arm`（见下面的工程注记），
14 个主臂 + 2 个自检臂 + 3 个校准臂 × 2 基准；逐臂独立随机流（共享流会让结果依赖臂顺序）。

**结果 —— 28 格全部 FAIL，且折均无一为正**：

| | `full` | `pure_e` |
|---|---|---|
| 最好的一个 | `responder_01` **−1.25%**（2/4 折）| `responder_01` **−0.94%**（2/4 折）|
| 同期相关最高的 `responder_03`（0.817）| −3.86%（0/4）| −4.53%（1/4）|
| 收益类 `responder_28`（0.694）| −2.54%（2/4）| −1.89%（2/4）|

**两道自检都过**：08-18 锚点（r00/r02 × 2 基准 × 6 个点估计）**最大偏差 0.000e+00**（逐位）；
`harness_ok = True`（负控制两基准均不过门禁，`known_negative_27` 两基准均为负）。

**⚠️ 本轮最值得记的方法学观察：「剥掉冻结系数让步」是一个精确的常数平移，不可能制造发现。**
`mean_delta_vs_frozen_baseline` 把「基准可在评估折重解 scale、候选必须冻结」这份让步剥掉，
08-18 就是用它把 `pure_e/responder_00` 从 +1.38% 读成 +3.92%。实测恒等式

```text
stripped(arm) ≡ mean_delta(arm) − mean_delta(null_frozen_scale)
全部 36 个臂的最大偏差：5.421e-20
```

⟹ 对每个臂减的是**同一个常数**：只改水平，不改排序，也不影响正折数 / 去最好折 / bootstrap CI。
剥完之后 `full` 有 15/18 个臂「转正」，但那不是 15 个发现 —— `responder_06` 剥完报 +1.22%
却是 **0/4 折**，`responder_20` 报 +0.37% 也是 **0/4 折**。⭐ 门禁里管用的是正折数、去最好折
和 CI，而这三样对常数平移免疫。

**⟹ Stage C 现在覆盖了全部 24 个族 / 47 列。** responder 这条线从「28% 的列被一条启发式挡着」
变成「全部测过」：**由证据关闭，不再由启发式关闭**。

⚠️ **定位要说清楚**：这一项本身**不满足**母条件（它属于被明令排除的「线性叠加 / 对预测值做
二层校准」机制族）⟹ 它买的是**关严**，不是收益。就算有格子过门槛也只能进 P10 Tier 2。

---

#### 发现 3 + 裁决（`REJECTED`）：用 responder 筛特征 —— 机制**没有兑现**，且先量后跑省下了那次 OOF

**这是三问里唯一没做过、也唯一符合母条件第 ② 条的用法**（用 responder 塑造表示，推理端完全
不需要 responder），同时命中 `ROADMAP.md:456` 给 v5 划的第 ③ 条范围项。

**判据（跑前钉死）**：梯子由 `responder_window_atlas` **自己的** `H_fit_is_equal_weight_MA`
判定派生（筛出 r00/r02/r03/r04/r05，排除 r01 RMSE 0.060 与 r06 0.192），**不看与 target 的
相关** —— 那个论证形式已被 `responder_targets_stage1.md:14-22` 证伪。
新判据 = 梯子上各带符号相关的平均（换的是**估计的方差**，不是被估计的量 ——
与 08-21 `selection_criterion_probe` 那三个臂机制不同）。

**决策规则（跑前钉死）**：重合 ≥ 190/200 ⟹ 判「与 base 不可区分」不跑 OOF；< 190 ⟹ 先验是
掉分，需机制假设才跑。

**结果**：五折重合 **170 / 175 / 180**（最小/中位/最大），history 40 列重合 18~31 ⟹ 低于决策线。
对照臂（全行 vs complete-case 的 `S_base`）**199~200/200** ⟹ 限制行集本身没挪动选列，隔离干净。

**⭐ 但真正做决定的是下面这个诊断，不是重合度这一个数**。两种互斥假设对 OOF 的预测不同：

```text
假设 A 边缘搅动   换掉的列挤在截断线附近（200th=0.00299 / 201st=0.00295，落差 1.33%）
                  ⟹ 效应很小、方向不定
假设 B 实质分歧   换掉的列里有高排名的 ⟹ 两个判据在读不同的东西
                  ⟹ selection_criterion_probe 的「分歧越大掉得越多」适用，先验是掉分
```

实测：**换掉的列原排名最高到 #16**，**换进来的低到 #314**（共 323 列），全局 Spearman 只有
**0.715~0.838**。⟹ **假设 A 被证伪，假设 B 成立**。

⭐⭐ **而我预注册时写下的机制恰恰是假设 A（「把估计方差压下去」）—— 机制没有兑现。**
按预注册规则，「需要一条不依赖『判据更好』的机制假设」这个条件不成立 ⟹ **不跑那次 OOF**。

**事后敏感性检查（不是预注册结果，只用于说明裁决稳健）**：各级 |corr| 中位数从 r00 的 0.0025
到 r05 的 0.0102，差 **4.0×** ⟹ 未标准化的平均确实会被幅度最大的那级偏向。逐级标准化后
fold 0 的重合度 175 → **182**/200 —— **仍在 190 之下** ⟹ 裁决不是这个规格选择造成的。

⚠️ 这一条**只否掉「窗口一致性平均」这一个判据**，不否掉「用 responder 打分」这个想法本身；
但它连同 08-21 的三向封死一起，把先验压得很低。

---

#### 发现 4 + 裁决（`REJECTED`）：把新选列喂给 NN —— **换特征集换不动天花板**，P9 范围项 ③ 结案

`ROADMAP.md:456,460-462` 给 v5 划的三条范围项里，第 ③ 条是「特征选择不再沿用为线性/树挑的
`|corr(feature, e)|` top-200」，重开条件是「三条至少改掉一条后**按原规格复验**」。发现 3 的
新选列正好就是那个「不再沿用」的东西 ⟹ 本轮顺手把这条也结了。

**设计**：`multitask_mlp.py` 新增 `--cross-selection-override`（**只覆盖 cross 块**，
market / history 不动，否则就是组合臂；不给＝行为逐位不变）。阶梯用 P9 **原规格**
`max_iter ∈ {12, 50, 150, 400}`，**不新选点**（P9 明写「挑一个让混合增益最大的 epoch 数
就是看结果选参」）。fold 0 的覆盖选列与基准判据重合 175/200，与探针实测一致。

⚠️ **环境自检必须换个做法**：这个臂的 12 档按设计就不该复现 08-19 的锚点（特征集变了）。
所以另跑一次**默认选列**的 12 档当锚点 —— 实测 `target_only` / `multitask` 对
17.4287% / 20.2833% 偏差 **0.000e+00**（逐位）⟹ 环境与 08-19/08-20 一致，曲线可解读。

**结果 —— 曲线形状原样保留，天花板没动**（独立 MLP peak / 生产 3s480 基准 peak）：

| max_iter | base `target_only` | base `multitask` | 新臂 `target_only` | 新臂 `multitask` | 较好者 Δ |
|---:|---:|---:|---:|---:|---:|
| 12 | 17.4% | 20.3% | 17.5% | **21.6%** | +1.4pp |
| **50** | 26.2% | **28.8%** | 24.9% | **27.4%** | **−1.4pp** |
| 150 | 6.5% | 5.7% | 4.4% | 6.9% | +0.4pp |
| 400 | 1.4% | 1.2% | 0.9% | 1.5% | +0.0pp |

**判定 `REJECTED`**：最好一档 **27.4%**，门槛 50%，差 22.6 个百分点；条件延长**未触发**。

⭐ **真正的信息不在那 ±1.4pp，在于倒 U 的形状一模一样**：换了 25 列输入之后，50 档仍是峰值、
150/400 档仍然崩溃。⟹ **这独立印证了 P9 自己的机制结论 —— 绑定约束是正则化
（早停关闭、无 LR 调度、无 dropout/LayerNorm），不是特征集。**
P9 当时是从曲线形状**推断**出「不是预算、是正则化」；本轮是直接**变更了另一个轴**
（特征集）而曲线不动 ⟹ 同一结论多了一条**正交**证据。

**⟹ v5 的可改项从 3 条收缩到 2 条**：① 能防过拟合的训练配方；② `asset_id` 用 embedding。
**③ 特征选择这条已经探过，量级不足。**

⚠️ **限制**：只测了 fold 0、只换 cross 块、只换了「窗口一致性」这**一种**新选列。
严格说否掉的是「这一套 MLP 配方 + 这一个替代选列」，不是「任何选列改动都无用」；
但连同发现 3（该选列本身机制未兑现）一起读，先验很低。

**工程注记（三条，都属于「不另写一份 / 不动生产路径」）**：

1. **`horizon_auxiliary_cache_probe.py` 做了一次行为保持的函数抽取**：门禁 + bootstrap 逻辑
   原本埋在 `main()` 里没法 import，照抄一份正是 CLAUDE.md §8 要防的。已抽成
   `evaluate_arm()`，`main()` 改为调它。**证明没改数值**：重跑后与抽取前的 JSON 逐字段对比，
   除 `elapsed_seconds` 外差异只有两处 `nan != nan`（Python 比较语义，两侧都是 NaN）。
   补测脚本的 08-18 锚点复现（逐位 0.000e+00）是第二重独立佐证。
2. **选列探针没有动 `select_features`**：它有 **97 处调用点**且在生产训练路径上，
   8/23 前一天不碰。探针自己算相关（因为那个函数只返回下标、拿不到分数），
   但**逐折硬断言**自算的 top-200 / top-40 与它逐位相同 —— 本轮 10 次比较全部相同。
3. `multitask_mlp.py` 新增 `--cross-selection-override`（**默认不给＝行为逐位不变**），
   为 P9 范围项 ③ 提供单轴入口；只覆盖 cross 块，market/history 不动，否则就是组合臂。

**门禁**：全量测试 **217 passed / 26 subtests**（原 174/26，本轮新增 43 个用例）。
生产目录 `strategies/v3_hybrid/model/` 一字节未动。
⚠️ **顺带订正一处文档漂移**：ROADMAP 多处记的是「174 passed / **27** subtests」，
实测（不含本轮新增文件）是 **174 / 26** —— subtest 数在本轮之前就已经差一个。

**证据**：`outputs/experiments/responder_family_grid.{json,md}`、
`responder_stage_c_fill{,_plan}.{json,md}`、`responder_selection_probe{,_plan}.{json,md}`；
单测 `tests/test_responder_{family_grid,stage_c_fill,selection_probe}.py`（14 + 14 + 15）。

**重新开放条件**：8/23 回补包若含 responder 列（**主办方原文未承诺** ——
`docs/data_description.md:173` 只说「标签回补 / 扩展训练数据」，没有字段清单）⟹
按原规格复验一次。**这一条应写进 8/23 当天 D0.2 的核查项。**

**后续问题**：选列探针否掉的是「窗口一致性平均」这一个统计量。若将来要再碰这条线，
应当先回答「为什么一个降方差的判据会把原排名 #16 的列丢掉」—— 本轮没有回答它。

### 2026-08-20 — `RESULT/INFRA`：公榜期那 3,217,458 行只能用一次 —— 封存 60,000 个 time_id 当密封测试集

**日期**：2026-08-20
**标签**：`RESULT`（基础设施与预注册；裁决本身 `BLOCKED_UNTIL_DATA_REFRESH`）

**问题**（用户提的：「接下来几天就是继续看 RUNBOOK 吗，感觉没什么意思」）。RUNBOOK 确实读不出
东西 —— 它已经写到「当天不需要做设计决策」的程度。但复查发现它**漏了一个取舍**。

**动机与机制**。主办方原文（`docs/data_description.md:172`）：「在公榜测试阶段，测试数据不提供
`responder_*`。在公榜截止后会发布标签回补数据，该部分数据将作为扩展训练数据使用」。实测两边
schema：`data/test/*.parquet` **326 列，无 `weight` / `target` / responder**；
`data/train/*.parquet` **375 列，带 `weight` / `target` / 47 个 responder**。
⟹ 8/23 回补的**就是公榜期的标签**，那段数据的两种用法互斥：

```text
公榜期  time_id 888,480–1,105,919   217,440 个 real time_id   3,217,458 行
现有 OOF 5 折验证行数合计            约 150 万行（每折约 30 万）
⟹ 公榜期评估行数约是现有 OOF 的 2.1×，而且是最近的一段
```

而现有 OOF 的检出下限是基准 peak 的 6.1%（1s160）/ 8.7%（3s480）—— 正是它把 `mkt323` (+1.09%)、
`phase_id` (+1.1%)、`responder_00` (+1.38%)、`lag3+lag10` (+0.38%)、扩展窗 (+1.08%) 全判成
「测不出来」。⟹ **若把回补数据全部拿去训练，8/23–8/31 最可能的结局是「重训了，但测不出有没有
用，于是按 RUNBOOK D6 维持现状」** —— 那是一个没有信息量的结局，而它是本项目最后一个大决定。

**实验设计与固定项**（跑之前钉死，`--emit-plan` 先落盘 `sealed_period_plan.json`，
sha256 记进每份裁决产物）：

```text
密封测试集   1,045,920 – 1,105,919   60,000 real time_id
  4 块 × 15,000
embargo      30 real time_id（= OOF 的 6 采样步 × sample_modulo 5，src/validation.py:40）
决策期训练   ≤ 1,045,889（比现在多 157,410 个 time_id，+17.7%）
最终交付     决定拍完后用 0–1,105,919 全量（+24.5%）重训一次
门禁         RUNBOOK D2 六道在 4 块上的映射（`≥4/5 折` → `≥3/4 块`）
             + 配对 block bootstrap（每块 25 chunk、2000 次、seed 2026、95%）
             + 第七道「超过检出下限」—— 标定前判 None，**不是**自动通过
```

⭐ **读数口径订正（设计初稿是错的）**。初稿写「`raw = pred / prediction_scale` 反解」。
那在 **slow/fast 下不成立** —— 最终值是 `clip(s_slow·slow + s_fast·fast)`，两个分量各有 scale，
除以单一 `prediction_scale` 还原不出 raw。正确做法反而更简单：`peak = A²/B` 对全局缩放**严格
不变**（`f→cf` ⟹ `A→cA`、`B→c²B`），限幅是唯一的非线性步骤 ⟹ **断言触限 0 行后直接算 peak**
就与拿 raw 算逐位等价。少一步除法，也少一个错。有回归用例把这条不变性和
「`optimal_scale` **不是**不变量」一起钉死。

**结果**（干跑，用当前数据、无标签）：

1. **推理链路**：官方 runner 全量 test **3,217,458 行**、`status=ok`、0 超时、
   `predict_total` 5.60 分钟、`max|pred| = 0.420450`、**触限 0 行**。
   ⭐ 与 ledger 08-18 走官方 runner 那次（`max|pred|=0.4204`、0 行触 clip）和 ROADMAP P0 的
   `0.4204497` 对上 ⟹ **推理路径确认无误，这是一个白送的自检**。
2. **判据链路**：`--synthetic-labels` 走通 join / 分块 / bootstrap / 判据，产物强制
   `adjudication_valid=false`（沿用 full-resolution smoke 的 `oof_valid=false` 先例）；
   self-vs-self 得 +0.00% / 0-of-4 / **FAIL** —— 空改动不过门禁，符合预期。
3. **密封段实测 856,319 行**，每块 **203,176 / 224,970 / 203,264 / 224,909**。
   ⚠️ 每块行数不等（摆动约 10%，且**交替**出现）—— 逐块 peak 是各自的比值不受影响，但读 pooled
   数时要记得块权重不均。这个交替模式与仓库已研究过的 `phase_id` 周期性方向一致，未深究。

**解释与限制**。这套东西**不会**补上离榜首 IC +20.8% 的差距 —— Tier 2 那批候选点估计全部
≤ +1.4%，就算全部确认也换不来那个量级。它买的是**让 8/23–8/31 那个最重要的决定变成有数的
决定**。⚠️ 最大的不确定性是：**这把尺子自己的检出下限仍然是未知数**。Tier 1 那六枪
（`production_slowfast` / `mkt_shrunk` / `mktwe` / `asset_adapter` / `r960` / `xs_shrunk`，
全部有已知公榜真值、盘上现成、每个约 6 分钟）就是去测它；若测出来也在 6% 以上，Tier 2 直接
不跑 —— 那也是一个干净的结论，只花 40 分钟。

⚠️ **本条只否掉「盲判」，不承诺「能判」。** 密封期是**一段**时期，不是多个 regime；它对
私榜期（9/1–9/30 前向实盘）的代表性仍然只能靠「它是我们能拿到的最近一段」来论证，
和 OOF 折一样存在时期错配。

**决策**：ROADMAP 新增 **P10 `PREREGISTERED`**；RUNBOOK 新增 **D0.4**（Tier 1 标定）与
**D4.5**（最终 100% 重训 + 三层回退）。8/23 之前不再动它。

**工程注记**：
- **不落提交格式 CSV**。官方 runner 必须给 `output_path`，脚本指向 `TemporaryDirectory`，
  读完随目录销毁；预测落 `outputs/cache/sealed_pred_<label>.npz`。理由有两条：CLAUDE.md §1.4，
  以及**盘上不留看起来像提交文件、8/31 可能被误传的东西**（P0 已经因为盘上三个 zip 写过一次警告）。
- 干跑第一次就撞到一个口径坑：`variant_submission.stage()` 要的是**模型产物目录本身**
  （生产在 `strategies/v3_hybrid/model/`，候选在 `outputs/candidates/<name>/` **本层**），
  给错一层只抛裸 `FileNotFoundError`。已加 `resolve_model_dir()` 两种布局都认 + 用例。

**证据**：`outputs/experiments/sealed_period_plan.{json,md}`、`seal_dryrun_gates.{json,md}`、
`experiments/sealed_period_eval.py`、`tests/test_sealed_period_eval.py`（11 个用例）。
全量测试 **112 passed / 22 subtests**。

**后续问题**：Tier 1 标定出的检出下限是多少？若它显著低于 OOF 的 6.1~8.7%，那 08-18 那批
「测不出来」的结论要按新下限逐条重读一遍，而不只是复验 `mkt323` 一个。

### 2026-08-20 — `REJECTED`：NN 独立能力阶梯 —— 预算确实掐住过它，但天花板只有 28.8%

**问题**（用户提的：「要不要单开 v5 先本地试试 NN」）。核对下来前提要修正一半：
NN **独立跑过**，`target_mlp_screen`(08-12) 与 `multitask_mlp_stage1`(08-19) 都报过独立 peak
（24.3% / 17.4~20.3% of 基准）。但两次都是 `max_iter=12`，而早停是**关掉的**
（`tol=0.0`、`n_iter_no_change=max_iter+1`），JSON 里 `iterations` 全部等于 12
⟹ **那不是收敛，是跑完预算被掐断**。cross 头只有 `(64,32)`、约 2.6 万参数，
对面是 6 片森林 × 480 棵树。⟹ 「NN 只有树的 20%」当时是**预算事实**，不是能力事实。

**设计**：单轴 epoch 阶梯 `{12, 50, 150, 400}`，**测量路径一行代码都没改** ——
直接用同一个 `experiments/multitask_mlp.py` 跑不同 `--max-iter`。
新增的只有编排/报告脚本 `experiments/nn_capacity_ladder.py`（`--emit-plan` 先落判据、
`--summarize` 出判定，没有预注册文件就拒绝汇总）。
⭐ 这带来一个白送的自检：**12 档必须复现 08-19 的 17.4287% / 20.2833%** ——
实测偏差 **0.00e+00**（逐位），环境与数据一致，曲线可解读。
判据跑前钉死：门槛 `max(两臂) ≥ 基准 50%`（用户 08-20 定），末档相对前一档仍 ≥ +5% 则追加 1200 档。
汇总时硬校验四档 `configuration` 除 `max_iter`/`label` 外逐项相同（单轴保护）。

**结果 —— 倒 U，峰值在 50 档**：

| max_iter | 独立 target_only | 独立 multitask | 较好者 | 相对上一档 | 混合增益（multitask）|
|---:|---:|---:|---:|---:|---:|
| 12 | 17.4% | 20.3% | 20.3% | — | +0.03% |
| **50** | 26.2% | **28.8%** | **28.8%** | **+42.0%** | +6.46% |
| 150 | 6.5% | 5.7% | 6.5% | −77.3% | +1.02% |
| 400 | 1.4% | 1.2% | 1.4% | −78.3% | +3.26% |

**两个方向相反的结论，都成立**：

1. ⭐ **「NN 没被公平测过」这个判断是对的** —— 12→50 档相对涨 **+42%**（20.3% → 28.8%）。
   之前那个 20% 确实被预算掐住了。
2. ❌ **但天花板是 28.8%，不是 50%**，而且 50 档之后**崩溃**（−77% / −78%）。
   ⟹ **绑定约束不是预算，是正则化** —— 这个配方没有任何阻止过拟合的机制：
   早停被显式关掉、无学习率调度、无 dropout / LayerNorm，只有 `alpha=1e-3` 的 L2。
   给它更多步数，它就去记忆训练段。

⭐ **辅助损失的符号随过拟合翻转**：欠拟合区（12/50 档）`multitask > target_only`，
过拟合区（150/400 档）`target_only > multitask`，翻转点恰好在曲线掉头处。
机制说得通 —— 辅助损失在容量饥饿时起**正则**作用，在网络开始记忆时变成**容量竞争**。
这是 08-19 P8「机制成立但增量不存在」的独立补充证据。

**⚠️ 方法学发现（本轮最有价值的一条）：单折 oracle 混合增益不可信。**
独立强度**单调**崩溃（28.8% → 6.5% → 1.4%），混合增益却**非单调**
（+6.46% → +1.02% → +3.26%）；400 档 MLP 独立只剩基准的 **1.4%**，
oracle 混合却仍报 **+3.26%**。两条曲线根本不同向 ⟹ 单折 + oracle 系数的读数被噪声主导。
**这追认了 08-19 要求「冻结系数 + 5 折」做终审的决定是必要的，不是形式主义。**

**⚠️ 不得据此重开 P8。** 50 档的混合增益 +6.46% 会过 08-19 Stage 1 的门槛，但那次预注册在
`max_iter=12`；**挑一个让混合增益最大的 epoch 数，正是仓库禁止的看结果选参**
（CLAUDE.md §5.1）。何况上一条已说明这个读数本身不可信。

**决策**：`REJECTED`，按预注册停止；**未触发**条件延长（末档 −78.3%，远低于 +5%）。
不动生产、不花公榜额度、不建 `strategies/v5_*`、不装 torch。

**⚠️ 适用范围**：本阶梯否掉的是 **sklearn `MLPRegressor` + 生产特征表示 + 这套预算**
这个**配方**，不是「NN 这个模型族」。三处对 NN 不利且本轮未动：特征按
`|corr(feature, e)|` 选的 top-200（为线性/树挑的判据）、`asset_id` 是 15 维 one-hot
而非 embedding、训练配方无调度/无 LayerNorm/无 dropout。

**⭐ 对 v5 的范围结论（这是本轮真正的产出）**：曲线已经把「预算」这条**排除掉了**。
8/31 之后若做 v5，值得改的是
① 能防过拟合的训练配方（早停 / LR 调度 / dropout / LayerNorm）——**曲线直接指向它**；
② `asset_id` 用 embedding；③ 特征选择不再沿用为线性/树挑的那 200 列。
**若 v5 只是「加算力 / 加宽网络」，可以直接不做。**
**重新开放条件**：上面三条至少改掉一条后按原规格复验；或 8/23 回补数据后基准变化。
**不得**只加 epoch 或只加宽网络重跑。

**证据**：`outputs/experiments/nn_capacity_ladder.{json,md}`、
`multitask_mlp_e{12,50,150,400}.{json,md}`、预注册 `nn_capacity_ladder_plan.{json,md}`；
单测 `tests/test_nn_capacity_ladder.py`（6 个）。全量测试 **101 passed / 22 subtests**。

### 2026-08-20 — `INCIDENT`（扩大范围，未造成损失）：被判毒的 OOF 缓存不止一份，而且「改名」不等于「封住」

**起因**：P6 磁盘清理执行完之后做收尾审计，顺手核了一遍「清理有没有留下悬挂引用」。

**发现 1 —— 08-18 那次 INCIDENT 漏点名了一份。** 当时的判据是「cache 时间戳早于
`experiments/v3_production_oof.py` 的首次提交（08-15 11:18）⟹ 出自从未入库的脚本版本」。
按同一判据扫整个 `outputs/cache/`：

| 缓存 | 产出时间 | 数组数 | checkpoint | 判定 |
|---|---|---:|:---:|---|
| `..._phasebal_prodwindow.npz` | **08-14 10:56** | 13 | 否 | ⬅ **签名与被判毒那份完全一致，当时没点名** |
| `..._phasebal_prodwindow_exact.npz` | 08-14 11:12 | 13 | 否 | 08-18 已判毒（实测差 3.37e-05）|
| `..._confirm_3s480_phasebal_prodwindow.npz` | 08-14 12:52 | 19 | 是 | 早于首次提交，但结构与已入库脚本一致 |
| `..._1s160_prodwindow_20260818.npz` | 08-18 15:21 | 19 | 是 | ✅ 唯一确认由当前代码产出 |

「13 数组、无 checkpoint」正是 08-18 用来佐证的那条旁证（「它早于 checkpoint 功能」）——
plain 那份**完全同签名且更早**，当时只是没被扫到。已一并隔离。

**发现 2 —— 光改名挡不住。** P6 把 `_exact` 改成 `.STALE-DO-NOT-USE.npz`，看起来是防呆，
但**四个实验脚本把它写死成 `--oof` / `--base-oof` 的默认值**：

```text
experiments/v3_residual_atlas.py:28        experiments/v3_market_round_scan.py:44
experiments/v3_asset_adapter_candidate.py:26   experiments/v3_residual_adapters.py:36
```

改名之后它们变成裸 `FileNotFoundError`，**不解释原因**。而旁边就躺着一个
`.STALE-DO-NOT-USE.npz` —— 8/23 赶工时最省事的「修复」就是把 `--baseline` 指过去，
毒缓存原地复活。⚠️ `v3_market_round_scan` 正好写在 P3 的重开条件里
（「市场块结构本身改变，或回补数据后原规格复验」），这不是假想路径。

**修复**：隔离改成**代码强制**而不是靠文件名和记性 ——
`src/oof_cache.py` 新增 `UNREPRODUCIBLE_CACHES` / `VERIFIED_CURRENT_CODE_CACHE` /
`assert_reproducible_cache()`：

- 三个路径都在名单里（含改名后的 `.STALE-DO-NOT-USE`）⟹ **把名字改回去也绕不过**；
- 报错直接给出现跑命令和已验证缓存的名字 ⟹ 不会引导人去找最省事的错解法；
- `load_oof_bundle()` 里也调了 ⟹ 走带校验入口的脚本自动受保护；
- 四个脚本的默认值改成 `required=True`（不再给默认）+ 调用守卫。

**防复发**：3 个回归用例（`tests/test_oof_cache.py`）钉住——名单覆盖、
文件存在也要拒绝、缺文件时的报错必须指向已验证的那份。全量测试 **95 passed / 22 subtests**。

**⚠️ 对已有结论的影响 —— 逐条核过，都不翻案**：

- 08-19 的多任务 Stage 1 用的基准是 `confirm_3s480`（**未隔离**，19 数组含 checkpoint）。
  即便按 `_exact` 那次量到的 ±2.4% 给基准整体加误差，实测增益 +0.0256% 也只在
  **+0.0250% ~ +0.0262%** 之间浮动 ⟹ 离 3% 门槛仍差两个数量级，**结论不动**。
- P4 扩展窗那一臂当年确实是拿 `_exact` 配对的，08-18 已记「重开训练窗轴前必须现跑基准重测」，
  本次不改变那条待办。

**顺带修掉的文档漂移**（P6 清理的副作用）：ROADMAP / RUNBOOK 里三处仍按旧文件名警告
「不要提交 `v3_hybrid_submission_20260813.zip`」，而它已改名为 `.PRE-SLOWFAST.zip`——
已改写成「改名就是防呆措施本身，不要改回去」。`experiments/lgbm_mt.py` 的 docstring 说
「数据已经缓存好，不用重扫 17G parquet」，而 `mt_aggregates.npz` 已被清理——
已注明 `load_or_build` 会自动重建，只是首次要多花一次全扫。

### 2026-08-19 — `RESULT`：slow/fast 顶点测出来了 —— 当前生产点已在这条线峰值的 99.96%

**接上条的预注册。** 第三点（t=2）已交，公榜 **S2 = 0.0039374211**。

**完整性检查先过**：`S2 < 2·S1 − S0 = 0.0042322660` ⟹ `a = −1.474225e−04 < 0`，曲线是凹的。
（这道检查不是形式主义：`a ≥ 0` 在数学上不可能由「三点都是同一组预测的线性变换且都没触限」
产生，出现就意味着某点触了限或哪次提交的模型身份与记录不符。）

**闭式解**：

```text
a  = −1.474225e−04     b = +2.646799e−04
t* = 0.897692          Score(t*) = 0.0041165516
```

⭐ **顶点在 t=0.898，也就是当前生产点 t=1 的**稍前**一点。结算下来：

| | 值 |
|---|---:|
| 当前点 t=1 处于这条线峰值的 | **99.9625%** |
| slow/fast 已捕获线上总可得增益的 | **98.70%** |
| 挪到真顶点还能多拿 | +0.0375%（绝对 +1.543e−06） |
| 半步收缩后 | +1.157e−06 |
| 预注册的采纳线 | 1e−05 |

⟹ **不改交付，私榜维持 t=1**，生产目录一字节未动。

**⭐ 这一枪的价值不在涨分，在于把轴关死。** 此前 slow/fast 的两个系数是从 OOF 的**相对模式**
搬过来的（`1.16 × c/a_global`），公榜上从来没验证过它是不是这条线的最优点 ——
「没测」和「测了发现已经最优」是两种完全不同的状态，后者才能结案。现在是后者。

**⭐ 顺带独立验证了一个方法。** 08-17 那次的做法是「只搬 OOF 的相对模式、保留公榜标定的
绝对水平」（因为 OOF 全局最优 scale 0.7296 与公榜标定的 1.16 差 59%，是本项目已知的
本地/公榜尺子分歧）。事后看，这个做法给出的 `(0.4496, 1.2530)` **落在最优点的 0.04% 以内**。
⟹ 该启发式在这一次上被独立确认；但它仍是**一次**验证，不是通则。

**⚠️ 两个值得记的观察**：

1. `S2 = 0.0039374211 < S0 = 0.0039977510` ⟹ **t=2 比完全不做 slow/fast 还差**。
   与 `t*≈0.9` 完全自洽（顶点两侧对称衰减，t=2 离顶点 1.10 而 t=0 只离 0.90）。
2. 曲率 `|a| = 1.474e−04` 相对 `S1−S0 = 1.173e−04` 是同量级 ⟹ **这条抛物线在顶点附近很平**。
   这既解释了为什么 t=1 已经拿到 99.96%，也解释了为什么半步收缩只损失 25% 的理论增益 ——
   平坦的顶点意味着「站得准」的边际价值本来就低。

**决策**：slow/fast 系数轴 `CLOSED`。**重新开放条件**：模型本身改变（森林重训 ⟹ 最优配比
要重解）或 8/23 回补数据后按原规格复验。**不得**在同一模型上继续搜第二个收缩系数、
换第三点位置重取、或去做 5 点平面顶点 —— 线上的余量只有 0.0375%，平面版的期望值不可能
支撑 3 次公榜额度。

**证据**：`outputs/experiments/slow_fast_vertex_solution.{json,md}`
（里面记着预注册 `slow_fast_line_geometry.json` 的 sha256 ⟹ 判据先于结果可核验）、
`experiments/ledger.csv` 2026-08-19 行。

### 2026-08-19 — `RESULT`：按外部 handoff 复盘，三处与仓库现状不符；并落盘 slow/fast 顶点预注册

**背景**：收到一份外部复盘 `HANDOFF.md`（2026-08-19），提出三条「没有真正关闭」的线
（A: slow/fast 顶点、B: 多任务辅助监督、C: 8/23 recency）。先逐条核对它引用的仓库事实。

**⚠️ 核出三处不符，都改变优先级或设计**：

| # | handoff 的说法 | 实际 |
|---|---|---|
| 1 | 今天第一优先是「P0 最终环境全量 wall-clock 复测」 | **08-18 就做完了**（4 核双路径 5.26 / 10.94 分钟，JSON 已落盘）。它读的是本文件 §2 更新前的旧版；加上 08-19 的打包审计 ⟹ **P0 已无剩余动作** |
| 2 | 私榜是 best-of-10 还是指定一个？（列为待核实） | 主办方原文 `docs/competition_description.md:201`：「共可以进行最多 `10` 次策略文件提交，**最终采用最新提交版本**」⟹ **不是 best-of-10** |
| 3 | 「你已核实 t=2 的 max\|pred\| = 0.4459」 | 仓库里**没有**这条记录，但**数是对的**：实测 0.445934、触限 0 行 |

**第 2 条的后果最大，而且它此前只在主办方原文里、没进本仓库任何文档**：
`ROADMAP.md:10` 与 `RUNBOOK_8_23.md:17` 都只写了「私榜共 10 次，至少留 3 次余量」，
**漏掉了决定性的后半句**。正确纪律是：

```text
8/31 最后一次上传的那份 = 最终答案
10 次是**上传失败的重试余量**，不是「交一组分散候选让主办方挑」
⟹ 高方差候选在这里**没有期权价值**；不存在「最后交个实验版试试」这种操作
```

已写进 ROADMAP §1 与 RUNBOOK §0，并在 RUNBOOK D6 加了 4 步收尾顺序。

**A 线（slow/fast 顶点）—— 预注册已落盘，等第三点**：

沿 `c(t) = (1−t)·(1.16,1.16) + t·(0.4496,1.2530)`，`pred(t)` 逐行线性 ⟹ `Score(t)` 是 t 的
**精确二次式**，且二次项系数 `a = −⟨d,d⟩_w/D` **恒为负**（构造决定，不是待检验假设）。
两个已交点 + 再取一点 ⟹ 闭式解顶点。⭐ 由 `S1−S0 = a(1−2t*)` 可导出

```text
gain(t*) = (S1 − S0) · (t*−1)² / (2t*−1)        只取决于顶点位置
```

⟹ **花名额之前就能把「S2 落在哪 ⟹ 拿到多少」整张表写死**，这正是 CLAUDE.md §5.1 要的形状。

⭐ **两件此前没有的实测**（都进了 `outputs/experiments/slow_fast_line_geometry.{json,md}`）：

1. **限幅几何**：t=0/0.5/1/1.5/2/2.5 触限 0 行，t=3 有 2 行；二分出 clip 边界 **t ≈ 2.6968**。
   t=2 的 `max|pred| = 0.445934` ⟹ handoff 那个数对，只是仓库里没有证据，现在有了。
2. **锚点交叉验证**：`submission_mkt_shrunk.csv` 的归属在 `public_replay_inventory` 里是
   **inferred（按模型名推断）**，不是 sha256 硬校验 —— 而整条抛物线都架在它是 t=0 上。
   从它按记录的变换重算 t=1，与盘上 slowfast CSV 逐行比：**max|Δ| = 5.0e-09**
   （正好是 8 位小数的舍入地板）、0 行超 1e-7 ⟹ **两份 CSV 相互印证，S0 锚点可信**。

**预注册的判据**（钉死，看到 S2 后不得改）：完整性 `S2 < 0.0042322660`（否则 a≥0 ⟹ 查触限/
查模型身份）；私榜**半步收缩** `c_used = c1 + 0.5(c*−c1)`，恰好拿到理论增益的 **75%**；
采纳线沿用 08-17 asset adapter 的「|Δ| < 1e-5 视为不可辨别」⟹ **t\* < 1.470
（约 S2 < 0.0041113）就不改交付**。

⚠️ **期望值要诚实**：上表显示典型情形只有 **+0.0%~+0.9%**；+2% 以上只在 S2 贴近 `a→0` 边界时
出现，而那时 t\* 已越过 clip 边界。花它的理由是**公榜名额 8/23 作废、不用即归零**，
且三点同源（ρ≈0.99）⟹ 顶点估计很精确 —— 不是因为它能翻盘。

**工程**：`slow_fast_csv.py` 加 `--scale-slow/--scale-fast`（成对给；都不给则走原常量推导，
输出逐位不变，`--dry-run` 已核）。新增 `experiments/slow_fast_vertex.py`：`--emit-plan` 出预注册、
`--s2` 解顶点，且**没有预注册文件时拒绝解**（判据必须先于结果落盘）。四个分支都测过：
采纳 / 不改交付 / `a≥0` 停止 / 缺预注册拒绝。

**C 线（8/23 流水线）—— 已端到端演练**：用当前数据、**全 file hash**（未用 `--no-file-hash`）
跑 `audit_data_release.py`，得 `comparison.changed = false`、`train.row_delta = 0`；
`retrain_extended.py` 随即以「audit does not prove a changed training split」**明确拒绝**。
⟹ 工具链与第一道闸门都验过，8/23 当天只需换 `--output` 日期。
⚠️ handoff 给的命令用 `data_release_20260812.json` 作基线，**已过期**，正确基线是 08-18 那份。

同时把 handoff §6 的判断写成 ROADMAP **P2-R 预注册臂**：P4 测的是往训练段**前端**加旧数据
（volume 轴，+1.08%、2/5 折、CI 跨 0），8/23 给的是往**后端**加新标签（recency 轴），
此前**根本没法测**（没有更近的标签），不是被否决。顺序钉死为
审计 → 现跑基准 → recency → 才轮到②类网格。

**B 线（多任务辅助监督）见下一条。**

**门禁**：全量测试 `92 passed / 22 subtests`（本轮新增 8 个用例）。生产目录与模型身份未改动。

### 2026-08-19 — `REJECTED`：多任务辅助监督 —— 机制**是**真的，增量**不存在**

**问题**（外部 handoff 的 B 线）：把 responder 从「输入特征」改成「共享 trunk 的辅助损失」，
能否补上生产 v3 的 target 残差？08-12 否掉的是前者（两阶段误差累积，且 runner 剥列 ⟹
线上不成立）；后者推理端只留 target 头，**完全不需要 responder**。
08-18 `horizon_auxiliary_cache_probe` 的重开条件（「不是换目标 / 线性叠加 / 对预测值做二层校准」）
字面满足 ⟹ 允许一次预注册筛选。

**立项依据**（`target_mlp_oracle_blend`，08-19，不训练）：从 `target_mlp_screen` 的逐折 A/B
反解交叉项 `C = 2B_e − (B_b+B_m)/2`，闭式算出两分量最优配比 ⟹ **折均 +6.97%、5/5 折、
去最好折 +4.86%**。⟹ 当年那个「等权集成 −54.49%」否掉的是**等权掺弱模型**
（`A₂≈0` 时 `Peak=½Peak₁ ⟹ −50%`，实测 −54.49% 就是这个签名），不是「MLP 没有独立信息」。

**设计**（`experiments/multitask_mlp.py`，跑前钉死）：λ=0.3 唯一超参；辅助目标是 08-18
`responder_window_atlas` 量出来的窗口梯子 5 个（H=1/2/4/7/10，夹着 target 的 H=5）；
用**多输出 `MLPRegressor` + 辅助列乘 √λ** 实现加权多任务（输出层线性 ⟹ 等价于该头损失权重 λ，
无正则时严格；有单测）—— **不需要 torch，不需要自写反向传播**。
基准是生产 3s480 OOF（modulo 5 / phase_balanced / train_window 78,960），按
`(time_id<<8)|asset_id` 连接、296,059 行**全部连上**。**关键是 `target_only` 对照臂**：
同架构、同种子、同迭代、共用同一个 market 头 ⟹ 两臂唯一差别就是 cross 头的损失。

**结果（fold 0，oracle 配比＝上界口径）**：

| 臂 | MLP 单独 peak / 基准 | 与基准 corr | oracle 给 MLP 的系数 | MLP 幅度占比 | blend 增益 |
|---|---:|---:|---:|---:|---:|
| `target_only` | 17.4% | 0.414 | −0.0036 | 1.7% | **+0.0248%** |
| `multitask` | **20.3%** | 0.421 | +0.0038 | 1.8% | **+0.0256%** |

⭐ **辅助损失确实起作用了**：MLP 自身 peak 从基准的 17.4% 提到 20.3%（**相对 +16.7%**），
残差信号还从 −7.17e−05 翻成 +7.14e−05（optimal c2 因此由负转正）⟹ 机制成立，不是没跑起来。

❌ **但它没有用**：即便在 oracle 上界口径下，blend 增益也只有 **+0.026%**，
**比 3% 门槛低约 115 倍**；最优配比给 MLP 的幅度占比只有 1.8%。

**为什么与立项时的 +6.97% 差 270 倍 —— 基准强度**：

```text
target_mlp_screen 的基准   fold0 peak 0.00069987   1 seed×160 轮 / modulo 10 / 窗 39,480 / 100 特征
本实验的基准（生产 3s480） fold0 peak 0.00105595   ⟹ 强 1.51×
MLP 相对强度               40.2% → 17.4~20.3%      与基准的 corr  0.24 → 0.41~0.42
```

⟹ **MLP 携带的那点信息，生产基准里已经有了。** 那 +6.97% 不是「MLP 的独立 alpha」，
是「1s160 弱基准的缺口」。这正是 CLAUDE.md §8.6/§8.7 点名的形状：
低相关的弱模型在弱基准上看着有增量，换成强基准就蒸发。
⚠️ 两次 screen 的 fold 版图不同（train_window 不同 ⟹ `first_valid_idx` 不同），
**不是配对比较**；上表只用于解释量级差异，不作为配对裁决。

**⚠️ 预注册里有一道门槛是我写错的，先说清楚**：Stage 1 原本的第二道是仓库惯用的 `2ΔA>ΔB`。
那条的前提是两个预测在**同一 scale 约定**下比较，而 oracle/frozen 配比会**重解系数**
（`A→cA`、`B→c²B`），peak 不变但 ΔA/ΔB 只反映整体缩放 —— 实测就报出 −51.77% / −76.75%
这种无意义的数。正确的机制分解是尺度不变的残差形式：

```text
Δpeak = (A_m − A_b·C/B_b)² / (B_m − C²/B_b)      分子=残差信号，分母=残差能量
```

已独立验算该恒等式（与直接相减相对差 **2.9e−13**）。⟹ 机制门槛与 `Δpeak>0` **等价**，
原来三道里有一道既冗余又口径错。已把它换成**本就该有的 3% 幅度门槛**（**收紧，不是放松**），
重跑后判定不变。**结论不依赖那道门**：+0.026% 对 3%，差两个数量级。

**决策**：`REJECTED`，按预注册停止 —— **不跑 Stage 2、不调 λ、不调 hidden、不换激活、
不换辅助目标集**。不动生产、不花公榜额度。
⚠️ **限制**：只测了 fold 0（预注册就是这么定的：符号筛不过即停）。严格说否证的是
「λ=0.3 + 梯子 5 个 + 这套 MLP 架构，在生产 3s480 基准上没有可部署增量」。
**重新开放条件**：8/23 回补数据后基准本身变化，按**原规格**复验一次；或出现能让 MLP 自身
peak 达到基准 70% 以上的模型族（当前只有 20.3%，差距不是靠辅助损失能补的量级）。

**证据**：`outputs/experiments/multitask_mlp_stage1.{json,md}`、
`target_mlp_oracle_blend.{json,md}`；单测 `tests/test_multitask_mlp.py`（8 个）。

### 2026-08-19 — `INCIDENT`（未造成损失）：交付链路有「模型身份」门禁，却没有「包内容身份」门禁

**背景**：仓库结构复查。ROADMAP 行动面板显示只剩 P0 动作 4 和等数据的 P1/P2，
但沿着「8/23 拿到数据之后会依次跑哪些命令」把链路走了一遍，发现三个洞，**都不在面板上**，
而且都属于同一类：**审计通过，但交出去/训出来的不是那个东西**（CLAUDE.md §8.2 的伤疤）。

**洞 1 —— 重训计划复现的不是当前生产架构。** `scripts/retrain_extended.py` 的 v3_hybrid
命令只传了轮数 / 种子数 / 特征数 / history：

```text
缺 --weighted-cross-section     ⟹ 截面块退回无权训练
缺 --market-model               ⟹ **整块行级市场森林消失**（公榜 +21.99% 的来源）
缺 --market-spec / --market-min-data-scale ⟹ 08-13 市场块容量收缩（+0.77%）丢失
```

两个都是 `store_true`，不传就是 `False` —— **不是「用默认值」，是另一个模型**。
按 ledger 反推，跑出来的等于 08-11 那版架构（公榜 0.0032523499，比生产低 **21.99%**）。
转正门禁确实会拦住它（`validate_meta` 有这几项），但那是在**几小时训练之后**，
而 8/23→8/31 只有 8 天。

**修法**：新增 `production_structure()`，从生产 `hybrid_meta.json` **派生**这些项
（CLAUDE.md §7「不在多处手工维护同一个数字」），再用 `assert_matches_public_baseline()`
与 `PUBLIC_BASELINE` 逐键对拍 ⟹ 生产 meta 与常量哪天分家，在**生成计划时**就红。
⚠️ 一个细节：`market_lgbm_params.min_data_in_leaf` 在 meta 里是**解析后的绝对值** 75580，
而 `train.py` 收的是倍数，且扩展数据后训练行数会变 ⟹ 只能传倍数。用生产数逐位核对：
`(12000/3.5e6) × 2,645,530 = 9070`（= `lgbm_params.min_data_in_leaf`），
`9070 × 8.333 = 75580` ⟹ `--market-min-data-scale 8.333` 正确。

**洞 2 —— slow/fast 没有生产者。** `strategies/v3_hybrid/train.py` 的 CLI 里**根本没有**
slow/fast 概念，`promote_v3_candidate` 也只**校验**不写入 ⟹ 任何重训候选的 meta 都必定缺这三键，
而 `main.py:222` 是 `PredictionTrail(int(window)) if window else None`——缺键**静默降级**。
RUNBOOK D1 的「坑 1」写着二选一，但当时两条路都得手改候选 JSON。
**修法**：按 `--scale` / `--blend-weight` 完全相同的形状，给 `promote_v3_candidate` 加
`--slow-fast-window/-slow-relative/-fast-relative`，默认取自 `PUBLIC_BASELINE`，
由 `stage_candidate()` 写进 staged meta 和 manifest。于是
(b) 沿用当前值 = 什么都不传；(a) 用新 OOF 重标定 = 显式传入（会因偏离基线要求 `--off-baseline`，
这正是「偏离必须是按下去的」）。

**洞 3 —— 提交包在装研究代码。** `make_submission.py` 是「除 `train.py` 外全收 `*.py`」，
于是 `strategies/v3_hybrid/temporal.py`（V4-T 研究模块）也进了私榜包。
用 AST 求 `main.py` 的本地 import 闭包实测是 `{main, features, lgbm_numpy, history}` ——
`temporal` 够不到。而 `audit_submission_zip` 只查**缺**文件、不查**多**文件 ⟹
08-19 对 `temporal.py` 的研究改动已经悄悄改变了提交包字节，没有任何门禁出声。
风险有两层：包字节随研究漂移（「验过的那份」≠「交出去的那份」），
以及包内 `.py` 在评测端就是 `sys.path` 上的顶层名字，哪天出现 `types.py` 会遮蔽标准库。

⚠️ **不能退回硬编码清单** —— `make_submission.py` 的注释记着写死清单曾漏过 `lgbm_numpy.py`。
**修法**：`SUBMISSION_MODULES` 声明入包集（包内容身份的唯一定义，与 `PUBLIC_BASELINE`
的模型身份分工对称），再与 AST 闭包**双向**对拍；策略目录里每个 `.py` 都必须被分类 ——
入包，或进 `EXCLUDED_MODULES` 并写明理由，未分类的一律硬失败。
`audit_submission_zip` 派生同一份声明并新增 `no_unexpected_modules` 检查；
`experiments/variant_submission.py` 里那份手抄的 `{"train.py"}` 副本也改成消费唯一定义。

**对现存包的实测**：`outputs/v3_hybrid_submission_20260818.zip`

```text
8 个模型文件 + main/features/history/lgbm_numpy   与生产逐字节相同（sha256）
4 个 slow_fast 键                                 齐全
public_baseline_drift                             []          ← 模型身份是对的
failed checks                                     ['no_unexpected_modules']
unexpected_modules                                ['temporal.py']
```

⟹ 那个包**不是坏模型**，它只是多带了一个不会被 import 的研究模块，而且盘上没有任何审计记录
（`audit_submission_zip.py` 默认只打印，不带 `--output` 就不落盘）。重打一次并落盘审计即可结案；
**不需要重测耗时**，执行路径上的四个模块一字节未动。

**门禁**：全量测试 `84 passed / 22 subtests`（原 73/18，新增 11 个用例）；
双后端 train/inference 一致性 `max|Δ| = 8.101e-09`（`--n-time-ids 500`，门槛 1e-6，两后端同值）。
生产目录与模型身份未改动。

### 2026-08-19 — `RESULT/INFRA`：本地 full-resolution 资源验证通过，但同真实时间跨度正式 OOF 不适合当前 30GB 机器

**问题**：继续测试 full-resolution training 是否可在本地安全执行；此前三次任务分别在约 26.1GB、
25.1GB、24.8GB 峰值被 OOM killer 终止。

**根因**：同一 fold 同时存在 full-resolution 特征/page cache、train/valid copies、Ridge/XS/market
设计矩阵、history blocks、LightGBM Dataset 和全量 OOF arrays；而保持生产真实训练跨度时，
modulo-1 fold 训练区间约 **394,800 real time_ids / 5.92m rows**，不是 sampled 版本的 1.18m rows。

**修复**：
- fixed production 200-feature 模式修复 history position 到原始 feature name 的映射；
- `DiskFeatureSubset` 支持 slice/int-array/bool mask，避免 NumPy 配对高级索引；
- 新增 `experiments/v3_fullres_resource_smoke.py`：固定生产 200 特征/statistics，跳过 Ridge、
  OOF arrays、checkpoint，只顺序训练 XS 和 market；
- full-resolution cache 使用 disk-backed memmap；LightGBM smoke 使用 col-wise、有限线程和 systemd 内存上限；
- 长任务全部移出当前会话，由 systemd service + journal/log 监控。

**本地结果**：短跨度 resource smoke（保持 78,960 real train time_ids，20,000 valid time_ids）
在 **1,182,292 train rows / 300,000 valid rows** 上完成：

```text
20 rounds：status=ok，max RSS≈11.5GB
160 rounds：status=ok，max RSS≈11.5GB
XS 160 rounds：约43.8s
market 160 rounds：约30.5s
```

该结果证明固定 200 特征的顺序训练路径安全，但**不是 full-resolution 同真实时间跨度的 OOF 分数**；
同真实跨度需要约 5.92m train rows，当前实现仍需进一步改成 chunked design writer，或转到 64GB+
云服务器。生产模型未修改。

**证据**：`outputs/experiments/v3_fullres_resource_smoke_160.json`；systemd journal
`quant-fullres-resource-smoke-local-160.service`；全量测试 `73 passed / 18 subtests`。

**2026-08-19 当前数据验证顺序（冻结）**：

1. `pytest -q`，必须维持 73 passed / 18 subtests；
2. 用 `data_release_20260818.json` 做快速 metadata audit，8/23 新包再做完整 hash；
3. 生产 v3 分别重跑 LightGBM/NumPy consistency，门槛 `max|Δ| < 1e-6`；
4. 仅在生产/推理代码或模型身份发生变化时重跑全量 4 核 runner；实验脚本变化不触发生产替换；
5. phase_id 只保留为弱筛选结果（pooled 约 +1.1%、3/5），不做 3s×480；
6. periodic 与 phase-balanced 当前 validation 行组成不同，禁止用 pooled 分数排序；
7. 本地不再硬跑 5.92m-row 同跨度 full-resolution OOF。正式复验需 chunked writer 或 64GB+ CPU
   服务器；任何 fixed-production 资源报告必须写 `oof_valid=false`。

### 2026-08-19 — `REJECTED`：剩余一层时序变化、截面秩和保留资产身份的 market panel

**问题**：二层修正失败后，把这些表示直接放进与 baseline history 共同训练的一层 XS 森林，或让
market 模型直接看到固定 `asset × feature` 面板，能否找到榜首级的新信号？

**实现与因果口径**：扩展 `strategies/v3_hybrid/temporal.py` 与
`experiments/temporal_multiscale.py`。原始 lag cache 仍逐行读取**全分辨率真实时间流**后才抽
modulo-5 行；每折使用自己的 robust transform。新增隔离臂：

- `x1_rank`：当前 history40 的 percentile rank / median-MAD z / tail；
- `f_lags`：lag3 + lag10；
- `f_changes`：delta3/5/10 + 一阶 acceleration；
- `f_volatility`：std5 + std20；
- `f_trend`：slope5 + slope20；
- `x2_change_rank`：`current-lag1` 的截面 rank / robust z / tail；
- `market_asset_panel`：按匿名 asset_id 固定展开的当前 panel、lag1 panel、delta panel和 presence，
  直接预测 market target，允许树学习跨资产交互。

**固定设计**：`modulo5 / phase_balanced / train_window=78,960 / embargo=6 / 5 folds /
1 seed × 160 rounds`。所有 XS 臂保留完整生产 baseline history；market panel 只替换 market，XS 固定。

**结果**：

| arm | Peak 相对 | 正折 | 去最好折 | 判定 |
|---|---:|---:|---:|:---:|
| `x1_rank` | −2.53% | 1/5 | 负 | ❌ |
| `f_lags` | **+0.38%** | 3/5 | 负 | ❌ |
| `f_changes` | −0.34% | 3/5 | 负 | ❌ |
| `f_volatility` | −0.56% | 2/5 | 负 | ❌ |
| `f_trend` | −1.64% | 2/5 | 负 | ❌ |
| `x2_change_rank` | −1.55% | 2/5 | 负 | ❌ |
| `market_asset_panel` | −7.24% | 2/5 | 负 | ❌ |

`market_asset_panel` 的 ΔA +16.88%，但 ΔB +49.44%，与手工 set summary 同样是预测方差膨胀。
`f_lags` 是唯一均值略正的臂，但效应只有 +0.38%、3/5，同号和 drop-best 都失败，远低于 3%
门槛和本实验检出能力；不得升级 3×480 或围绕 lag 长度做网格。

**Responder 决策**：没有重新启动。仓库已经存在比计划中的“强模型终审”更完整的证据：
`responder_nonlinear_reaudit_phasebal_prodwindow` 用 323 特征严格 OOF + 冻结非线性二层为
−11.70%/−17.03%；`responder_vs_v3_nonlinear_audit` 对强 v3 控制后的 responder 增量为
−7.76% 或 −6.77%；`horizon_auxiliary_cache_probe` 的 responder_00/02 也无通过臂。重复训练不会
回答新问题。

**决策**：当前数据上关闭这些规格。重新开放条件只有 8/23 回补数据按原规格复验，或出现真正
不同的模型族；不允许继续调 top-k、lag 长度、树轮数或容量。

**证据**：`temporal_change_families_1s160.{json,md}`、`temporal_change_rank_1s160.{json,md}`、
`structural_signal_screen_market_asset_panel_1s160.{json,md}`。

### 2026-08-18 — `RESULT/REJECTED`：集合级 market 分布摘要未打赢生产 market

**问题**：榜首差距是否来自当前逐行市场森林看不到的完整截面分布；直接用每个 `time_id` 的
mean/std/分位数/正值比例及其 lag、delta、|delta| 预测市场 target，能否替换生产 market 块？

**基础设施**：新增 `src/oof_cache.py`，统一验证 OOF schema、逐 `time_id` fold 完整性、
validation 有限值、报告/cache 身份和 sha256。新增 `experiments/structural_signal_screen.py`，固定复用
当前 3 seeds × 480 rounds 生产等效 OOF，支持 `market_set`、`xs_rank`、
`xs_residual_select` 三个严格折内筛选臂；不生成 CSV、不修改生产。

**设计**：`modulo5 / phase_balanced / train_window=78,960 / embargo=6 / 5 folds`；每折只在训练段
按 feature 截面均值与 market target 的相关性选 top-20，构造集合摘要并训练 1 seed × 160 rounds
强正则 LightGBM。验证时只替换现有 `market`，XS 与其他生产组件固定。

**结果**：折均 Peak 相对生产基准 **−16.02%**，**0/5** 折为正，去最好折仍为负；
ΔA +10.15%，但 ΔB +44.61%，明显违反 `2ΔA>ΔB`。

**解释**：集合摘要确实提高了与 target 的协方差，但预测方差膨胀更快；当前手工分布摘要丢失了
逐资产结构，不能替代现有 row-level market 森林。这不是轮数不足的边界失败，而是五折一致的结构失败。

**决策**：`market_set` 当前规格 `REJECTED`，不跑 3×480、不做参数网格。重新开放条件仅限：
使用保留资产身份的 permutation-invariant/set 模型，或 8/23 回补数据后按完全相同规格复验；
不得只调整 leaves、轮数或 top-k。

同一套 harness 随后完成两个严格增量 XS 臂：只用每折训练窗内**更早的严格 OOF 行**拟合
生产残差修正，fold 0 因没有更早 OOF 明确置为 no-op。`xs_rank`（percentile rank / robust z / tail）
为 **−4.64%、1/5** 正折；`xs_residual_select`（按生产残差重新选 top-32 raw deviation）为
**−3.74%、0/5** 正折；两者都是 ΔA 小涨但 ΔB 涨得更多。⟹ 这两种低容量二层修正当前规格也
`REJECTED`，不升级 3×480。重新开放仅限把 rank/tail 直接并入一层生产森林、由完整训练行共同学习；
不得调整二层轮数或 top-k 网格。

**证据**：`outputs/experiments/structural_signal_screen_market_set_1s160.{json,md}`、
`structural_signal_screen_xs_1s160.{json,md}`；全量测试 `71 passed / 18 subtests`。

### 2026-08-18 — `REJECTED`：拆掉树前面的线性选列筛子 —— 截面块变差、市场块测不出

**问题**：生产在 **LightGBM 前面**叠了线性单变量筛子（`strategies/v3_hybrid/train.py:329/346`）：

```text
xs_selected       = top-200 by |corr(feature, e)|  无权  → 截面 LGBM 块 **和** 市场 LGBM 块
history_positions = top-40  within xs_selected           → history 块
```

判据是 `strategies/v1_ridge/train.py:86-108` 的 |加权 Pearson 相关| 排序 ⟹
**123 个原始特征从未进过任何模型**。边际相关是线性、单变量的，对「单独看没用、交互才有用」
的特征完全失明 —— 而那正是 GBM 存在的理由。

**这一刀还是任意的**（partition_008 只读实测）：

```text
|corr(feature, e)|  1st=0.01010  50th=0.00621  150th=0.00391
                  200th=0.00299 ← 截断线   201st=0.00295   260th=0.00128   323rd=0.00003
落差 200th→201st = 1.33%，无断崖；保留组最小值 0.00299
```

**为什么此前没测**：`ab_featsweep` 测过 feat323，但那是**线性 Ridge 时代**
（+4.4%、7/10 折、未晋级）—— 对线性模型加弱相关列主要是加方差，是另一个问题。
`joint_recalibration_plan.json` 里 Ridge 有 200/323 两档、**LGBM 那 9 格没有 feature_count 档**。

**做法**：给 `v3_production_oof.py` 拆出三个独立开关（`--ridge/-xs/-market-feature-count`）——
此前一个模块常量同时驱动截面块和市场块，动一个数就是组合臂。
⭐ 默认值下与 HEAD 版**逐位相同**（19 个数组 max|Δ|=0.0，隔离沙箱对拍）。
⭐ **单变量保证已实证**：`history_positions` 是 `xs_selected` 内的 top-40，而 top-200 的 top-40
== 全 323 的 top-40 ⟹ 三个臂的 history 原始列名与基准**逐折完全相同**（脚本断言，全 True）。

**结果**（1 seed × 160 轮 × 5 折，基准折均 peak 0.00140840，1,461,732 行）：

| 臂 | 设计列 XS/market | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 检出下限 | 配对 CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `xs323` | 484 / 561 | −1.413e−05 | **−1.00%** | 2/5 | −2.22e−05 | −1.75% | −2.30% | 3.40e−05 | [−4.9e−05, +1.9e−05] |
| `mkt323` | 361 / 807 | +1.533e−05 | **+1.09%** | 3/5 | **+4.79e−06** | +0.20% | −0.41% | 2.20e−05 | [−1.2e−05, +3.2e−05] |
| `both323` | 484 / 807 | +1.462e−06 | +0.10% | 2/5 | −1.72e−05 | — | — | 4.13e−05 | [−4.5e−05, +3.7e−05] |

**三个臂全不过门禁。** 最好的一格 `mkt323` 过了「折均为正」「去最好折为正」「2ΔA>ΔB」三道，
但 3/5 折不足 4/5、+1.09% 远低于 3% 门槛、效应只有**检出下限的 0.70×**、配对 CI 跨 0。

⭐ **内部一致性佐证**：`xs323`(−1.00%) + `mkt323`(+1.09%) ≈ +0.09%，而 `both323` 实测
**+0.10%** —— 两个效应几乎精确可加。说明测的是真效应（只是太小），不是噪声乱跳。

**解读**：被丢的 123 列 |corr| 全在 0.003 以下，在 118 万训练行上树找不出可靠的交互信号；
而 `feature_fraction`（截面 0.7 / 市场 0.4）本来就在随机丢列，放宽只是稀释了分裂机会。
方向上截面块**变差**、市场块**略好**，与 ROADMAP §3.2「两块容量方向相反」同调，但两边都在噪声内。

**决策**：`REJECTED`，选列宽度轴关闭。⚠️ 这条**不需要公榜裁决** —— 拒绝改动就是保持现状。
**重新开放条件**：8/23 回补数据后训练行数显著增加（弱信号在更多行上才可能被树找到），
按原规格复验一次；届时把 `--xs-feature-count` / `--market-feature-count` 补进 P2 冻结矩阵。

证据：`outputs/experiments/feature_screen_1s160.{json,md}`；
基准 `outputs/cache/v3_production_oof_1s160_prodwindow_20260818.npz`（当前代码现跑，见下条）。

### 2026-08-18 — `INCIDENT`（未造成损失）：`*_exact` 那份 OOF cache 出自**已不存在的代码版本**

**现象**：给 `v3_production_oof.py` 加三个选列开关后，按惯例拿
`outputs/cache/v3_production_oof_phasebal_prodwindow_exact.npz` 做「默认值下逐位相同」的
回归验证，结果 **`market_ridge` 差 3.37e-05**（约折均 peak 的 2.4%），
`market` / `prediction_raw` / `prediction` 跟着差；而 `e_lgbm` / `xs_lgbm` 只差 ~2.8e-17。

**根因不是我的改动**：

```text
cache 时间戳                              2026-08-14 11:12
v3_production_oof.py 的**首次提交**        2026-08-15 11:18   ← 比 cache 晚
最近一次提交                              2026-08-18 09:59
```

⟹ 那份 cache 是由**从未提交过的脚本版本**产出的，无法复现，也不该当基准。
另一条旁证：它的 JSON **连 `rounds` 段都没有**，而当前脚本必写 —— 它早于 checkpoint 功能。

**证实**：把 HEAD 版脚本抽到隔离沙箱（符号链接 `src`/`strategies`/`data`，不污染仓库）
同线程数跑一遍，与加了开关的版本对拍 —— **19 个数组全部逐位相同，max|Δ| = 0.0**。
⟹ 重构无副作用；差异 100% 来自那份陈旧 cache。

**影响面（需要注意，但本轮不追）**：`v3_recency_ladder` 的扩展窗臂就是拿这份 cache 做配对
基准的（NOTES P4 收官那条）。那次结论是「+1.08%、2/5 折、CI 跨 0 ⟹ 测不出效应」，
而 3.37e-05 相对折均 peak 0.00140840 约 2.4% —— **与被测效应同量级**。
⟹ P4 扩展窗那一臂的配对可能跨了两个代码版本。结论方向（测不出）大概率不变，
但若 8/23 后要重开训练窗轴，**必须用当前代码现跑基准重测**。

**防复发**：
1. 任何配对比较前，先核对基准 cache 的产出代码版本 —— 时间戳早于脚本提交就是红旗；
2. `experiments/feature_screen_compare.py` 的 docstring 里写死了这条，且本实验的基准一律现跑；
3. 改公共实验脚本后，回归对照要跟 **HEAD 现跑**比，不跟历史 cache 比。

### 2026-08-18 — `INCIDENT`（未造成损失）：三道交付门禁都不认识 slow/fast，丢键会**静默**交出旧模型

**背景**：推进 P0 时顺手核了一遍打包链路。slow/fast 是 2026-08-18 转正的，而候选与转正前生产的
**唯一差别就是 meta 里那几个 `slow_fast_*` 键**（6 片森林 + 冻结岭回归 hash 逐字节相同、未重训）。

**问题**：`scripts/promote_v3_candidate.PUBLIC_BASELINE` —— 整条链路唯一的「公榜模型身份」
定义 —— 只有 9 个键，**一个 slow/fast 都没有**。三处消费者因此全是瞎的：

| 关口 | 改之前 |
|---|---|
| `make_submission.check_v3_hybrid_meta` | 遍历 PUBLIC_BASELINE ⟹ 键不在表里就不核 |
| `promote_v3_candidate.validate_meta` | 结构项逐条硬编码 ⟹ 没有 slow/fast 那条 |
| `audit_submission_zip.audit` | 只核 scale / iterations / seeds |

而 `main.py:222` 是：

```python
self.slow_fast = PredictionTrail(int(window)) if window else None
```

⟹ `slow_fast_window` 缺失时 slow/fast 被**静默关掉、不抛任何错**，模型退回单一 scale 1.16 的
旧行为。三道门禁全放行 + 运行时不报错 = **交出去的是低 2.93% 的那份，而且没有任何东西会提示**。
这正是 CLAUDE.md §8.2 那条伤疤（「公榜 CSV 曾由临时 override 生成，而候选 meta 仍保留占位值」）
换了个位置重演。

**实证**：仓库里现存的 `outputs/v3_hybrid_submission_20260813.zip` 就是转正前的旧模型 ——
旧审计**八项全 PASS**；加了 `--expect-public-baseline` 才被拦下（三个 slow/fast 键 drift）。
⟹ 不是假想风险，是 8/31 伸手就能拿错的那个文件。

**顺带查出另外两个洞**（同属「审计通过但包是坏的」）：

- `audit_submission_zip` 只核 `lgbm_model_files` 在不在包里，**`market_model_files` 一个都不核** ——
  市场森林是架构的一半（公榜 +21.99% 的来源），漏打包照样 PASS；
- 不核 `main.py` 顶层无条件 import 的 `features` / `lgbm_numpy` / `history`，少一个 `Model`
  就装不起来 ⟹ 整份提交判无效。

**修复**：三个 slow/fast 数值键进 `PUBLIC_BASELINE`（保持「唯一定义」不变，不在别处抄第二份）；
`make_submission` 的取值表同步加三项（它自带的键集同步守卫本来就会当场炸，设计是对的）；
`validate_meta` 加结构项并接上 `--off-baseline` 逃生口；`audit_submission_zip` 加
`--expect-public-baseline`（全表核对）、市场森林在包检查、三个必需模块进 `REQUIRED`。
**缺键一律判为偏离，不判为通过** —— `float(None)` 会 TypeError，所以统一落成 NaN 再比。

**防复发**：4 个新回归用例（`tests/test_submission_packaging.py`），逐键覆盖「丢键」和「写错值」
两种破坏，并断言 `--off-baseline` 仍能放行。夹具一律从 `PUBLIC_BASELINE` 派生 ⟹ 以后往那张表
加键，夹具不会悄悄落后。全套 **64 passed / 18 subtests**。

**留给用户的口子**：`--off-baseline`。8/23 回补数据后若重训出不带 slow/fast 的候选，
用它显式放行 —— 偏离必须是按下去的，不是漏掉的。

### 2026-08-18 — `RESULT`：P0 交付闭环 —— 4 核下两条路径全量实测并落盘

**为什么要重测**：ROADMAP §2 记着 `predict_total = 5.15 分钟` / NumPy 兜底 10.44 分钟，但这两个数
**只写在 ROADMAP 和 ledger 里，没有任何落盘的 runner JSON，也没有记录线程数**。开发机 32 核、
P0 动作 3 要的却是「接近私榜环境的 4 核」⟹ 那两个数不能替 4 核背书。

**做法**：新增 `scripts/verify_delivery_runtime.py`。走官方 runner 的 `run_loaded_model`
（**不是** `run_strategy` —— 后者会 `to_csv`）⟹ 全程不写任何提交文件。线程数与
`OMP_NUM_THREADS` 不符直接退出，避免把口径记错。兜底那条用 **import shim 让 `import lightgbm`
抛 ImportError**，复刻评测机没装 lightgbm 的真实路径，并断言实际选中的后端确实是 `numpy`。

| | LightGBM 主路径 | NumPy 兜底 |
|---|---:|---:|
| `predict_total` @ 4 线程 | **5.26 分钟** | **10.94 分钟**（2.08×）|
| wall clock | 6.20 分钟 | 11.81 分钟 |
| model init | 0.36 s | 0.37 s |
| 单步最大 / 平均 | 0.682 s / 1.47 ms | 0.658 s / 3.06 ms |
| 行数 / 调用 | 3,217,458 / 214,538 | 同 |
| 超时 / 非有限值 / 触 clip | 0 / 0 / 0 | 0 / 0 / 0 |
| `max\|pred\|` | 0.4204497 | **0.4204497（相同）** |

⭐ **本轮最有价值的一条**：兜底跑起来是**单核 100%**（纯 numpy 树遍历不并行，RSS 4.56 GB）
⟹ **4 核评测机不会比 32 核开发机更慢**。此前「兜底 2 倍慢」是已知的，但「兜底会不会随核数
进一步恶化」从没验过 —— 现在这个风险从「未知」变成「已量化且不随核数恶化」。

两条路径 8 个模型文件的 sha256 与 promotion manifest 逐字节一致；两后端全量 321 万行的
`max|pred|` 完全相同，与 staging 对拍 2.082e-16 吻合。

⚠️ 两处与 ROADMAP 旧记录对不上，已按实测更新：耗时 5.15→5.26 / 10.44→10.94（旧值无线程记录，
不追因）；`check_consistency` 同参数复测是 **4.019e-09** 而非记录的 8.111e-09 ——
两个数都远低于 1e-6 门槛，未追因，以实测为准。

证据：`outputs/experiments/delivery_runtime_lightgbm_4t.{json,md}`、
`delivery_runtime_numpy_fallback_4t.{json,md}`。

### 2026-08-18 — `REJECTED`：`responder_00`/`responder_02` 的 Stage-C 空白格已填，仍不补 target 残差

**这个格子为什么空着**（核对当时的本地工作笔记 `NEXT_STEPS_horizon_auxiliary_oof_validation.md` 时发现 —— 该文件未入库、现已不在盘上，本条的结论证据是 `outputs/experiments/horizon_auxiliary_cache_probe.{json,md}`）：
2026-08-14 的 responder 重新审计只把 **8 个通过族**送进了 Stage C。查
`responder_predictability_reaudit_phasebal_prodwindow.json` 的 cluster 明细：

```text
cluster 24 = {responder_00}  mean_peak 0.02945  5/5 折  drop-best 0.02857  pass=false
cluster 22 = {responder_02}  mean_peak 0.00141  5/5 折  drop-best 0.00130  pass=false
```

七项 check 里**只有 `multi_member_family` 一项为 false**（它们是单成员族），其余六项全过。
⟹ 这两个「更短窗口」候选被挡在 Stage C 门外**不是因为证据，是因为一条稳健性启发式**。

**方法**：不训练任何新模型。把两份现有缓存按 `(time_id<<8)|asset_id` 连接 ——
auxiliary 取 `responder_oof_phasebal_prodwindow_f323.npz` 的严格 OOF 预测（Ridge、全 323 特征），
基准取 `v3_production_oof_confirm_3s480_phasebal_prodwindow.npz` 的强 v3 3s480 OOF。
组合系数**只用 fold 0..k−1 拟合、冻结到 fold k**（4 个评估折），门禁运行前写死。
全程 **2.2 秒**。

⚠️ 两份缓存的折边界错开约 90 个 time_id（各自的 `rolling_time_folds` 建在不同的 unique
time_id 列表上），604 行（0.041%）折号不一致 ⟹ 丢掉，剩 1,460,308 行。

**结果**（相对 = Δ折均 / 基准 peak 折均）：

| 基准 | 臂 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 检出下限 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | `null_frozen_scale` | −6.71e−05 | −3.84% | 0/4 | −8.54e−05 | 0 | 0 | 4.07e−05 |
| full | `responder_00` | −5.58e−05 | −3.19% | 2/4 | −8.04e−05 | −3.59% | −7.83% | 7.00e−05 |
| full | `responder_02` | −5.64e−05 | −3.23% | 3/4 | −8.19e−05 | +9.46% | +18.58% | 7.34e−05 |
| pure_e | `null_frozen_scale` | −2.69e−05 | −2.54% | 0/4 | −3.59e−05 | 0 | 0 | 2.97e−06 |
| **pure_e** | **`responder_00`** | **+1.47e−05** | **+1.38%** | **3/4** | **−3.65e−06** | +0.85% | **−2.01%** | 3.38e−05 |
| pure_e | `responder_02` | −3.92e−05 | −3.69% | 1/4 | −6.80e−05 | +15.20% | +34.64% | 2.58e−05 |

**harness 已校准**（这是能相信上表的前提）：负控制（`responder_00` 预测在每个 `time_id`
内打乱）两个基准下都不过门禁；已测族 `responder_27` 复现出明确负结果（−3.72% / −3.78%），
方向与 08-14 的 −18.81% 一致。

**唯一有内容的一格是 `pure_e/responder_00`**，但它**不过**：

- 过了门禁 1（折均为正）、2（3/4 折）、5（`2ΔA>ΔB`）、6（配对 CI 下界 +9.04e−06 > 0）；
- **没过** 3（去最好折 −3.65e−06，整个效应挂在一折上：逐折 +6.98e−05 / +5.34e−06 /
  +3.21e−05 / **−4.84e−05**）、4（+1.38% < 预注册的 3%）、7（**只有检出下限的 0.43×**）。
- 机制是 **ΔB −2.01% 而 ΔA 只有 +0.85%** ⟹ 减方差不是加信号，与 `corr(e_lgbm)=−0.099`
  合读正是 CLAUDE.md §8.6 点名的那种「低相关的弱模型」。

⭐ **顺带量化了一个一直没被拆开的东西**：`null_frozen_scale` 臂（不加任何 auxiliary、
只把 scale 冻结在过去折）就已经是 **−3.84%（full）/ −2.54%（pure_e）**。也就是说
`asset_blend_check` 那类「基准可在评估折重解最优 scale、候选必须冻结」的比较里，
**有约 2.5~3.8 个百分点是让步本身**，不是候选变差。剥掉让步后
（`mean_delta_vs_frozen_baseline`）：`pure_e/responder_00` 是 **+3.92%**，
`full/responder_00` 只有 +0.64%，`responder_02` 转负。以后读这类表要先看 null 臂。

**决策**：`REJECTED`，horizon auxiliary 方向关闭。空白格已填，不再为 responder 选择、
目标替换或 combiner 形式花时间。**不动生产、不花公榜额度。**

**限制**：缓存里的 auxiliary 是 **Ridge 强度**（与 08-14 对那 8 个族用的是同一把尺子，
可比），负结果严格说只证否「Ridge 强度的 auxiliary 无增量」；基准也不含 slow/fast 后处理。
本探针是**准入筛不是终审** —— 但先验很硬：被测过的 8 个族可预测性高 8~460× 尚且 −18.81%。

**重新开放条件**：8/23 回补数据带来新的 responder 列，或出现不是「换目标 / 线性叠加 /
对预测值做二层校准」的新机制（沿用 `responder_reaudit_20260814.md` 的原条件）。
证据：`outputs/experiments/horizon_auxiliary_cache_probe.{json,md}`。

### 2026-08-21 — `RESULT`：长窗 w512 公榜裁决 0.0041833953（+1.662%）—— 真涨，但只有本地的 0.22×

**日期**：2026-08-21　**标签**：`RESULT`（公榜实测；生产**仍未转正**）

**结果**：`0.0041150085 → 0.0041833953`，**+1.662%**，新最好成绩。
同 scale 1.16、两份 CSV **均 0 行触限**（max|pred| 0.4204 → 0.4021）⟹ 二次式精确，
这个比较**不依赖任何近似**。

**峰值口径（排除 scale 混淆）**：
- `Σp²(新)/Σp²(旧) = 1.012795` ⟹ 预测幅度**上升** 1.28%，不是下降
  ⟹ **「最优 scale 上移、1.16 处低估了新模型」这个猜想不成立**。
- 峰值增益 **+1.72~1.74%**，把 `B_old` 拨 ±8% 结论不翻 ⟹ 与固定 scale 的 +1.66% 一致，
  **没有隐藏的 scale 效应**。
- 折算：分数 +1.66% ≈ **IC +0.83%**；与榜首的 IC 差距 **+20.8% → +19.7%**，
  **填掉约二十分之一**。

**⚠️⚠️ 迁移率 0.22× —— 本项目连续第二次③类本地高估**：

| 改动 | 本地 | 公榜 | 迁移率 |
|---|--:|--:|--:|
| 相位采样（08-10） | +1.84%（判「测不出来」） | +2.86% | **1.6×**（低估）|
| history40（08-11） | +10.10% | ≥+23.47% | **2.3×**（低估）|
| 市场森林 + 带权截面（08-13） | +18.30% | +21.99% | **1.20×**（低估）|
| slow/fast（08-17） | +5.77% | +2.93% | **0.51×**（高估）|
| **长窗 w512（08-21）** | **+7.77%**（截面块 peak） | **+1.72%**（全模型 peak） | **0.22×**（高估）|

即便给「截面块只占分 58.8%」打折（0.588 × 7.77% ≈ 4.6%），实测仍只有 **0.37×**。

⭐ **符号翻转本身是信号**。两次高估的共同点是**在已有块上加派生量**
（slow/fast = 预测自身的慢/快拆分；长窗 = 已选 history 列的长窗摘要）；
三次低估都是**加新的信息通道或新分量**（相位采样、history40、行级市场森林）。
⟹ **本地尺子对「派生量」类改动系统性乐观**，这一条应进③类先验。
⚠️ 5 个样本、事后归纳，**不是**已验证的规律；写在这里是为了下次可以**先验注册**再验。

**这一枪问出来的东西**：确认档只到 `PASS_BUT_BELOW_DETECTION_FLOOR`（0.89× 下限）——
本地已经答不出「幅度多大」。公榜答了：**+1.66%**。这正是花这一枪的理由。

**决策**：生产**仍未转正**（`--activate` 未加，`strategies/v3_hybrid/model/` 一字节未动）。
是否转正由用户定，权衡是：
- 收益：+1.66% 公榜实测，门禁全过（一致性 4.019e-09、双后端 1.39e-16、耗时无回归、0 触限）；
- 代价：新增一个跨 `predict` 状态 = 模型身份变更；本项目的伤疤多数是交付事故。
- ⚠️ 私榜是 9 月另一时段，而本次迁移率 0.22× 说明本地对这类改动乐观 —— 但公榜 +1.66%
  本身就是**测试期实测**，不再是本地外推。

**证据**：`experiments/ledger.csv` 2026-08-21 行、`outputs/promotions/v3_hybrid_long512/`、
`outputs/experiments/long_window_{ladder,confirm}.{json,md}`、`long512_*_4t.{json,md}`。

---

### 2026-08-21 — `DELIVERY`：长窗 w512 候选已训练并通过全部本地门禁（**未转正、未上榜**）

**日期**：2026-08-21　**标签**：`RESULT`（交付验证；生产目录一字节未动）

**做了什么**：把 `long_window_confirm` 的结论落成一个可提交的候选
`outputs/candidates/v3_hybrid_long512` → staging `outputs/promotions/v3_hybrid_long512`。

**实现要点**：
1. ⭐ **新增 `history.AssetLongWindow`**。不能把 `AssetHistory` 的 window 调大 —— 它是 O(window)，
   window=512 时在线每步要搬 512×40×15 ≈ 307K 个元素 × 21.4 万次，跑不动。
   改用**持久累积和相减**（与 `main.PredictionTrail` 同一套路）。
   ⭐ 逐位一致是**构造上的**：离线 `np.cumsum(dtype=float64)` 本身就是定序累加，
   在线维持持久 running total —— 实测 `np.array_equal` 为 True、max|Δ| = 0。
   ⚠️ **分块重起**的 cumsum 与整段**不**相同（实测 False），那正是 `history.py` 开头警告的写法；
   离线批处理必须把上一批的 `running` 作为 cumsum 的**第一行**接上（`r+(s0+s1)` ≠ `(r+s0)+s1`）。
2. ⭐ **长窗块只进截面设计**。训练日志自证：截面 2,645,530 × **441**（361+80）、
   市场 2,645,530 × **561**（= raw 200 + 截面块 **361**，不是 441）⟹ 市场设计一列未动。
3. **只重训截面森林**，市场森林与冻结岭回归按 `reuse_forest` 原样复用
   （hash 与生产**逐字节相同**，且 `reuse_forest` 硬校验选列 + 4 组统计量一致 ⟹ 选列未变）。
4. **只多一个 meta 键 `long_window`**（复用 history 那 40 列与同一套统计量 ⟹ 不扩输入契约）。

**本地门禁全部结果**：

| 门禁 | 结果 |
|---|---|
| 缺键时与旧模型逐位不变 | `check_consistency` 生产模型仍是 **4.019e-09**（两后端），与 ROADMAP 记录一致 |
| train/inference 一致性（交付配置） | **4.019e-09**（两后端）|
| 双后端对拍（staging） | **1.39e-16**（生产那次 2.082e-16）|
| 全量 runner @ 4 线程 LightGBM | `predict_total` **5.33 分钟**（生产 5.26，+1.3%）|
| 全量 runner @ 4 线程 NumPy 兜底 | **10.55 分钟**（生产 10.94，**−3.6%**）|
| 行数 / 超时 / 非有限值 / 触 clip | 3,217,458 / 0 / 0 / 0（max\|pred\| 0.402099 < 0.5）|
| 跑批用的包 vs staged manifest | 8 个文件**逐字节一致** |
| 单测 | **174 passed / 27 subtests** |

⭐ **一致性数值与生产完全相同（4.019e-09）不是加载错模型** —— 已直接对拍两个模型排除：
森林特征数 361 vs 441、预测 max|Δ| = 5.11e-02、corr 0.9796。
相同是**良性**的：该指标衡量浮点路径一致性，残差由两模型**完全相同**的岭回归路径主导。
⟹ 反过来说这是更强的结论：**长窗块对训练/推理差异的贡献是 0**（它逐位一致）。

⭐ **触 clip 0 行、max|pred| 0.402099** ⟹ 与公榜 0.0041150085 在**同一 scale 1.16** 下
可直接比大小，二次式精确成立，**不依赖任何近似**。

**顺带修掉的三个登记缺口**（都属「加了身份键但门禁不认识」，与 08-18 slow/fast 那次同类）：
- `promote_v3_candidate.PUBLIC_BASELINE` 加 `"long_window": None`（榜上那份没有它）
  ⟹ 带长窗的候选**必然**被判 drift、必须显式 `--off-baseline`。这是**有意的**：
  把「悄悄换了模型身份」与「明知故犯地换」区分开。
- `make_submission.check_v3_hybrid_meta` 与 `audit_submission_zip.public_baseline_drift`
  **两张取值表**都要同步 —— 我只改了后者，`make_submission.py:170` 的防呆当场报
  「缺 long_window」并说清该改哪。那正是 08-13 加键漏改导致 KeyError 的同一个坑。
- `verify_delivery_runtime` 的 `meta_identity` 漏打 `long_window`。报告号称打印「meta 身份」，
  漏一个身份键等于报告在说谎。已补。

**⚠️ 过程中的一次自查**：核对「跑批包 vs manifest」时第一版查错了键（manifest 用的是
`staged_files`），查找返回空 dict ⟹ 循环一次没跑、`ok` 保持初始 True ——
**空循环的真值不是验证**。已改为先 `assert files` 再比对，实测 8/8 一致。

**决策**：**不转正**（未加 `--activate`，生产目录一字节未动）。
下一步是公榜一枪，**CSV 由用户生成**（CLAUDE.md §1.4）。
公榜是唯一还活着的外部裁判（8/23 停更），而确认档只到 `PASS_BUT_BELOW_DETECTION_FLOOR`
（0.89× 检出下限）⟹ **本地已经给不出更多信息了，只有公榜能回答「幅度多大」。**

**证据**：`outputs/promotions/v3_hybrid_long512/promotion_manifest.json`、
`outputs/experiments/long512_{lightgbm,numpy_fallback}_4t.{json,md}`、
`tests/test_asset_long_window.py`（11 用例）。

---

### 2026-08-21 — ⭐ `RESULT`（确认档）：长窗 w512 —— 五道全过，但幅度低于 3s480 的检出下限

**日期**：2026-08-21　**标签**：`RESULT`（裁决 `PASS_BUT_BELOW_DETECTION_FLOOR`；**未**上公榜、生产未动）

**问题**：`long_window_ladder`（筛选档 1s160）给出 `w512` +6.80% / 5-of-5 / 五道全过。
在**确认档**（3 种子 × 480 轮）上还成立吗？

**设计**（预注册 `long_window_confirm_plan.json`，sha256 `11eb76283c69…`，先于结果落盘）：
只留 `base` 与 `w512` 两臂（`w64`/`w4096` 两个评价器打架，带上就是多重比较捞鱼）；
fold 版图与筛选档**完全相同**；`base` 本次现跑；市场块**不参与不改动**。
⚠️ 预注册**写死了三档裁决**，其中中间那档是跑之前就预判最可能的：
`PASS`（五道过且 ≥8.7%）/ **`PASS_BUT_BELOW_DETECTION_FLOOR`** / `REJECTED` /
`INVALID_LINEAR_CROSSCHECK_FAILED`。

**结果**：

| 指标 | 值 |
|---|--:|
| pooled Δpeak | **+7.77%** |
| 逐折 | +13.38 / +6.76 / +1.91 / +8.43 / +11.48 |
| 正折 | **5/5** |
| 去最好折 | **+6.49%** |
| 配对 bootstrap CI 下界 | **+4.18%** |
| 检出下限倍数 | **0.89×**（+7.77% vs 8.7%）|

五道门槛**全过**。裁决 **`PASS_BUT_BELOW_DETECTION_FLOOR`**。

**解释与限制**：
1. ⭐ **符号非常干净**：5/5 折为正、去最好折仍 +6.49%、**配对 bootstrap CI 下界 +4.18%**
   （排除 0 且离 0 不近）。⟹ **「有没有效」这个问题有答案了：有。**
2. ⚠️ **但「多大」没有答案**：+7.77% 只有 3s480 检出下限的 **0.89×**。
   按预注册与仓库惯例（`mkt323` +1.09% 是 0.70× 被判「测不出来」），
   **这只够作为花一次公榜额度的理由，不构成晋级依据。**
3. ⭐ **筛选→确认迁移率 1.14×**（+6.80% → +7.77%），**未衰减**。
   与本项目③类改动「本地系统性低估」的历史一致（公榜迁移率 1.20× / 1.6× / 2.3×）。
   ⚠️ 但 slow/fast 那次是 **0.51×（本地高估）**，所以不能当规律用。
4. ⭐ **基准更强而增益仍在**：3s480 的 base 比 1s160 强 2.8%~21%（逐折不等），
   增益不但没被吃掉反而略升 ⟹ 这不是「弱基准撑出来的」，正是 P8 栽跟头的反面。
5. ⭐ **线性对拍 10/10 逐位相同**（线性不依赖树超参）⟹ 两次运行数据路径一致，
   排除了「换了个跑法」这一类解释。
6. ⚠️ **只测了截面块**。生产还有市场块（其设计里也含 history160），
   「长窗要不要同时进市场块」是**另一个问题，本实验不回答**。
7. ⚠️ **全模型效应会明显小于 +7.77%**：截面块只占分 58.8%（ROADMAP §3.9）。
   粗估全模型分数增益约 **+2%~+5%**，与 slow/fast（公榜 +2.93%）同量级 —— **粗估，不是测量**。
8. ⚠️ **本探针的 float64 cumsum 不能进生产**。`strategies/v3_hybrid/history.py` 刻意不用 cumsum
   （为保离线整块与在线逐 time_id 逐位一致）⟹ 生产实现必须换成定序求和并过 `check_consistency`。

**决策**：生产不动、meta 不改。这是今天唯一走到确认档的线索，**是否花 8/23 前的公榜额度由用户定**。
若要花，前置工作是：① 生产端 512 步滚动状态（定序求和，非 cumsum）；
② 只重训截面森林（市场森林按 `train.py` 的 `reuse_forest` 原样复用，与 `mkt_shrunk` 同做法）；
③ CSV 由用户生成（CLAUDE.md §1.4）。

**重新开放条件**：8/23 回补数据后按原规格复验；或市场块同改后作为独立实验测。

**证据**：`outputs/experiments/long_window_confirm.{json,md}`、`..._plan.json`、
`tests/test_long_window_confirm.py`（7 用例，含「检出下限必须是 8.7% 而非 6.1%」的防复制粘贴断言）。

**⚠️ 过程记录（工具可靠性）**：本次运行期间收到 **3 次内容与落盘日志不符的后台通知**
（2 次数字被篡改、1 次在 fold 4 还在拟合时就报「completed exit 0」）。
⟹ 本轮所有数字均从落盘文件核对后才采用；后续判定只认产物文件与 `kill -0`，不认通知正文。

---

### 2026-08-21 — ⭐ `RESULT`（筛选档 PASS）：长窗列数阶梯 —— 信号在 512 那一档，之前是被 240 列的估计代价淹掉了

**日期**：2026-08-21　**标签**：`RESULT`（1s160 筛选档五道门槛全过；**尚未**做 3s480 确认，**尚未**上公榜）

**问题**：`long_history_probe`（同日）一次加了 **240 列**（40 特征 × 窗口 {64,512,4096} ×
{滚动均值, 偏离}），pooled +0.69%、`REJECTED`。但那个设计把「信号」与「估计代价」焊死了 ——
分不清「长窗里没信号」和「有信号但被 240 列的估计方差淹掉」。
（⚠️ 当天曾用 ΔA/ΔB 声称「有信号但付不起方差」，**那个读数是错的并已收回**。）
拆成单窗口各 **80 列**，列数降到 1/3，若某档由负转正 ⟹ 信号集中在那个窗口。

**跑前声明的两处设计变更**（预注册 `long_window_ladder_plan.json`，sha256 `4d9aeb2b44fa…`）：
1. 主评价器从线性改为**树**（LightGBM 1s×160，只跑截面块）—— `long_history_probe` 唯一的实质
   缺陷就是线性内层 alpha 选参（fold 4 两臂都选到梯底 1e-6 造出 −19.49%），换树后这类脆弱性消失。
2. 线性降为次臂，用**新常量** `WIDE_ALPHA_LADDER`（1e-8~1e+2，11 档），
   **绝不就地改** `function_class_probe.ALPHA_LADDER`（它被两个已结案实验 import，
   改了会让复跑与落盘产物对不上 —— 08-18 那类事故的形状）。有单测断言旧常量未变。

**结果（树，主判据）**：

| 臂 | pooled Δpeak | 逐折 | 正折 | 去最好折 | bootstrap CI 下界 | 超检出下限 | 判定 |
|---|--:|---|--:|--:|--:|:--:|:--:|
| `w512` | **+6.80%** | +7.30 / +1.87 / +7.10 / +7.92 / **+12.26** | **5/5** | **+5.84%** | **+3.87%** | ✅ 1.12× | **✅ 五道全过** |
| `w4096` | +3.43% | −0.63 / +0.57 / +7.15 / +4.67 / +6.36 | 4/5 | +2.32% | +0.71% | ❌ | ✅ 但低于下限 |
| `w64` | +4.51% | +10.16 / −0.07 / +10.39 / +0.85 / −0.53 | 3/5 | +2.76% | −0.05% | ❌ | ❌ |

⭐⭐ **预注册的机制预测兑现了。** 在**同一个评价器（线性）**上做同一件事：

```
240 列（三窗合一，long_history_probe）  →  +0.69%
 80 列（单窗 512，本实验）             →  +5.75%
```

⟹ **信号一直在，是被 240 列的估计代价淹掉的。** 这正是列数阶梯要问的那个问题，
现在有答案了 —— 而 `long_history_probe` 当时给不出。

**解释与限制**：
1. ⭐ **`w512` 是唯一两个评价器都认的臂**（树 +6.80% 5/5 pass、线性 +5.75% 4/5 pass）。
   `w64` 线性过树不过（+9.96% vs +4.51%/3-of-5），`w4096` 树过线性不过（+3.43% vs −0.33%/1-of-5）
   ⟹ **「哪个窗口」只在 512 上被稳健识别**，w64/w4096 不要采信。
2. ⭐ **加宽梯子生效**：全部 40 次线性拟合**无一撞端**（旧梯子下 1e-6 就是底），
   `long_history_probe` 那个 −19.49% 的坏折**消失** ⟹ 追认那确实是选参失效。
3. ⚠️ **这是筛选档**（1 种子 × 160 轮、**只有截面块**），生产是 3 种子 × 480 轮 × 两片森林。
   本项目③类改动的迁移率历史区间是 **0.51×~2.3×** ⟹ 生产端可能是 +3.5%，也可能是 +15%。
4. ⚠️ **只超检出下限 1.12 倍**（+6.80% vs 6.1%）。对照 `mkt323` 是 0.70× 被判「测不出来」；
   这次刚过线，不是绰绰有余。
5. ⚠️ **三个臂 = 多重比较**。预注册已声明单臂通过需外部确认；不得据此直接晋级。
6. ⚠️ **本实验的滚动均值用 float64 cumsum**（研究口径）。`strategies/v3_hybrid/history.py`
   刻意**不用** cumsum（为了离线整块与在线逐 time_id 逐位一致）⟹ **若要进生产，
   必须换回定序求和的实现并过 `check_consistency`**，不能照搬本脚本。

**决策**：**不**动生产、**不**改 meta。下一步按仓库既有阶梯走：
`筛选(1s160) → 确认(3s480) → 公榜`。确认档作为**新的预注册实验**另开
（只有 base 与 w512 两臂，不再带 w64/w4096 —— 它们已被第 1 条排除）。

**证据**：`outputs/experiments/long_window_ladder.{json,md}`、`..._plan.json`、
`tests/test_long_window_ladder.py`（11 用例，含「旧 ALPHA_LADDER 未被改动」的断言）。
`base` 臂的树 peak 五折与同日 `selection_criterion_probe` **逐位相同**
（2.5263e-03 / 2.9779e-03 / 2.8497e-03 / 2.1639e-03 / 1.8691e-03）⟹ 两个独立脚本互相验证。

**后续问题**：窗口只测了 64/512/4096 三个点，512 是内部极值还是单调段上的一点未知；
但**不得**在筛选档上继续搜窗口（那是看结果选参）—— 若 3s480 确认通过，
窗口精调只能在确认档上作为独立预注册实验做。

---

### 2026-08-21 — `REJECTED`：选列准则探针 —— 单变量筛子的诊断成立，但换掉它更差

**日期**：2026-08-21　**标签**：`REJECTED`（三个臂、五道门槛全败）

**问题**（用户提出）：`select_features` 是单变量筛子（`|加权 Pearson|` 取 top-N），
忽略特征间相关结构 —— 共线的强特征会被重复选进来，单独弱但有独立贡献的列会被丢掉。
另有一处更具体的错配：`history_positions`（`v3_production_oof.py:412`）按
**`|corr(xs_dev[t], e[t])|`（当期）** 选 40 列，而这 40 列进模型的形态是
`previous`(lag1) / `difference` / `rolling_mean(5)` / `rolling_deviation` —— **看起来全是滞后量**。

**跑前的可行性测量**（`train_partition_008`，同生产口径）：
top-40 重合 当期 vs lag1 = **24/40**、当期 vs rollmean5 = 25/40、lag1 vs rollmean5 = **35/40**
⟹ 两个滞后准则彼此高度一致、却都与当期不一致 ⟹ **真实区分，不是抽样噪声**。据此立项。

**设计**（预注册 `selection_criterion_probe_plan.json`，sha256 `a86d4de02c25…`，先于结果落盘）：
评价器 = LightGBM **1 seed × 160 轮、只跑截面块**（不能用线性 —— 对 `lasso200` 臂那是循环论证）。
四臂共用一次数据加载与 fold 版图；四臂的 history 列取并集**只流式扫一次**再切片
（AssetHistory 四个块逐列独立 ⟹ 等价，有回归用例钉住）。
⚠️ 判据**显式剔除 `2ΔA>ΔB`**（同日 `long_history_probe` 刚证明它混着共同尺度、且是两分量判别式），
只用尺度不变的 `peak = A²/(B·D)`。

**结果**：

| 臂 | pooled Δpeak | 正折 | 去最好折 | bootstrap CI 下界 | 与 base 的 history 重合 |
|---|--:|--:|--:|--:|--:|
| `lasso200` | **−1.29%** | 2/5 | −2.59% | −6.47% | 200 列重合 107~119/200 |
| `hist_lag1` | **−4.38%** | **0/5** | −4.88% | −7.10% | 17~24/40 |
| `hist_roll5` | **−10.39%** | **0/5** | −11.25% | −12.47% | 12~21/40 |

⭐ **单调关系**：与 base 分歧越大，掉得越多（`lasso200` 分歧最小掉 1.29%、
`hist_roll5` 分歧最大掉 10.39%）。⟹ 不是「换了列碰巧不好」，是
**当期准则挑出来的那 40 列确实更好**。`hist_lag1` **0/5 折**尤其干净，不是噪声。

**解释与限制**：
1. ⭐ **诊断是对的，代价不存在。** LASSO 换掉了 **45%** 的 200 列（重合 107~119/200），
   滞后准则换掉了 **40%~70%** 的 history 40 列 —— 分歧巨大且真实，
   但树的 OOF peak **一个都没变好**。
2. ⭐⭐ **选列这条轴现在被三个方向同时封死**：
   `feature_screen_1s160`「**更宽**（323 全给）」→ 打平；ledger `hist_c80`「history **更宽**
   （top-80 超集）」→ 公榜 0.00%；本实验「**换准则**」→ 全负。
   ⟹ 对 `feature_fraction=0.7` 的树，**前置筛子的选择质量不是绑定约束**。
3. ⚠️ **为什么「错配」的直觉落空（事后解释，不是本实验的发现）**：history 四个块里
   `difference = 当前 − previous` 与 `rolling_deviation = 当前 − rolling_mean` **都含当前值**，
   所以那个块并非「纯滞后」。按当期相关选列，对这两个子块反而是匹配的。
   要验证得做四子块消融，本轮**没做**，因此这只是解释、不是结论。
4. ⚠️ `lasso200` 在 fold 4 的 `uni∩lasso` 只有 68/200（其余折 107~119）——
   fold 4 在今天三枪里都是异常折（线性 alpha 选到梯底、r 塌到 0.34），原因未查。

**决策**：`REJECTED`。**不**试 Rank IC / 分组检验 / SHAP：
Rank IC 与分组检验同属「单变量 + 单调性」一族，而选列发生在 `robust_transform_fit`
**之后**（已裁尾/中位数中心化/IQR 缩放/±10 截断），厚尾早被处理；
SHAP 是循环的 —— 树本来就看得到全部列，SHAP 只能剪掉树认为不重要的，**不能加信号**，
而「全给 323 列」已经打平 ⟹ 剪枝不是瓶颈。
用户提的「滚动样本外验证」**本来就在做**（`rolling_time_folds` + embargo 6，选列只在训练折内拟合）。

**重新开放条件**：history 块的**用法**改变（例如换成序列/权重共享提取器）⟹
选择准则应随用法重新匹配，届时按原规格复验；或 8/23 回补数据后基准变化。

**证据**：`outputs/experiments/selection_criterion_probe.{json,md}`、`..._plan.json`、
`tests/test_selection_criterion_probe.py`（12 用例）。

**顺带补的基础设施**：`tests/test_select_features.py`（9 用例）——
`select_features` 有 **77 个调用点、此前 0 个测试**。钉住返回升序、`count` 两端截断、
零权等价于删行、`|corr|` 双向、零方差列不进 top-N、不修改入参。与本实验结论无关，
但它是研究代码里被依赖最多的函数。

---

### 2026-08-21 — `REJECTED`：长历史窗探针 —— 信息**确实存在**，但付不起估计它的代价

**日期**：2026-08-21　**标签**：`REJECTED`（两个函数类家族均未过预注册门禁）

**问题**：生产 history 块只有 `window=5`，而 slow/fast 证明**预测**在 K=2000 真实步尺度上
仍有可用结构（公榜 +2.93%）。中间那段从没被看过 —— `experiments/temporal_multiscale.py:49`
写着 `MAX_LAG = 20`，temporal 全族（`t1_lags`/`t2_state`/`t3_full`/`f_lags`/`f_changes`/
`f_volatility`/`f_trend`）的滞后都止步于 **20 个真实步**。**20 到 2000 是 100 倍的未测跨度。**

**为什么主臂敢用线性**：同日 `function_class_probe` 实测同一输入上线性 r=0.611 / 核 r=0.798 /
树 r=1.000，三者互相 ρ≈0.6~0.7 ⟹ 读到的是同一个东西 ⟹ **线性是「信息在不在」的有效探测器**。

**设计**（预注册 `long_history_probe_plan.json`，sha256 `f9158ae04449…`，先于结果落盘）：
`base` = 与生产截面块逐列相同的 `[xs_dev200 ‖ history160 ‖ asset_id]`；
`long` = base + 40 个 history 特征 × 窗口 {64, 512, 4096}（观测数）× {滚动均值, 当前−滚动均值}
= **240 列**。滚动均值严格滞后、无历史取 0，与 `history.py` 边界规则一致。
五折行对齐以 `target`/`weight` 逐位相同验证通过；`linear_base` 五折 IC 与 `function_class_probe`
的线性臂**逐位相同** ⟹ 两个独立脚本复现出同一设计矩阵。

**结果**：

| 函数类 | pooled 配对增益 | 正折 | 去最好折 | ΔA | ΔB | 2ΔA>ΔB | bootstrap CI 下界 |
|---|--:|--:|--:|--:|--:|:--:|--:|
| `linear` | **+0.69%** | 4/5 | −0.49% | **+3.36%** | **+15.40%** | ❌ | −5.96% |
| `rff` | −0.04% | 3/5 | −0.68% | −2.03% | −3.61% | ❌ | −2.47% |

逐折线性增益：**+3.25% / +4.54% / +1.42% / +1.74% / −19.49%**。

**解释与限制**：
1. ⚠️⚠️ **判据 4（`2ΔA>ΔB`）在本设计下无效 —— 我第一版的解读是错的，在此收回。**
   实测恒等式：逐折 `A比/√(B比) ≡ IC比`。fold 0 的 A 比是 **0.9555**（A 其实是**降**的），
   B 比 0.8564 降得更多，于是 IC 反而涨 ⟹ 表里的 ΔA/ΔB **混着两臂岭回归解的共同尺度**
   （`A→cA`、`B→c²B` 时 IC 不变而 ΔA、ΔB 都非零）。
   更根本：`2ΔA>ΔB` 是**两分量配比**的判别式，本实验是**嵌套模型**（long ⊃ base）比较，
   这道判据从一开始就不该进预注册。**ROADMAP 2026-08-19 记着 P8 已栽过同一个坑，本次重犯。**
2. ⛔ **由此作废的推论**：「ΔA=+3.36% 说明长窗里确实有信号」与
   「零方差代价的完美提取器上界 = +3.36% IC」**两条都无效**。
   本实验**没有**给出「长窗里有多少信号、且可与估计代价分离」的有效测量 ——
   想要那个数，得做单窗口消融或列数阶梯，本轮没做。
   ⟹ **序列模型的价值仍未定价**，不能说它「最多补五分之一」，也不能说它值得做。
3. ⚠️ **预注册缺陷（如实记）**：fold 4 两臂的内层 alpha 都选到梯子**最底**（1e-5 / 1e-6，
   ≈ 不正则），该折 −19.49% 多半是**选参失效**而非「长历史有害」。
   ⭐ 但**结论对它稳健**：去掉 fold 4 后 pooled = **+2.77%**（仍不过 3%），
   再去最好折 = +2.16%；而 `2ΔA>ΔB` 那道无论去不去 fold 4 都过不了。
   ⚠️ 与 `function_class_probe` 的缺陷是镜像的（那次 RFF 全钉在梯**顶**）⟹
   **`ALPHA_LADDER` 两端都太窄**，下次复验必须先加宽。
4. ⚠️ RFF 臂有约 **2.5% 的种子级抖动**（同一 fold 0 同一设计，第一枪 r=0.826、本枪 r=0.847，
   差别只来自随机投影抽样）⟹ RFF 的单折 r 不能当精确量读。

**决策**：`REJECTED`（判据 1/3/5 各自独立失败，IC 与 peak 都是尺度不变量，裁决可靠）。
**不**加宽 alpha 梯子重跑、**不**调窗口集合 —— 那是看结果选参。
与 `function_class_probe` 合读：**「换提取器」已定价（oracle 集成 +0.91%）；
「换输入」只测到「按现在这个 240 列的加法，净效果不过门槛」，其可分离的信号量仍未测。**

**重新开放条件**：出现一个**权重共享**的提取器（1D 卷积 / 序列模型），能在拿到 ΔA 的同时
把 ΔB 控制在 2ΔA 以内 —— 且要先加宽 `ALPHA_LADDER` 重跑本实验作为配对基准；
或 8/23 回补数据后基准变化，按原规格复验。

**证据**：`outputs/experiments/long_history_probe.{json,md}`、`long_history_probe_plan.json`。

**后续问题**：本实验一次加了 240 列，无法把「信号」与「估计代价」分开。
若将来重开，正确的下一步是**列数阶梯 / 单窗口消融**（64 / 512 / 4096 各一臂，各 80 列），
用尺度不变的 Δpeak 逐档读 —— 列数减到 1/3，估计代价大致同比下降，
若某一档由负转正，就说明信号集中在那个窗口。**先加宽 `ALPHA_LADDER` 再跑。**
⚠️ 并且**不要**再用 `2ΔA>ΔB` 判嵌套模型；嵌套比较只能看 Δpeak / ΔIC。

---

### 2026-08-21 — `REJECTED`：函数类探针 —— 树之外的函数类在同一批输入上没有增量

**日期**：2026-08-21
**标签**：`REJECTED`（`rff_full` / `rff_pca64` 两臂均未过预注册门禁；实验有效）

**问题**：诊断 D 观察到树对时间/截面维的三次扩容全部以「ΔB 涨幅是 ΔA 的三倍」失败
（history 列 40→80 公榜 **0.00%**；`market_asset_panel` **−7.24%**，ΔA +16.88% 而
ΔB +49.44%；`peer_leadlag` −2.31%/−1.73%）。那是**函数类吃不下**的指纹。
换一个函数类、输入一列不改，能不能拿到树拿不到的东西？

**动机与机制**：随机傅里叶特征 + 加权岭回归是**光滑、全局、各向同性**的，与 LightGBM
的轴对齐分段常数在函数空间里尽可能不同 ⟹ 集成所需的 ρ 应该小。纯 numpy/scipy，
推理端只是一次矩阵乘，若过门槛交付成本几乎为零。

**实验设计与固定项**（预注册 `outputs/experiments/function_class_probe_plan.json`，
sha256 `7290281815e6…`，先于结果落盘）：
- 输入与生产截面块**逐列相同**：`[xs_dev200 ‖ history160 ‖ asset_id]`，逐折在训练段内
  拟合 `robust_transform_fit` / `select_features` / history 选列；asset_id 对光滑核改 one-hot。
- 标签 `e_tr`、**带权**；validation 上复刻生产的逐 time_id 无权零均值投影。
- fold 版图沿用 `rolling_time_folds(5, 78960, embargo 6)` + `modulo 5` + `phase_balanced`。
- 基准是**生产强度**的 `e_lgbm`（3 种子 × 480 轮）—— P8 的教训：`target_mlp_screen` 拿
  1s160 弱基准（比生产弱 1.51×）算 oracle 得 +6.97%，实测只有 +0.026%，差 270 倍。
- 阴性对照：同一设计上的普通加权岭回归。
- 五道门槛：pooled `r > ρ+0.05`、隐含集成增益 ≥ +3% IC、≥4/5 折 `r_f>ρ_f`、
  去最好折仍 ≥3%、对照臂不过判据 1。

**结果**：

| 臂 | pooled r | pooled ρ | oracle 集成增益 | r>ρ 折数 | 去最好折 |
|---|--:|--:|--:|--:|--:|
| `rff_full` | **+0.7983** | +0.7022 | **+0.91%** | 5/5 | +0.70% |
| `rff_pca64` | +0.6982 | +0.6379 | +0.31% | 5/5 | +0.23% |
| `linear`（对照） | +0.6105 | +0.5918 | +0.03% | 3/5 | +0.00% |

判定：两个 RFF 臂都过了判据 1/3/5，**卡在判据 2/4**（+0.91% vs 门槛 +3%）。
对照臂未过判据 1 ⟹ **实验有效**，测到的确实是函数类而不是早已存在的 ridge 块。

**解释与限制**：
1. ⭐ **方向是真的，量不够。** 5/5 折 `r > ρ` ⟹ 最优配比给 RFF 的是**正权重**（加信号 ΔA），
   不是负权重对冲（减方差 ΔB）。这是这个项目少见的「符号全对但幅度差 3.3 倍」。
2. ⭐⭐ **P9 的「NN 天花板 28.8%」是配方产物，不是能力事实。** 一个**没有任何训练配方**
   的 RFF-岭回归就拿到生产截面块 IC 的 **79.8%**，而 P9 的 sklearn MLP 只有 54%（IC 口径）。
   ⟹ 「非树模型族天生弱」这个印象被否掉了。
3. ⭐⭐⭐ **但拿到 80% 恰恰说明没用** —— ρ 随 r 一起涨到 0.70。三种函数类
   （线性 0.61 / 核 0.80 / 树 1.00）互相 ρ≈0.6~0.7 ⟹ **它们在读同一个东西**。
   ⟹ **这 361 列里的信息基本被榨干了**，换提取器换不出增量。
4. 树相对线性在**同一设计**上只强 **1.31×** IC，而 RFF 补掉了其中大部分 ⟹
   非线性在这批输入上总共只值约 30% IC，且已被生产模型吃掉。
5. ⚠️ **预注册缺陷（如实记）**：alpha 阶梯上界 1e-1 太低，两个 RFF 臂在 10 次拟合中有 9 次
   选到边界值 ⟹ 内层想要**更强**的正则而梯子没给。这是真截断。但 +0.91% 要涨 3.3 倍才过门槛，
   逐折模式（+1.94/+0.74/+0.88/+0.29/+1.00）没有任何迹象支持那个幅度。
6. ⚠️ 内层选参在折间不稳：线性臂 fold 4 选到 1e-5 而其余折是 1e-1，且该折 r 从 0.57~0.77
   塌到 0.34。RFF 在同一折稳在 0.77 ⟹ 核方法对这个不稳定性明显更稳健。
7. ⚠️ 增益是**最优配比的 oracle 上界**，不是可实现值（`peer_leadlag` 实测样本内 +2.358% /
   样本外 −1.280%）。真实可得只会更低。

**决策**：`REJECTED`。**不**围绕 RFF 调 D / 调带宽 / 加宽 alpha 梯子重跑 —— 那正是
「挑一个让增益最大的超参」。本轮真正的产出是把两个假设分开了：
「换提取器」已被否，剩下的只有「**换输入**」（更长的历史窗）。

**重新开放条件**：输入表示本身改变（不再是这 361 列）后按原规格复验一次；
或 8/23 回补数据后基准变化。

**证据**：`outputs/experiments/function_class_probe.{json,md}`、
`outputs/experiments/function_class_probe_plan.json`、`tests/test_function_class_probe.py`（10 用例）。
五折的行对齐均以 `target`/`weight` 逐位相同验证通过。

**后续问题**：`history_window=5` 只看到「上一次观测 + 5 步滚动均值」，而 slow/fast 说预测在
**K=2000 步**尺度上仍有结构。5 与 2000 之间从没被任何函数类看过。下一个该问的是
「**更长的历史窗里到底有没有信息**」，而且该用**线性模型**先问（便宜、且本轮已证明
线性/核/树在同一输入上读到的是同一个东西 ⟹ 线性够用来判断信息在不在）。

---

### 2026-08-21 — `INCIDENT`（未爆）：OOF 缓存的 `e_target` 列是全 NaN

**发现路径**：函数类探针原打算用「自算 `e_va` 与 cache 的 `e_target` 一致」做行对齐断言，
运行时报 `差 nan > 1e-9`。

**根因**：`experiments/v3_production_oof.py:511` 把 `e_target` 写成占位 `e_tr[:0]`，
紧接着 512 行 `if name == "e_target": continue` 又跳过赋值 ⟹ 该列从 `np.full(n, np.nan)`
初始化起**从未被写入**。所有 OOF 缓存（含 `confirm_3s480`）的这一列都是全 NaN。

**影响范围**：`src/oof_cache.py:19` 仍把 `e_target` 列在 `COMPONENT_COLUMNS` 里，
`tests/test_oof_cache.py:24` 造合成缓存时也带它。已逐个扫过 `experiments/*.py`，
**当前没有任何脚本消费它** ⟹ **没有污染过任何已有结论**。

**风险**：下一个把它当标签用的实验会静默拿到全 NaN。`e_tr` 本来就可以由
`target − 逐 time_id 无权截面均值` 现算，成本可忽略。

**规避**：探针改用**确实被写入**的 `target`/`weight` 做逐位对齐断言（五折全部 max|Δ| = 0）。

**建议门禁**（未实施，待用户决定）：`src/oof_cache.load_oof_bundle` 对
`COMPONENT_COLUMNS` 里的每一列断言「不是全 NaN」，或把 `e_target` 从该元组移除。

---

### 2026-08-18 — `RESULT`：重建测试补测落盘 —— 数字全部确认，但口径此前没写清楚

**证据缺口**：ROADMAP:165 与本文件上一条用「重建 R² 只 0.883、单步 u 不存在」关闭了
horizon 分解方向。但全仓库检索（排除 `.venv/.git/data`）显示 **`0.207` 只出现在
`NOTES.md` 里** —— `responder_window_atlas.py` 只算自相关和错位相关，其 JSON 没有
reconstruction 字段，没有任何脚本能产出那张表。⟹ 关闭理由当时建立在无法复现的测量上。
新增 `experiments/responder_reconstruction.py` 补测（不改 atlas、不覆盖其产物）。

**结果**（全量 9 分区、约 1,322 万行配对，耗时约 20 秒）：

| 设计 | R² 中心化 | R² 非中心化 | NOTES 记录 | 复现 |
|---|---:|---:|---:|:--:|
| `responder_00 @ +1..+5`（纯 u 假设）| **0.2010** | 0.0125 | 0.207 | ✅ |
| `responder_02 @ 0..+3` | 0.8245 | 0.0781 | 0.818 | ✅ |
| `responder_03 @ −1..+1` | 0.8392 | 0.1574 | 0.835 | ✅ |
| `responder_04 @ −4..−2` | 0.7349 | 0.2239 | 0.732 | ✅ |
| **全部合并**（15 个回归元）| **0.8873** | 0.8461 | 0.883 | ✅ |

五格全部落在 ±0.006 内 ⟹ **NOTES 的记录被确认，ROADMAP 的关闭理由现在有产物支撑。**
纯 u 假设要求 R²→1，实测 0.201 ⟹ `responder_00` 不是 target 所聚合的那个单步增量，
这一条站得住。

⚠️ **但口径此前完全没写，而它的影响很大**：NOTES 那张表是**带截距的中心化 R²**。
responder 带很大的非零均值（`responder_03` 加权均值 +0.502 而 std 只有 0.262），
所以换成项目指标口径（无截距、分母 `Σw·y²`，与 `src/metric.py` 的
`Score = 1 − Σw(y−ŷ)²/Σw·y²` 一致）后同一个设计从 0.84 掉到 0.16。
两套都已落盘。判定复现用中心化那一套（那才是「张成了多少」的正确度量）；
非中心化一列只作诊断，别拿它当「按比赛指标能兑现多少」。

⚠️ 一处对不上但不影响结论：NOTES 写的是「299 万行」，本次全量配对是 1,322 万行。
R² 值逐格吻合，故判为当时的行数记述问题，不是口径问题。

证据：`outputs/experiments/responder_reconstruction.{json,md}`。

### 2026-08-18 — `REJECTED`：per-asset ridge **叠加**到生产截面块上也不行

**问题**：per-asset ridge 单独比生产截面块低 50.5%，但「单独更弱」≠「叠加无用」——
只要足够去相关，组合仍可能加分。（responder 那条的 blend 版本 A0 早已测过：
`multi` 臂折均 −14.08%、0/5 折，已结案。per-asset 这条确实没测过。）

**设计**：基准 = 生产 `e_lgbm` **单独**（同批验证行上的最优单 scale），不是共享 ridge；
候选 = 两系数组合 `c1·e_lgbm + c2·ridge`，系数按扩展窗口只用 fold 0..k−1 拟合；
ridge 预测按 time_id 投影成无权零均值（与 `main.py` 里 `e_lgbm -= e_lgbm.mean()` 同口径）。
⚠️ 按 CLAUDE.md 伤疤 §6，**只以配对 peak 增量判决**，corr 仅作诊断。
另设 `shared` 臂作**对照**：若共享 ridge 也能叠加，那说明是「ridge 补树」而非
「per-asset 结构补树」。

**结果 —— 全部为负**：

| 臂 | Δ折均 | 相对 | 正折 | corr(与 e_lgbm) |
|---|---:|---:|---:|---:|
| `shared`（对照）| −1.254e−05 | −1.19% | 2/4 | +0.570 |
| `per_asset_k0` | −4.135e−05 | **−3.91%** | **0/4** | +0.593 |
| `per_asset_k1` | −4.525e−05 | −4.28% | 0/4 | +0.631 |
| `per_asset_k10` | −2.679e−05 | −2.53% | 2/4 | +0.616 |
| `per_asset_k100` | −1.076e−05 | −1.02% | 1/4 | +0.617 |

**解释**：与 `e_lgbm` 的相关是 **0.57~0.63**，不是「低相关」—— ridge 与树在很大程度上
重叠，叠加只带来估计方差。更能说明问题的是：**per-asset 臂比 `shared` 对照还差**
（−3.91% vs −1.19%），且 κ 越小（越 per-asset）越差 ⟹ 那些逐资产系数
在树已经吃掉这块结构之后，纯粹是噪声。

**决策**：`REJECTED`。至此 per-asset 这条轴**替换与叠加两种用法都关闭**。
**重新开放条件**：截面块换成表达不了 asset 交互的模型族。
证据：`outputs/experiments/asset_blend_check.{json,md}`。

### 2026-08-18 — `RESULT`：资产**确实**异质（线性载荷差很多），但生产的树已经吃掉了这块

**问题**：15 个资产共用同一套系数。它们该不该有各自的线性载荷？

**设计**：口径对齐现有 OOF（5 折 / modulo 5 `phase_balanced` / 训练窗 78,960 / embargo 6）；
目标是截面分量 `e`，设计矩阵是 200 列截面去均值；三臂 `shared` / `per_asset(κ=0)` /
`shrunk(κ 阶梯)`，κ→∞ 逐位退化为 shared（断言实测 −1.08e−20，通过）。

**结果 —— 异质性是真的**：

| κ（向共享收缩） | Δ折均 | 相对 | 正折 |
|---|---:|---:|---:|
| 0（纯 per-asset） | +2.475e−04 | **+95.8%** | **5/5** |
| 1 | +2.239e−04 | +86.7% | 5/5 |
| 10 | +9.028e−05 | +35.0% | 5/5 |
| 100 | +4.649e−05 | +18.0% | 5/5 |
| 1000 | +3.813e−05 | +14.8% | 4/5 |
| ∞ | −1.08e−20 | −0.00% | 0/5（退化检验）|

整条 κ 阶梯全过门槛；资产间**系数余弦相关只有 +0.419**（远不是 1）；
逐资产 peak 最大/最小 **12.91×**。⟹ 在**线性模型内**，资产异质性又大又稳。

**⚠️ 但决定性的对照否掉了它的可部署价值**：

| 模型 | 折均 peak |
|---|---:|
| 共享 ridge | 0.00025823 |
| **per-asset ridge** | 0.00050570 |
| **生产 LGBM 截面块** | **0.00102218** |

per-asset ridge 比共享 ridge 高 **+95.8%**，但比生产的截面块**低 50.5%**。
原因很直接：生产截面块是 LightGBM，而 `asset_id` **本来就是它的 categorical 特征**
（`d_tr_xs = [xs_dev ‖ history ‖ asset_id]`，`categorical_feature=[cat]`）——
树可以在 asset 上分裂，早就能表达资产特异行为；只有**线性**模型表达不了。

⟹ 这个 +95.8% 是「线性模型补回了树本来就有的能力」，**不是生产模型的未开采余量**。
同时它也解释了两件旧事：为什么每资产**标量** scale 在公榜上不可辨别（树已经吃掉了）、
为什么稀疏 asset×feature **残差**交互全负（残差里已经没有这块结构了）。

**决策**：`per-asset 载荷`轴**关闭**。诊断价值达成 —— 它把「资产是否异质」（是）与
「生产模型是否漏掉了它」（没漏）分开了。
**重新开放条件**：若截面块换成表达不了 asset 交互的模型族（例如纯线性/无 categorical 的结构）。
证据：`outputs/experiments/asset_loading_diagnostic.{json,md}`。

### 2026-08-18 — `RESULT`：responder 是一把**窗口长度梯子**；但 horizon 分解缺前提

**问题**：target 已确认是等权 MA(H=5)。主办方说 responder「覆盖多个预测窗口」。
有没有哪个 responder 是**更短窗口**版本？若有，就能把 target 拆成 horizon 分量分别定收缩。

**方法**：全量 1,320 万行，逐 asset、按**真实 time_id 步长**配对（复合键 searchsorted），
逐分区流式累加加权相关的充分统计量。两条独立证据：
(a) 每个 responder **自己的**自相关曲线对 `(H−k)/H` 做整条 RMSE 拟合 ⟹ 读出 `H_j`；
(b) 与 target 的错位相关 `corr(target_t, r_{j,t+k})`，`k ∈ [−12,12]`。

⚠️ **方法学修正**：第一版用「归零点」读 `H_j`（第一个 |ac|<0.05 的 lag），**会误读** ——
`responder_00` 的 ac 本来就全 ≈0，被报成 H=2；`responder_02` 的 ac=(0.49, 0.08, ~0) 明明是
MA(2)，却因 ac2=0.08>容差 报成 H=3。改成整条曲线 RMSE 拟合（target 上验证过的做法）才对。

**结果 —— 一把干净的窗口梯子**：

| | ac1 | 拟合 H | RMSE | 与 target 峰值 shift | 峰值 corr |
|---|---:|---:|---:|---:|---:|
| `responder_00` | 0.06 | **1** | 0.024 | +5 | +0.275 |
| `responder_02` | 0.49 | **2** | 0.025 | +1 | +0.625 |
| `responder_03` | 0.70 | **4** | 0.045 | 0 | +0.817 |
| **`target`** | 0.82 | **5** | 0.025 | — | — |
| `responder_04` | 0.84 | **7** | 0.046 | −4 | +0.854 |
| `responder_05` | 0.89 | **10** | 0.054 | −9 | +0.774 |

拟合 RMSE 0.024~0.054 ⟹ 这几个确实是**等权 MA 窗**；峰值 shift 随 H 单调
（+5 → +1 → 0 → −4 → −9），正是同源嵌套窗该有的样子。target 的 H=5 被第二次独立确认
（本次 ac=0.822/0.594/0.378/0.167/−0.010，拟合 RMSE 0.025）。

**决定性的重建测试**（加权 R²，299 万行）：

| 设计 | 加权 R² |
|---|---:|
| `responder_00` @ shift +1..+5（**纯 u 假设**） | **0.207** |
| `responder_02` @ shift 0..+3 | 0.818 |
| `responder_03` @ shift −1..+1 | 0.835 |
| `responder_04` @ shift −4..−2 | 0.732 |
| **全部合并** | **0.883** |

**解释**：若 `responder_00`（H=1）就是底层增量 `u`，则 `target=(1/5)Σu_{t+h}` 意味着
5 个错位应给出 R²→1。实测只有 **0.207** ⟹ **`responder_00` 不是 target 所聚合的那个 u**。
（若同源，更短的窗只会给出**更多**信息而不是更少。）
全部 responder 合并也只到 0.883 ⟹ 约 **12%** 的 target 不被任何 responder 张成
（与那份材料引用的「17.3% 正交」同量级）。

**决策**：窗口梯子是**真实的结构发现**并已量化；但 horizon 分解所需的「单步 u」
**在 responder 里不存在**，该路线的干净版本告吹。且 responder 只在 train 里
（runner 会剥掉 `responder_` 前缀列）⟹ 永远不能当推理输入，唯一用法是当训练目标，
而那条已被 A0 否决（外层 −14.44%、0/5 折）。⟹ **不推进，记录结构事实备查。**

**重新开放条件**：回补数据带来新的 responder 列，或找到 `H_j=1` 且与 target 高度同源的量。
证据：`outputs/experiments/responder_window_atlas.{json,md}`。

### 2026-08-18 — `REJECTED`：P4 收官 —— 训练数据量已经**饱和**，两侧都没有余量

接上条（缩短方向已否）。扩展窗臂跑完五折（1 seed × 160 轮，配对基准是同口径的
`v3_production_oof_phasebal_prodwindow_exact`）：

| fold | 基准(滑动 78,960) | 扩展窗 | 相对 | 加了多少数据 |
|---:|---:|---:|---:|---|
| 0 | 0.00086514 | 0.00086514 | **+0.00%** | +0（对照，必须逐位相同）|
| 1 | 0.00163381 | 0.00156762 | −4.05% | +25% |
| 2 | 0.00143389 | 0.00172649 | +20.41% | +50% |
| 3 | 0.00149302 | 0.00153761 | +2.99% | +75% |
| 4 | 0.00161617 | 0.00142134 | −12.05% | +100% |

折均 **+1.08%**、**正折 2/5**、去最好折 **−3.84%**、配对 bootstrap 95% CI
**[−6.71e−05, +1.03e−04]**（跨 0）、检出下限 8.52e−05（基准 peak 的 6.1%）⟹ **不过门槛**。

**两侧合读才是结论**：

| 方向 | 效应 | 一致性 | 读法 |
|---|---|---|---|
| 减数据（40k/60k）| −24.54% / −9.50% | 0/5、1/5 折，CI 远离 0 | **测得出来的有害** |
| 加数据（+25%~+100%）| 逐折 −12% ~ +20% 乱跳 | 2/5 折，CI 跨 0 | **纯散布，测不出效应** |

⟹ **78,960 已在收益递减点之后：拿走数据会掉分，加数据什么也换不来。**
这个不对称本身就是饱和的signature —— 缩短方向效应大且同号，加长方向只有散布。

**决策**：P4 整条结案，维持 78,960，不再碰训练窗轴。**不花公榜名额**
（空动作即保持现状；②类量反风险只在采纳改动时才需要公榜背书）。
**重新开放条件**：8/23 回补数据显著增加训练期长度后，按原规格复验一次。

⚠️ 过程中修掉一个真 bug，见下条 INCIDENT。
证据：`outputs/experiments/v3_recency_expanding_ladder_1s160.{json,md}`、
`v3_recency_expanding_1s160.{json,md}`。

### 2026-08-18 — `INCIDENT`：`v3_production_oof.py` 在峰值时刻白占 6 GB，且我把「省时间」当成了「省内存」

**现象**：扩展窗臂连续三次被内核 OOM killer 杀掉（fold 2/fold 3），日志里**没有**
`MemoryError`、没有 traceback —— 因为这台机器 **swap=0**，OOM killer 直接 SIGKILL。

**我的判断错误**：我建议「改跑 1 seed × 160 轮省资源」。seeds/rounds 控制的是**串行训练
多少棵森林**（每棵训完即释放），**完全不影响峰值内存**。峰值由**设计矩阵**决定，
只跟「训练行数 × 列数」有关，而扩展窗恰恰在放大训练行数。实测：1s160 每折 171~319s、
3s480 每折 729~867s（快 4 倍），但两者死的位置几乎一样，差别纯属分配器运气。
⟹ **「省时间」与「省内存」是两个轴，不能混用。**

**真 bug**：训练 market 森林（全流程最大的一次分配）时，`transformed_train/valid`、
`xs_tr/va`、`history_tr/va` 都已并进设计矩阵、不再被引用，却要等到折末才 `del`，
在峰值时刻白占约 6 GB。改成 market 训练**之前**就释放：

| fold | 训练行 | 修复前峰值 | 修复后峰值 | 降幅 |
|---:|---:|---:|---:|---:|
| 2 | 1.77M | 12.23 G | 7.39 G | −4.84 G |
| 3 | 2.06M | 13.68 G | 8.04 G | −5.63 G |
| 4 | 2.35M | 15.12 G | 8.70 G | **−6.43 G** |

修复后五折一次跑通（fold 4 用时 319s），此前 fold 3 必崩。该修复对以后每次 OOF 运行都有效。

**防复发**：`load_rows` 的 3.42 GB 全程常驻是另一个已知常量；下次再遇到 OOM，
先算「训练行数 × 列数 × 4 B」，不要指望调 seeds/rounds。

### 2026-08-18 — `REJECTED`：训练窗缩短明显有害；但阶梯单调，指向**加数据**那一侧

**问题**（P4）：未来更像最近的训练窗，还是完整历史窗？

**设计**：⚠️ 关键陷阱 —— `rolling_time_folds` 的 `first_valid_idx = train_window + embargo`，
直接改 `--train-window` 会把**验证段**一起挪走，各臂落在不同数据上、配对失效。
所以给 `v3_production_oof.py` 加了 `--train-truncate`：**固定 fold 版图、只砍训练段前端**。
比较脚本对 `time_id/asset_id/fold/target/weight` 逐位断言（已通过，1,461,732 行）。
另加 `--freeze-min-data` 拆混淆：窗口变短 ⟹ 行数少 ⟹ `min_data_in_leaf` 自动变小 ⟹
有效容量被动改变。3 seeds × 480 rounds，生产口径。

**结果**（基准 = 生产窗 78,960，逐折 peak 均值 0.00160444）：

| 臂 | Δ折均 | 正折 | 去最好折 | ΔA | ΔB | 配对 CI |
|---|---:|---:|---:|---:|---:|---|
| `w60000_scaled` | **−9.50%** | 1/5 | −12.64% | −3.17% | **+6.49%** | [−3.18e−04, −1.04e−04] |
| `w40000_scaled` | **−24.54%** | 0/5 | −26.39% | −3.95% | **+23.17%** | [−5.59e−04, −2.43e−04] |
| `w40000_frozen` | **−24.19%** | 0/5 | −27.54% | −6.41% | **+15.50%** | [−5.41e−04, −2.38e−04] |

检出下限（配对 bootstrap 半宽）≈ 1.39e-04 = 基准的 8.7%。效应远超它 ⟹
这是**测得出来的负结果**，不是「测不出来」。

**机制**：ΔA 只是轻微为负（−3~−6%），ΔB 却大幅为正（+6.5%~+23.2%）。
⟹ 缩短窗口**不太损失信号，主要是把预测方差抬起来** —— 训练数据少、系数更噪。
`frozen` 臂验证了这一点也验证了它不是全部：冻结 `min_data_in_leaf` 把 ΔB 从 +23.2% 压到 +15.5%，
但 ΔA 从 −3.95% 掉到 −6.41%，净效果几乎不变（−24.5% vs −24.2%）⟹
**结论对容量混淆是稳健的**。

**决策**：`REJECTED`，维持 78,960。
⚠️ 关于②类晋级限制：这里**不需要**公榜裁决 —— 空动作就是「保持现状」，
②类的量反风险只在**采纳**改动时才需要公榜背书，**拒绝**改动不需要。
不为一个本地 −24.5%、0/5 折的方向花公榜名额（8/23 就停更）。

**⭐ 但阶梯是单调的，反方向没测**：78,960 是当前 fold 版图允许的**最大**滑动窗，
而滑动窗在后面几折**丢掉了本来可用的历史**。改成**扩展窗**（用到 embargo 之前的全部数据）：

| fold | 滑动窗（现在） | 扩展窗 | 多出 |
|---:|---:|---:|---:|
| 0 | 78,960 | 78,960 | +0 |
| 1 | 78,960 | 98,699 | +19,739 |
| 2 | 78,960 | 118,438 | +39,478 |
| 3 | 78,960 | 138,177 | +59,217 |
| 4 | 78,960 | 157,916 | +78,956 |
| 合计 | 394,800 | 592,190 | **+50%** |

fold 0 完全不变、验证段完全不动 ⟹ **与现基准天然配对**，是严格的「只加数据」臂。
一次运行（约 50 分钟）即可。数据量轴既然单调，这一侧值得测。

证据：`outputs/experiments/v3_recency_ladder_3s480.{json,md}`、`v3_recency_w*.{json,md}`。
⚠️ `w60000_frozen` 臂在运行中被中止、未产出；但 40,000 档的 scaled/frozen 两臂结论一致，
且 60,000 与 40,000 同号同向，缺这一臂不影响结论。

### 2026-08-17 — `RESULT`：slow/fast 公榜 +2.93%（新最好），但迁移率 0.51× —— 项目首次本地高估

| 提交 | 公榜 | Δ vs 0.0039977510 | 相对 |
|---|---:|---:|---:|
| slow/fast 纯 CSV 变换 | **0.0041150085** | +1.173e-04 | **+2.93%** |
| asset adapter 候选 | 0.0039908352 | −6.92e-06 | −0.17% |

**slow/fast**：08-13 以来第一次上涨，而且**没重训、没改模型** —— 只对生产 CSV 做后处理。
输出触限 0 行（max 0.420450）⟹ 与 0.0039977510 可直接比大小，不依赖任何近似。

**⚠️ 迁移率 0.51×，本地高估 2 倍。** 本地 OOF +5.77%（6/6 预注册门槛、3/4 折、bootstrap CI
排除 0、第二份 cache 复现 +5.87%、全分辨率三窗合并 +5.93%）→ 公榜只有 +2.93%。
**这是项目第一次出现本地高估** —— 此前③类迁移率是 1.20× / 1.6× / 2.3×，全是本地低估。
两个候选解释：(a) 系数在训练期 OOF 上拟合，测试期 regime 不同；
(b) 测试期 slow 只占预测方差 10.0%，而 OOF 采样格上约 17%（全分辨率下滚动均值噪声更小，
本来就该更小）⟹ 可修正的量本身更少。
⟹ **迁移率区间要改写**：不再是「本地系统性低估」，而是「①类后处理可能高估、③类结构可能低估」。

**asset adapter**：Δ=−6.9e-06，按 NEXT_STEPS §3 的预注册判据「|Δ| < 1e-5 视为不可辨别」⟹
不可辨别。同一条判据规定「不能根据一次公榜分数重新搜索 scale 或 asset 系数」⟹
**asset scale 轴关闭**。本地 +1.99%/3-of-4 没有迁移。

**⚠️ 交付状态**：slow/fast **不在任何模型产物里**。当前若直接打包私榜，拿到的是
0.0039977510 那一档，不是 0.0041150085。要带上它必须在 `main.py` 实现逐 asset 的
自身预测滚动均值（新的跨 `predict` 状态 ⟹ 模型身份变更 + promotion 全套门禁 + 在线一致性测试）。

**未花的一张牌**：`Score` 沿 `c(t) = (1−t)·(1.16,1.16) + t·(0.4496,1.2530)` 是 t 的**精确二次式**，
现有 t=0 / t=1 两个公榜点。但 `Score(c)=2cᵀV − κ·cᵀSc` 有三个未知量（V_s, V_f, κ），
两点定不下来 —— **再花一个公榜名额取第三点即可解出顶点**。已核实 t=1.5 / t=2 都不触限
（max|pred| 0.4283 / 0.4459 < 0.5），t=2 杠杆最长、条件数最好。

证据：`experiments/ledger.csv` 最后两行。

### 2026-08-17 — `INCIDENT`（未造成损失）：`variant_submission.py` 两处与 `--model-dir` 不配套

用 `--model-dir` 交 asset adapter 候选时被挡住，查出两个问题，都属 08-13 那类
「候选 meta 与实际跑的不一致」的**同族**风险：

1. **守卫过期**：`if not overrides: raise SystemExit("至少要覆盖一个参数…")`。
   这条写在 `--model-dir` 加进来之前 —— 当时「变体」只能靠覆盖 meta 产生。
   但换一套模型产物、用它**自己的** meta 跑，本身就是合法变体。
   改为 `if not overrides and args.model_dir is None`。
   ⚠️ 绕过它的土办法（`--scale 1.16`）**不能用**：那会把 scale 真写进入包 meta，
   候选若本来是别的值就被静默改写 —— 正是 08-13 事故的形状。
2. **`check()` 读错 meta**：限幅体检用的是 `strategies/v3_hybrid/model/hybrid_meta.json`
   的 `prediction_clip`，即**生产**模型的值；但 `--model-dir` 下入包的是另一个模型。
   两者今天都是 0.5 所以没出事，改为读**入包那份**的 staged meta。

**加固**：stage 之后打印**实际入包**的模型身份（`blend_weight` / `num_iteration` /
`prediction_scale` / `prediction_clip` / `market_lambda` / 模型文件数 / `asset_cross_scales` 个数）。
干跑核实 adapter 候选入包为：15 个资产 scale、scale 1.16、clip 0.5、blend 1.0、480 轮、λ=0.5、3+3 个模型文件。

**教训**：工具加了新入口（`--model-dir`）之后，要把**所有**读 meta 的地方都跟着切到入包那份，
不能只改产模型的那一处。

### 2026-08-17 — `HYPOTHESIS`（预注册，未启动）：recency / 训练窗阶梯

**问题**：未来是不是更像**最近**的训练窗，而不是完整历史窗？

**为什么现在做它**：market 侧六条路全关、temporal 族全族关闭、①类只剩 slow/fast 待公榜裁决 ——
训练窗是少数几条**从没跑过**的轴（NEXT_STEPS P2 列了但一直没执行）。

**预注册设计（先写死，跑之前不改）**：

- 三档窗口：`{40,000 / 60,000 / 78,960}` 个**采样** time_id（78,960 = 当前生产值，作基准臂）。
- 其余逐项固定：480 轮 × 3 种子、λ=0.5、`blend_weight=1.0`、scale 1.16、history40/window5、
  `sample_modulo=5` / `phase_balanced`、embargo 6、5 折 rolling。
- **只扫窗口，不同时扫权重函数**（NEXT_STEPS §5 明写）。同时搜两个会让归因失效；
  recency weighting 若要测，必须在窗口结论定下来之后单独一轮。
- ⚠️ 窗口变短 ⟹ 每折训练行数变化 ⟹ `min_data_in_leaf` 按 `MIN_DATA_FRAC × 行数` 自动跟着变，
  这是**混淆项**。预注册两个子臂：(a) `min_data_in_leaf` 随行数缩放（默认行为）；
  (b) 冻结为 78,960 窗下的绝对值。两臂都报，不允许只报好看的那个。

**判据**：同一批 rolling fold、配对 peak 增量、折均 >0、≥4/5 折同号、去最好折 >0、
相对 ≥ +1%、`2ΔA > ΔB`。

**⚠️ 晋级限制**：这是第②类「拟合紧密度」轴（数据量/近因性）。本项目在该轴上已三次本地↔公榜
量反（alpha、轮数 160/320/960、history 宽度 c80）⟹ **本地结果不单独晋级**，
必须由公榜或回补标签裁决（CLAUDE.md §8.1）。本地若为负也不能直接结案，要记检出下限。

**停止条件**：三档都在噪声地板内（用 offset 对照估）⟹ 记「测不出来」并关闭，不扩格。

### 2026-08-17 — `REJECTED`：market 同口径复测 —— 直接回归在每个 α 上都输

**问题**：`market_model` 用的是旧口径（10 折/modulo 10/训练窗 39,480）和旧基准
（岭回归**单独**产的 m̂）。换到今天的口径、对今天的**两分量** m̂，直接回归 `m_t` 还输吗？

**设计**：5 折 / modulo 5 `phase_balanced` / 训练窗 78,960 / embargo 6；**无权**截面均值
（`mt_aggregates.npz` 的 `m` 与 `xbar` 都是加权的，不可用，另建 `xbar_unweighted_m5pb.npz`）；
fold 划分**直接读 OOF cache**，保证与基准逐折对齐；指标与四条判据逐字沿用 `market_model.py`。
**预注册预期：仍然输。**

**结果**：基准（今天的两分量 m̂）逐折 peak_m 均值 **0.00114815**。候选：

| α | 1e4 | 1e5 | 1e6 | 1e7 | 1e8 | 1e9 |
|---|---:|---:|---:|---:|---:|---:|
| 相对 Δ | −74.37% | −55.33% | −31.26% | **−22.55%** | −38.71% | −71.97% |
| 正折 | 0/5 | 0/5 | 0/5 | 1/5 | 1/5 | 0/5 |

**解释**：比旧口径的 +2.0%/6-of-10 差得多，因为今天的基准（ridge + 行级 LGBM 两分量）
比当年的 ridge-only 强得多。⟹ 预注册预期兑现，**market 侧最后一个窄问题也关闭**。

**决策**：`REJECTED`。重新开放条件：市场块换成结构上不同的模型族，或回补数据后原规格复验。
证据：`outputs/experiments/market_direct_recheck_3s480.{json,md}`。

### 2026-08-17 — `SUPERSEDED`：我说「下一个方向大概率是 market」，那个判断是错的

**我当时的推理**：①测到市场块每提升一点 R² 值 **2.46 倍**的分（`w_m/w_e`），
而市场块的收割率只有截面块的 **1/2.62**（R²_m 0.00104 vs R²_e 0.00273）；
又看到 `mt_predictability` 里「直接回归 m_t 能拿 0.0026」比今天的 0.00104 高，
于是把它当成一条待查的疑点，说「如果那个 +22% 存在，大概率在市场块」。

**错在两处**：

1. **把价值论证当成了可得性论证。** 2.46× 说的是「**如果**你拿到一点市场 R² 改进，
   它值 2.46 倍的分」，它完全没说这个改进**拿得到**。这两件事必须分开说。
2. **没先查仓库。** `market_model.md` 早就把那条「疑点」拆掉了：`mt_predictability` 的
   「13% 余量」是**口径假象** —— (a) 设计矩阵用**加权**特征截面均值，而推理端拿不到
   `weight`，那个数根本不可交付；(b) 基准在 `prediction_scale=0.5` 下算 R²（预测被砍半、
   R² 被压低），而直接回归是最小二乘、自带最优缩放。改成无权 + 尺度无关 `peak` 后，
   `direct_u` 只有 +2.0%、6/10 折、符号 p=0.754、去最好折 −6e-06，**四条判据全否**，
   整条 α 阶梯只有 1e6 一格为正（典型的单格假象）。

**真实情况：market 是被查得最彻底的块，不是最开放的块。**

| 市场侧尝试 | 结果 | 证据 |
|---|---|---|
| 直接回归 `m_t`（无权、尺度无关） | 四条判据全否 | `market_model.md` |
| LightGBM 打 `m_t`（6 容量臂、无泄漏、3 种子） | 对两个基准**全部 INCONCLUSIVE** | `lgbm_mt_v2.md` |
| 滞后截面均值增量 | ΔR²_m = +0.00005 ⟹ Δ总分 ≈ +0.000037 | `mt_lagged.md` |
| 市场块容量 | 公榜 +0.77%，轴已耗尽 | `experiments/ledger.csv` |
| 市场森林独立轮数 | 160/240/320/400/480 **全 FAIL**，`Selected: None` | `v3_market_round_scan_phasebal_prodwindow.md` |
| 解开 `market_lambda` | OOS −1.62%、2/4 折 | 今天 |

**收割率低还有另一种读法**（我上次权重给低了）：`m_t` 是市场收益，
**它可能本来就接近内在可预测上限**。六次独立进攻全部失败，支持这个读法而不是「还有富矿」。

**决策**：不把 market 当下一条主线。用户点名的同口径复测仍然跑（`market_direct_recheck`），
因为 `market_model` 用的是旧口径（10 折/modulo 10/训练窗 39,480）和旧基准（岭回归单独产的 m̂），
今天的基准是两分量 blend —— 那个窄问题确实还没答过。

**教训**：给方向性建议之前先把仓库里同题的实验查完；「价值高」和「有余量」是两个命题。

### 2026-08-17 — `RESULT`：慢分量与快分量需要差 2.8 倍的 scale（当日最大发现）

**问题**：②测到「预测比信号平滑得多」（信号 `m_t` 在真实 lag 5 归零，预测 `m̂` 在 lag 50 仍有
0.324）。target 在 5 步后没有记忆 ⟹ 预测里持续远超 6 步的成分能不能降权？

**动机与机制**：逐 asset 做**因果** trailing mean 把每块拆成 slow/fast，四个分量张成一个线性
空间，于是「全局单 scale / 两块各自 scale / slow-fast / 块×slow-fast」是同一个 Gram 的四个
投影，闭式可解且严格嵌套 —— 增量可归因，不会把 M1 的收益记到 M2 头上。

**实验设计与固定项**：基线一律是**同一批训练折上解出的全局单 scale**（不是 1.16、不是 pooled
最优）；扩展窗口，fold k 的系数只用 fold 0..k−1 拟合。6 条预注册门槛（折均>0、≥3/4 折、
去最好折>0、相对≥1%、K 连续段全为正、block bootstrap CI 下界>0）。
⚠️ 诚实声明：K∈{10,25,50,100,200} 在写脚本前已被探索过，K 阶梯不是干净的预注册；
抗选择保护是 K 连续段门槛 + 第二份 cache 复现 + 全分辨率口径核对。

**结果**：

| 模型 | 系数 | OOS（K=400） | 正折 | 判定 |
|---|---:|---:|---:|---|
| `M1` 两块各自 scale | 2 | −0.23% | 2/4 | ❌ 6/6 门槛全否 |
| `M2` slow/fast | 2 | **+5.77%** | 3/4 | ✅ **6/6 全过** |
| `M3` 块×slow/fast | 4 | +5.81% | 3/4 | 主 cache 过但 CI 下界仅 +5.6e-08；复现 cache 否 |

- pooled 系数：基线单 scale 0.7296，而 `slow`=**0.2828**、`fast`=**0.7881**。
- 机制：ΔA +8.07% / ΔB +11.21%，`2ΔA > ΔB` 通过 —— 不是「砍掉死方差」，而是
  **两个分量需要差 2.8 倍的 scale，单一 scale 同时高估 slow、低估 fast**。
- 逐折（K=400）：f1 +8.48%、f2 +7.51%、**f3 −2.44%**、f4 +8.68%。唯一为负的 f3 正是
  NOTES 早已记过的哑弹时段。
- 不是静态每资产偏置：把 slow 换成因果 expanding mean（=静态偏置）只剩 +1.41%，K 有内部最优。
- 复现 cache（1 种子×160 轮）：M2 同样 PASS、同一个 K=400、+5.87%。

**全分辨率口径核对**：OOF 的 trailing mean 走采样格（每 ~5 个真实 time_id 一个点），
生产走全分辨率。取 fold 2/3/4 验证段末尾各 20,000 个连续真实 time_id 重训重测，
**系数从主实验冻结取来、窗口内不做任何拟合**（= 部署时的真实处境），合并三窗后：

| K（采样步） | 全分辨率 | 采样格 |
|---:|---:|---:|
| 100 | +9.91% | +10.51% |
| 200 | +9.07% | +7.94% |
| **400** | **+5.93%** | +4.64% |
| 800 | +3.47% | +2.83% |

两种分辨率**逐 K 几乎重合**，且全分辨率在选中 K 上的 +5.93% 与主实验 OOS +5.77% 吻合
⟹ **采样格口径没有骗人**。合并后 CI 仍跨 0（90 万行只有主实验约 12% 的数据量），
所以效应大小的主证据仍是 5 折 OOF，口径核对只回答「换口径会不会翻向」。

**解释与限制**：`M2` 是稳的那个，`M3` 多两个参数并不更好。asset adapter 之上再测，
点估计更大（+7.49%）但 CI 含 0，且该臂 fold 0 无法因果适配、拟合集是混合的。

**决策**：**停在实验报告，不建候选、不改 `main.py`、不改 meta。** 上线需要在推理端新增
逐 asset 的自身预测滚动均值（跨 `predict` 状态，属模型身份变更）+ promotion 全套门禁，
而离 8/31 只剩 14 天、ROADMAP P0 是交付闭环。是否推进由用户决定。
证据：`outputs/experiments/v3_slow_variance_3s480.{json,md}`、
`v3_slow_variance_1s160_replication.{json,md}`、`v3_fullres_slow_probe_fold{2,3,4}.{json,md}`、
`v3_fullres_slow_probe_summary.{json,md}`；预测缓存
`outputs/cache/v3_fullres_slow_probe_fold{2,3,4}_predictions.npz`。

**后续问题**：K 的内部最优（采样步 400 ≈ 真实步 2000）机制是什么？为什么比 target 的
记忆长度（~6 步）大三个数量级还有效？

### 2026-08-17 — `REJECTED`：z-score，temporal 族至此全部关闭

**问题**：`(current − rolling_mean) / rolling_std` 是用户特征清单里唯一没被 V4-T 覆盖的一项 ——
`baseline` 有未归一化的 `deviation5`，`t2_state` 有裸 `std5/std20`，但两者的**比值**没测过。

**设计**：新增 `zscore5` 原子（`strategies/v3_hybrid/temporal.py`，在线与离线**共用一份实现**
`zscore_from`，`std5` 下限 1e-3；已被 `test_lag_cache_reconstruction_matches_online` 覆盖），
单臂 `t5_zscore` = baseline + zscore5，其余与 08-12 那次筛选逐项相同
（1 seed × 160 rounds、5 折、modulo 10/phase_balanced），门禁也相同。

**结果**：`+0.70%`、3/5 折、去最好折 **−3.14e-06**、`2ΔA>ΔB` ✅ → **不过门槛**。

**解释**：机制信号是干净的（2ΔA>ΔB 通过），但幅度低于 +1% 且去最好折翻负，与 `t2_state`
的 +0.18% 同量级。⟹ **temporal 族（lag2/lag5、EWM、std、slope、gap、z-score）至此全部关闭**，
`stop temporal expansion` 的结论覆盖整族。

**重新开放条件**：换到完全不同的模型族（例如非树模型），或回补数据显著改变波动结构。
证据：`outputs/experiments/temporal_zscore_screen.md`。

### 2026-08-17 — `RESULT`：slow/fast 可以只靠改 CSV 来验证，不需要重训

**发现**：`data/test/*.parquet` 带 `row_id`/`time_id`/`asset_id`，且 `row_id` 与提交 CSV
逐行对齐（已实测）。又因为 trailing mean 是**线性**的，
`slow(m̂) + slow(ê) = trailing_mean(m̂ + ê)` ⟹ M2 的 slow 就是**整份原始预测**的滚动均值。

于是 slow/fast 是一次**纯 CSV 变换**：

```text
raw = pred / 1.16                     （当前 CSV 触限 0 行，max|pred|=0.404663 ⟹ 可精确反解）
slow = 逐 asset、按真实 time_id 步长 K=2000 的因果滚动均值
new  = clip(scale_slow·slow + scale_fast·(raw − slow), ±0.5)
```

⚠️ **不能搬 OOF 的绝对系数**：OOF 全局最优 scale 是 0.7296，公榜标定是 1.16（差 59%，
本项目已知的本地/公榜尺子分歧）。只搬相对模式：

```text
scale_slow = 1.16 × 0.2828/0.7296 = 0.4496
scale_fast = 1.16 × 0.7881/0.7296 = 1.2530
```

最坏情况 `max|raw| × 1.2530 = 0.4371 < 0.5` ⟹ **仍然 0 行触限**，二次式精确成立、
与现有公榜分可直接比大小、不依赖任何近似。

**意义**：不重训、不改模型、不改 meta，一次公榜提交即可裁决 slow/fast 是否迁移。
公榜每天 5 次（`docs/competition_description.md:193`）、8/23 停更 ⟹ 名额不紧张。
CSV 由用户执行；AI 不生成公榜 CSV。

### 2026-08-17 — `REJECTED`：解开 `market_lambda=0.5`

`hybrid_meta.json` 里 λ=0.5 标着「先验、不拟合」。解开它（三系数
`c_r·m̂_ridge + c_l·m̂_lgbm + c_e·ê`，扩展窗口 OOS）得 **−1.62%、2/4 折、去最好折 −6.17%**，
且拟合出的 ridge 系数翻负、逐折不稳（−0.033 / −0.314 / −0.277 / +0.019）。
与 08-10「分量配比值 +0.02%」一致。**λ=0.5 不要动。**
重新开放条件：市场块结构再次改变（例如市场森林独立轮数落地）后重测。

### 2026-08-17 — `RESULT`：market 块的收割率只有 cross 块的 1/2.6，但一点 R² 值 2.46 倍的分

**问题**：market 块与 cross 块各自还剩多少？精力该往哪块投？

**动机与机制**：oracle 替换（把 `m̂` 换成真实 `m`、把 `ê` 换成真实 `e`）得到的两个 Score
本身只是方差拆分的复述，必然分别 ≈ `w_m` 与 `w_e`。可比较的量是同一框架下的
`Score ≈ w_m·R²_m + w_e·R²_e`：`w` 是兑换率，`R²` 是收割率。

**实验设计与固定项**：只读 `v3_production_oof_confirm_3s480_phasebal_prodwindow.npz`
（其 `xs_spec`/`market_spec`/轮数/种子数逐项等于生产 `hybrid_meta.json`）。
15 个逐 time_id 二阶矩预聚合，逐折与 block bootstrap（块长 500，1000 次）复用同一函数。
断言：能量拆分必须复原 1，`peak(y,y)` 必须为 1，矩法结果必须与 `src/metric.py` 对拍。

**结果**：

| 量 | 值 |
|---|---:|
| `m + ê`（market 完美） | 0.71226871 |
| `m̂ + e`（cross 完美） | 0.29108476 |
| 兑换率 `w_m/w_e` | **2.460×** |
| `R²_market` | 0.00104244 |
| `R²_cross` | 0.00272704 |
| 收割率之比 cross/market | **2.62×**（CI [1.62, 4.82]，5/5 折同向） |
| 当前占分 market : cross | **41.2% : 58.8%**（CI [27.5%, 53.1%]） |

**解释与限制**：cross 块的收割效率是 market 块的 2.6 倍，但 market 块每提升一点 R²
值 2.46 倍的分 —— 两个方向的乘积接近 1，OOF 上**没有**给出明确的偏向。
`⟨m̂,ê⟩_w/D = 1.151e-4` 实测不为 0，旧笔记的 `⟨m̂,ê⟩ ≡ 0` 只是近似。
这是 OOF 尺子，占分比不等于公榜上的占分比。

**决策**：不据此重新分配研究方向；记录为分块诊断基线。
证据：`outputs/experiments/v3_block_ceiling_3s480.{json,md}`。

**后续问题**：market 占分在逐折上从 28.7% 单调升到 57.1%，是否与 regime 漂移有关。

### 2026-08-17 — `REJECTED`：时间平滑 —— 预测比信号平滑得多，方向相反

**问题**：预测里的噪声是否不如信号平滑？若是，时间平滑就有免费收益。

**动机与机制**：`f_t = ŷ_t + ρ·ŷ_{t−Δ}` 的最优 ρ 有精确闭式解
`ρ* = (ψ̃ − r)/(1 − ψ̃·r)`，`ψ̃ = A_lag/A`（信号侧残留）、`r = R/B`（预测自相关）。
`r` 明显低于信号自相关 ⟹ ρ*>0 有收益；接近 ⟹ ρ*≈0。

**实验设计与固定项**：两个必须先绕开的坑 ——
（a）`phase_balanced` + modulo 5 的采样网格最小真实间隔是 **4**，缓存**测不了 lag 1**；
把缓存上的相邻差当 ac1 去比 0.836 必然假阳性（`mt_lagged.py` 记过同型陷阱）。
所以信号侧另用**窄列全分辨率**扫描（只读 time_id/asset_id/weight/target 四列）。
（b）`gain(ρ*)` 是被最大化出来的量，**恒 ≥0**；判据只认扩展窗口的样本外配对增量。

**结果**：

- 信号侧全分辨率 ac1：无权 `m_t` = **0.8373**（加权口径 0.8358，复核了 `mt_diagnostics`
  的 0.836）；逐 asset cross `e` = **0.7875**（本项目首次测量）。两条都在真实 lag 5 归零。
- 同一真实 lag 4 上：信号 `m_t` 已衰减到 0.183，而预测 `m̂` 的自相关仍有 **0.794**。
- 最小可测 lag 4 的**样本外**相对增益：market −16.37%、cross −0.51%、raw −6.14%。
  19 个 lag 的 OOS 值围绕 0 散布、无一致符号；唯一出现大幅正值的是成对行最少
  （146k，仅 1/10）的 lag 19/31。

**解释与限制**：方向与假设相反 —— 预测不是「噪声不如信号平滑」，而是**整体过度平滑**
（真实 lag 50 时信号 ac ≈ −0.03，预测仍有 0.28）。lag→0 时 ψ̃ 与 r 同时 →1、楔子只会更小，
且预测自相关在成对行 ≥50 万的 lag 上单调不增 ⟹ 最小可测 lag 的增益是 lag-1 的上界。
未做全分辨率实测，结论依赖这条单调性。

**决策**：`REJECTED`，不升级全分辨率确认。**重新开放条件**：模型结构发生会降低预测平滑度的
变化（例如去掉 history40 一类持久特征），或回补标签后信号自相关形状改变。
证据：`outputs/experiments/v3_temporal_smoothing_3s480.{json,md}`。

### 2026-08-17 — `REJECTED`：分 phase 的 scale 与混合比，测不出真异质性

**问题**：`phase = time_id % 10` 线上确定已知，最优 scale 该不该分 phase？

**动机与机制**：纯①类后处理参数，Score 未触 clip 时是精确二次式，闭式可解、零公榜配额。
与被否掉的 asset×regime / asset×magnitude 有本质区别：那些条件在估计量上，phase 是一口钟。

**实验设计与固定项**：预注册在先。主臂分 phase scale、次臂分 phase 两系数混合；
收缩网格 κ ∈ {0, 0.5, 1, 2, 5, 10, ∞}；**基线是同一批训练折上解出的全局解**
（不是 1.16、不是 pooled 最优 —— `conditional_blend` 的教训）；扩展窗口评估，
LOFO 只作探针。6 条门槛：折均 >0、≥3/4 折为正、去最好折 >0、相对 ≥1%、
κ∈{1,2,5} 整段为正、block bootstrap 95% CI 下界 >0。κ→∞ 退化为全局解由断言检查。

**结果**：两臂均 `FAIL`，6 条门槛主臂过 1 条、次臂过 0 条。

- pooled `a_p` 跨度 0.537~0.992（84.8%），异质性几乎全在 `A_p`（82.6%）而非 `B_p`（3.3%）。
- 但**逐折 `a_p` 的两两 Spearman 秩相关只有 +0.097**（复现 cache 上 +0.063）。
- 方差分量：观测到的 `A_p` 跨 phase 方差 2.19e-9，bootstrap 抽样方差 1.74e-9，
  比值 1.26×，超额方差 95% CI 含 0 ⟹ **测不出真异质性**（不是「没有差别」）。
- OOS 随 κ 单调改善到 κ→∞（全局常数）：κ=0 时 −5.32%，最佳有限 κ=10 只有 +0.06%。

**解释与限制**：预注册时估的 `A_p` 相对噪声 ~26%（每 phase 约 9,870 个 time_id）
足以解释全部观测离散度。换第二份 cache（1 种子 ×160 轮）重跑，`a_p` 的 Spearman 高达
**0.988** —— 但两份 cache 用的是**同一批行、同一套 fold**，这只能排除模型专属假象，
**不能**证明可外推；能回答外推的是逐折 Spearman 与超额方差，两者都是否定的。

**决策**：`REJECTED`。不改 `main.py`、不改 `hybrid_meta.json`、不建候选目录。
**重新开放条件**：回补标签或扩展数据把每 phase 的有效样本量提高到能压下 `A_p` 的抽样噪声，
或出现独立于本 OOF 的 phase 证据。若将来通过，可部署形式是对已标定全局值做乘性修正
`scale_p = 1.16 × (a_p / a_global)`，只搬相对模式 —— OOF 全局最优 scale 是 0.7296，
与公榜标定的 1.16 差 59%，绝对值不可搬。
证据：`outputs/experiments/v3_phase_scale_3s480.{json,md}`、
`v3_phase_scale_1s160_replication.{json,md}`。

### 2026-08-13 — `RESULT`：第二市场分量与 weighted XS 进入生产架构

**问题**：岭回归市场分量是否还能由非线性模型补充？截面块是否应按比赛权重训练？

**设计**：固定 Ridge、history40、480 轮和 scale 1.16；预注册比较行级市场模型、截面训练权重和
组合臂。市场模型使用 `[raw | xs_dev | history | asset_id]` 预测 `y`，最终只取逐 time_id 无权均值；
截面模型预测零均值残差。

**结果**：`mkt_we` 本地 5 折 +18.30%，公榜从 0.0032523499 提升到 **0.0039673997
（+21.99%）**。训练可复现、双后端和端到端门禁通过。

**解释**：新增市场模型和 weighted XS 基本可叠加；市场模型本身不应带权，因为训练行级 `y` 的
加权目标与最终无权截面均值并不对齐。

**决策**：结构通过。证据：`outputs/experiments/lgbm_market_row.md`、
`lgbm_weight_select.md`、`combo_market_weight.md` 和 ledger。

### 2026-08-13 — `RESULT`：市场容量收缩小胜，普通调参轴基本耗尽

**问题**：两片森林继承同一套旧 SPEC 是否造成明显容量错配？

**设计**：只重训被调的一片森林，另一片逐字节复用；比较 market moderate/shrunk、XS shrunk 和
960 轮，全部在同 scale 1.16 下公榜裁决。

**结果**：

| 变化 | 公榜变化 |
|---|---:|
| market moderate | +0.49% |
| market shrunk | **+0.77%** |
| XS shrunk | −9.84% |
| 960 轮 | −5.20% |

**解释**：市场模型最终只使用截面均值，过多容量会浪费在被投影掉的截面噪声上；截面模型则需要
更大容量。方向符合预注册机制，但最大收益低于 2% 关注门槛。

**决策**：`mkt_shrunk` 转正；市场容量、截面容量和统一轮数轴结案。唯一未利用的不对称是市场森林
独立轮数，列入 backlog，不自动执行。

### 2026-08-13 — `INCIDENT`：候选 meta 与实际公榜 override 曾不一致

**现象**：历史公榜 CSV 使用 `blend_weight=1.0`、`scale=1.16` 的临时 override，而候选目录仍保存
训练占位值；旧 promotion 没有校验 blend weight，可能转正成另一模型。

**修复**：promotion 和 packaging 增加完整结构门禁、非退化烟测、双后端逐元素对拍、staging 复用
校验和原子转正。

**长期规则**：模型身份包括所有 meta 参数和文件 hash；“预测有限”不能证明“模型正确”。详见
[`research_history/delivery-and-incidents.md`](research_history/delivery-and-incidents.md)。

### 2026-08-12 — `REJECTED`：Responder 可预测，但不补 target 残差

47 个 responder 聚为 24 族；其中 8 族可由 feature 样本外预测，但把其预测加入 target Ridge 后
5 折全部为负，折均 −20.64%。因此停止多任务 NN；只有扩展数据显著改变前置 B/C 门禁时才能重开。

证据：`outputs/experiments/responder_analysis.md`、`responder_predictability.md`、
`responder_residual_increment.md`。

### 2026-08-12 — `RESULT/REJECTED`：V4 结构筛选

- 多尺度资产历史三臂全部否决。
- 压缩 market regime +1.34%、4/5 折，机制干净但未过 +3% 门槛；只保留扩展数据原规格复验资格。
- target-only MLP 单模太弱，50/50 集成 −54.49%，关闭参数搜索。

证据：`outputs/experiments/temporal_multiscale_screen.md`、`target_mlp_screen.md`。

### 2026-08-11 — `RESULT`：history40 成为最大一次结构跃迁

每资产 previous/difference/rolling mean/rolling deviation 接入 LGBM 截面块，公榜达到
0.0032523499。换到 Ridge 上同类历史则为负，说明“一个特征机制有效”不代表它对所有模型族有效。

完整演变见 [`research_history/features-and-signals.md`](research_history/features-and-signals.md)。

## 6. 稳定事实与未解问题

### 稳定事实

- 数据包含 15 个资产、323 个匿名特征；完整 schema 以 manifest 和主办方文档为准。
- 推理端拿不到 `weight`，所以市场/截面分解和最终投影必须使用无权截面均值。
- 市场共同分量约占 target 方差 68% 左右，是主要信号来源，但分区长期漂移方向不稳定。
- LightGBM 文本模型与 NumPy 解析器已在真实数据和合成边界用例上对拍。
- 当前生产目录已于 2026-08-13 转正，不再是旧的 160 轮无 history 模型。

### 未解问题

- **（仍开着）** 为什么多个拟合紧密度参数在本地与公榜量反；需要真实回补标签按时期分解。
  ⟹ 回答动作已就位：RUNBOOK D0.3，`experiments/public_replay.py` + 盘上 21 份历史公榜 CSV。
- **（仍开着）** 公榜期与私榜期的 regime 差异是否会削弱市场模型；本地 fold 3 曾接近哑弹。
  ⟹ 8/23 之后没有外部尺子，只能靠 D0.3 校准后的本地尺子给逐折权重。
- ~~当前生产模型在最终目标硬件上的 LightGBM/NumPy wall-clock 和超时余量。~~
  `RESOLVED`（2026-08-18，4 核实测）：5.26 / 10.94 分钟，0 超时 / 0 非有限值 / 0 触 clip；
  兜底是单核绑定 ⟹ 4 核评测机不会比 32 核开发机更慢。
  证据：`outputs/experiments/delivery_runtime_{lightgbm,numpy_fallback}_4t.{json,md}`。
- ~~市场森林独立截短能否在不损失市场 alpha 的前提下降低过拟合和耗时。~~
  `CLOSED_FAIL`：160~480 全部不过门槛，480 是这一族里最好的但只有 3/5 折、去最好折 +0.81%。
  证据：`outputs/experiments/v3_market_round_scan_phasebal_prodwindow.md`。

## 7. 历史入口

- 验证框架、A/B、scale、公榜校准：
  [`research_history/validation-and-calibration.md`](research_history/validation-and-calibration.md)
- Ridge → v3 → history → 双森林生产模型：
  [`research_history/model-evolution.md`](research_history/model-evolution.md)
- 特征、history、responder、temporal、MLP：
  [`research_history/features-and-signals.md`](research_history/features-and-signals.md)
- 推理优化、promotion、打包和事故：
  [`research_history/delivery-and-incidents.md`](research_history/delivery-and-incidents.md)
- 重构前逐字原文：
  [`research_history/source_snapshots/`](research_history/source_snapshots/)
