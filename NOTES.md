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

## 2. 当前研究上下文（2026-08-13）

### 生产模型

当前生产为 `strategies/v3_hybrid/model/`，来源候选 `v3_hybrid_mkt_shrunk`，公榜
**0.0039977510**。核心结构：

```text
ridge market component
  blended with λ=0.5
row-level unweighted LGBM market component
+
weighted LGBM cross-sectional component (replace, blend_weight=1.0)
+
asset history40 features
→ prediction_scale 1.16 → clip 0.5
```

精确配置以 `strategies/v3_hybrid/model/hybrid_meta.json` 为准；不要从本节复制参数生成模型。

### 当前问题

1. **交付风险**：当前模型的精确全量 wall-clock 尚需在最终环境复测；NumPy 双森林兜底约 15 分钟。
2. **数据更新**：8/23 若收到回补数据，必须先审计 train split 是否改变，再决定重训。
3. **本地尺子**：alpha、轮数和 history 宽度曾与公榜量反；回补标签优先用于重建评估可信度。
4. **剩余研究轴**：市场森林独立截短有机制依据且无需重训，但未排期，也不得在无裁判时替换生产。
5. **扩展数据复验**：V4-R regime 是唯一保留的原规格结构复验；其他 V4 和 responder 路线关闭。

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
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src strategies scripts tests timeseries_api experiments examples
```

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

**这个格子为什么空着**（核对 `NEXT_STEPS_horizon_auxiliary_oof_validation.md` 时发现）：
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

- 为什么多个拟合紧密度参数在本地与公榜量反；需要真实回补标签按时期分解。
- 公榜期与私榜期的 regime 差异是否会削弱市场模型；本地 fold 3 曾接近哑弹。
- 当前生产模型在最终目标硬件上的 LightGBM/NumPy wall-clock 和超时余量。
- 市场森林独立截短能否在不损失市场 alpha 的前提下降低过拟合和耗时。

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
