# NOTES.md — 会过期的工作笔记

（规则类内容在 CLAUDE.md；优先级在 ROADMAP.md。本文件随时改写，不保证常新。）

## 仓库结构（2026-07-26 重构后）

```
data/                    # 主办方数据，20G，只读，gitignore
docs/                    # 主办方赛题/数据说明，只读
examples/                # 主办方示例，只读
timeseries_api/          # 主办方本地 runner，只读（vendoring：入库以便 8/23 更新时 git diff）

strategies/lightgbm_baseline/   # ★ 主办方增补包（约 8/06 发布），只读原文，勿改
                                #   要重构就新开目录，原件留着 8/23 重发时 diff

src/                     # 离线专用公共库（提交包不含）
  metric.py              #   加权零均值 R² —— 全项目唯一实现
  io.py                  #   FEATURE_COLUMNS / train_files / load_time_sample
  validation.py          #   partition_folds / rolling_time_folds（含 offset）/ rolling_fold_chunk_size

strategies/v1_ridge/     # 自包含：整个目录 = 提交包内容
  features.py            #   ★ 预处理+推理唯一实现，train.py 和 main.py 都 import 它
  train.py               #   训练（可 import src/；main.py 不可以）
  main.py                #   交付推理件，只依赖 numpy + 同目录 features.py
  model/baseline_model.json

experiments/             # 离线实验脚本 + 台账
  walk_forward.py        #   训练窗口对比（分区级 3 折，绝对分口径，已被下面那个取代）
  walk_forward_rolling.py#   ★ P0 主力：time_id 滚动多折 + 配对 A/B + 噪声地板
  mt_diagnostics.py      #   m_t 共同分量占比 + 自相关衰减形状
  mt_predictability.py   #   ★ 分数拆解（择时 vs 截面）+ 直接回归 m_t 测可预测性
  phase_diagnostic.py    #   ★ 拆开训练相位与评估相位，量 sample_modulo 的混叠效应
  walk_forward_history.py#   因果历史特征验证（依赖先跑 walk_forward.py）
  history_features.py    #   因果 lag/rolling 特征构造（未接入主模型）
  ledger.csv             #   ★ 提交台账：本地分 vs 公榜分校准

scripts/
  check_consistency.py   #   ★ 断言 train 与 main 预测逐元素一致（改口径必跑）
  make_submission.py     #   只读打包 main.py+features.py+model/ → zip

outputs/                 # 生成产物，csv/zip gitignore；experiments/ 小结入库
```

## 数据事实（从 parquet/manifest 核实）

| 字段 | 训练 | 测试 | 说明 |
|---|---|---|---|
| row_id, time_id, asset_id | ✅ | ✅ | 索引；15 个 asset；分区顺序 = 时间顺序 |
| feature_000..322（323 个） | ✅ | ✅ | 唯一可用输入，匿名 |
| weight | ✅ | ❌ | 主办方给定，仅训练/验证用 |
| responder_00..46（47 个） | ✅ | ❌ | 未来构造，绝不可当输入（=泄露），只能当辅助目标 |
| target | ✅ | ❌ | 预测目标 |

- 训练 1322 万行 / 9 分区；测试 322 万行 / 3 分区。float32 + zstd。
- E_w[y²] = 1.1757，sd(y) = 1.0784。
- 推理端约束：4 核 / 12GB / 无 GPU / 无网络；超时该 time_id 置 0。

## 常用命令

```bash
# 训练（生成 strategies/v1_ridge/model/baseline_model.json）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python strategies/v1_ridge/train.py

# 训练/推理口径一致性（改预处理后必跑）
.venv/bin/python scripts/check_consistency.py

# 时序验证
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward_history.py

# 滚动配对 A/B（P0-2）—— 结论一律看每折的 Δ，不看绝对分
#   配置名在 experiments/walk_forward_rolling.py 的 CONFIGS 里，第一个是 baseline 臂
WF="OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward_rolling.py"
$WF                                                              # 单配置，复现 walk_forward_rolling.{json,md}
$WF --configs baseline,baseline --disable-fit-cache --label selfcheck   # 自检：Δ 必须恒为 0
$WF --configs baseline,feat323 --label feat323                   # 一次 A/B
$WF --configs baseline,feat323 --label feat323_offhalf --fold-offset half  # 噪声地板（P0-3）
#   噪声地板 = |mean(Δ)_off0 − mean(Δ)_offhalf|；效应没明显超过它就等于没测出来
#   同名报告已存在会直接报错退出，要覆盖得显式 --force
#   scale 是后处理旋钮 → 同一 (feature_count, alpha) 的多个 scale 臂共用一次拟合，很便宜

# 过度收缩联合网格（18 臂只要 6 次拟合，配置名由 CONFIGS 里的循环生成）
ARMS=$(.venv/bin/python -c "import sys;sys.path[:0]=['experiments','strategies/v1_ridge','.'];\
import walk_forward_rolling as w;print('baseline,'+','.join(n for n in w.CONFIGS if n.startswith('g')))")
$WF --configs "$ARMS" --label shrinkgrid

# 分数拆解 + m_t 可预测性（P2-2）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/mt_predictability.py

# 相位诊断：训练相位 vs 评估相位拆开报分
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/phase_diagnostic.py

# 本地全量顺序推理（≈21.7 万次 predict 调用，约 2 分钟）→ 公榜交这个 CSV
.venv/bin/python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir strategies/v1_ridge --output outputs/submission.csv

# 私榜提交包（zip，不是公榜用的）
.venv/bin/python scripts/make_submission.py --strategy v1_ridge
```

## 当前模型参数（v1, 2026-08-08 更新）

```
算法              加权 Ridge（lsqr），400 设计列 = 200 原始 + 200 截面去均值
train_partitions  4        ← walk_forward 推荐 4，已统一
sample_modulo     5        ← 训练抽样；先别改
prediction_clip   0.5      ← 实际从未触发（预测最大 0.354，要 scale>1.41 才会碰到）
intercept         ~-0.004  ← 置 0 在单折和滚动配对两套框架下都测不出差异
NaN 预处理        nanquantile ← 修复了 NaN→0 污染统计量的 bug
```

|  | 旧版（公榜 0.00151886） | **现在的默认值（2026-08-08 落地）** |
|---|---|---|
| feature_count | 200 | 200 |
| ridge_alpha | 2e6 | **5e5** |
| prediction_scale | 0.6424 | **0.8** |
| 滚动 10 折 @modulo 10（仅相位 0） | 0.00152539 | **0.00175314**（+14.9%） |
| 滚动 10 折 @modulo 5（相位 0+5） | 0.00083224 | **0.00102767**（+23.5%） |

依据 `ab_shrinkgrid` / `ab_shrinkgrid_m5`：两个采样口径都单调指向 alpha=5e5（各 8/10 折同号）；
最优 scale 本地给 0.759（modulo 10）/ 0.820（modulo 5），公榜给更高，取中间的 0.8，
精确值留给公榜两点定抛物线。限幅体检：raw 幅度 0.367，clip(0.5) 要到 scale≈1.362 才生效。

⚠️「323 特征略优」那条旧结论**已被联合网格否定**，见下面「过度收缩」一节。

⚠️ **`train.py` 打印的 `validation_score` 不能用来选配置**：它训练用
`validation_sample_modulo=10`（只有相位 0）、评估用 `sample_modulo=5`（相位 0+5），
正是「相位」一节里测出有害的那种训练/评估错配，而生产模型没有这个错配。
新配置在这个数上是 0.00067794（旧版 0.00089825），方向与两把滚动尺子相反 —— **以滚动配对 Δ 为准**。

## P1 拆分测试结果（2026-08-07 单折口径，⚠️ 已被下一节取代，保留作对照）

| 配置 | validation_score | vs 基线 | 当时的结论 |
|---|---|---|---|
| 基线（200特征, scale=0.5, 真截距） | 0.00089825 | — | — |
| 只改截距=0 | 0.00089825 | ±0 | 无影响 |
| 只改 323 特征 | 0.00089307 | -5.2e-6 | 拖后腿 |
| 只改 auto-scale | 0.00094467 | +4.6e-5 | 有效，保留 |

## 滚动配对 A/B 结果（2026-08-07/08，P0-2/P0-3 框架，10 折）

产物：`outputs/experiments/ab_*.{json,md}` —— selfcheck、scale_auto(+offhalf)、p1_stable(+offhalf)、
featsweep、alphasweep、scalesweep、meandev、f323_scale、shrinkgrid。
每个 md 自带完整配置与每折数字，结论一律以产物为准。复跑命令见上面「常用命令」。
baseline = window4 + nanquantile + 200 特征 + alpha 2e6 + scale 0.5 + 真截距。

自检（`ab_selfcheck`）：同配置两臂关缓存，Δ 逐位为 0；且各次运行的 baseline 臂都
逐位复现 `walk_forward_rolling.json`（max|Δ| = 0.000e+00）。

**尺子本身的分辨率。** 噪声地板 = 同一个 A/B 把 fold 边界平移半段后 mean(Δ) 的漂移量。
关键点：**地板不是框架的常数，它取决于你在测什么改动 —— 两臂共享的计算越多，Δ 越安静。**

| 两臂共享什么 | SE(Δ) | 噪声地板 |
|---|---:|---:|
| 一切（同配置自检） | 0（严格逐位相等） | — |
| 同一次拟合，只换后处理旋钮（intercept） | 1.24e-05 | **9.9e-06** |
| 只共享 fold 切分，各自拟合（feature_count） | 3.54e-05 | **2.16e-05** |
| 还多一个自己在抖的派生参数（a*） | 5.11e-04 | **2.2e-04** |

对照：绝对分的 SE = 2.11e-04，且均值在两套边界间漂移 1.44e-04 —— 绝对值确实没法比这个量级。
ROADMAP 想要的 6e-6 仍未达到（最好 ~1e-5）；各自拟合那类实验按 2.16e-5 的地板算。

**各改动的结论**（baseline = 200 特征 / alpha 2e6 / scale 0.5；地板按共享程度取）：

| 改动 | 证据文件 | mean(Δ) | 同号 | vs 地板 | 判定 |
|---|---|---:|---:|---:|---|
| scale 0.75 | `ab_scalesweep` | +2.73e-04 | 8/10 | +27× | ✅ 有效 |
| scale 1.00 | `ab_scalesweep` | +2.81e-04 | 8/10 | +28× | ✅ 有效 |
| scale 1.50 | `ab_scalesweep` | −4.94e-04 | 6/10 | −49× | ❌ 过头了 |
| alpha 5e5 | `ab_alphasweep` | +2.16e-04 | 8/10 | +9.8× | ✅ 有效 |
| alpha 4e6 | `ab_alphasweep` | −1.29e-04 | 8/10 | −5.9× | ❌ 更差 |
| feat323 @alpha 2e6 | `ab_featsweep` | +5.9e-05 | 7/10 | +2.7× | ⚠️ 假象，见下 |
| feat100 @alpha 2e6 | `ab_featsweep` | −1.74e-04 | 8/10 | −7.9× | ❌ |
| meandev r=10 | `ab_meandev` | −3.93e-04 | **10/10** | −17.9× | ❌ **P2-1 被否决** |
| intercept=0 | `ab_p1_stable` | +1.81e-05 | 5/10 | +1.8× | ⚪ 测不出 |
| auto_inner scale | `ab_scale_auto` | −7.6e-05 | 6/10 | −0.35× | ⚪ 测不出 |

### 过度收缩：三个旋钮修的是同一个毛病

alpha 太大、预选太狠、scale 太小 —— 这三个「发现」都是**模型被压得太扁**的不同表现，
**收益不可加，必须放在同一个网格里比**。联合网格 `ab_shrinkgrid`（18 臂 6 次拟合）：

因为分数关于 scale 精确二次（残差 ~1e-13），每个配置的最优 scale 是**解出来**的不是扫出来的：

| 特征 | alpha | 最优 scale | 该处分数 |
|---:|---:|---:|---:|
| **200** | **500,000** | **0.759** | **0.00175839** |
| 200 | 250,000 | 0.696 | 0.00175515 |
| 323 | 500,000 | 0.697 | 0.00173015 |
| 323 | 2,000,000 | 0.849 | 0.00168135 |
| 200 | 2,000,000 | 0.883 | 0.00164787 |

**教训：`feat323` 的 +5.9e-5 是 alpha=2e6 过度正则的假象。** alpha 调到 5e5 后，
323 特征反而略差于 200。单独扫某一个旋钮会得出错误结论。

### 内层估 a\* 为什么失败

每折估出来 0.34~1.78（sd=0.41），`corr(a*, Δ) = −0.50` —— 估得越大结果越差，这是估计噪声的
特征而非真实的时变最优。fold 5 估出 a\*=1.78，那折分数从 +0.00043 掉到 **−0.00387**。

但要分清两件事：**「每折重估 scale」这个流程没用 ≠「scale 该取多少」这个问题不重要**。
我当初把两者混为一谈，得出了「auto-scale 没有证据支持」的错误框架性判断。
固定 scale 扫描（共享拟合、地板 1e-5）证明 scale 影响极大。

**为什么单折那张表被取代**：单折框架连 5.1e-5 都分辨不了，表里的 −5.2e-6 / ±0 都在它自己的
分辨率以下，从来就没有测量支持；而 +4.6e-5 是「a\* 在打分的同一批数据上估」的样本内产物
（a\* 按定义就是让那批数据分数最大的值，所以那个差必然 ≥ 0，是上界不是估计）。

**限制**：本框架用 sample_modulo=10 + 4/9 窗口，生产用 modulo=5 + 4 分区。
n=10 的符号检验功效很低（10/10 才 p=0.002，7/10 只有 p=0.34），别只看 p 值。

## 本地 ↔ 公榜校准（2026-08-08）

同一个模型（200 特征 / alpha 2e6），只改 scale，两次公榜提交：

| scale | 本地滚动 10 折 | 公榜实测 | 公榜/本地 |
|---:|---:|---:|---:|
| 0.5000 | 0.00133767 | 0.00128602 | 0.961 |
| 0.6424 | 0.00152539 | 0.00151886 | **0.996** |

### ⚠️ 这个吻合很可能是巧合，别当地基（2026-08-08 降级）

我当初据此写下「本地绝对分就近似等于公榜分，后面可以只在本地定夺」——**这条撤回**。

理由：`phase_diagnostic` 显示，同一个模型在 modulo 10（只有相位 0）上得 0.00133767，
在 modulo 5（相位 0+5）上只有 0.00071851。公榜是**全相位**的，按理该更接近后者，
实际却贴着前者。更可能的解释是**公榜测试期本身更好预测**：公榜从 time_id 888,480 起，
紧邻本地最后一折（839,105–888,475），而那折恰是十折里最好的（modulo 5 下 0.00111，
十折均值只有 0.00072）。可预测性在往后变高，本地十折均值被前面那些难的时期拖低了。

**仍然有效的**：配对 Δ（那本来就是这个框架的用途）。
**不再有效的**：拿本地绝对分当公榜分的代理。

### 抛物线解法本身是对的

`Score(a) = 2aA − a²B`，不触限幅时精确成立（实测残差 ~1e-13）。
旧模型两点定出：`A = 0.00165067, B = 0.00145858` → 公榜顶点 **a\* = 1.1317，峰值 0.00186804**。

⚠️ **私榜交付用本地最优，不用公榜最优** —— 本地是 10 个时期的平均，
公榜是单一测试集的精确值，后者等于对公榜过拟合。

## 公榜两点法：分数关于 scale 是精确二次式（2026-08-08 已验证）

`Score(a) = 2aA − a²B`，其中 `A = Σw·y·raw/Σw·y²`、`B = Σw·raw²/Σw·y²` 是测试集上的常数。
不触限幅时精确成立 —— **同一模型交两个不同 scale，就能解出它的公榜最优 scale 与峰值**。
峰值 `= A²/B = IC²`，与 scale 无关，是**唯一能公平比较不同模型的量**。

**外推验证**：用 scale 0.5 与 0.6424 两点拟合，外推到区间外的 1.13，
预测 0.00186804 / 实测 **0.00186805**，差 9.3e-09。三点最小二乘残差 ~6e-10。

| 模型 | 公榜最优 scale | 峰值 | IC |
|---|---:|---:|---:|
| 200 特征 / alpha=2e6 | 1.1317 | **0.00186805** | 0.04322 |
| 200 特征 / alpha=5e5 | 0.8139 | 0.00150896 | 0.03885 |

限幅检查：测试集 raw 幅度 0.354（alpha 2e6）/ 0.425（alpha 5e5），
clip(0.5) 分别自 scale 1.411 / 1.178 起生效，超过就不再是二次式。

## ⚠️ 本地与公榜在 alpha 上量反了（未解决）

上表说 alpha 2e6 的峰值比 5e5 高 **23.8%**。而三把本地尺子全说反话：

| 尺子 | 2e6 / 5e5 峰值比 | 产物 |
|---|---:|---|
| 公榜 | **1.238** | `experiments/ledger.csv` |
| 相位隔离（训练相位 0+5，验证相位 1+6） | 0.935 | `ab_phase_gen_alpha` |
| modulo 10（只评训练相位） | 0.870 | `ab_shrinkgrid` |

### 两个假设都试过，都不成立

**假设 1：本地只在模型训练过的相位上评估。** ❌
造了 `--holdout-phase`，训练相位 0+5、验证相位 1+6（模型完全没见过），
方向没变（`ab_phase_gen_alpha`，2e6/5e5 = 0.935）。

**假设 2：本地外推距离太短。** ❌
公榜要往训练期后外推 214,538 个 time_id，而本地每折验证段只有 49,345 个。
把验证段拉长 3.3 倍重测：

| 验证段跨度 | 占公榜 | 最优 alpha | 2e6/5e5 峰值比 | 产物 |
|---:|---:|---:|---:|---|
| 49,345 | 23% | 500,000 | 0.908 | `ab_shrinkgrid_m5` |
| 164,490 | 77% | 500,000 | 0.914 | `ab_horizon_f3` |
| **246,740** | **115%** | 500,000 | 0.929 | `ab_horizon_f2` |
| 214,538（公榜） | 100% | ≥ 2e6 | **1.238** | `ledger.csv` |

**决定性的是第三行**：验证段拉到比公榜还长 15%，本地依然说小 alpha 赢。
所以差异与「预测多远」无关。本地三点的斜率是每增 1 万 time_id 动 +0.0010，
要够到 1.238 还需再加 296 万个 time_id —— 现有全部训练数据的 3.3 倍，不可能。

**剩下的候选**：公榜测试期本身的分布/regime 与全部训练期数据不同，
而更强的收缩对分布漂移更稳健。**这个在本地测不了** —— 没有那段时期的标签。

### 什么时候能解决：2026-08-23

主办方文档：「**公榜截止 & 标签回补 8月23日**：公榜停止更新，
发布扩展训练数据供最终模型训练与本地验证使用」。
拿到公榜测试期的标签后，就能在本地直接复现公榜评分，这个矛盾会被一次性诊断清楚。

### 在那之前的工作规矩

**凡是影响「拟合紧密度」的参数 —— alpha、特征数、模型复杂度、树深、轮数、学习率 ——
本地结论一律不可信，改用公榜两点法比峰值。**

预算够：公榜到 8/23 还有约 75 次（5/天），每个模型 2 次拿到精确峰值 → 约 35 组配置。

**不受影响、仍可在本地定夺的**：纯后处理旋钮（scale、clip）、
结构性诊断（分数拆解、相位效应、m_t 可预测性）。

⚠️ 分析脚本里判「趋势上移」用 `<=` 会把「没动」判成「上移」，已在结论里纠正。
自动判据要用严格不等号。

## 相位：`time_id % 10`（`phase_diagnostic`，2026-08-08）

`sample_modulo` 不是中性抽样。modulo 10 只取 `time_id % 10 == 0`（相位 0），
modulo 5 取相位 0 和 5。把训练相位和评估相位拆开：

| 场景 | 训练相位 | 评估相位 | 分数 |
|---|---|---|---:|
| A | 0 | 0 | 0.00133767 |
| C | 0+5 | 只看相位 0 | **0.00097096** |
| B | 0+5 | 0+5 | 0.00071851 |

**两个效应都成立**：

1. **相位 5 的行难预测 2.1 倍**（同一模型：相位 0 得 0.00097096，相位 5 只有 0.00045755）
2. **训练里混进相位 5，模型在相位 0 上也掉 27%**（A→C）。多给一倍数据反而变差，
   说明不同相位的「特征→target」关系不一样，混在一起拟合会互相稀释

旁证：p008 上按相位统计，`mean_w(y)` 是个平滑的周期-10 循环
（相位 0 = −0.028 → 相位 4 = +0.005 → 相位 9 = −0.030），而 `E_w[y²]` 只差 ±2%。
`time_id` 大概率编码了某种周期为 10 的结构。

**由此暴露的口径错配**：测试集是连续 time_id（**全部 10 个相位**），
而生产模型 `sample_modulo=5` **只在 2 个相位上训练**。这是 ROADMAP 的新主线项，尚未处理。

## P2-2：分数是从哪来的（`mt_predictability`）

`Score = share_m·R²_m + share_e·R²_e`（m = 逐 time_id 加权截面均值，e = y − m）。
恒等式残差 **5.89e-16**，说明拆解是精确的。

| | 均值 | 占总分 |
|---|---:|---:|
| share_market（择时块占方差） | 0.721 | — |
| R²_择时 | 0.001565 | — |
| R²_截面 | 0.000713 | — |
| **择时贡献** | **0.001138** | **85%** |
| **截面贡献** | **0.000200** | 15% |

三条结论：

1. **那 72% 的共同分量不但吃得到，它就是主菜** —— 85% 的分来自择时。
   `ab_meandev` 从反面独立佐证：压制择时分量，10/10 折掉分。
2. **这个可预测性是结构性的**：单变量 IC 在训练段与验证段之间的相关性 **+0.720**。
3. **但线性择时已近饱和**：专门回归 m_t 的样本外 R²=+0.00177（最优 alpha=1e6，8/10 折为正），
   比模型现在隐含的 0.00157 只高 13%，换算总分 0.001338 → 0.001473。
   要继续吃这块，只剩「加滞后输入」和「上非线性」两条路。
4. **被浪费的是截面块**：占 28% 方差却只出 15% 的分，R²_截面 只有 R²_择时 的一半。

## 工程坑（都踩过）

1. **train.py 在不同 BLAS 线程数下训出不同模型。** 带
   `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` 可逐位复现；不带则 coef 相对差 4.75e-04，
   预测差到 1.6e-04。根因：`Ridge(solver="lsqr", tol=1e-4)` 是迭代解法且停得太早，
   线程数一变求和顺序就变。**直接威胁 10 月答辩「复现 0.00119」那条验收标准，待修（收紧 tol）。**
2. **公榜交 CSV（`row_id,target`），私榜才交 zip。** 公榜每天最多 5 次成功评分，私榜共 10 次。
   `scripts/make_submission.py` 是私榜用的，公榜要跑 runner 出 CSV。
3. **提交 CSV 的精度**：主办方示例是 8 位小数、64.1 MB；直接 `to_csv` 写 float64 是 97.6 MB。
   float32 根本没那么多有效位，多出来的全是垃圾位。
4. **配对 A/B 的 fit 缓存键必须由 `FIT_KEYS` 生成。** 曾硬编码成 `(feature_count, ridge_alpha)`，
   导致 meandev 四个臂全部命中 baseline 的缓存、Δ 恒为 0。
   **Δ 严格等于 0 是这类 bug 的显式信号**，看绝对分就发现不了。

## 时间线

- 8/23 公榜截止 + 标签回补（主办方重发包 → 对 docs/examples/timeseries_api 跑 git diff）
- 8/31 策略文件提交截止（私榜策略共 10 次）
- 9 月 实盘评估；10 月 答辩（交付物最好逐行自己重写，验收标准：复现 0.00119）
