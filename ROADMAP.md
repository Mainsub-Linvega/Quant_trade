# ROADMAP.md — 当前状态与行动面板

> **状态日期：2026-08-31（私榜策略文件已提交，赛程收官）。** 本文件只描述当前有效状态和未来动作。完整探索过程见
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

### 🛑 8/29 封板作废 —— 两份存量交付件都会撞穿官方总超时

**2026-08-29 实测（云端真机，`experiments/thread_default_probe.py`）**：
`main.py` 从未显式设置 `num_threads`，lightgbm 默认 `-1` = 用**全部可见核**。
评测容器 `os.cpu_count() = affinity = 128`，而 CPU 配额只有 4 核 ⟹ 起 128 线程挤 4 核：

| | 每次 `booster.predict`（15 行 × 3 森林 × 480 轮）|
|---|---:|
| 默认（不传 `num_threads`）| **99.249 ms** |
| 显式 `num_threads=4` | **1.355 ms** |
| **比值** | **73.27×** |

**⭐ 同日第二次云端跑把推算换成了端到端实测**（探针新增三臂，量整条 `Model.predict`）：

| 臂 | ms / 次 |
|---|---:|
| 出厂设置（`num_threads=4`）| 4.743 |
| **抹掉 `num_threads`（改动前的提交包）** | **353.330** |
| numpy 兜底 | 11.035 |

加性开销 **+348.59 ms/次**（端到端 74.50×；微基准只量截面森林，端到端还含市场森林
——两片都是 3×480 ——所以 353 ≈ 121×2 + 取列/岭回归，对得上）。
折到云端全量实测锚点（lightgbm `mean_predict` 3.393 ms、total 821.0 s）：
**351.98 ms/次 = 20.98 h vs 官方总预算下限 2.98 h ⟹ 704%**
⟹ 约第 **30,476** 次调用（**14.2% 处**）撞 `total_timeout`，
`timeseries_api/runner.py:186` 置 `aborted_after_timeout`，
**其后 85.8% 的 `time_id` 全部保持默认值 0**。

⚠️ 本条订正了同日早些时候的两个数（**6.04 h / 49.4% 处**）：那是只按截面森林的微基准
推算的，漏了市场森林那一半。方向没变，量级更糟。

✅ **numpy 兜底不受影响**（这是此前唯一没量过的一格）：不钉线程时兜底/主路径 = **2.33×**，
比钉 4 线程时的历史读数 2.78× 还小 ⟹ 纯 numpy 树遍历确实不吃 BLAS 线程。
⟹ 万一评测端没有 lightgbm，兜底那条路**本来就是安全的**。

### ✅ 8/29 云端真机实测：超时风险关闭

`delivery_cloud_20260829_unpinned`（**主件 zip，128 核真机，`--threads 0` 不钉线程，
`thread_env` 四个全 `None`**）—— 这是第一次在「什么环境变量都不给」的条件下量提交包自己：

| 项 | 实测 | 限额 | 占比 |
|---|---:|---:|---:|
| `model_init` | 4.66 s | 180 s | **2.59%** |
| `total_seconds` | **910.2 s** | 10,726.9 s | **8.49%** |
| `mean_predict` | **3.81 ms** | 50 ms | **7.62%** |

对照没修版本的推算（352 ms/次 → 20.98 h → 14% 处撞闸丢 86%）⟹ **余量 11.8 倍**。
3,217,458 行 / 0 非有限 / 0 触 clip / 0 超时 / `aborted=False`；峰值 RSS **11.05 GB**
（比本地 11.35 还低，与 8/23「真机比开发机宽裕」一致）。16 条只红两条：
`zip_audit_passed`（**假警报** —— `.gitignore:106` 排除 `outputs/cloud/`，云端没有
`delivery_cloud_py311_4t.json` 这份评测环境真值 ⟹ 报「版本无法核」；同一份 zip 本地审计
13/13 全过）与存量的 `peak_rss_has_headroom`。

⚠️ **我预告的判据写错了一条，记下来**：跑前我说「`predictions_sha256` 必须等于
`524e14e0…`」。**逐位相同只对同机比较成立。** 实测云端 `21698601…`，`max|pred|`
相对差 **1.35e-08** —— 正落在 8/23 已量过的「同后端换机器 2.3e-8」那条轴上（§浮点差异分解），
不是新现象。⟹ **「线程数不改预测」是靠本地那次干净 A/B 证明的**（同机、同模型文件、
唯一变量是 `main.py` 改没改，两次指纹都是 `524e14e0…`）；跨机器那一跑同时变了机器、
numpy 版本和线程设置，**三个变量混在一起，本来就回答不了这个问题**。
⟹ 判据要跟着比较的**轴**走：同机比指纹，跨机比 `max|pred|` 的相对差量级。

⭐ `单步最大 2.833 s` 与 8/23 那次的 2.802 s 几乎相同 ⟹ 那个「单点离群」**不是偶然**，
是这台机器上可重复的现象（补上了第二次观测）。新规则下没有 per-step 硬闸，无害。

### ⭐⭐ 8/29 重打包后的交付件（本节取代下方 8/27 封板表）

| 角色 | 文件 | sha256 | 状态 |
|---|---|---|---|
| **主件** | `outputs/v3_hybrid_submission_20260829.zip` | `d934d246…` | 审计 **13/13**（无 `--off-*`）|
| **兜底件** | `outputs/v3_hybrid_submission_20260829FALLBACK.zip` | `5f91bca9…` | 审计 **13/13**（无 `--off-*`）；全量验证 16 条只红 `peak_rss_has_headroom`，指纹 `fe527e41…` = 公榜 0.0041833953 那组 |
| ⛔ 不可交 | `20260827` / `20260828FALLBACK` 及更早 | — | 装的是没有 `num_threads` 的 `main.py`（`ada6a2c2…`）|

⭐ **主件的差异面正好是一个文件**：与封板的 `20260827.zip` 逐条目比对，
**12 个条目逐字节相同**（7 个模型文件 + `requirements.txt` + 另外 3 个 .py），
**只有 `main.py` 变了**（`ada6a2c2…` → `68ff745e…`）。
⟹ 「只改了该改的」不是自述，是逐条 hash 得出的。

⚠️ 8/31 上传顺序：主办方**采用最新提交版本**（不是 best-of-10）⟹ **主件必须是最后上传的那一份**。

⟹ **`20260827` 主件与 `20260828FALLBACK` 兜底件都不可交**（两者的 `main.py` 逐字节相同，
sha `ada6a2c2…`）。修复已落在 `strategies/v3_hybrid/main.py`（`_PREDICT_NUM_THREADS = 4`，
探测式启用），全量对拍确认 **`predictions_sha256` 逐位不变**（`524e14e0…`）
⟹ **只改速度、不改模型身份**，证据 `outputs/experiments/delivery_src_lgbm_4t_numthreads.{json,md}`。
**待用户执行 `make_submission.py` 重打包**，之后本节下表的 sha 全部作废、需重填。

⚠️ **为什么四次全量交付验证都没抓到**：那四次的「4 线程」全部来自命令行前缀
`OMP_NUM_THREADS=4`，而**环境变量不随提交包走**。测量本身没错，错的是它的关键口径
由一个不在交付件里的外部开关决定 —— CLAUDE.md §8.11 的新形状，已补为伤疤规则 17。

### ⭐ 8/31 交哪一份（2026-08-27 封板，⚠️ 已被上一节推翻，sha 待重填）

| 角色 | 文件 | 状态 |
|---|---|---|
| **主件** | `outputs/v3_hybrid_submission_20260827.zip`（sha `d1ee32ae…`，13 文件）| 审计 **13/13 全过**，两条后端均从 zip 跑完全量推理 |
| **兜底件** | `outputs/v3_hybrid_submission_20260828FALLBACK.zip`（sha `5f3bdc58…`，5,835,403 B，13 文件）| **2026-08-27 已打并审计 13/13 全过**（`public_baseline_drift: []`，**未用任何 `--off-*` 开关**）。由备份 `model_before_20260824_150921` 现打 = 长窗 w512，公榜真值 **0.0041833953** |
| ⛔ **一律不可交** | `20260824` / `20260822` / `20260819` / `20260818` / `20260813.PRE-SLOWFAST` | **全部缺 `requirements.txt`**（8/23 新增硬要求）；`20260819` 另缺 `long_window` |

⚠️⚠️ **本表推翻了本文件下方多处旧说法**（§P0-B「8/31 只能交 20260822」、§P0「只有 20260819」、
§P-D45「三层回退最后一层是 20260822」、§P6「8/31 要交的 20260819」）。
按 CLAUDE.md §7 旧文不删，但**以本表为准**。
根因值得记：**主办方 8/23 新增的一条硬要求，让盘上所有存量交付件一次性集体失效** ——
不是我们的模型变了，是交付件的**合格定义**变了。8/31 收尾顺序见 RUNBOOK §D6 的上传日卡片。

**兜底件的三道归属检查（2026-08-27 实跑，伤疤规则 11）**：

1. **对已知真值**：8 个模型文件的逐条目 sha256 与 `20260822.zip` 同名条目**逐字节全同**
   ⟹ 装的确实是公榜 0.0041833953 那个长窗 w512 模型。
2. **口径边界**：兜底件 vs 主件的差异**只落在 `model/*`**（7 个文件；
   `baseline_model.json` 两边同为 `54dc6afb…` —— 冻结岭回归未随扩展数据重训），
   4 个 `.py` 与 `requirements.txt` 逐字节相同 ⟹ 执行代码零差异，差别只在森林权重。
3. **训练规模对得上身份**：兜底 `train_rows = 2,645,530`（扩展数据**前**）
   vs 主件 `3,289,030`（+24.3%）。

⚠️ **打包 + 审计只花了不到一分钟，这是对的、也是要看清楚的**：
`make_submission.py` 只做文件搬运 + 13 个身份键核对 + **15 行 × 1 次 predict 的烟测**；
`audit_submission_zip.py` 是**纯元数据**（开 zip、算 sha256、解析 meta/requirements），
一次模型推理都没有。⟹ **审计 13/13 ≠ 交付验证**（`CLAUDE.md §6`：烟测只证明能跑）。
真正跑满 3,217,458 行的是 `verify_delivery_runtime.py --from-zip`，见下。

**兜底件的交付验证（2026-08-27，`--from-zip`，lightgbm 主路径，4 线程）**：
`predict_total` **5.32 分钟** / wall 6.27 / 峰值 RSS **11.60 GB**、
`rows 3,217,458` / `calls 214,538` / 0 超时 / 0 非有限值 / 0 触 clip，
**13 条 check 只红 `peak_rss_has_headroom`**（同主件，是 08-23 立案的存量风险，
余量线 9.60 GB 历次实测 10.93–11.60 GB **从未达标过**；`peak_rss_under_limit` 始终为真）。
证据：`outputs/experiments/delivery_zip_fallback_lgbm_4t.{json,md}`。

⭐⭐ **两条计划外的归属检查，都比原计划那条更强**：
1. **`--manifest auto` 自己扫出了 `v3_hybrid_long512/promotion_manifest.json`**
   （8 文件逐字节相同）—— 这个匹配不是我们指定的，是脚本在 `outputs/promotions/*` 里
   扫出来的 ⟹ 独立于 sha256 人工对拍，再次确认兜底件就是长窗 w512 那版。
2. **`predictions_sha256 = fe527e41…` 与三份历史读数逐位相同** ——
   `delivery_4c12g_lightgbm`（08-23，4 核/12 GB cgroup，**源目录**）、
   `delivery_local_py313_4t`、
   `delivery_runtime_lightgbm_4t_production_control_20260824`（08-24 转正前的生产对照臂）。
   ⟹ 「zip → 解压 → 全量推理」这条链路**对预测零偏移**，且它产出的就是公榜
   **0.0041833953** 那一组预测。这与 08-27 主件那次（`524e14e0…` 对上 08-24 源目录）
   是同一形状的证据，现在两份交付件**各自独立**都有了。

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
| train/inference 一致性 | **`1.098e-08`**（两后端同值，2026-08-23 实测）⟹ 见下方注 | `scripts/check_consistency.py` |

> **注：`train/inference 一致性` 这个数怎么读。**
> 口径是 `--partition-index 8 --n-time-ids 2100`（**新默认**）。
> ⭐ 旧默认 50 太窄：每 asset 只有 50 个观测 ⟹ 长窗 512 的环形缓冲只填到 **9.8%、从未回绕**，
> slow/fast 的 2000 步窗只填到 **2.5%、左端从未移动** ⟹ 测的是「还没热起来的模型」，
> 而榜上跑的是热的那个。见 §4 P12（已归档至 `research_history/delivery-and-incidents.md`）。
>
> ⚠️ 此前那条「记 `8.111e-09`、同参数复测对不上、**未去追因**」**已追因**：
> `max\|Δ\|` 随 `--n-time-ids` **单调增长**
> （50→4.019e-09、200→7.603e-09、600→**8.117e-09**、1000→8.770e-09、1500/2100→1.098e-08、3000→1.630e-08），
> 而同参数重复跑**逐位相同**（n=50 / n=600 各验两次）
> ⟹ 那两个数的差别是**当时用了不同的窗口**（8.111e-09 对应约 600），**不是不确定性**。
>
> ⚠️ `check_consistency.py` 是 **slow/fast-aware** 的 —— 训练端没有该后处理的概念，
> 不补上会永久报红 9.4e-02。

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
- 2 种子旧候选只节省 5.45% 全量推理时间。
  ⚠️⚠️ **2026-08-24 当日两次订正**：先记「D0.3 复算 `submission_slowfast_t2.csv` = 0.0039374211，对 slowfast −4.32% ⟹ 2 种子掉 4.32%」—— **那是错的归属**。同日 B1 用**同一次训练的前 2 片森林**做干净测量：**3→2 只掉 0.30%**（peak 0.0041768 vs 0.0041895）。
  ⟹ `submission_slowfast_t2.csv` 的 −4.32% **不是种子数造成的**，它与 slowfast 的差别另有来源，**归属未知**，不得再当作「2 种子」的证据。
  ⚠️ 这次错误的形状正是 `CLAUDE.md §8.10`（归属断言必须当场核）—— 规则当天写下、当天被我自己违反：我把 ROADMAP 的标签「2 种子旧候选」直接当成了事实。

## 3. 当前有效的研究判断

1. **结构收益仍是主要来源。** history、第二市场分量和带权截面训练贡献远大于后续容量微调。
2. **市场块和截面块的容量方向相反。** 市场块收缩小幅有益，截面块收缩显著有害。
3. **480 轮已形成内部极值。** 旧结构 320 轮下降，新结构 960 轮下降；统一轮数轴结案。
4. **普通②类调参余量很小。** 2026-08-13 容量扫描只得到 +0.77%，低于 2% 关注门槛。
5. **本地尺子不能稳定判断拟合紧密度。** alpha、轮数、history 宽度都出现过本地与公榜量反。
6. **Responder 三种用法全部走完，这条轴现在由证据关闭**（2026-08-22 收口）。可预测不等于能补
target 残差；此外 Stage C 此前有 **14 / 47 列**是被 `multi_member_family` 这条**启发式**挡着、
从未测过的，现已补测，**28 格无一过门禁**。用 responder 打分选列（唯一没做过、
也唯一不在母条件排除项里的用法）在前置测量处否掉：重合 170~180/200 低于决策线，且换掉的是原排名
#16 的列 ⟹ 预注册的降方差机制**未兑现**。证据：`responder_stage_c_fill.md` /
`responder_selection_probe.md`。
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

> **2026-09-01 整理**：本节曾有 1,169 行，其中 23 个课题已结案。
> 它们的完整证据卡片已按主题**逐字**迁入 [`research_history/`](research_history/README.md)，
> 这里只留结论与去向。课题编号的含义见 [`GLOSSARY.md`](GLOSSARY.md) §1。

### 4.1 仍然开着的课题

### P-TIME 运行时间限制（2026-08-29 主办方补全）—— 已对账，余量 4.97×

主办方把 `docs/competition_description.md` 的「运行时间限制以最终发布环境为准」换成三个硬数字
（`docs:161,166-172`），并**新增一节「重要 Note」**要求策略代码内部手动设 `num_threads`。
两件事的处置不同：前者我们已达标、只是补上门禁；后者是**未关闭的归属缺口**，见下。

```text
--model-init-timeout-seconds 180
--total-timeout-seconds (0.05 + a) * n_time_id + b     # a、b 均未给值，但都 ≥ 0
```

取 `a = b = 0` 即**预算下限**，按它判永远比真实评测更严。公榜 `n_time_id = 214,538`
⟹ 总预算下限 **10,726.9 s**。

| 环境 / 后端 | `model_init`（180 s）| `total_seconds`（10,726.9 s）| `mean_predict`（50 ms）|
|---|---:|---:|---:|
| 本地 4c/12G · LightGBM | 0.36 s / 0.20% | 376.8 s / 3.5% | 1.51 ms / 3.0% |
| 本地 4c/12G · NumPy 兜底 | 0.36 s / 0.20% | 702.3 s / 6.5% | 3.05 ms / 6.1% |
| 云端真机 · LightGBM | 0.96 s / 0.54% | 821.0 s / 7.7% | 3.39 ms / 6.8% |
| **云端真机 · NumPy 兜底（最坏）** | 0.81 s / 0.45% | **2,158.6 s / 20.1%** | 9.45 ms / 18.9% |

- ⟹ **最坏 4.97× 余量**，三条闸没有一条接近。
- ⭐ **预算随 `n_time_id` 线性缩放** ⟹ 9 月实盘期变长**不改变**这个占比。此前记的
  「实盘期更长会放大兜底总耗时」就此消解（分子分母同步涨），与内存那条
  「峰值由分区大小决定、不随运行长度增长」同向。
- ⚠️ **超总闸的后果是悬崖不是线性损失**：`timeseries_api/runner.py:180-198` 一旦
  `elapsed_total` 超限就置 `aborted_after_timeout = True`，**剩余全部 `time_id` 填 0**。
  且 `elapsed_total` 从 `run_loaded_model` 起算、**含 parquet I/O**
  ⟹ 该与之比较的是 `total_seconds`，不是 `predict_total_seconds`。
- **门禁**：`verify_delivery_runtime.py` 此前把 `total_timeout_seconds` 写死 `None`
  ⟹ `not_aborted` 一直没有失败的机会（CLAUDE.md §8.11）。现新增 `--total-timeout`
  （默认按公式下限开闸）与 `model_init_under_limit` / `total_under_budget` /
  `mean_predict_under_budget` 三道判据。⭐ **可失败性当场验证**：`--total-timeout 5`
  跑一次，`not_aborted` 与 `predict_calls_expected` 双红；而 `row_count_correct`、
  `zero_non_finite`、`zero_clip_rows` **照样全绿** ⟹ 这种悬崖式失效**只有 `not_aborted`
  抓得到**。主件两条路径重跑，16 条 check 只红存量的 `peak_rss_has_headroom`，
  `predictions_sha256` 与既有读数逐位相同（`524e14e0…` / `567265de…`）⟹ 加门禁没改预测。
  证据：`outputs/experiments/delivery_zip_{lgbm,numpy}_4t_timegates.{json,md}`。

**🛑 已关闭（2026-08-29，结论是「必须重打包」）：`num_threads` 归属缺口。**
云端**端到端**实测 **74.50×**（微基准 73.27× 是当天第一个信号，只量了截面森林），
远超预注册的 1.2× 判据 ⟹ 走重打包分支。详见本文件 §1 顶部。
`num_threads` 排在 `validate_features` **之前**探测：两者代价不对等 ——
少了后者只是慢一点，少了前者会撞穿总时限。原始记录如下。

**⚠️ 原「未关闭」记录：`num_threads` 归属缺口。** 主办方要求策略代码**内部**手动设最大线程数，
而 `strategies/v3_hybrid/main.py:436` 的 `booster.predict(...)` 没有传 `num_threads`；
我们历次「4 线程」读数**全部来自外部环境变量** `OMP_NUM_THREADS=4`，**它不随提交包走**。
本机复现不出劣化（默认/`=4` 为 0.90× / 0.97×），因为 libgomp 尊重 `sched_getaffinity`，
cpuset 型限核下 `-1` 自己收敛；危险的是 cgroup **配额型**限制（不改 affinity），
而云端 `os_cpu_count = 128` 的默认行为**我们从未观测过**（那两次都设了环境变量）。
⟹ 原待办「云端跑 `experiments/thread_default_probe.py`（不设环境变量）」
**已于 2026-08-29 执行完毕**，结果就是上面那条 🛑：端到端 74.50×。本段仅存档当时的推理。
判据先写死：**默认/`num_threads=4` ≲ 1.2× ⟹ 按原样交；≫ 1.2× ⟹ 补 `num_threads` 重打包。**

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

### 4.2 已结案课题索引

| 课题 | 结论 | 归档去向 |
|---|---|---|
| **P12** | 转正门禁漏掉 `long_window`、一致性窗口过窄；同型事故的第五个现场，也是第一个「零参数可达」的。生产预测一位未变。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P11** | 评测环境资源门禁首次真机实测，四条环境 × 后端组合全部通过，结论是「别改模型」。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P0-B** | 私榜包重打并落盘审计 `passed: true`。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P0** | 私榜交付闭环打通，四个动作全部完成并落盘审计。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P1-R** | 修尺子：21 份有公布分数的历史 CSV **全部离线复现**（最大偏差 1.9e-09）⟹ 尺子可信；且公榜排名多半是 regime。 | [验证与标定](research_history/validation-and-calibration.md) |
| **P-REQ** | 主办方 8/23 新文档要求交付包内含 `requirements.txt`，而工具链从没接上；08-27 结案。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P-B1** | 种子数 3→10 `REJECTED`。「本地公榜协议」首次实跑。 | [模型演进](research_history/model-evolution.md) |
| **P-ROB** | 稳健性劈成两个轴：一个免费、一个修不了。23 份历史 CSV 全量体检。 | [验证与标定](research_history/validation-and-calibration.md) |
| **P-D45** | 最终交付件用 `extended_full` 全量重训（12 分区 / 16,445,150 行 / 含密封段）。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P-D4** | 扩展候选转正门禁机械正确性全过；**未转正** —— `--activate` 需用户明确授权。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P-V4R** | V4-R 压缩 market regime 按原规格做扩展数据复验 ⟹ `REJECTED`，轴关闭。 | [特征与信号](research_history/features-and-signals.md) |
| **P-RESP** | responder 轴的重开条件已消化，按原规格复验 ⟹ `REJECTED`，这次关严了。 | [特征与信号](research_history/features-and-signals.md) |
| **P2-E** | 扩展数据固定结构重训，密封期 6/7 道门通过。 | [模型演进](research_history/model-evolution.md) |
| **P2-D2B** | 原 D2 的预注册写法**因果不成立**（验证段全部落在新数据之前），已改设计。伤疤规则 11 的第四个现场。 | [验证与标定](research_history/validation-and-calibration.md) |
| **P1** | 8/23 回补包是**公榜窗口的标签回填**、不是新特征数据；四项差异已修。 | [交付与事故](research_history/delivery-and-incidents.md) |
| **P2** | ⚠️ `SUPERSEDED`（2026-09-01 标注）—— 原状态一直挂 `BLOCKED_BY_P1`，但 P1 已于 08-24 `CLOSED`，其内容也已由 P2-E 的 `RESULT` 实际回答。 | [模型演进](research_history/model-evolution.md) |
| **P3** | 市场森林独立轮数 `CLOSED_FAIL`；实验其实早已跑完，只是状态没同步。 | [模型演进](research_history/model-evolution.md) |
| **P4** | recency / 训练窗阶梯两侧都已结案；缩短方向是**测得出来的**负结果。 | [模型演进](research_history/model-evolution.md) |
| **P7** | slow/fast 抛物线顶点闭式解出，判定**不改交付**。 | [模型演进](research_history/model-evolution.md) |
| **P8** | 多任务辅助监督 `CLOSED_FAIL`：⭐ **机制是真的，增量不存在**。 | [特征与信号](research_history/features-and-signals.md) |
| **P9** | NN 独立能力阶梯 `REJECTED`，天花板 28.8%，曲线在 <50% 门槛处掉头。 | [特征与信号](research_history/features-and-signals.md) |
| **P10** | 密封期尺子标定完成：排序 Spearman ρ=1.0（满分），区间几乎没有分辨力 ⟹ **用序不用值**。 | [验证与标定](research_history/validation-and-calibration.md) |
| **P6** | 磁盘清理 `CLOSED`（用户已执行）：`outputs` 21G → 3.5G。 | [交付与事故](research_history/delivery-and-incidents.md) |

> 每条的完整口径、门禁读数与证据路径都在归档文件里，按课题编号搜索即可。
> 归档时的原始标题（含状态与日期）一并保留，例如：
> `P12 — 转正门禁补 `long_window` + 一致性窗口扩宽 —— `INCIDENT`（未爆）/ `RESULT`（2026-08-23）`

## 5. 已结案项目

#### 8/23 · ⭐ **截面块窄 peer 对**（`xs_peer_pair_probe` / `xs_peer_pair_confirm_3s480`）

**`PASS_BUT_BELOW_DETECTION_FLOOR`，但当前设计不可部署**（关键限制见下）。
诊断先行：`asset_grouping_diagnostic.py` 用 OOF cache 算 15 资产两两 `e` 相关，逐 time_id
零和约束把均值机械压到 −1/14≈−0.071，**真正的信号是偏离这条基线的对子**—
—`(0,6)+0.18`、`(2,14)+0.13`、`(1,13)+0.12`，层次聚类在 k=3~5 稳定成群，且"模型解释前的
e"与"解释后残差"两个相关矩阵几乎逐位相同（生产模型完全没碰这部分结构，因为 `asset_id`
categorical 分裂看不到"另一个资产这一刻在干什么"）。特征＝partner 资产上一采样 time_id 的 `e`，
只加 3 对、1 列。1s160 筛选档 `REJECTED`（pooled +2.39%、4/5 折、去最好折 +1.31%、CI 下界
−0.49%、0.39× 检出下限）；3s480 确认档翻盘：**pooled +3.29%、5/5 折单调递增（+1.71%→+5.04%）、
去最好折 +2.93%、`2ΔA(+4.46%)>ΔB(+2.18%)`、bootstrap CI 下界 +2.30%（清楚为正）**，
六道门禁过五道，只差检出下限（0.38×）——与长窗 w512 confirm
档当年同一个桶（`PASS_BUT_BELOW_DETECTION_FLOOR`，那次 0.89× 后来公榜验证涨了 +1.66%）。⚠️⚠️
**部署路径当场查出走不通，未动 `train.py`**：特征 `peer_e_lag1` 由**真实 target**
反推（`e = y − 截面均值`），而 `main.py:22` 的 `forbidden={"weight","target",...}` 在
`timeseries_api/runner.py` 交给 `predict()` 前就把 target 剥掉——推理时这个量**不存在**，
不是工程不便，是信息本身拿不到。诊断/两次探针全部用的是 OOF cache 里的**真实**
`e`/`e_lgbm`（训练期标签已知），这是只在离线分析里成立的 oracle 量，不能原样进候选，
否则训练用真值、推理被迫换成别的东西（比如自身历史预测），是 CLAUDE.md §8.4
那类"训练推理不同口径"的坑。**本轮因此未生成候选、未碰公榜、未改生产**。⟹ **重新开放条件**：
把特征换成"模型自身对 partner 的历史预测值"（沿用 `PredictionTrail` 的因果状态模式，与 slow/fast
同构）。⭐⭐ **该条件已于同日定价（`xs_peer_deployable_probe`）**：前置只读测量显示把搭档量从真实
`e_j` 换成模型自身 `ê_j` 后，驱动相关 `\|均值\|` 0.01664 → 0.00401（**存活 24.1%**）／当期口径
0.00318（19.1%），**同号数从 6/6 掉到 4/6 和 2/6**；根因是 `corr(e_j, ê_j)` 逐资产只有
**0.023~0.098**，`ê_j` 是 `e_j` 的极弱代理。缓存探针（零训练、验证段 ê 覆盖 100%、逐有向对拆 6
列给 `evaluate_arm`）四臂结果：`oracle_lag1` **+0.69%**（3/4 折、CI 下界为正，但去最好折翻负、
0.44× 检出下限 ⟹ **未过门禁**）、`deployable_lag1` **−3.21%（0/4）**、`deployable_now`
**−2.41%（1/4）**、阴性对照 `shuffled_lag1` −1.76%。⚠️ 按**预注册原文**（"oracle 过"）判
**`INCONCLUSIVE_NO_DETECTION_POWER`**：线性尺子对该机制检出力不足，
可部署臂的阴性结果**不得**升级为"没效果"。⭐ 事后旁证（不在预注册里）：相对阴性对照，oracle
**+2.45pp**、`deployable_lag1` **−1.45pp**、`deployable_now` −0.65pp ⟹ 尺子能把 oracle
与噪声列分开，而两个可部署臂落在噪声列同侧或更低。⟹ **实操结论：不推进**；树版另需在训练段生成
`ê`（训练段覆盖实测 0%/25%/50%/75%/100%，直接换列会引入伪时间信号），代价是扩展 fold 版图的
OOF（小时级），只有可部署臂过门禁才值得

**证据入口**：`asset_grouping_diagnostic.{json,md}`（2026-08-23 补落盘）、`xs_peer_pair_probe.{json,md}`、`xs_peer_pair_confirm_3s480.{json,md}`、`xs_peer_deployable_{plan,probe}.json`

#### 8/23 · **截面块市场态交互探针**（`xs_market_state_probe`）

`REJECTED`：给截面块（XS LGBM）加一列 `market_pred_t`（训练折内拟合的行级 LGBM 打 y
的截面均值），让树自己学 `asset_id × market` 交互，六道门禁只过 1 道。pooled **+0.77%**（门槛
3%）、**3/5 折**、去最好折 **−0.23%**（翻负）、bootstrap CI 下界 **−1.43%**（跨 0）、仅 6.1%
检出下限的 0.13×；`2ΔA>ΔB` 通过（ΔA +2.46%/ΔB +3.02%），说明有一丝真信号但方差撑不住。⭐ **与
8/14 `asset × observable regime`（同样 3/5 折、跨期不稳）落在同一个坑**——
本轮换了机制（训练时输入列 vs 训练后 2-bin adapter；market_pred_t vs 预测截面 RMS；树自己学交互
vs 手工分箱），结果仍然不稳，说明这不是那次实现细节的问题，
是"市场态"这条信息通道本身在当前数据上就稀薄不稳（与市场块样本外 R² 仅 0.0018 的已知事实吻合）。
⟹ **"截面块对市场态完全瞎"这条架构缺口由代码事实转为实验证据关闭**，不建候选，生产未动

**证据入口**：`xs_market_state_probe.md` / `xs_market_state_interaction_plan.json`

#### 8/22 · **P9 范围项 ③：把新选列喂给 NN**

`REJECTED`，且**这一条结案本身就是产出**：按 P9 **原规格**（`max_iter ∈ {12,50,150,400}`，
不新选点）复跑，天花板 28.8% → **27.4%**，门槛 50%，条件延长未触发。⭐ 真正的信息不在那 ±1.4pp，
而在**倒 U 形状一模一样** —— 换了 25 列输入后 50 档仍是峰值、150/400 档仍然崩溃 ⟹ **独立印证 P9
的机制结论：绑定约束是正则化不是特征集**（P9 当时是从形状*推断*，
本轮是变更另一个轴而曲线不动）。⟹ **v5 可改项由 3 条收缩到 2 条**。环境自检：另跑一次默认选列的
12 档当锚点，对 17.4287%/20.2833% 偏差 **0.000e+00**（该臂自己的 12 档按设计不该复现锚点）。⚠️
只测 fold 0、只换 cross 块、只换一种新选列

**证据入口**：`nn_capacity_ladder_respsel.md`

#### 8/22 · ⭐ **Stage C 补测（14 个空白格）**

`REJECTED`：Stage B 七道 check 里未通过的 **16 个族全部只错
`multi_member_family`**（单成员族），因证据不过 **0 个** ⟹ 此前是启发式在挡路。08-18 只补了
r00/r02，剩 14 个从未测过且与 Stage C 冻结的 8 个代表交集为空。本轮 14 臂 × 2 基准 = **28
格无一过门禁**，折均无一为正（最好 `responder_01` −1.25%/−0.94%；同期相关最高的 `responder_03`
−3.86%/−4.53%）。两道自检全过：08-18 锚点复现 **0.000e+00**（逐位）、`harness_ok=True`。⭐
方法学：**「剥掉冻结系数让步」是精确的常数平移**（36 臂恒等式偏差 5.4e−20）⟹ 只改水平不改排序，
剥完「转正」的臂里有 0/4 折的 ⟹ 不可能制造发现。⟹ **Stage C 现覆盖全部 24 族 / 47 列**。⚠️
本机制属母条件明令排除的「线性叠加」族，价值是关严不是收益

**证据入口**：`responder_stage_c_fill.md`

#### 8/22 · **responder 监督的选列判据**

`REJECTED`（前置测量即结案，**未花那次 OOF**）：梯子由 `responder_window_atlas` 自己的
`H_fit_is_equal_weight_MA` 派生（r00/02/03/04/05 + target），**不看与 target 的相关**。五折重合
**170/175/180**/200 低于预注册决策线 190；对照臂（全行 vs complete-case）199~200/200 ⟹
隔离干净。⭐⭐ 真正做决定的是 churn 诊断：**换掉的列原排名最高 #16、换进来的低到 #314**（共 323
列）、全局 Spearman 0.715~0.838 ⟹ 不是截断线附近的搅动，是全局实质分歧 ——
**而预注册写下的机制恰恰是「边缘搅动/降方差」⟹ 机制未兑现**，按规则不跑 OOF。事后敏感性：
逐级标准化后 175→182，仍低于 190 ⟹ 裁决稳健。自检：自算相关与 `select_features` 的 top-k 10
次比较全部逐位相同（**未改动那个 97 处调用点的生产函数**）

**证据入口**：`responder_selection_probe.md`

#### 8/22 · ⭐ **responder 族群表**（`RESULT`）

只读 parquet row-group 统计（不加载数据）：缺失数是**窗口指纹**、取值域是**维度指纹** ⟹ 47
列切成 **8 个维度族，7/7/7/7/3/7/5/4 = 47**，且 **8 族只用 3 条截断梯子**（422/934/2397 → 4 族；
526/1035/2490 → 2 族；4/9/24 → 2 族）⟹「同缺失数 = 同窗口，不同维度」有了直接证据。⭐ 与
`responder_analysis.py` 的 24 族聚类是**正交的两把刀**：那把按 `1−\|corr\|`
聚出来的其实是**窗口组**（cluster 13 横跨两个量纲不同的维度但窗口相同），本表是**维度组** ——
这解释了为什么现有聚类会把 a 族与 e 族的每个成员切成单成员族，进而被 `multi_member_family`
系统性筛掉。⚠️「像什么」是解读不是主办方语义

**证据入口**：`responder_family_grid.md`

#### 8/21 · ⭐ **长窗 w512 公榜裁决**

**0.0041833953（+1.662%，新最好）**，同 scale 1.16、两份 CSV 均 0 行触限 ⟹ 不依赖任何近似。
峰值口径 +1.72~1.74%（B_old 拨 ±8% 不翻），Σp²(新)/Σp²(旧)=1.0128 ⟹ **无隐藏 scale 效应**。⟹
分数 +1.66% ≈ **IC +0.83%**，与榜首 IC 差距 +20.8% → **+19.7%**，填掉约二十分之一。⚠️⚠️
**迁移率 0.22×**（确认档截面块 +7.77% → 公榜全模型 +1.72%），按占分 58.8% 打折仍只 0.37× ——
**连续第二次本地高估**（slow/fast 0.51×），此前③类全是低估（1.20×/1.6×/2.3×）

**证据入口**：ledger / `long_window_confirm.md`

#### 8/21 · ⭐ **长窗 w512 确认档（3s480）**

**`PASS_BUT_BELOW_DETECTION_FLOOR`**：pooled **+7.77%**、**5/5 折**、去最好折 +6.49%、配对 CI
下界 **+4.18%**，五道门槛全过；但只有 3s480 检出下限的 **0.89×** ⟹ **方向可信、幅度测不出**。
筛选→确认迁移率 1.14×（未衰减）；基准更强而增益仍在；线性对拍 10/10 逐位相同。⚠️
只测截面块（占分 58.8%），全模型粗估 +2%~+5%；探针的 cumsum 口径不得进生产。⟹
只够作为花一次公榜额度的理由，**不构成晋级依据**；生产未动

**证据入口**：`long_window_confirm.md` / `..._plan.json`

#### 8/21 · ⭐ **长窗列数阶梯**

**筛选档 PASS**（1s160，只截面块）：`w512` pooled **+6.80%**、**5/5 折**、去最好折 +5.84%、
bootstrap CI 下界 +3.87%、**超检出下限 1.12×**，五道全过。⭐⭐ 预注册机制预测兑现 ——
**同一评价器(线性)上 240 列 +0.69% → 80 列 +5.75%** ⟹ 信号一直在，是被 240
列的估计代价淹掉的。`w512` 是唯一两个评价器都认的臂。⚠️ 仅筛选档、迁移率历史区间 0.51×~2.3×、
只超下限 1.12×、三臂多重比较 ⟹ **必须先过 3s480 确认**，生产未动

**证据入口**：`long_window_ladder.md` / `..._plan.json`

#### 8/21 · **选列准则探针**

`REJECTED`：三臂五门槛全败。`lasso200` −1.29%（2/5）、`hist_lag1` **−4.38%（0/5）**、`hist_roll5` **−10.39%（0/5）**。⭐ 与 base 分歧越大掉得越多（单调）⟹ 不是碰巧，是**当期准则挑的那 40 列确实更好**。⭐⭐ 选列轴至此**三向封死**：更宽（323 全给）打平、history 更宽（c80 超集）公榜 0.00%、换准则全负 ⟹ 对 `feature_fraction=0.7` 的树，前置筛子的选择质量**不是绑定约束**

**证据入口**：`selection_criterion_probe.md` / `..._plan.json`

#### 8/21 · **长历史窗探针**

`REJECTED`，但**给出了定价**：窗口 {64,512,4096} 的长窗块 **ΔA=+3.36%（信号是真的）而
ΔB=+15.40%**，`2ΔA<ΔB` ⟹ 估计方差是所拿信号的 4.6 倍。pooled +0.69%、4/5 正折、去最好折 −0.49%、
bootstrap CI 下界 −5.96%。⚠️ **判据 4 无效并已收回**：逐折 `A比/√(B比) ≡ IC比` ⟹ ΔA/ΔB
混着两臂解的共同尺度（fold 0 的 A 比 0.9555，A 其实是降的），且 `2ΔA>ΔB` 是两分量判别式、
不适用于嵌套模型 —— **P8 栽过的坑本次重犯**。据它推出的「完美提取器上界 +3.36% IC」作废；
本实验**未**给出长窗信号的有效定价。裁决不受影响（判据 1/3/5 独立失败，IC/peak 尺度不变）。⚠️
fold 4 内层 alpha 选到梯底致 −19.49%，但去掉它 pooled 仍只有 +2.77%，结论稳健

**证据入口**：`long_history_probe.md` / `..._plan.json`

#### 8/21 · **函数类探针**

`REJECTED`：换函数类换不出增量。RFF-岭回归在**与生产截面块逐列相同**的 361 列上拿到
`r=0.798`（P9 的 sklearn MLP 只有 0.54 ⟹ 「NN 天花板 28.8%」是配方产物不是能力事实），但 `ρ`
同步涨到 0.702 ⟹ oracle 集成只有 **+0.91%**，门槛 +3%。线性对照 0.611、核 0.798、树 1.000
三者互相 ρ≈0.6~0.7 ⟹ **三种函数类在读同一个东西，这 361 列已榨干**。⭐
剩下的唯一方向是**换输入**（更长历史窗），不是换模型。⚠️ 预注册缺陷：alpha 梯子上界 1e-1 太低，
10 次拟合 9 次选到边界

**证据入口**：`function_class_probe.md` / `..._plan.json`

#### 8/17 · **slow/fast 分离**

OOF +5.77%（6/6 门槛、3/4 折）、复现 +5.87%、全分辨率合并 +5.93%；**未建候选**；可纯改 CSV 验证

**证据入口**：`v3_slow_variance_3s480.md` / `v3_fullres_slow_probe_summary.md`

#### 8/17 · z-score（temporal 最后一项）

`REJECTED`：+0.70%、3/5 折、去最好折为负 ⟹ **temporal 族全族关闭**

**证据入口**：`temporal_zscore_screen.md`

#### 8/18 · **P4 训练窗（收官）**

`CLOSED`：减数据明确有害（−24.5%/0-of-5），加数据测不出（+1.08%/2-of-5、CI 跨 0）⟹ **数据量已饱和**，78,960 维持

**证据入口**：`v3_recency_expanding_ladder_1s160.md`

#### 8/18 · **slow/fast 转正**

⭐ 公榜两次独立确认 0.0041150085（CSV 后处理版 / 官方 runner 版逐位同分）；未重训、仅 4 个 meta 键；耗时 5.15 分钟反比前身更快

**证据入口**：ledger / `promotion_manifest.json`

#### 8/18 · per-asset **叠加**（blend）

`REJECTED`：全部为负，per-asset 臂（−3.91%、0/4）比 `shared` 对照（−1.19%）**还差**；corr 0.57~0.63 非低相关 ⟹ 替换与叠加两种用法都关闭

**证据入口**：`asset_blend_check.md`

#### 8/18 · per-asset 完整载荷

`RESULT`→关闭：线性内异质性大（+95.8%、5/5 折、系数相关仅 +0.419），但 per-asset ridge 仍比生产 LGBM 截面块**低 50.5%** —— 树的 `asset_id` categorical 早已吃掉这块

**证据入口**：`asset_loading_diagnostic.md`

#### 8/18 · **选列宽度（树前面的线性筛子）**

`REJECTED`：323→200 的 \|corr\| 单变量筛子装在 LGBM 前面、123 列从未进过模型，且截断处落差仅 1.33% 无断崖。拆开测三个单变量臂：`xs323` **−1.00%**（2/5）、`mkt323` **+1.09%**（3/5，但只有检出下限的 0.70×、CI 跨 0）、`both323` +0.10%。⭐ 两效应精确可加（−1.00+1.09≈+0.10 实测 +0.10）⟹ 测的是真效应，只是太小。**回补数据后按原规格复验一次**

**证据入口**：`feature_screen_1s160.md`

#### 8/18 · **`*_exact` OOF cache 失效**

`INCIDENT`：该 cache（08-14 11:12）早于脚本首次提交（08-15 11:18）⟹ 出自已不存在的代码版本，与当前输出差 `max\|Δ(market_ridge)\|=3.37e-05`（折均 peak 的 2.4%）。已用当前代码现跑替代基准并落盘。⚠️ P4 扩展窗臂曾以它配对，效应同量级，重开训练窗轴前必须现跑基准重测

**证据入口**：NOTES / `v3_production_oof_1s160_prodwindow_20260818.json`

#### 8/18 · **responder_00/02 的 Stage-C 空白格**

`REJECTED`：这两个短窗口候选此前被 `multi_member_family` **启发式**（而非证据）挡在 Stage C 外。缓存探针补测（不训练、2.2 秒）：最好一格 `pure_e/responder_00` +1.38%、3/4 折、CI 下界为正，但**去最好折为负、只有检出下限的 0.43×**、机制是 ΔB −2.01% 减方差 ⟹ 不过门禁。负控制与已测族 `responder_27`（−3.7%）校准通过

**证据入口**：`horizon_auxiliary_cache_probe.md`

#### 8/18 · **重建测试补测落盘**

`RESULT`：NOTES 的 0.207/0.818/0.835/0.732/**0.883** 在全量 1,322 万行上五格全部复现（±0.006）⟹ 下一行的关闭理由现在有产物支撑。⚠️ 口径是**带截距中心化 R²**（此前未记）；换成项目指标口径（无截距、分母 Σw·y²）同一设计从 0.84 掉到 0.16

**证据入口**：`responder_reconstruction.md`

#### 8/18 · responder 窗口图谱

`RESULT`：测出窗口梯子 H=1/2/4/**5(target)**/7/10；但重建 R² 只 0.883、单步 u 不存在 ⟹ horizon 分解缺前提，不推进。⚠️ 重建那张表当时只写在 NOTES、无产物，已于同日补测确认

**证据入口**：`responder_window_atlas.md` / `responder_reconstruction.md`

#### 8/18 · P4 训练窗缩短

`REJECTED`：60k −9.50%（1/5）、40k −24.54%（0/5），机制是 ΔB 抬升而非 ΔA 丢失；维持 78,960。阶梯单调 ⟹ **扩展窗（+50% 数据）未测**

**证据入口**：`v3_recency_ladder_3s480.md`

#### 8/17 · **slow/fast 公榜裁决**

⭐ **+2.93%（0.0041150085），新最好成绩**，08-13 以来第一次上涨，且未重训未改模型。⚠️ 迁移率 **0.51×**（本地 +5.77%），**项目首次本地高估**

**证据入口**：ledger / `v3_slow_variance_3s480.md`

#### 8/17 · asset adapter 公榜裁决

`REJECTED`：Δ=−6.9e-06，按预注册 \|Δ\|<1e-5 判为**不可辨别** ⟹ asset scale 轴关闭，不再调

**证据入口**：ledger

#### 8/17 · **market 同口径复测**

`REJECTED`：直接回归 `m_t` 在**每一个** α 上都输，最好 −22.55%、1/5 折 ⟹ market 侧六条路全关

**证据入口**：`market_direct_recheck_3s480.md`

#### 8/17 · 市场森林独立轮数

`CLOSED_FAIL`：160~480 全 FAIL、`Selected: None`（实验早已完成，本次只是同步面板）

**证据入口**：`v3_market_round_scan_phasebal_prodwindow.md`

#### 8/17 · 解开 `market_lambda`

`REJECTED`：OOS −1.62%、2/4 折、去最好折 −6.17%，λ=0.5 保持不动

**证据入口**：`v3_slow_variance_3s480.md`

#### 8/17 · market/cross 分块天花板

占分翻转为 41.2%:58.8%；收割率之比 2.62× 与兑换率 2.46× 接近抵消

**证据入口**：`v3_block_ceiling_3s480.md`

#### 8/17 · 时间平滑（lag 平滑器）

`REJECTED`：预测比信号平滑得多，最小可测 lag 的 OOS 增益为负

**证据入口**：`v3_temporal_smoothing_3s480.md`

#### 8/17 · 分 phase scale / 混合比

`REJECTED`：两臂 6 门槛均未过；`A_p` 离散度测不出超过抽样噪声

**证据入口**：`v3_phase_scale_3s480.md`

#### 8/13 · 市场块容量

shrunk 赢 +0.77%，方向真实但接近饱和

**证据入口**：`experiments/ledger.csv`

#### 8/13 · 截面块容量

shrunk −9.84%，保留 loose

**证据入口**：`experiments/ledger.csv`

#### 8/13 · 轮数 960

−5.20%，480 内部极值

**证据入口**：`experiments/ledger.csv`

#### 8/13 · 市场模型 + weighted XS

公榜 +21.99%，进入生产架构

**证据入口**：`combo_market_weight.md` / ledger

#### 8/12 · Responder A/B/C

可预测但不补 target 残差，停止多任务 NN

**证据入口**：`responder_*.md`

#### 8/12 · V4 temporal / MLP

仅 regime 保留扩展数据复验资格

**证据入口**：`temporal_multiscale_screen.md` 等

#### 8/11 · 每资产 history40

公榜大幅提升，已进入生产

**证据入口**：`history_peak_lgbm_scoped.md` / ledger

#### 8/10 · phase-balanced Ridge

公榜 +1.97%，已进入生产 Ridge

**证据入口**：ledger

#### 8/8–10 · 验证框架、严格求解器、A/B 分解

已形成当前研究判定规则

**证据入口**：`research_history/validation-and-calibration.md`


完整失败路径和结论翻转见 [`research_history/`](research_history/README.md)，不要从本表反推实验细节。

## 6. 更新规则

- 当前生产模型变化时，同时更新本节、promotion manifest 引用和 ledger；不要只改文字。
- 新任务进入行动面板时必须写状态、目标、动作、验收条件和证据。
- 结案任务移入“已结案项目”，长推导进入主题历史，ROADMAP 不保留整段实验日志。
- 每次更新在下表追加一行，不回写成没有日期的“现在”。

#### 2026-08-23

⚠️ **`INCIDENT`（未爆，8/23 前拦下）：重训计划漏掉 `long_window`，并给这类漏键装了机械门禁**。
核证据时发现 `strategies/v3_hybrid/train.py:335` 的 `--long-window` **默认 0（＝关闭）**，而
`scripts/retrain_extended.py` **从未传过它**，`production_structure()` 派生的 8 个键与
`BASELINE_CHECKED_KEYS` 的 7 个键里**也都没有它** ⟹ 8/23 跑 D1 会训出一个**没有长窗**的候选，
而长窗正是 08-21 转正、公榜实测 **+1.662%** 的那块结构。转正门禁最终会拦下（`PUBLIC_BASELINE` 含
`long_window: 512`），但那是在**几小时训练之后**，而 `BASELINE_CHECKED_KEYS`
上面那行注释写的恰恰是「8/23 之前就要红，而不是训练几小时之后才红」—— 该守卫在 08-21
长窗转正后没有同步。⚠️⚠️ **这是同一类事故的第四次**（08-18 slow/fast 丢键 → 08-19
`--weighted-cross-section`/`--market-model` 漏传 → 08-21 `long_window` 漏进 `PUBLIC_BASELINE` →
本次漏进重训计划）⟹ 逐次补洞已被证明不够，新增 `tests/test_model_identity_key_coverage.py`：遍历
`PUBLIC_BASELINE` 全部 13 个键，断言四个消费者（audit / retrain / delivery / 打包）都覆盖，
覆盖不了的必须写进显式豁免表**并附理由**（沿用 `make_submission.EXCLUDED_MODULES`
那套「偏离必须是按下去的」）。⭐ **验收方式是先让门禁红**：打补丁前实测
`AssertionError: ['long_window'] != []`，补丁后转绿 —— 证明它真会抓，不是恒真断言。
顺带查明交付报告那一处 08-21 已补过（该臂本就是绿的）。dry-run 走真实 CLI 复核：v3 命令现含
`--long-window 512`，值从生产 meta 派生。⚠️ 过程中踩到一个小坑并写进 RUNBOOK：挑 v3
命令不能用「字符串含 `v3_hybrid`」，候选目录名 `v3_hybrid_extended_fixed` 里也含它。全量 **261
passed / 28 subtests**（原 254/26）。生产目录与模型身份未动，未执行任何重训。

#### 2026-08-23

⭐ **peer 对轴收口：重新开放条件已定价，判 `INCONCLUSIVE` 而非 `REJECTED`**。8/23 的
`xs_peer_pair_confirm_3s480` 在 3s480 上过五道门禁（+3.29%/5-of-5/CI 下界 +2.30%），但特征是
oracle（由真实 target 反推的 `e`，推理端被 `forbidden` 剥掉）。ROADMAP
留的"换成模型自身预测"至今没有数字，本轮补上：只读前置测量 —— 驱动相关 `\|均值\|` 0.01664 →
0.00401（**存活 24.1%**）、同号 **6/6→4/6**；根因 `corr(e_j, ê_j)` 只有 0.023~0.098。⚠️⚠️
**"换列重跑 6 分钟"这条捷径经查不成立**：`e_lgbm` 只在 `fold>=0` 行有值，训练段覆盖是
**0%/25%/50%/75%/100%**，fold 0 两臂等价、fold 1-3 覆盖率与时间强相关 ⟹ 树会学到伪时间信号。
改用缓存探针（零训练 2.7 秒、验证段 ê 覆盖 100%、逐有向对拆 6 列喂 `evaluate_arm`，
既有函数一行未改）：oracle +0.69%（未过门禁）、deployable −3.21%/−2.41%、阴性对照 −1.76%。⭐
**自查一处预注册与实现不一致**：预注册写"oracle **过门禁**"、初版代码写成"oracle **为正**"，
而实测恰好落在两者之间 —— 按严格的那条判 `INCONCLUSIVE`，代码已对齐预注册而不是反过来。⭐
顺带补上 `asset_grouping_diagnostic` 的产物缺口（此前只 print，ROADMAP 自己标着"无产物文件"），
并加一道与 ROADMAP 记录的三对相关值的对拍断言；同时订正一处弱论证：
"两个相关矩阵几乎逐位相同"是算术必然（模型只解释约 0.4% 方差），该结论真正的支撑是 `asset_id`
categorical 分裂看不到别的资产当刻值这个代码事实。全量 **254 passed / 26 subtests**（新增 9）。
生产目录与模型身份未动。

#### 2026-08-23

⭐ **P11 评测环境资源门禁 `RESULT`**：核出交付验证从来没有内存口径，而官方环境是 4 核 / **12
GB**（`docs/competition_description.md:158-159`）、超限可判提交无效、且私榜截止后无法修改。
首次在 `MemoryMax=12G` + `AllowedCPUs=0-3` 下走官方 runner 全量：峰值 RSS **11.47 GB = 上限的
95.6%**、cgroup `memory.events` 记到 `max 990`（990 次顶到上限被迫回收、`oom_kill 0`）。⭐⭐
**决定性的一步是做归属**：新写 `measure_harness_memory.py`，用零预测桩模型走同一条
`run_loaded_model`，测得 harness 单独就要 **11.09 GB**。⚠️ 但 harness 臂自己跑三次就摆动
11.09–11.47 GB，与「模型净增」同量级 ⟹ 只能说**模型贡献 ≤ 约 0.5 GB、与跑间波动不可分辨**。1
秒间隔追踪定位到峰值在 **18.0s / 36.8s = 49% 处**（分区加载段），后半程遍历 214,538 个 time_id
与最后的 `pd.concat` 一点没涨 ⟹ **峰值由分区大小决定、不随运行长度增长**，9
月更长的实盘期不会推高它。⟹ **不要为内存改模型**。NumPy 兜底同口径 **11.55 GB / 10.90 分钟**，
两条路径除 `peak_rss_has_headroom` 外门禁全过。⭐ **同日用户在真实评测机（JupyterHub）
上把兜底也跑完了**：峰值 **10.93 GB（91.1%）比本地还低 0.62 GB**、**33.78 分钟 / wall 36.28
分钟**、3,217,458 行 / 0 超时 / 0 非有限 / 0 触 clip ⟹ **lightgbm 万一不可用，兜底不会 OOM，
只会慢**。用户实测该机超 12 GB 即 OOM ⟹ 官方那条 12 GB 是硬限（`FACT`）。⚠️
我跑前按本地比例外推 25 分钟、实测 33.78，**低估 35%** —— 兜底是单核绑定而主路径吃 4 线程，
两者对「云端单核更慢」的敏感性不同（云端/本地：主路径 2.25× vs 兜底 3.10×）⟹ CLAUDE.md
§5.7「代理量不可跨结构搬用」在**跨环境**上同样成立。⭐ 单步耗时把风险排序反了：兜底单步最大仅
**0.050 s**，是主路径 **2.802 s** 的 1/56 ⟹ 单步超时风险在主路径不在兜底。⭐ 浮点差异分两轴：
同机换后端 1.4e-15、同后端换机 2.3e-8，跨机器主导但传到 Score 仍约 1e-8 可忽略。跨 predict
状态也定价了：`AssetLongWindow` 固定 2.46 MB，`PredictionTrail` 唯一无界但实测 **314.6
B/time_id**（40 万步线性），公榜期仅 64 MB、余量还能吃 8.4 倍 ⟹ 安全。顺带查明两份交付报告长期判
FAIL 是 `--manifest` 默认值写死 `slowfast` **比错了对象**（生产与 `long512` manifest 8
文件逐字节全中），改为 `auto` 解析并新增非循环的 `model_matches_public_baseline`（复用
`audit_submission_zip.public_baseline_drift`，不另抄表）。另订正 `VmHWM`
在本机内核上非严格单调（221.49→220.94 MB），`peak_rss_bytes()` 叠模块级高水位保证不低报。全量
**241 passed / 26 subtests**（新增 14）。生产目录与模型身份未动。

#### 2026-08-23

⭐ **截面块窄 peer 对：确认档 `PASS_BUT_BELOW_DETECTION_FLOOR`，但部署路径查出走不通**。
用户提出"15 个资产能不能分组估计"，诊断出 3
对残差共动明显偏离零和基线的资产（`(0,6)(2,14)(1,13)`），1s160 筛选档 `REJECTED`（+2.39%、4/5
折）但机制干净（`2ΔA>ΔB`）；升级到 3s480 确认档翻盘：pooled +3.29%、5/5 折单调递增、bootstrap CI
下界 +2.30%，六道门禁过五道，只差检出下限——与长窗 w512 当年 confirm 档同一个桶。用户提议趁 8/23
公榜停更前花一次配额验证，动手前查 `main.py`/`timeseries_api/runner.py` 的 `forbidden` 集合，
发现特征依赖的 `e`（真实 target 反推）在推理时不存在（target 被剥掉）——诊断和探针全部用的是 OOF
cache 里训练期已知标签的 oracle 量，不能直接进候选。**未生成候选、未碰公榜、未改生产**，
原样记录，重新开放条件是换成模型自身历史预测值（causal，同 slow/fast 的 PredictionTrail 模式）
后从头验证。

#### 2026-08-23

**截面块市场态交互探针 `REJECTED`**：核出代码事实——
截面块设计矩阵对市场态完全不可见（无市场块预测值/聚合量），新写
`experiments/xs_market_state_probe.py`（评价器用树，不用线性，因为假设的机制是
`asset_id×market_pred_t` 的非线性分裂）跑满 5 折 1s160 筛选档。pooled +0.77%、3/5 折、
去最好折翻负、CI 下界跨 0，六道门禁仅过 1 道 ⟹ `REJECTED`。与 8/14
`asset × observable regime`（同样 3/5 折不稳）互相印证：两套不同机制（训练时输入列 vs 训练后
adapter）在同一条信息通道上得到同一个不稳定结论，判定该架构缺口已由证据关闭。
生产目录与模型身份未改动。

#### 2026-08-13

文档体系重构；以已转正的 `mkt_shrunk` 和公榜 0.0039977510 重建当前状态。

#### 2026-08-17

三项纯 OOF 后处理诊断结案（分块天花板 / 时间平滑 / 分 phase A·B）；占分比 60.1%/39.9% 标 `SUPERSEDED`。生产目录与模型身份未改动。

#### 2026-08-18

**`slow/fast` 转正为生产基线**（公榜 0.0041150085，+2.93%）；`check_consistency.py` 改为 slow/fast-aware 以免永久报红；P4 与 per-asset 两条轴结案。

#### 2026-08-17

顺着时间平滑的否定结论找到 `slow/fast` 分离（OOF +5.77%、全分辨率核对 +5.93%）；本节第 11 条被同日测量证伪并改写。`market_lambda` 结案。仍未建候选、未改生产。

#### 2026-08-17

公榜第一更新为 0.0060（用户报告）；market 侧同口径复测 `REJECTED`，六条路全关，我上一轮「下一个方向是 market」的判断标 `SUPERSEDED`；P3 同步为 `CLOSED_FAIL`，新增 P4 recency 预注册。

#### 2026-08-18

核对当时的本地工作笔记 `NEXT_STEPS_horizon_auxiliary_oof_validation.md`（未入库、现已不在盘上；
结论证据见 `outputs/experiments/horizon_auxiliary_cache_probe.{json,md}`）：引用数字全对，
但立项论证漏引 08-14 的同机制否决；发现 `responder_00/02` 从未进 Stage
C（被单成员族启发式挡住）。缓存探针补测 ⟹ `REJECTED`；重建测试补测落盘 ⟹ NOTES 数字确认、
口径澄清为中心化 R²。生产目录与模型身份未改动。

#### 2026-08-18

选列宽度轴 `REJECTED`（三个单变量臂全不过，两效应精确可加）；发现 `*_exact` cache 出自已不存在的代码版本并落盘替代基准；`RUNBOOK_8_23.md` 与 `public_replay.py`（21 份 CSV 全归属）就位，8/23 当天无需再做设计决策；与公榜第一的差距订正为 +45.8%（IC +20.8%）。

#### 2026-08-18

**P0 推进到 `AWAITING_USER`**：4 核下 LightGBM（5.26 分钟）与 NumPy 兜底（10.94 分钟）全量实测并落盘，兜底确认为单核绑定 ⟹ 不随核数恶化。修掉三道交付门禁都不认识 slow/fast 的缺口（丢键会静默交出低 2.93% 的旧模型），补 4 个回归用例；现存 `v3_hybrid_submission_20260813.zip` 已被新审计判为旧模型。只剩用户执行打包 + zip 审计。

#### 2026-08-19

**当前数据剩余结构搜索与 full-resolution 资源验证收官**：rank/change/lag/volatility/trend、market set/panel 全未过门禁；phase_id 仅弱 +1.1%、3/5，不升级；periodic 比较因 validation 组成不同不作裁决。修复 disk-backed loader、fixed history 映射和后台 systemd 监控；短跨度 fixed-200 双森林 160 轮 smoke 成功（max RSS≈11.5GB，`oof_valid=false`），同跨度正式 OOF 因 5.92m rows 暂缓。全量测试 73 passed / 18 subtests。

#### 2026-08-19

补三道交付/重训缺口（本轮由仓库结构复查发现，均**不在**原行动面板上）：① `retrain_extended.py`
的「固定结构重训」计划缺
`--weighted-cross-section`/`--market-model`/`--market-spec`/`--market-min-data-scale`，
跑出来的是 08-11 架构（比生产低 21.99%），现改为从生产 `hybrid_meta.json` 派生并与
`PUBLIC_BASELINE` 对拍；② `train.py` 没有 slow/fast 概念 ⟹ 重训候选必缺三键且会被 `main.py`
静默降级，现由 `promote_v3_candidate` 在 staging 写入（`--slow-fast-*`，默认即公榜值）；③
提交包此前「除 train.py 外全收 `*.py`」，把研究模块 `temporal.py` 也装了进去，
审计只查缺文件不查多文件 —— 现由 `make_submission.SUBMISSION_MODULES` 唯一声明 + `main.py` 的
AST import 闭包双向对拍。全量测试 **84 passed / 22 subtests**。

#### 2026-08-19

按外部 `HANDOFF.md` 推进 A/B/C 三线，并核出它三处与仓库不符（P0 的 wall-clock 复测 08-18
已完成；私榜是「**最新提交版本生效**」不是 best-of-10 —— 此前 ROADMAP/RUNBOOK 都漏了这半句；t=2
的 max|pred|数对但仓库无证据）。**P0 结案**（用户 20260819 包审计 `passed:true`、零漂移、
无多余模块）。**P7 预注册落盘**：slow/fast 顶点闭式解 + 限幅几何实测（clip 边界 t≈2.6968）+
锚点交叉验证（max|Δ|=5.0e-09）；增益闭式 `(S1−S0)(t*−1)²/(2t*−1)`，诚实期望只有 +0.0%~+0.9%。
**P8 `CLOSED_FAIL`**：辅助损失机制成立（MLP 自身 peak +16.7%）但对生产基准增量仅 +0.026%；
顺带订正一道写错的机制门槛（收紧）。**P2-R** recency 预注册臂立项（P4 测的是 volume 轴，
不覆盖）。**P1 端到端演练通过**（全 hash 审计 + 闸门正确拒绝）。全量测试 **92 passed / 22
subtests**。

#### 2026-08-19

**P7 结案**：slow/fast 顶点第三点已交（`S2 = 0.0039374211`）。完整性检查 `a = −1.474e−04 < 0`
通过；`t* = 0.897692`、`Score(t*) = 0.0041165516` ⟹ **当前生产点已处在这条线峰值的 99.9625%**，
slow/fast 捕获了线上总可得增益的 **98.70%**；半步收缩后增益 1.157e−06 < 预注册 1e−05 线 ⟹
**不改交付、生产不动**。⟹ 该轴从「没测」变成「顶点已测出」。顺带确认了 08-17「只搬 OOF
相对模式、保留公榜标定绝对水平」这个做法 —— 它落在最优点的 0.04% 以内。

#### 2026-08-20

**P6 清理后的收尾审计**，查出两件事：① 08-18 判毒的 OOF 缓存**不止一份** ——
`..._phasebal_prodwindow.npz`（08-14 10:56、13 数组、无 checkpoint）
与被判毒那份**签名完全一致**，当时漏点名，已一并隔离；② **改名挡不住** ——
四个实验脚本把毒缓存写死成 `--oof` 默认值，改名后只报裸 `FileNotFoundError`，而旁边就是
`.STALE-DO-NOT-USE` 文件，赶工时最省事的「修法」就是指回去。⟹
隔离改成代码强制：`src/oof_cache.assert_reproducible_cache`（含改名后路径，`load_oof_bundle`
内也调用），四个脚本改 `required=True` 并接上守卫，3 个回归用例钉住。
逐条核过对已有结论无影响（多任务 Stage 1 用的是未隔离的 `confirm_3s480`，且 ±2.4%
基准误差下增益仍在 +0.025%~+0.026%）。顺带修掉 P6 清理造成的文档漂移（旧 zip 名 ×3、`lgbm_mt`
docstring）。全量测试 **95 passed / 22 subtests**。

#### 2026-08-20

**P9 NN 独立能力阶梯 `REJECTED`**：单轴 epoch 阶梯（12/50/150/400，测量路径复用
`multitask_mlp.py` 一行未改，12 档对 08-19 锚点偏差 0.00e+00）。曲线倒 U：**峰值 28.8% @ 50
档**（相对 12 档 +42% ⟹ 此前确实被预算掐住），但随后崩溃到 6.5% / 1.4% ⟹
**绑定约束是正则化不是预算**，天花板 28.8% < 50% 门槛。辅助损失符号随过拟合翻转（欠拟合区
multitask 更好，过拟合区反之）。⚠️ 方法学发现：**单折 oracle
混合增益不可信**（独立强度单调崩溃而混合增益非单调；400 档独立仅 1.4% 却报 +3.26%）⟹ 追认
08-19「冻结系数 + 5 折终审」的必要性；**不得**据 50 档的 +6.46% 重开 P8。v5
范围因此重定为「训练配方 / embedding / 特征选择」三条，**加算力不在其中**。全量测试 **101 passed
/ 22 subtests**。

#### 2026-08-20

**P10 密封期尺子 `PREREGISTERED`**：核出 RUNBOOK 漏掉的一个取舍 —— 8/23
回补的**就是公榜期的标签**（实测 test 分区 326 列无 weight/target/responder，train 375
列全有），那段 3,217,458 行**只能用一次**：当训练数据或当干净测试集。而它的评估行数约是现有 OOF
5 折合计的 **2.1×**，正是 OOF 那个 6.1~8.7% 的检出下限把
`mkt323`/`phase_id`/`responder_00`/`lag3+lag10`/扩展窗全判成「测不出来」⟹ 全部拿去训练的话，
8/23–8/31 最可能的结局是「重训了但测不出，按 D6 维持现状」。⟹ 预注册封存最后 **60,000 real
time_id**（4 块 × 15,000、embargo 30、实测 856,319 行），六道门禁按 `≥3/4 块` 映射，
第七道「检出下限」标定前判 `None` **而非自动通过**。⭐ 读数口径订正：
初稿的「`raw = pred/prediction_scale` 反解」在 slow/fast 下是错的（两个分量各有 scale），
改为「断言触限 0 行后直接算 peak」—— peak 对全局缩放严格不变，有用例钉死。干跑双通过：官方
runner 全量 test 3,217,458 行、0 超时、`max|pred|=0.420450` 触限 0 行（与 ledger 08-18
那次对上）；合成标签走通判据链路且强制 `adjudication_valid=false`、self-vs-self 判 FAIL。
**不落任何提交格式 CSV**（runner 输出只进临时目录）。全量测试 **112 passed / 22 subtests**。

#### 2026-08-21

**函数类探针 `REJECTED`，但把两个假设分开了**：诊断出树对时间/截面维扩容的三次失败都是「ΔB
涨幅是 ΔA 三倍」的**函数类**指纹后，用
RFF-岭回归在**一列不改**的生产截面设计上做预注册对照（含线性阴性对照、生产强度 3s480 基准、
五折行对齐以 target/weight 逐位验证）。结果：`r=0.798`、`ρ=0.702`、oracle 集成 **+0.91%**、5/5
折 `r>ρ`（**符号全对、幅度差 3.3 倍**），判据 2/4 不过 ⟹ `REJECTED`。⭐
两个方向相反的结论都成立：① **P9 的「NN 天花板 28.8%」被否** —— 一个没有任何训练配方的核方法就到
79.8%；② **但到 80% 恰恰说明没用** —— ρ 随 r 同步涨，线性/核/树互相 ρ≈0.6~0.7，在读同一个东西。⟹
「换提取器」这条线关闭，只剩「**换输入**」（`history_window=5` 与 slow/fast 的 K=2000
之间是空的）。顺带查出 `INCIDENT`（未爆）：所有 OOF 缓存的 `e_target` 列是**全
NaN**（`v3_production_oof.py:512` 显式 continue 跳过赋值），`src/oof_cache.py:19` 仍把它列在
COMPONENT_COLUMNS —— 当前无脚本消费，未污染任何结论。另订正我自己写错的集成增益表：恒等式是
`1+(r−ρ)²/(1−ρ²)`，oracle 恒 ≥ 0 且关于 r=ρ 对称（单测在跑数据前抓到）。⚠️⚠️
**还查出一个测试门禁漏洞**：`NOTES §4` 文档的 `unittest discover -s tests` **静默少跑 36
个用例**并照样报 `OK` —— 7 个模块是 pytest 风格（裸 `def test_x()` 无 TestCase 子类），unittest
只收 TestCase 子类。被跳过的头两个正是 **`test_sealed_period_eval`(11) 与 `test_oof_cache`(6)**
—— P10 密封期尺子和缓存出处隔离的把关用例。pytest 实测 **122 passed / 22 subtests**（含本轮新增
10），unittest 只有 86。`NOTES §4` 已改为 pytest 并写明该陷阱。

#### 2026-08-21

**长历史窗探针 `REJECTED` —— 两条路都定价完毕**：核出 temporal 全族的
`MAX_LAG = 20`（`experiments/temporal_multiscale.py:49`）⟹ 20 到 slow/fast 的 K=2000 之间是 100
倍未测跨度，本枪补上。40 个 history 特征 × 窗口 {64,512,4096} × {滚动均值, 偏离} = 240 列，
与生产设计配对。结果 pooled **+0.69%**、4/5 正折、去最好折 −0.49% ⟹ `REJECTED`。⚠️⚠️
**本轮最重要的自查**：我最初把 ΔA=+3.36%/ΔB=+15.40% 读成「有信号但付不起方差」，
**那是错的并已收回** —— 逐折 `A比/√(B比) ≡ IC比`，ΔA/ΔB 混着共同尺度（fold 0 的 A 比 0.9555，A
是降的）；且 `2ΔA>ΔB` 是**两分量配比**判别式，不适用于**嵌套模型**比较，
从一开始就不该写进预注册。**这正是 P8 已记录过的同一个错误**（见 2026-08-19 行）。⟹
「序列模型值不值得做」**仍未定价**；要定价得做列数阶梯/单窗口消融，本轮没做。⚠️
两枪各暴露一个镜像的预注册缺陷：`function_class_probe` 的 RFF 全钉在 alpha 梯**顶**，本枪线性
fold 4 钉在梯**底**（−19.49%）⟹ `ALPHA_LADDER` 两端都太窄，复验前必须加宽；但去掉 fold 4 后
pooled 仍只有 +2.77%，**结论稳健**。⟹ 合读：「换提取器」已定价（oracle +0.91%）；「换输入」
只证明**按 240 列这个加法**净效果不过门槛。8/31 前追平榜首仍不现实，但「长窗有没有可用信号」
这个问题**尚未被回答**。

#### 2026-08-21

**选列准则探针 `REJECTED`** —— 用户指出 `select_features` 是单变量筛子、忽略特征间相关结构；
另发现 `history_positions` 按**当期**相关选列却把它们当**滞后量**用。
跑前实测确认分歧真实（top-40 重合：当期vs lag1 = 24/40，而 lag1 vs rollmean5 = 35/40 ⟹
两个滞后准则彼此一致、都与当期不一致）。四臂预注册（评价器必须是**树** —— 对 LASSO
臂用线性评价是循环论证；四臂 history 列取并集只扫一次，等价性有用例钉住）。结果：**换掉 45% 的
200 列（LASSO）或 40~70% 的 history 40 列（滞后准则），树的 peak 一个都没变好**，
且分歧越大掉得越多。⟹ **诊断成立、代价不存在**。选列轴三向封死。⚠️ 事后解释（非本实验发现）：
history 的 `difference` 与 `rolling_deviation` **都含当前值**，该块并非纯滞后，
当期准则对它们反而匹配 —— 要证实需四子块消融，本轮未做。顺带补
`tests/test_select_features.py`（9 用例）：该函数有 **77 个调用点、此前 0 个测试**。

#### 2026-08-21

⭐ **长窗列数阶梯：今天第一个 PASS（筛选档）**。把 `long_history_probe` 那 240 列拆成单窗口各 80
列，`w512` 在树上 pooled **+6.80% / 5-of-5 / 去最好折 +5.84% / CI 下界 +3.87%**，
五道门槛全过且**超检出下限 1.12×**。⭐⭐ 最有说服力的一条：**同一个评价器（线性）上，240 列给
+0.69%、80 列给 +5.75%** —— 预注册时写下的机制（『信号被估计代价淹掉』）**跑之前就写死、
跑之后兑现**。两处跑前声明的变更都生效：主评价器改树 ⟹ 线性 alpha 选参那类脆弱性消失（40
次拟合无一撞端，`long_history_probe` 的 −19.49% 坏折消失，追认为选参失效）；`WIDE_ALPHA_LADDER`
另起常量 ⟹ 旧 `ALPHA_LADDER` 一字未动（有单测断言），两个已结案实验仍可复现。⚠️ 克制读法：
筛选档（1s160/单块）、迁移率历史 0.51×~2.3×、只超下限 1.12×、三臂多重比较、且本脚本的 cumsum
口径**不能**照搬进生产（`history.py` 刻意不用 cumsum 以保离线/在线逐位一致）。⟹ 下一步是 **3s480
确认**（只留 base 与 w512 两臂），生产目录未动。

#### 2026-08-21

**长窗 w512 走完确认档**：3 种子 × 480 轮、只截面块、fold 版图与筛选档相同、base 现跑。结果
pooled **+7.77% / 5-of-5 / 去最好折 +6.49% / 配对 CI 下界 +4.18%**，五道门槛全过，但 **0.89×
检出下限** ⟹ 预注册三档裁决里的中间那档 `PASS_BUT_BELOW_DETECTION_FLOOR`（跑前就预判最可能）。⭐
三条正面旁证：筛选→确认迁移 **1.14× 未衰减**；**基准更强（+2.8%~21%）而增益仍在**（P8
栽跟头的反面）；**线性对拍 10/10 逐位相同**（线性不依赖树超参 ⟹ 排除『换了个跑法』）。⚠️
三条克制：只测截面块（占分 58.8%，全模型粗估 +2%~+5%，与 slow/fast 同量级）；slow/fast
那次迁移率是 0.51× 本地高估 ⟹ 不能拿 1.14× 当规律；探针的 float64 cumsum
**不得**进生产（`history.py` 刻意不用 cumsum 以保离线/在线逐位一致）。⟹ **是否花 8/23
前的公榜额度由用户定**；生产目录仍一字节未动。⚠️ 工具可靠性：本轮收到 3
次与落盘日志不符的后台通知（2 次数字错、1 次提前报完成），所有数字均已从产物文件核对。

#### 2026-08-21

**长窗 w512 候选已就绪，等公榜裁决**（生产未动）。新增 `history.AssetLongWindow`：`AssetHistory`
是 O(window)、512 窗在线跑不动，改用**持久累积和相减**（同 `PredictionTrail`）；
逐位一致是构造上的 —— 离线 `np.cumsum(float64)` 本身定序累加、在线持久 running total，实测
max|Δ|=0，而**分块重起**的 cumsum 不同（正是 `history.py` 警告的写法）。11 个用例钉住「离线整块
≡ 离线分批 ≡ 在线逐 time_id」。长窗块**只进截面设计**（训练日志自证：截面 441 列、市场仍 561 =
raw200 + 截面块 **361**）；只重训截面森林，市场森林与冻结岭回归 hash 与生产**逐字节相同**；
只多一个 meta 键。门禁：缺键时生产仍 4.019e-09、交付配置 4.019e-09（两后端）、双后端 1.39e-16、4
核全量 LightGBM **5.33 分钟**（+1.3%）/ NumPy 兜底 **10.55 分钟**（−3.6%）、0 超时 / 0 非有限值
/ **0 触 clip**（max|pred|0.402099）⟹ 与公榜 0.0041150085 同 scale 可直接比大小、二次式精确。⭐
一致性数值与生产相同（4.019e-09）经直接对拍排除「加载错模型」（森林 361 vs 441 特征、预测
max|Δ|5.11e-02）—— 相同是良性的，说明**长窗块对训练/推理差异贡献为 0**。
顺带补三个登记缺口（`PUBLIC_BASELINE`/两张取值表/`verify_delivery_runtime` 的身份摘要），其中
`make_submission.py:170` 的防呆当场抓到我漏改第二张表。全量 **174 passed / 27 subtests**。

#### 2026-08-21

**长窗 w512 公榜裁决：0.0041833953，+1.662%，新最好成绩**（生产**仍未转正**）。峰值口径 +1.72%，
且对 `B_old` 的 ±8% 扰动稳健；`Σp²` 实测**上升** 1.28% ⟹ 排除「最优 scale 上移导致 1.16
处低估」。⟹ 折成 **IC +0.83%**，与榜首差距 +20.8% → +19.7%。⚠️⚠️
**本项目连续第二次③类本地高估**：迁移率 0.22×（slow/fast 那次 0.51×），而 08-13
之前③类全是**低估**（1.20× / 1.6× / 2.3×）。**符号翻转本身是信号** ——
两次高估的共同点是「在已有块上加派生量」（预测的慢/快拆分、特征的长窗摘要），
而三次低估都是「加新的信息通道或新分量」（相位采样、history40、行级市场森林）。⟹
本地尺子对**派生量**类改动系统性乐观，这一条应写进③类的先验。⟹ 是否转正由用户定：+1.66%
是真涨且门禁全过，但代价是新增一个跨 predict 状态。

#### 2026-08-22

**responder 轴收口：三种用法全部走完，由证据关闭**。① 族群表 —— 缺失数/取值域两个指纹把 47
列切成 8 个维度族（7/7/7/7/3/7/5/4），8 族只用 3 条截断梯子，且与现有 24
族聚类**正交**（那把切的是窗口组）。② Stage C 的 **14 个空白格**填满 —— 未通过 Stage B 的 16
个族**全部只错 `multi_member_family`**、因证据不过 0 个 ⟹ 此前是启发式在挡路；补测 28
格**无一过门禁**，08-18 锚点逐位复现 0.000e+00。③ 用 responder 打分选列（唯一没做过、
也唯一不在母条件排除项里的用法）在**前置测量**处结案、**省下那次 OOF** —— churn
诊断显示换掉的是原排名 #16 的列、Spearman 0.72~0.84 ⟹ **预注册的降方差机制未兑现**。⭐
方法学产出一条：**「剥掉冻结系数让步」是精确的常数平移**（36 臂恒等式偏差 5.4e−20），
只改水平不改排序，剥完「转正」的臂里有 0/4 折的 ⟹ 不能当发现读；08-18 那个被反复引用的
`pure_e/responder_00` +3.92% 也是同一回事。工程：`horizon_auxiliary_cache_probe`
的门禁+bootstrap 抽成 `evaluate_arm()` 供复用，**抽取前后 JSON 逐字段相同**（仅两处
`nan!=nan`）；`multitask_mlp` / `nn_capacity_ladder`
各加一个**默认不改变行为**的入口（后者复跑默认路径，JSON 只多一个新键、数值逐字节不变）；
**未改动 `select_features`**（97 处调用点、在生产训练路径上），改为逐折硬断言自算相关与它的
top-k 逐位相同。全量测试 **217 passed / 26 subtests**（原 174/26，新增 43）。⚠️ 顺带订正：
多处文档记的「27 subtests」实测是 **26**，本轮之前就已差一个。

#### 2026-08-22

**P9 范围项 ③ 结案**：把发现 3 的新选列（与原判据重合 175/200）喂给 `multitask_mlp`
按原规格复跑整条阶梯 —— 天花板 **28.8% → 27.4%**（门槛 50%），`REJECTED`、条件延长未触发。⭐
关键读数是**曲线形状不变**（50 档峰值、150/400 崩溃照旧）⟹ 给「绑定约束是正则化不是特征集」
补上一条**正交**证据。⟹ v5 可改项 3 → 2 条。工程：`--cross-selection-override` 默认不改变行为；
因该臂的 12 档按设计不复现 08-19 锚点，另跑一次默认选列的 12 档做环境自检（偏差
0.000e+00），`nn_capacity_ladder` 相应新增 `--anchor-label` / `--summary-label`（默认行为不变，
已复跑验证）。

#### 2026-08-22

**P0-B 结案**（用户执行）：重打私榜包 `v3_hybrid_submission_20260822.zip` 并落盘审计 —— `passed: true` / `public_baseline_drift: []` / `unexpected_modules: []` / `missing: []` ⟹ 长窗 w512 转正后的模型身份已装进包里（此前 `20260819` 那份缺 `long_window` 键，会**静默**交出低 1.66% 的旧模型）。⚠️ 盘上现有**五个** v3 zip，8/31 只能交 `20260822`。

#### 2026-08-27

⭐ **封板：回退层重建 + 8/31 上传日卡片**。核出 P-REQ 一个当时没记的副作用 —— `requirements.txt`
这条 8/23
新增的硬要求让**盘上所有存量交付件一次性集体失效**：`20260824`/`20260822`/`20260819`/`20260818`/`20260813.PRE-SLOWFAST`
实测**一份都不带**它 ⟹ RUNBOOK D4.5 的「三层回退」当场塌成一层，而文档还写着「时间不够就交
20260822」。⚠️ 形状是 `CLAUDE.md §8.10` 的近亲：不是归属断言过期，
是**合格定义**在我们背后变了。⟹ ① 新增 §1「8/31 交哪一份」作为唯一权威表，
下方六处旧说法逐条标注过期（按 §7 不删原文）；② 回退层由「挑一份旧 zip」改为「用备份
`model_before_20260824_150921` 现打」——跑前已只读核实该备份 13 个身份键全等于
`PUBLIC_BASELINE`、`baseline_model.json` 与生产同为 sha `54dc6afb…` ⟹ **不需要
`--off-baseline`**（若脚本要开关就是我核错了，停下来查）；③ RUNBOOK §D6 新增 8/31
上传日卡片（主件 sha256、审计命令、顺序纪律、中止判据、以及「`peak_rss_has_headroom`
红不算中止理由」这条例外）。本轮**只改文档，未动任何 `.py`、未动生产目录、未跑新实验** ⟹
不需要重跑测试。

#### 2026-08-27

**兜底件落地并通过全量交付验证**（用户打包，AI
核对）：`v3_hybrid_submission_20260828FALLBACK.zip`（sha `5f3bdc58…`）内容审计 **13/13 全过**、
**未用任何 `--off-*` 开关**（跑前那份只读核算因此得到确认）；`--from-zip` 全量推理
`predict 5.32 分钟 / wall 6.27 / 峰值 11.60 GB`，13 条 check 只红
`peak_rss_has_headroom`（存量风险）。⭐⭐ **两条计划外的归属检查比原计划那条更强**：①
`--manifest auto` 自己扫中 `v3_hybrid_long512` 的 manifest（8 文件逐字节相同），独立于人工
sha256 对拍；② `predictions_sha256 = fe527e41…` 与**三份**历史读数逐位相同（08-23
`delivery_4c12g_lightgbm` 源目录、`delivery_local_py313_4t`、08-24 转正前生产对照臂）⟹
兜底件跑出的就是公榜 **0.0041833953** 那一组预测。⚠️ 顺带记一条方法学：**打包 +
审计不到一分钟是对的** —— `make_submission` 只做文件搬运 + 13 个身份键核对 + 15
行烟测，`audit_submission_zip` 是纯元数据、零推理 ⟹ 「审计 13/13」与「交付验证」是两件事，
本次身份之所以可信是因为**逐字节 hash**，不是因为审计过了。

#### 2026-08-29

🛑⭐⭐ **`INCIDENT`（8/31 前 2 天拦下）：交付件从不设 `num_threads`，会在评测机上撞穿总超时并丢
86% 的 `time_id`**。云端真机探针（`experiments/thread_default_probe.py`）：评测容器
`os.cpu_count() = affinity = 128` 而 CPU 配额只有 4 核，lightgbm 默认
`num_threads=-1`（用全部**可见**核）⟹ 起 128 线程挤 4 核。微基准 **99.249 ms vs 1.355 ms =
73.27×**；同日第二次跑把推算换成**端到端实测**（量整条 `Model.predict` 三臂）：出厂
`num_threads=4` **4.743 ms** / 抹掉它 **353.330 ms** / numpy 兜底 **11.035 ms**，加性开销
**+348.59 ms 每次**。折到云端全量锚点 = **351.98 ms/次 = 20.98 h vs 官方总预算下限 2.98
h（704%）** ⟹ 约第 **30,476** 次调用（**14.2% 处**）撞 `total_timeout`，其后 **85.8% 的
`time_id` 全部保持默认值 0**。⚠️ 本条**订正同日早些时候的 6.04 h / 49.4% 处** ——
那是只按截面森林的微基准推的，漏了市场森林那一半，方向没变、量级更糟。✅ **numpy
兜底不受影响**（此前唯一没量过的一格）：不钉线程时兜底/主路径仅 **2.33×**，纯 numpy 树遍历不吃
BLAS 线程 ⟹ 万一评测端没有 lightgbm，那条路本来就安全。**修复**：`main.py` 新增
`_PREDICT_NUM_THREADS = 4` + `_probe_predict_kwargs()` 逐参数探测（`num_threads` 排在
`validate_features` 前面 —— 两者代价不对等，少了前者是撞总闸、少了后者只是慢一点）；同机 A/B
确认 `predictions_sha256` **逐位不变**（`524e14e0…`）⟹ **只改速度、不改模型身份**。重打包
`20260829.zip`（`d934d246…`）与封板的 `20260827.zip` 逐条目比对：**13 个条目里 12 个逐字节相同，
只有 `main.py` 变了**（`ada6a2c2…` → `68ff745e…`）；兜底件同步重打为
`20260829FALLBACK.zip`（`5f91bca9…`，指纹 `fe527e41…` = 公榜 0.0041833953 那组）。
**云端真机零环境变量全量验证**（`--threads 0`，`thread_env` 四个全 `None` ——
第一次在「什么都不给」的条件下量提交包自己）：`model_init` **4.66 s**（2.59%）/ `total` **910.2
s**（**8.49%**）/ `mean_predict` **3.81 ms**（7.62%）/ 峰值 RSS **11.05 GB** / 3,217,458 行 / 0
非有限 / 0 触 clip / 0 超时 ⟹ **余量 11.8 倍，超时风险关闭**。⚠️⚠️
**四次全量交付验证全都没抓到这个洞**，因为那四次的「4 线程」全部来自命令行前缀
`OMP_NUM_THREADS=4`，而**环境变量不随提交包走** —— 测量本身没错，
错的是它的关键口径由一个不在交付件里的外部开关决定 ⟹ **伤疤规则 17**：
「这个读数依赖的每一个口径，是不是都在交出去的那件东西里？」便宜自检 = 解压到干净目录、
不带任何环境变量跑。⚠️ 同日另两条：**伤疤规则 18** —— 本地 Python **3.13** 比评测机 **3.11.15**
新，`verify_delivery_runtime.py --from-zip` 里一处**跨行 f-string**（PEP 701，3.12+）
本地跑了四天没事、云端开跑即 `SyntaxError`，且 `ast.parse(feature_version=(3,11))` **抓不到**它
⟹ 改用 `tests/test_eval_python_compat.py` 的 token 级扫描，版本口径取自评测机实测产物；
**伤疤规则 19** —— 我预告「云端 `predictions_sha256` 必须等于 `524e14e0…`」，
**判据本身写错了**：逐位指纹只在**同机**比较成立，实测云端 `21698601…`、`max\|pred\|` 相对差
**1.35e-08**，正落在 08-23 已量过的「同后端换机器 2.3e-8」那条轴上；跨机那一跑同时变了机器、
依赖版本和被测开关，本来就回答不了「这个开关改没改预测」（那由同机 A/B 回答）⟹ **定判据前先问：
这次比较只变了一个东西吗？**
证据：`outputs/cloud/thread_default_probe.json`、`outputs/cloud/delivery_cloud_20260829_unpinned.{json,md}`、`outputs/experiments/delivery_src_lgbm_4t_numthreads.{json,md}`。

#### 2026-08-31

⭐⭐ **结案：私榜策略文件已上传，主件 `v3_hybrid_submission_20260829.zip`（sha `d934d246…`，
5,856,322 B / 13 文件）**。上传日复核全部只读实跑：内容审计 **13/13 `passed: true`** 并落盘
`outputs/experiments/audit_submission_20260829_main_uploadday.json`（文件名写 zip 日期而非
`final`，按 08-28 那次的教训）；全量 unittest **259 passed**；zip 13 个条目与
`strategies/v3_hybrid/` 源目录**逐字节全同**（`main.py` `68ff745e…`，含
`_PREDICT_NUM_THREADS = 4`）；`git status` 与 `git diff docs/ examples/ timeseries_api/` 均空。
首次把官方 `submission_and_evaluation.md`「最终交付要求」7 条**逐条对包核过**（根目录 `main.py`
/ 源码+模型 / `requirements.txt` / 无 conda 缓存 / 不联网装依赖 / `class Model` 可导入 / 仅
`Path(__file__).resolve().parent` 无绝对路径），此前只核过其中第 3 条。⚠️ **本轮唯一「发现」
最后作废，值得留档**：我按伤疤规则 17 推「代码从哪个目录被导入」这个口径不在我们量过的东西里 ——
官方文档写「评测系统会导入 `~/submit/main.py`」、检查清单要求「`~/submit` 内容与 ZIP 一致」，而
RUNBOOK/ROADMAP 从没提过这个路径，云端三次全量跑的是 `outputs/delivery_verify/…` ⟹ 担心
JupyterHub 上残留旧 `~/submit`（`ada6a2c2…` 那版无 `num_threads` 的 `main.py`）被误用。
**用户当天向主办方问清：提交策略的存放路径与开发环境是分开的，选手看不到** ⟹ 该 `~/submit`
只是选手侧自检目录，整条假想路径不存在，已记入 `CLAUDE.md §4` 人工核实事实、RUNBOOK §D6
卡片改写为「不是门禁」。**教训是形状对、结论错：归属检查不能靠读文档措辞推，
要去问那个系统的真值来源。** 存量风险照旧接受：`peak_rss_has_headroom` 红（云端 11.05 / 12 GB =
92.1%，`under_limit` 为真；归属早已做完 —— 零预测桩模型单跑 harness 就要 11.09 GB，
峰值在分区加载段、不随实盘期长度增长）。本轮**只改文档，未动任何 `.py`、未动生产目录、未打新包**
⟹ 不需要重跑测试。

