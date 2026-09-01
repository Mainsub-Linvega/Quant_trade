# 推理、晋级、打包与工程事故

本文件保存会直接影响交付正确性的工程经验。当前交付动作以 [`../ROADMAP.md`](../ROADMAP.md) 为准。

## 1. 提交包约束塑造了策略结构

评测包只包含策略目录 Python 文件和 `model/`，不能依赖仓库 `src/`。因此生产策略使用同目录模块，
并保留纯 NumPy 推理后端作为 LightGBM 兼容兜底。

这一约束是合理的，但也造成仓库内多个策略都有 `main.py`、`train.py`、`features.py` 等同名模块。
跨策略工具若依赖 `sys.path` 顺序，容易命中 `sys.modules` 中已有的错误模块。

长期规则：跨策略加载一律按文件路径和唯一模块名；测试需要覆盖同一进程连续加载多个策略。

## 2. LightGBM 文本模型的 NumPy 兜底

评测环境 LightGBM 版本未知。仅仅 `import lightgbm` 成功，不能证明它会正确读取 v4 文本模型。
生产模型因此：

1. 无条件解析一份 `NumpyForest`；
2. 尝试加载 LightGBM Booster；
3. 用固定种子合成输入逐元素对拍；
4. 自动模式下不一致则退到 NumPy；显式 LightGBM 模式则直接失败。

解析器锁定的关键语义包括：

- 数值分裂使用 `<=`；
- 缺失值类型和默认方向；
- 分类特征的整数截断和 bitset；
- leaf value 已包含 shrinkage；
- threshold 按 float64 解析和比较。

测试同时覆盖合成批次、真实测试分区、非法模型文本、重复资产和越界资产。

## 3. `INCIDENT`：训练与推理数学等价但不逐位等价

训练端和推理端曾分别使用两个数学等价的截面去均值函数。由于 float32 求和顺序不同，输入相差
一个 ULP；树在阈值附近因此翻到另一叶子，输出差被放大到约 2.85e-03，影响约 0.52% 行。

修复：LGBM 分支训练和推理共用同一个 `cross_sectional_deviation` 实现；Ridge 保持原线性路径。
一致性误差随后降到约 4e-08。

长期规则：对阶跃模型，“数学等价”不构成推理一致性证据，必须逐预测对拍。

## 4. 端到端推理性能：树不是唯一成本

旧测量只计时 `booster.predict`，低估了完整 runner。实际每次只有约 15 行，小批 pandas 取列、
稳健变换、截面去均值和历史状态维护占据主要成本。

2026-08-13 的等价优化包括：

- 整帧一次转 float32，再按缓存位置切列；不支持时永久退回 `.loc`；
- `AssetHistory.transform_online` 使用定槽环形缓冲；
- 探测后使用 `validate_features=False`，兼容旧 LightGBM；
- 旧候选在新代码上逐位不变。

单森林 history 候选的局部测量从约 2.205 ms/call 降到约 0.979 ms/call；加入第二森林后的
`mkt_we` 全量 `predict_total` 记录为 6.23 分钟。当前 `mkt_shrunk` 的最终全量 wall-clock 仍应在
打包前重新测量。

NumPy 双森林兜底估计约 15 分钟，是独立风险，不能用 LightGBM 主路径耗时代替。

## 5. `INCIDENT`：公榜 CSV 与候选 meta 不是同一模型

### 现象

历史公榜高分 CSV 由临时副本通过 `blend_weight=1.0`、`prediction_scale=1.16` 生成；训练候选目录的
meta 仍保存本地占位值 `blend_weight=0.5`、`prediction_scale=0.856`。旧 promotion 会覆盖 scale，
却没有校验 blend weight。

如果直接转正，代码和模型文件都能正常运行，烟测也会通过，但提交的是另一模型。

### 证据

留档 CSV 重算：

- blend 1.0 与已交 CSV 的最大差约 5e-09，只是 CSV 舍入底噪；
- blend 0.5 与已交 CSV 最大差约 0.121，相关性也明显下降。

### 修复

- `PUBLIC_BASELINE` 明确完整生产结构；
- promotion 新增 blend、market、权重和模型数量校验；
- packaging 在烟测前检查 meta；
- staging 可复用但必须核对来源候选；
- 激活采用 incoming + 原子替换 + backup；
- smoke test 使用非退化随机特征和连续 time_id；
- 对 blend/market lambda 做敏感性断言。

长期规则：模型身份是“代码 + meta + 所有模型文件”，不能只看策略名字或预测是否有限。

## 6. Promotion 流程

当前流程：

```text
candidate
→ stage_candidate（复制并重写允许的后处理项）
→ validate_meta
→ NumpyForest / LightGBM 自检
→ 结构敏感性检查
→ 写 promotion_manifest
→ 用户显式 activate
→ production backup + atomic replace
→ 再验证 production
```

2026-08-13 的生产 staging：

- 来源：`outputs/candidates/v3_hybrid_mkt_shrunk`；
- staging：`outputs/promotions/v3_hybrid_s1.16_w1_3seed`；
- production：`strategies/v3_hybrid/model/`；
- 双后端最大差：`2.498001805406602e-16`；
- blend 和 market lambda 敏感性检查均非零；
- 当前生产文件 hash 与 staging manifest 一致。

### 当前生产 hash

| 文件 | SHA-256 |
|---|---|
| `baseline_model.json` | `54dc6afba78b16cb47ef06f1392901690b4161d93c37ce0357cda2cdf31ed2fd` |
| `hybrid_meta.json` | `777bbeaad1ccc6460670a8b7354d1fb9b739e81824a676938b611361f1016d20` |
| `lgbm_seed2026.txt` | `582a1e93117b1b1105b06899bdbab76e6c4d4f58643ff40101905953b4866168` |
| `lgbm_seed2027.txt` | `9bdc8a5300a607c710199b0836a97b049d8543001128a1e3f65bdf84cfdcb195` |
| `lgbm_seed2028.txt` | `5f57f744f48b352cb3841d05d070c74ac1d838f5fb349c7ba9cac8904b01557c` |
| `lgbm_market_seed2026.txt` | `c055f119b5539380c8e7b929f7bee0fe540657ae2e54e7c67708bef2ea86552c` |
| `lgbm_market_seed2027.txt` | `591e13692eb283a69710050484545bf4deae56f6b88b2207fe08e8f0ff87458a` |
| `lgbm_market_seed2028.txt` | `91dcef3a63fb6b8818d7757dedef8046e07d037023b12fce7dcae84401bce4f9` |

Hash 是 2026-08-13 的身份记录；未来合法转正后应更新本表，而不是继续称其为当前值。

## 7. 私榜打包

`scripts/make_submission.py` 只读复制策略 Python 文件（排除训练模块）和模型目录，随后：

1. 校验生产 meta；
2. 运行非退化 smoke test；
3. 排除缓存、pyc 和 promotion manifest；
4. 生成 zip。

用户负责执行打包；AI 不生成 zip。打包后还应运行 `scripts/audit_submission_zip.py`，检查：

- 必需文件；
- 禁止文件（尤其 `train.py`）；
- 模型大小和数量；
- 可导入性；
- 包内模型 meta；
- 路径布局。

## 8. Runner 超时语义

本地 `timeseries_api/runner.py` 在 `predict()` 返回后才比较 elapsed time，因此是“事后判超时”，
不是可中断的硬超时。超时步骤可能已经推进模型内部历史状态，然后输出被替换为零。

主办方文件只读，本项目不修改它；最终风险验收应在独立进程外层设置 wall-clock 限制，并明确区分：

- runner 的 `predict_total_seconds`；
- 完整进程 wall clock；
- 模型初始化时间；
- 单步尾延迟；
- LightGBM 主路径与 NumPy 兜底。

## 9. 数据更新审计

主办方原始目录只读。2026-08-12 已保存数据快照：

- `outputs/data_audits/data_release_20260812.json`

收到新包时先比较文件、行数、schema 和 SHA-256。只有 audit 明确指出 train split 变化，
`scripts/retrain_extended.py` 才允许进入执行路径。该门禁防止把“文件日期变化”误当成“新增训练数据”。

---

## 从 ROADMAP §4「行动面板」归档（2026-09-01）

收官后整理。ROADMAP §4 曾积累 **1,169 行**，其中绝大多数是**已结案课题的完整证据卡片** ——
一个叫「行动面板」的章节里 80% 是历史。下列条目按原文**逐字**迁入本文件，
ROADMAP §4 只保留一行结论与指回这里的链接。
课题编号（`P0`…`P12`、`P-XXX`）的含义见 [`../GLOSSARY.md`](../GLOSSARY.md) §1。

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
- ~~**⭐ 单步耗时把风险排序反过来了**~~ `SUPERSEDED`（2026-08-29 被主办方补全的规则推翻，
  见 §P-TIME）：私榜**没有 per-step 硬闸**，只设 `model-init` 和 `total` 两道，
  50 ms 只是总预算公式里的**平均**系数 ⟹ 那 2.802 s 只贡献总账里的 2.802 s，
  不再是一个失败模式。原文保留如下。
  **⭐ 单步耗时把风险排序反过来了**：兜底总耗时是主路径的 2.78×，但**单步最大只有 0.050 s，
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
  ⚠️ **2026-08-27 订正**：本句已过期（该包缺 `requirements.txt`）。交哪一份以 §1「8/31 交哪一份」为准。
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
  ⚠️ **2026-08-27 订正**：本句已过期（该包缺 `requirements.txt`）。交哪一份以 §1「8/31 交哪一份」为准。


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
  ⚠️ **2026-08-27 订正**：本句已过期（该包缺 `requirements.txt`）。交哪一份以 §1「8/31 交哪一份」为准。
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
  ⚠️ **2026-08-27 订正**：本句已过期（该包缺 `requirements.txt`）。交哪一份以 §1「8/31 交哪一份」为准。
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


### P-REQ — 交付包缺 `requirements.txt` —— `RESOLVED`（2026-08-25 发现，**2026-08-27 结案**）

**主办方 8/23 新文档的硬要求，我们的工具链从没接上。**
`public_release_20260823/docs/submission_and_evaluation.md` 的「最终交付要求」第 3 条：

> 3. ZIP **必须**包含 `requirements.txt`，用于记录 Python 包及版本。

提交前自检清单里也列了这一项。而 `outputs/v3_hybrid_submission_20260824.zip`
只有 **12 个文件**（4 个 `.py` + 8 个模型文件），**没有 `requirements.txt`**。

- `scripts/make_submission.py` 从不打包它（docstring 明写「入包内容：`SUBMISSION_MODULES`
  声明的那几个 `*.py` + `model/`」）；
- `scripts/audit_submission_zip.py:39` 的 `REQUIRED` 里也没有它
  ⟹ **审计 11/11 全过，却漏掉一条明写的硬要求。**

⚠️ **这是新要求**：`requirements` 在旧包的 `competition_description.md` /
`data_description.md` 里出现 **0 次**，是 8/23 那份**新增**文档带来的。
⟹ 我 8/24 做 §8.8 的更新包审计时，抓了新文档的评分公式与 80/20 规则，
**没有把它的打包要求与我们的打包代码对一遍** —— 数据审计做了，**文档要求审计没做**。

**真实依赖面**（已核，用于给 freeze 结果做合理性检查）：
`numpy` 是唯一硬依赖；`lightgbm` 是**延迟 import**（`main.py:274`，拿不到就走 numpy 兜底）；
其余全是标准库（`json` / `pathlib` / `re`）。**无 pandas / scipy / sklearn。**

**待办**：

1. ~~AI 侧~~ ✅ **2026-08-25 完成**。`make_submission.py` 新增 `SUBMISSION_EXTRA_FILES`
   （按策略声明的非 `.py` 交付物）+ `check_requirements()` 闸门；
   `audit_submission_zip.REQUIRED` 改为**派生**它，并新增两道 check
   （`requirements_covers_dependencies` / `requirements_matches_eval_env`）；
   `verify_delivery_runtime.py` 新增 `--from-zip`（解压到 `outputs/delivery_verify/<stem>/`
   走官方 runner，落盘 zip sha256 + check `zip_audit_passed`）。
   全量 **343 passed / 43 subtests**（新增 26 项）。生产目录与提交包未动。
   ⭐ **归属检查**（伤疤规则 11，独立于被测量本身）：`requirements.txt` 的 numpy/lightgbm
   版本必须等于 `outputs/cloud/delivery_cloud_py311_4t.json` 里**真实评测机**实测的
   `1.24.3` / `4.3.0`；第三方 import 根由 AST 现算（实测 `{numpy, lightgbm}`）而非维护清单。
   **负控制实跑**：本机 `.venv` 的真实 freeze 过闸门 → 当场炸
   （`numpy==2.5.1 != 1.24.3`）；`--off-env-baseline` 是有意偏离的唯一出口。
   ⚠️ 区分两台机器的是 **numpy**（lightgbm 两边都是 4.3.0），已用单测钉死这一前提。
   证据：`NOTES.md` 2026-08-25 条目。
2. ~~用户侧~~ ✅ **2026-08-27 完成**。在评测机 base 环境（`/opt/conda/bin/python`，
   `pip check` → `No broken requirements found.`）生成，落到
   **`strategies/v3_hybrid/requirements.txt`**（223 条，`sha256 db645ebd…`，进版本控制）。
   `numpy==1.24.3` / `lightgbm==4.3.0` 与 `delivery_cloud_py311_4t.json` 的 `environment`
   块逐字相同 ⟹ 归属检查通过（`problems` 空、`env_drift` 空）。
   ⚠️ **命令与文档字面不同，是门禁逼出来的**：文档写 `pip freeze`，但 conda 装的 numpy
   在 freeze 里渲染成 `numpy @ file:///home/conda/feedstock_root/…/numpy_1682210216651/work`
   —— **记不下版本**（`1682210216651` 是构建时间戳）、写死绝对路径（违反交付要求第 7 条）、
   且让归属检查瞎掉（lightgbm 两边同为 4.3.0，numpy 是唯一判别项）。
   ⟹ 实际用的是 **`python -m pip list --format=freeze`**，同一套元数据、`name==version` 形状。
   同时修掉一个真 bug：conda-forge 构建根 `/home/conda/feedstock_root/` 被
   `_TEAM_PATH` 的 `/home/<user>/` 误判为队伍路径，改为按前缀显式豁免构建根
   （`/home/jovyan/…` 仍拦得住，两条回归各覆盖一面）。全量 **345 passed**。
   证据：`NOTES.md` 2026-08-27 条目。
3. ~~用户重新打包 + 交付验证~~ ✅ **2026-08-27 完成**。
   **`outputs/v3_hybrid_submission_20260827.zip`，sha256 `d1ee32ae…`，13 个文件，
   内容审计 13/13 全过** —— 这是现在唯一可交的那份。
   证据：`outputs/experiments/audit_submission_20260827.json`、
   `delivery_zip_lgbm_4t.json`、`delivery_zip_numpy_4t.json`。

   ⭐ **三道归属检查（伤疤规则 11），全部实跑**：
   - **对已知真值**：新包 vs `20260824.zip` 逐条目 sha256 —— 12 个原有文件**逐字节相同**，
     唯一差异就是新增的 `requirements.txt` ⟹ 模型身份不可能变。
   - **对已知真值（更强）**：从 zip 解压出的模型跑全量推理，`predictions_sha256`
     `524e14e0…`（lightgbm）/ `567265de…`（numpy 兜底）与 08-24 用**源目录**跑的
     `delivery_runtime_*_4t_full_20260824` **逐位相同** ⟹ 打包链路零偏移。
   - **口径边界**：`requirements.txt` 的 numpy/lightgbm 版本 == 评测机实测真值（见待办 2）。

   **两条后端的交付验证读数**（`rows 3,217,458` / `calls 214,538` / `runner_messages` 空 /
   `zero_timeouts` / `zip_audit_passed`，13 条 check 只红 `peak_rss_has_headroom` 一条）：

   | 后端 | model_init | predict_total | wall | 峰值 RSS |
   |---|---:|---:|---:|---:|
   | lightgbm | 0.39 s | 6.05 min | 7.03 min | 11.57 GB |
   | numpy 兜底 | 0.33 s | 10.09 min | 10.89 min | 11.54 GB |

   ⚠️ `peak_rss_has_headroom`（余量线 = 12 GB 的 80% = 9.60 GB）**在每一次量过内存的运行里
   都是红的** —— 本地 11.32–11.57 GB、真实评测机 10.93 GB。这是 08-23 立项时就记在案的
   **存量风险**（见 NOTES 08-23「交付链路从来没量过内存」），不是本次引入的回归；
   `peak_rss_under_limit`（< 12 GB）始终为真。

⚠️ **`20260824.zip`（sha `015ab10e…`）已作废**：它缺 `requirements.txt`，
自 08-25 起内容审计判 FAIL（红 `required_files_present` 与 `requirements_covers_dependencies`）。
**不加豁免开关** —— 加开关就是再造一个「审计过了但缺硬要求」的洞。
⟹ 现在唯一可交的是 **`20260827.zip`（sha `d1ee32ae…`）**，模型身份与 `20260824` 逐字节相同。

⭐ **2026-08-27 补记：这条新要求的杀伤半径比结案时写的大一圈。**
结案时只点了 `20260824` 作废，但同一条判据对**所有**存量交付件成立 ——
实测 `20260822` / `20260819` / `20260818` / `20260813.PRE-SLOWFAST` **一份都不带**
`requirements.txt`（`unzip -l | grep -c requirements` 全为 0）。
⟹ **RUNBOOK D4.5 的三层回退当场塌成一层**，而它塌得毫无动静：
文档还写着「时间不够就交 20260822」，那份包却已经不合格。
形状是 `CLAUDE.md §8.10` 的近亲 —— 不是归属断言过期，是**合格定义**在我们背后变了，
而所有引用它的文档都没跟着动。⟹ 已按 §1「8/31 交哪一份」重建单一权威表，
并把回退层由「挑一份旧 zip」改为「用备份模型现打」（RUNBOOK D5 有命令，
已核该备份不需要 `--off-baseline`）。


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
  ⚠️ **2026-08-27 订正**：本句已过期（该包缺 `requirements.txt`）。交哪一份以 §1「8/31 交哪一份」为准。
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


### P6 — 磁盘清理（**清单由 AI 出，删除/改名由用户执行**，CLAUDE.md §1.1）

- **状态**：`CLOSED`（2026-08-20，用户已执行；本条为执行后复查）。
- **复查实测**（2026-08-20）：`outputs` **21G → 3.5G**、`/` 空闲 **64G → 81G**。三项可回收项
  （`fullres_rows_mod1` / `mt_aggregates.npz` / `nvim.log`）已不在盘上；两项改名封存已生效
  （`..._exact.STALE-DO-NOT-USE.npz` 67M、`v3_hybrid_submission_20260813.PRE-SLOWFAST.zip` 5.6M）。
  ⭐ **「绝对不要删」三项全部完好**：`outputs/submission_*.csv` **22 份 / 1.4G**（D0.3 的全部原料）、
  `outputs/promotions/` 57M（含 `backups/`）、`data_release_20260818.json`；
  8/31 要交的 `v3_hybrid_submission_20260819.zip` 仍是 **5,819,904 B**，与 P0 结案记录逐字节一致。
  ⚠️ **2026-08-27 订正**：本句已过期（该包缺 `requirements.txt`）。交哪一份以 §1「8/31 交哪一份」为准。
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


