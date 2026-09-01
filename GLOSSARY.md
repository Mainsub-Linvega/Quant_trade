# GLOSSARY.md — 名词表

这个仓库有五份长文档、19 条伤疤规则和一套自造的编号体系。本表是**唯一入口页**：
每条一行说清是什么，附「详见」指针。**不在这里维护任何生产数值** —— 数值一律看
`strategies/v3_hybrid/model/hybrid_meta.json`、`outputs/promotions/*/promotion_manifest.json`
和 [`experiments/ledger.csv`](experiments/ledger.csv)。

---

## 1. 编号体系：看到 `P7`、`D4.5` 时去哪里找

| 记号 | 含义 |
|---|---|
| `P0`…`P12` | **ROADMAP 课题号**，按开题先后编，与重要性无关。见 [`ROADMAP.md`](ROADMAP.md) §4 |
| `P-TIME` `P-REQ` `P-B1` `P-ROB` `P-D4` `P-D45` `P-V4R` `P-RESP` | 同为 ROADMAP 课题，但在数字序号用完后临时开的**带名课题**。两套编号并存是历史产物，不是分类 |
| `P1-R` `P2-E` `P2-D2B` `P0-B` | 某个课题的**子课题或复验轮次**（`-R` = 修尺子，`-E` = 扩展数据，`-B` = 第二轮） |
| `D0`…`D6` | **RUNBOOK 执行日**，8/23→8/31 收官期每天该做什么。见 [`RUNBOOK_8_23.md`](RUNBOOK_8_23.md) |
| `v1_ridge` / `v2_lgbm` / `v3_hybrid` / `v4_mlp` | **策略代际**。`v3_hybrid` 是交付基线；`v2` 是教学基线；`v4` 被否决、保留留证 |
| `§8 第 N 条` | 指 [`CLAUDE.md`](CLAUDE.md) §8 的**伤疤规则**第 N 条 |

## 2. 信息标签：每条结论头上的那个词

定义在 [`CLAUDE.md`](CLAUDE.md) §3，全项目统一使用。

| 标签 | 含义 |
|---|---|
| `FACT` | 由用户本人核实。**只有用户能新增**，AI 不得自行升级观察为事实 |
| `RESULT` | 在明确数据、切分、指标口径下得到的实验结果 |
| `HYPOTHESIS` | 尚未验证的机制或方向 |
| `REJECTED` | 已被实验否决，附否决口径、失败幅度与**重新开放条件** |
| `SUPERSEDED` | 曾经有效，被更新的模型/数据/证据替代。旧结论不删除，只标记并链到新结论 |
| `INCIDENT` | 工程事故：根因 + 防复发门禁。`INCIDENT（未爆）` = 交付前拦下 |
| `CLOSED` / `CLOSED_FAIL` | 课题结案（成功 / 走完流程但结论是否定） |

## 3. 评价口径：这个项目怎么判一个改动值不值

| 术语 | 含义 |
|---|---|
| **加权零均值 R²** | 比赛指标。唯一实现在 [`src/metric.py`](src/metric.py)，别处不得复制 |
| **A / B 分解** | 把分数拆成 `A` = 预测与目标的加权协方差（信号量）、`B` = 预测能量。分数变高可能只是 `B` 变大，不一定是发现了信号 |
| **`peak = A²/B`** | 在**最优 scale** 处能达到的分数，等于 `IC²`。⚠️ 它只对**全局缩放**不变 —— `blend_weight`、`long_window`、种子数都不是缩放，换了就是另一个模型（伤疤规则 9）。详见 [`ARCHITECTURE.md`](ARCHITECTURE.md) §11 |
| **`2ΔA > ΔB`** | 判「这是新信息还是只是更激进」的机制门槛：信号增量要跑赢能量增量 |
| **迁移率** | 公榜增量 ÷ 本地增量。历史上低到 0.22×，所以本地绝对分不用于排序生产候选。详见 ARCHITECTURE §5.3 |
| **公榜 / 私榜** | 公榜 = 赛程中每日可见的排行榜（8 月底停更）；私榜 = 最终评测，只提交一次 ZIP |
| **密封期尺子** | 公榜停更后自建的离线裁决工具。标定结论方向相反的两条：**排序** Spearman ρ=1.0（可信），**区间**几乎没有分辨力 ⟹ **用序不用值**。详见 ARCHITECTURE §5.4 |
| **OOF** | out-of-fold，用交叉验证的样本外预测拼出的全段预测，用于比较组件而不泄漏 |
| **embargo** | 训练段与验证段之间强制空出的 `time_id` 数（默认 6），防止时序泄漏。见 [`src/validation.py`](src/validation.py) |
| **配对增量 / 同号折数 / 去最好折** | 判据三件套：只看同一切分下的成对差、有多少折同向、剔掉最好那折后是否还成立 |
| **归属检查** | 伤疤规则 11：每个测量都要配一个**能失败**、且**独立于被测量本身**的检查，回答「它量的是不是那个东西」。三种形式：对已知真值 / 零对照 / 口径边界 |
| **零对照** | 归属检查的一种：构造上必然给 0 的那一格，实测若不为 0 说明测量管道有问题 |

## 4. 模型部件：`v3_hybrid` 是由什么拼的

结构总述见 [`README.md`](README.md) §模型是什么，来历见 ARCHITECTURE §3。
**所有数值以 `hybrid_meta.json` 为准，本表只解释名字。**

| 名字 | 是什么 |
|---|---|
| **截面块** | 主森林。输入是当前 `time_id` 的截面设计矩阵（原始特征 ‖ 截面去均值 ‖ history ‖ 长窗），带权拟合 |
| **市场块** | 第二片森林，逐 `time_id` 预测**市场共同分量**（占 target 方差约 68%），**不带权** |
| **`market_lambda`** | 市场块与 Ridge 市场估计的混合系数 |
| **`blend_weight`** | 截面块与 Ridge 截面估计的混合权重。**属于模型身份**，改了就是另一个模型 |
| **history40 / `history_window`** | 每个资产自身特征的滚动历史列。见 [`strategies/v3_hybrid/history.py`](strategies/v3_hybrid/history.py) |
| **long512 / `long_window`** | 长窗（512 步）滚动均值与偏离，2026-08-21 进入截面块 |
| **slow / fast** | 对模型自身预测做逐资产因果滚动均值，再线性重组。窗口以**真实 `time_id` 步**计 |
| **`prediction_scale` / `prediction_clip`** | 最终后处理：整体缩放与截断 |
| **种子数** | 同结构训多个随机种子取平均。**属于模型身份**（伤疤规则 9） |
| **numpy 兜底** | [`lgbm_numpy.py`](strategies/v3_hybrid/lgbm_numpy.py) 的纯 NumPy 树遍历，评测端若无可用 lightgbm 时走它。两条路都过交付验证 |

## 5. 流程黑话：候选怎么变成榜上模型

完整门禁链见 [`CLAUDE.md`](CLAUDE.md) §6。

| 术语 | 含义 |
|---|---|
| **候选（candidate）** | 训出来但未采纳的模型，落在 `outputs/candidates/<名字>/` |
| **staging → promotion → 转正** | 候选进暂存区 → 过 meta 结构校验 / 双后端对拍 / 训练推理一致性 / 端到端耗时 → **用户确认后**原子替换 `strategies/v3_hybrid/model/` 并留备份。脚本：[`scripts/promote_v3_candidate.py`](scripts/promote_v3_candidate.py) |
| **模型身份** | `blend_weight`、`prediction_scale`、轮数、种子数、history、市场模型、训练权重的**总和**。烟测只证明「能跑」，不证明「是榜上那个模型」 |
| **对拍** | 同一输入下 LightGBM 主路径与 NumPy 兜底逐位比较 |
| **交付验证** | 在钉死核数下跑完整官方 runner 全量推理 + 内存峰值，产出可审计 JSON。[`scripts/verify_delivery_runtime.py`](scripts/verify_delivery_runtime.py) |
| **`predictions_sha256`** | 逐行预测的指纹。⚠️ **只在同机比较里成立** —— 跨机器有已量过的浮点轴（2.3e-8），指纹必然不同（伤疤规则 19） |
| **ledger** | [`experiments/ledger.csv`](experiments/ledger.csv)，每次改动一行：日期 / 改了什么 / 本地分 / 公榜分 / keep 还是 rollback |
| **主办方原文** | `docs/`、`examples/`、`timeseries_api/` —— **只读，不修改**（CLAUDE.md §1.3）。不随本仓库分发，见 [`UPSTREAM.md`](UPSTREAM.md) |

---

## 想快速上手，按这个顺序读

1. 本表 + [`README.md`](README.md)（约 200 行）—— 赛题、交付模型、仓库地图
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) §0–§3 —— 为什么长成这个结构
3. [`ROADMAP.md`](ROADMAP.md) §1–§3 —— 收官时的状态与仍然开着的口子
4. [`CLAUDE.md`](CLAUDE.md) §8 —— 19 条伤疤规则，本项目最贵的那部分
5. 需要细节时再进 `outputs/experiments/`（原始证据）与 `research_history/`（主题史）
