# Research History — 研究历史与原文档案

本目录保存项目的研究演变、失败路径和工程事故。它不是“当前状态”的来源：

- 当前生产与行动面板：[`../ROADMAP.md`](../ROADMAP.md)
- 当前研究上下文：[`../NOTES.md`](../NOTES.md)
- 人与 AI 协作规则：[`../CLAUDE.md`](../CLAUDE.md)

## 组织方式

历史采用**主题为主、时间为辅**的组织方式：

| 文件 | 内容 |
|---|---|
| [`validation-and-calibration.md`](validation-and-calibration.md) | 时序验证、A/B 分解、scale、公榜校准与本地量反 |
| [`model-evolution.md`](model-evolution.md) | Ridge、v3_hybrid、history、双森林和生产模型演进 |
| [`features-and-signals.md`](features-and-signals.md) | 市场/截面信号、history、responder、temporal、MLP |
| [`delivery-and-incidents.md`](delivery-and-incidents.md) | 推理一致性、性能优化、promotion、打包与工程事故 |

## 近期专项记录

2026-08-14 的 v3 严格 OOF、残差地图与信号搜索单独保留如下（索引于 2026-08-15 更新）：

| 文件 | 核心结论 |
|---|---|
| [`v3-residual-atlas-2026-08-14.md`](v3-residual-atlas-2026-08-14.md) | 建立完整生产架构 OOF 与 market/cross residual atlas |
| [`v3-market-round-and-asset-confirm-2026-08-14.md`](v3-market-round-and-asset-confirm-2026-08-14.md) | market480 在 3-seed 确认中通过；per-asset XS adapter 稳定通过 |
| [`v3-residual-signal-search-2026-08-14.md`](v3-residual-signal-search-2026-08-14.md) | asset×magnitude/regime 条件化跨时期不稳；静态 PCA 因子失败 |
| [`v3-sparse-and-raw-state-search-2026-08-14.md`](v3-sparse-and-raw-state-search-2026-08-14.md) | sparse asset×feature 与 raw dispersion gate 均失败；当前最佳仍为基础 per-asset XS scale |

主题文件是便于检索的整理版，不替代原始实验报告。需要核对细节时，按以下顺序查看：

```text
代码 / 模型 / manifest / ledger
→ outputs/experiments 下的 JSON 与 Markdown
→ 本目录主题历史
→ source_snapshots 中的原始长文
```

## 重构前原文快照

2026-08-13 文档重构前，三个入口文件已逐字保存到
[`source_snapshots/`](source_snapshots/)。快照只读，不再追加内容。

| 原文件 | 快照 | 行数 | SHA-256 |
|---|---|---:|---|
| `CLAUDE.md` | `source_snapshots/CLAUDE.pre-doc-refactor.md` | 144 | `9a2ecd6923da3a70ffccf50da37585f3b47ec6fbf7634b83e9b404a3e1af5384` |
| `ROADMAP.md` | `source_snapshots/ROADMAP.pre-doc-refactor.md` | 946 | `261bee2983d508f1bffbebbe0af5a62e58edb40dcf2e222926188383714929bb` |
| `NOTES.md` | `source_snapshots/NOTES.pre-doc-refactor.md` | 2549 | `1f21c12183b549396ee15699fbe711beab4ea96c86c15adfcb6f336831b9676b` |

校验命令：

```bash
sha256sum research_history/source_snapshots/*.md
```

## 历史条目的解释

- 历史中出现的“当前”“下一步”“生产模型”只对标注日期有效。
- 被后续结果替代的判断会标成 `SUPERSEDED`，但保留当时推理和为何会错。
- 被否决的方向会标成 `REJECTED`，同时记录重新开放条件。
- 分数必须结合模型版本和 scale 阅读；非最优 scale 的较低分不等于模型退步。
- 主题整理无法承载原始长文的每个数字；完整推导始终可在快照和实验报告中恢复。
