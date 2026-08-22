# responder 监督的选列判据 —— 预注册（`responder_selection_probe_plan`）

> 判据先于结果落盘。结果产物里记本文件的 sha256。

**问题**：用 responder 的窗口梯子把选列判据的估计方差压下去，选出来的 200 列会不会与现状不同？不同到能被本地尺子测出来吗？

## 为什么这不被 `selection_criterion_probe` 覆盖

那次三个臂换的是**被估计的量**（LASSO 多元系数、lag1、rollmean5）；本判据换的是**估计的方差** —— 同一个量、多次测量取平均。机制不同。

## 为什么这个机制是被允许的

responder_reaudit_20260814.md:93-100 的母条件排除的是「换目标 / 线性叠加 / 对预测值做二层校准」。用 responder 给 feature 打分三者都不是，属该条点名认可的 representation 一类；且推理端完全不需要 responder。同时命中 ROADMAP.md:456 给 v5 划的第 ③ 条范围项。

## 梯子（从 `responder_window_atlas` 派生，不硬编码）

准入：α 族（`unit_interval`）成员中 `responder_window_atlas` 自己判定 `H_fit_is_equal_weight_MA == True` 的；**不看与 target 的相关** —— responder_targets_stage1 已证伪该论证形式

| 成员 | 拟合 H | 拟合 RMSE |
|---|---:|---:|
| `responder_00` | 1 | 0.024 |
| `responder_02` | 2 | 0.025 |
| `responder_03` | 4 | 0.045 |
| `responder_04` | 7 | 0.046 |
| `responder_05` | 10 | 0.054 |
| `target` | 5 | — |

被准入判据挡掉的 α 族成员：
`responder_01`（RMSE 0.060）, `responder_06`（RMSE 0.192）

## 判据

- 现状：`|corr(feature_j, e_target)| —— 现状，strategies/v1_ridge/train.py:86-108`
- 新判据：`score_j = mean_k corr(feature_j, e_k)，e_k = 第 k 级标签的逐 time_id 无权截面残差`

## 决策规则（跑前钉死）

```text
重合 ≥ 190/200   ⟹ 不跑 OOF，结案
重合 < 190/200   ⟹ 先验是掉分；需机制假设才跑
```

- ≥ 线：重合 ≥ 190/200 ⟹ 改动幅度低于 1s160 的 6.1% 检出下限，判「与 base 不可区分」，**不跑 OOF**，结案
- < 线：重合 < 190/200 ⟹ 落进 selection_criterion_probe 已量过的区间（lasso200 重合 68~119/200 → −1.29%，且分歧越大掉得越多是单调的）⟹ 先验是掉分；只有能写出一条不依赖「判据更好」的机制假设时才跑 1s160 五折

## 自检

- 自算相关的 top-200 / top-40 必须与 select_features 逐位相同（逐折断言）
- S_base 与 S_new 用同一批 complete-case 行 ⟹ 唯一差别是标签
- 另报全行 S_base 与 complete-case S_base 的重合，证明限制行集本身没挪动选列

## 限制

- v3_production_oof.py 不支持长窗（截面设计 361 列，生产是 441 列）⟹ 结论不能直接外推到生产结构
- v3_production_oof.py:390-402 假设 xs/market 两个宽度是同一判据的嵌套 top-N；换判据若破坏嵌套会 AssertionError
- 选列是模型身份的一部分（train.py:469-478 的 reuse_forest 硬校验）⟹ 换选列则两片森林都得重训；PUBLIC_BASELINE 里目前没有选列身份
- 本脚本只做前置测量，不训练、不产生任何可晋级的候选
