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
