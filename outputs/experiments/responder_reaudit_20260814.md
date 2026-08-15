# Responder 重新审计阶段报告（2026-08-14）

## 问题

此前“responder 对 target 没用”的结论是否受以下因素影响：

1. periodic/modulo10 与生产 phase-balanced/modulo5 口径不同；
2. sampled `train_window` 没按采样密度换算，实际历史跨度不同；
3. target-only top-200 选列提前过滤了 responder 信息；
4. 只检查了线性换目标和线性残差叠加，没有检查非线性二层关系；
5. 弱 Ridge 基线上的增量未必能迁移到强 v3。

## 本轮执行

### 1. 生产采样口径复验 responder 可预测性

- `sample_modulo=5`
- `sampling=phase_balanced`
- `train_window=78,960` sampled time_ids（约等于 40 万原始 time_ids）
- 5 folds，embargo 6

结果：原来的 8 个 responder 家族仍全部通过可预测性门禁，且 market/cross 两块都可预测。
证据：`responder_predictability_reaudit_phasebal_prodwindow.{json,md}`。

### 2. 线性残差增量复验

同一生产等效窗口下，将 8 个 responder 家族的严格 OOF 预测线性叠加到 target Ridge：

- 平均增益 **-18.81%**
- 1/5 折为正
- 去最好折仍负

结论：线性残差叠加继续否决。证据：
`responder_residual_increment_reaudit_phasebal_prodwindow.{json,md}`。

### 3. 换训练目标 / responder 投影复验

一个重要反例：若 modulo5 仍错误沿用 `train_window=39,480`，实际历史跨度只有约生产的一半，
`multi` 会显示 **+13.39% PASS**。把窗口改为生产等效的 78,960 后：

- inner-selected **-14.44%**，0/5 折为正；
- multi **-14.08%**，0/5 折为正。

结论：短窗口正结果是训练历史跨度混入后的假象，不支持 responder 换目标。
证据：`responder_targets_phasebal_projection.md` 与
`responder_targets_phasebal_prodwindow_projection.md`。

### 4. 去掉 target-only 选列偏置

使用全部 323 个 feature，为 target 和 47 个 responder 生成严格 Ridge OOF 预测；不存在 top-200
选列过滤。缓存口径：phase-balanced/modulo5、train_window 78,960、5 folds。

- complete-case 2,643,374 行；排除 2,156 行（0.0815%）；
- OOF 1,460,912 行；
- 缓存：`outputs/cache/responder_oof_phasebal_prodwindow_f323.npz`。

### 5. 非线性二层门禁

诊断性 expanding-window 二层模型有局部正结果，但对 regime 和采样 grid 高度敏感；不能作为裁决。
正式脚本 `experiments/responder_nonlinear_reaudit.py` 使用更严格的固定历史校准：只在最早 OOF fold
训练二层模型，冻结后评估后续四折。

| capacity | target-only peak | +responder peak | responder 增量 | +folds | drop best |
|---|---:|---:|---:|---:|---:|
| strong | 0.00089236 | 0.00078794 | **-11.70%** | 1/4 | -20.00% |
| current | 0.00082037 | 0.00068067 | **-17.03%** | 2/4 | -25.61% |

结论：在不利用未来 fold 重新校准的诚实门禁下，非线性二层 responder 增量失败。
证据：`responder_nonlinear_reaudit_phasebal_prodwindow.{json,md}`。

### 6. 与强 v3 history OOF 的对照

生成 1 seed × 160 rounds 的严格 v3 history OOF（同 phase-balanced/modulo5、train_window 78,960），
再将 responder OOF 预测作为二层输入。无论 strong/current 二层容量：

- 所有整体候选均低于原 v3 OOF；
- responder 相对同容量 `[v3 + target Ridge]` 对照为 **-7.76% / -6.77%**；
- drop-best 同样为负。

证据：`v3_history_oof_phasebal_prodwindow.{json,md}`、`responder_vs_v3_nonlinear_audit.json`。

## 裁决

**当前不开放 GPU 多任务 responder 训练。**

更准确的结论不是“responder 与 target 无关”，而是：

- responder 的确高度可预测；
- 但目前能从可见 feature 恢复的 responder 成分，没有稳定补充强 v3 的 target 残差；
- 短窗口、逐折重新校准和弱基线都能制造看似很大的正结果；
- 一旦恢复生产等效历史跨度、固定过去校准并换成强 v3，对应增量消失。

## 重新开放条件

只有以下任一条件成立才重新进入 GPU 阶段：

1. 8/23 回补数据后，固定本轮脚本与门槛原样复验，强 v3 OOF 上 responder 增量转为至少 4/5 正、
   drop-best 正且相对增益 ≥3%；
2. 找到新的 responder 使用机制，不是换目标、线性叠加或对预测值做二层校准，例如具有明确因果
   约束的 representation/distillation，并先在强 v3 OOF 上通过同样门禁。
