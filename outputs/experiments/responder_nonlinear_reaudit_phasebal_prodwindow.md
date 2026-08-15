# Responder 非线性重新审计

**问题**：严格 OOF 的 responder 预测，能否在同容量 target-only 非线性对照之上提供稳定增量？

- 采样：modulo 5 / phase_balanced
- 训练窗口：78,960 sampled time_ids；embargo 6
- 一级模型：全部 323 个特征，target + 47 responder 共享 Ridge 设计
- 二级模型：只在 OOF fold 0 训练，冻结后评估 fold [1, 2, 3, 4]

| capacity | target-only peak | +responder peak | relative | +folds | drop best | pass |
|---|---:|---:|---:|---:|---:|:---:|
| `strong` | 0.00089236 | 0.00078794 | -11.70% | 1/4 | -20.00% | ❌ |
| `current` | 0.00082037 | 0.00068067 | -17.03% | 2/4 | -25.61% | ❌ |

**STOP** — do not open responder multi-task training

限制：本门禁的 target-only 对照仍是 Ridge OOF 的二层校准，不等同于当前生产 v3。只有与强 v3 OOF 配对后仍增益，才允许进入 GPU 多任务模型。
