# A1：每资产历史特征 —— peak 口径复现

**问题**：每资产历史特征在尺度无关的 peak 口径下还提高样本外表现吗？

**为什么重测**：原实验 walk_forward_history 是在固定 prediction_scale=0.5 下算绝对分的，而固定低 scale 对 B 膨胀的惩罚不到 peak 的一半；history 臂恰恰把设计矩阵从 400 列加到 560 列。这与 08-10 把「市场模型重建 13% 余量」判成幻觉的机制相同。

## 原实验（07-23）的结论与它的口径

- baseline 0.00080097 → history 0.00082130（**+2.54%**，3/3 折为正，报告写 `Accepted: True`）
- 口径：`weighted_zero_mean_r2 @ prediction_scale 0.5` —— **不是尺度无关的 peak**
- 3 折、按分区划分、legacy 求解器 tol 1e-4/max_iter 100

## 本次配置

- 折 5 × train_window 39,480，embargo 6，sample_modulo 10
- 特征 200 列 + history 40 列 × 4 块，窗长 5
- 求解器 lsqr tol=1e-8 max_iter=2000（严格档，与 train.fit_model 一致）
- history 列：第 0 折训练窗上按加权 |corr| 选一次，全折通用（预注册）

## lgbm 臂

- baseline peak 折均 **0.00177542** → history **0.00192421**

| 口径 | 配对 Δ 均值 | 相对 | 正折 | 去掉最好一折 | 符号检验 p |
|---|---:|---:|---:|---:|---:|
| **peak（判据口径）** | +1.488e-04 | **+8.38%** | 5/5 | +7.73% | 0.062 |
| 固定 scale 0.5（旧口径） | +1.053e-04 | +7.22% | 5/5 | +6.76% | 0.062 |

- 逐折 Δpeak：['+7.801e-05', '+8.783e-05', '+1.953e-04', '+1.916e-04', '+1.912e-04']
- ΔA +6.17%，ΔB +3.74%，`2ΔA > ΔB` 成立

## 判据（由 `verdict()` 判，不是报告里的评语）

**lgbm 臂 —— ✅ PASS**
- ✅ 1_paired_delta_positive
- ✅ 2_survives_drop_best_fold
- ✅ 3_relative_gain_at_least_1pct

