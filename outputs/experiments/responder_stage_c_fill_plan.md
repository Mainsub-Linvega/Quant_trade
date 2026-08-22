# Stage C 补测 —— 预注册（`responder_stage_c_fill_plan`）

> 判据先于结果落盘。结果产物里记本文件的 sha256。

**问题**：Stage B 的 multi_member_family 是启发式而非证据。被它挡掉、且从未进过 Stage C 的 14 个 responder，其严格 OOF 预测能不能补强 v3 的 target 残差？

**定位**：结案，不是找收益。本机制属于 responder_reaudit_20260814.md:93-100 母条件**明令排除**的「线性叠加 / 对预测值做二层校准」一族 ⟹ 任何过门槛的臂都只能成为 P10 Tier 2 候选，不得据此重开 responder 线。

## 反面先验（跑之前就写下）

1. 已测的 8 个族可预测性比 target 高 8~460×，仍为 −18.81%、1/5 折
2. 08-18 补测的 r00/r02 最好一格只有 +1.38%、去最好折为负、0.43× 检出下限
3. 不得按「与 target 相关高」解读：responder_03 相关 0.817 却是 A0 全场最差 −15.47%

## 臂

- **主臂（14）**：`responder_01`, `responder_03`, `responder_04`, `responder_05`, `responder_06`, `responder_10`, `responder_13`, `responder_17`, `responder_20`, `responder_28`, `responder_29`, `responder_30`, `responder_42`, `responder_43`
- **自检臂（2）**：`responder_00`, `responder_02` —— 点估计必须复现 08-18 落盘值
- **校准臂（3）**：`null_frozen_scale`, `negctrl_shuffle`, `known_negative_27`
- **基准**：`full`（prediction_raw）, `pure_e`（e_lgbm）

## 门禁

- `1_mean_delta_positive`：折均 Δpeak > 0
- `2_at_least_3_of_4_folds_positive`：≥3/4 评估折为正
- `3_survives_drop_best_fold`：去掉最好一折后仍为正
- `4_relative_gain_at_least_3pct`：相对增益 ≥ 3%
- `5_two_delta_A_exceeds_delta_B`：2ΔA > ΔB
- `6_paired_bootstrap_ci_lower_bound_positive`：配对 block bootstrap CI 下界 > 0
- `7_exceeds_detection_floor`：折均超过该臂自己的检出下限

**harness 门**：negctrl_shuffle 两基准均不得通过门禁，且 known_negative_27 两基准相对增量均 < 0；不满足则整轮作废、不解读任何数字

**复现门**：点估计不经过 bootstrap ⟹ 应逐位相同；CI 因逐臂换随机流会不同，不作自检项

## 多重比较纪律

1. 过门槛的臂只能成为 P10 Tier 2 候选；不建候选模型、不碰生产、不花提交额度
2. 必须报告过门槛的臂落在哪个维度族；集中在某族才算机制信号，散落按噪声读
3. 读表先看 null_frozen_scale —— 那 2.5~3.8 个百分点是冻结系数的让步，不是效应

## 限制

- 缓存里的 responder OOF 是 Ridge 强度 ⟹ 准入筛，不是终审
- 基准在评估折上重解最优 scale、候选用冻结系数 ⟹ 对候选不利，null_frozen_scale 量化该让步
- 基准不含 slow/fast 后处理 ⟹ 与 slow/fast 的交互未验证
- v3 基准缓存 v3_production_oof_confirm_3s480_phasebal_prodwindow.npz 未被隔离但**未经现跑复验**（RUNBOOK_8_23.md:174-180）⟹ 不得用于晋级裁决
