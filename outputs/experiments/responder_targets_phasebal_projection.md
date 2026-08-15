# A0：`responder_*` 换训练目标 —— 外层判决

**用 responder 当训练目标，预测 target 的样本外 peak 能不能超过用 target 训练？**

- 指标：peak = A²/B（尺度无关），在 target 上评
- 公平性：每个候选目标在训练段内标准化成零加权均值/单位加权方差 ⟹ ROADMAP 的 alpha_eff 规则的等价实现；X、y 均加权中心化 ⟹ 截距恒 0（坑 1）
- 防选择偏置：阶段 1 那折已排除；候选只在内层选，外层只评被选中的
- 短名单（预注册）：`responder_04`、`responder_28`、`responder_05`、`responder_29`、`responder_06`
- 来源 stage1 JSON sha256：`0fcff71f2f0466360ed3ee02ca70a863822e67af6159e4a66a9552e391b37e7e`

## 逐折

| 折 | 内层选中 | baseline peak | 选中臂 peak | multi peak |
|---:|---|---:|---:|---:|
| 0 | `responder_04` | 0.00042050 | 0.00052604 (+25.10%) | 0.00049393 (+17.46%) |
| 1 | `responder_04` | 0.00045629 | 0.00086749 (+90.12%) | 0.00082817 (+81.50%) |
| 2 | `responder_04` | 0.00057340 | 0.00065927 (+14.98%) | 0.00050730 (-11.53%) |
| 3 | `responder_05` | 0.00077914 | 0.00069651 (-10.61%) | 0.00077569 (-0.44%) |
| 4 | `responder_28` | 0.00056370 | 0.00059261 (+5.13%) | 0.00056195 (-0.31%) |

## inner_selected 臂

- baseline 折均 **0.00055861** → 本臂 **0.00066838**
- 配对 Δ 均值 +1.098e-04（**+19.65%**），4/5 折为正，符号检验 p=0.375
- 去掉最好一折：+6.16%
- 整条 alpha 阶梯为正：❌

## multi 臂

- baseline 折均 **0.00055861** → 本臂 **0.00063341**
- 配对 Δ 均值 +7.480e-05（**+13.39%**），2/5 折为正，符号检验 p=1.000
- 去掉最好一折：+0.10%
- 整条 alpha 阶梯为正：✅

## 判据（由 `verdict()` 判，不是报告里的评语）

**inner_selected —— ❌ 不过**
- ✅ 1_paired_delta_positive
- ✅ 2_survives_drop_best_fold
- ❌ 3_positive_across_alpha_ladder
- ✅ 4_relative_gain_at_least_5pct

**multi —— ✅ PASS**
- ✅ 1_paired_delta_positive
- ✅ 2_survives_drop_best_fold
- ✅ 3_positive_across_alpha_ladder
- ✅ 4_relative_gain_at_least_5pct

