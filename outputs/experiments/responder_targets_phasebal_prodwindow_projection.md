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
| 0 | `projection` | 0.00117270 | 0.00102774 (-12.36%) | 0.00109252 (-6.84%) |
| 1 | `responder_04` | 0.00079331 | 0.00078946 (-0.48%) | 0.00074996 (-5.46%) |
| 2 | `responder_04` | 0.00122376 | 0.00088604 (-27.60%) | 0.00099646 (-18.57%) |
| 3 | `projection` | 0.00085256 | 0.00079798 (-6.40%) | 0.00054985 (-35.51%) |
| 4 | `responder_28` | 0.00116616 | 0.00095509 (-18.10%) | 0.00108620 (-6.86%) |

## inner_selected 臂

- baseline 折均 **0.00104170** → 本臂 **0.00089126**
- 配对 Δ 均值 -1.504e-04（**-14.44%**），0/5 折为正，符号检验 p=0.062
- 去掉最好一折：-17.96%
- 整条 alpha 阶梯为正：❌

## multi 臂

- baseline 折均 **0.00104170** → 本臂 **0.00089500**
- 配对 Δ 均值 -1.467e-04（**-14.08%**），0/5 折为正，符号检验 p=0.062
- 去掉最好一折：-16.56%
- 整条 alpha 阶梯为正：❌

## 判据（由 `verdict()` 判，不是报告里的评语）

**inner_selected —— ❌ 不过**
- ❌ 1_paired_delta_positive
- ❌ 2_survives_drop_best_fold
- ❌ 3_positive_across_alpha_ladder
- ❌ 4_relative_gain_at_least_5pct

**multi —— ❌ 不过**
- ❌ 1_paired_delta_positive
- ❌ 2_survives_drop_best_fold
- ❌ 3_positive_across_alpha_ladder
- ❌ 4_relative_gain_at_least_5pct

