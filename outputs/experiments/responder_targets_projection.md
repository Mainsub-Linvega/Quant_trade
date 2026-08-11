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
| 0 | `responder_29` | 0.00201702 | 0.00224996 (+11.55%) | 0.00195956 (-2.85%) |
| 1 | `responder_29` | 0.00387559 | 0.00345392 (-10.88%) | 0.00327723 (-15.44%) |
| 2 | `target` | 0.00092161 | 0.00092161 (+0.00%) | 0.00075710 (-17.85%) |
| 3 | `responder_04` | 0.00136105 | 0.00125835 (-7.55%) | 0.00102143 (-24.95%) |
| 4 | `projection` | 0.00222091 | 0.00182853 (-17.67%) | 0.00279130 (+25.68%) |

## inner_selected 臂

- baseline 折均 **0.00207924** → 本臂 **0.00194248**
- 配对 Δ 均值 -1.368e-04（**-6.58%**），1/5 折为正，符号检验 p=0.375
- 去掉最好一折：-11.02%
- 整条 alpha 阶梯为正：❌

## multi 臂

- baseline 折均 **0.00207924** → 本臂 **0.00196132**
- 配对 Δ 均值 -1.179e-04（**-5.67%**），1/5 折为正，符号检验 p=0.375
- 去掉最好一折：-13.95%
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

