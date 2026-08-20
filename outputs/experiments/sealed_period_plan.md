# 密封期尺子 —— 预注册

注册日期：**2026-08-20**（标签到手**之前**）

**问题**：把公榜期尾段封存成密封测试集，标定它的检出下限，再决定哪些「测不出来」的候选值得复验

## 切分

```text
test 期        888,480 – 1,105,919   217,440 个 real time_id
密封测试集     1,045,920 – 1,105,919   60,000 个
  block 0      1,045,920 – 1,060,919   15,000 个
  block 1      1,060,920 – 1,075,919   15,000 个
  block 2      1,075,920 – 1,090,919   15,000 个
  block 3      1,090,920 – 1,105,919   15,000 个
embargo        30 real time_id（= OOF 的 6 采样步 × sample_modulo 5）
决策期训练     ≤ 1,045,889
```

## 门禁

| # | 判据 |
|---|---|
| 1 | 块均相对增益 > 0 |
| 2 | ≥3/4 块为正（D2「≥4/5 折」的映射） |
| 3 | 去掉最好一块后仍 > 0 |
| 4 | 块均相对增益 ≥ 3% |
| 5 | 2ΔA > ΔB（pooled） |
| 6 | 配对 block bootstrap 的 CI 下界 > 0 |
| 7 | 超过检出下限 —— ⚠️ 该值由 Tier 1 标定，标定前判 None 而非自动通过 |

配对 block bootstrap：每块 25 个 chunk、重抽 2000 次、seed 2026、95% percentile。

## 读数口径

比较用 peak = A²/B（尺度不变），不用单点分（CLAUDE.md §5.5）。官方 runner 输出已乘 scale、已限幅；触限 0 行时 peak 与 raw 上的 peak 逐位等价，所以不反解 raw —— slow/fast 下两个分量各有 scale，除以单一 prediction_scale 是错的。

## 阶段顺序

1. D0 审计 + D0.3 修尺子（不变）
2. D0.4 用 Tier 1 盘上现成候选标定密封期尺子的检出下限
3. D1/D2 重训 + 裁决 —— 训练止于 decision_train_time_id_max，评分在密封期
4. D3+ Tier 2 复验（排在 recency 之后，RUNBOOK 顺序不变）
5. D4.5 决定拍完后，最终交付件用 100% 数据重训，只过 D4 转正门禁（机械正确性）
6. D5/D6 不变

⚠️ D4.5 的风险：最终交付件训练在没有任何评估覆盖过的数据上。缓解：与刚被密封期验过的是同一结构，且 D4 覆盖机械正确性。回退：D4.5 任一门禁不过就交决策期那份（它被密封期评过），再不行交当前生产 —— 三层都有落盘产物。

## Tier 1 — 零重训成本，盘上现成

| 候选 | 模型目录 | 公榜真值 | 这一枪问什么 |
|---|---|---:|---|
| `production_slowfast` | `strategies/v3_hybrid` | 0.0041150085 | 基准，必须先打 |
| `mkt_shrunk` | `outputs/candidates/v3_hybrid_mkt_shrunk` | 0.0039977510 | 密封期能不能重现 +2.93% 这个已知差 |
| `mktwe` | `outputs/candidates/v3_hybrid_r480_pb_hist_mktwe` | 0.0039673997 | 第三个标定点 |
| `asset_adapter` | `outputs/candidates/v3_asset_cross_3s480_shrink500` | 0.0039908352 | OOF 说 +1.99%、公榜说 −0.17% —— 密封期站哪边 |
| `r960` | `outputs/candidates/v3_hybrid_r960_pb_hist_mktwe` | 0.0037609312 | 负控制（−5.20%） |
| `xs_shrunk` | `outputs/candidates/v3_hybrid_xs_shrunk` | 0.0035771492 | 负控制（−9.84%） |

⟹ 六个已知公榜真值 + 块级方差 = **这把尺子的检出下限**。它决定 Tier 2 值不值得花重训。

## Tier 2 — 每个一次 3s480 重训

- `mkt323` —— 市场块选列 323（ROADMAP 已写明「回补数据后按原规格复验一次」，+1.09%/3-of-5）
- `v4r_regime` —— V4-R 压缩 market regime（ROADMAP §3.7 唯一保留的原规格复验项）
- `phase_id` —— phase_id 作特征（+1.1%、3/5 折）
- `lag3_lag10` —— lag3+lag10（+0.38%、3/5 折、drop-best 为负）
- `responder_00` —— pure_e/responder_00 辅助（+1.38%、3/4 折、drop-best 为负）

⚠️ 只有 Tier 1 标定出的检出下限低于各自点估计时才开跑。

## 明确不做

- 不生成提交格式 CSV，不打包 zip，不 commit（CLAUDE.md §1.2 / §1.4）
- 不动生产目录和 hybrid_meta.json
- 8/23 之前不打公榜枪
- 不用密封期反复调参：候选清单预注册，每个候选只打一次分
- 不因为看到密封期结果改块数、改门槛、或往清单里加项
- 不把 Tier 2 重训排到 recency 前面
