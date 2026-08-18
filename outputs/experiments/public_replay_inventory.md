# 公榜分数离线复算（inventory）

> 第一个产出必须是「离线复算 == 公布分数」。对不上先修复算器，不解释现象 —— 否则后面的拆解是在解释 bug

## 配对清单

- 盘上 CSV **21** 份，全部已归属（sha256 4 / ledger 原文 2 / 推断 15 / 未归属 0）
- ledger 有公榜分数的行：**31**，其中 10 行没有对应的逐行 CSV
- 只剩指纹、CSV 已删：**14** 份，其中 **8** 份模型已不在仓库 ⟹ 永久不可复算

| CSV | 归属 | ledger 日期 | 公布分数 | 行数 | row_id 覆盖 |
|---|---|---|---:|---:|---:|
| `submission_asset_cross.csv` | inferred | 2026-08-17 | 0.0039908352 | 3,217,458 | 100.0000% |
| `submission_hist_c80_s116.csv` | inferred | 2026-08-11 | 0.003252246 | 3,217,458 | 100.0000% |
| `submission_hist_r160_s116.csv` | inferred | 2026-08-11 | 0.0029039065 | 3,217,458 | 100.0000% |
| `submission_hist_r320_s116.csv` | inferred | 2026-08-11 | 0.0031385676 | 3,217,458 | 100.0000% |
| `submission_hist_r480_s090.csv` | inferred | 2026-08-11 | 0.003056081 | 3,217,458 | 100.0000% |
| `submission_hist_r480_s116.csv` | inferred | 2026-08-11 | 0.0032523499 | 3,217,458 | 100.0000% |
| `submission_market_s130.csv` | inferred | 2026-08-10 | 0.0014991067 | 3,217,458 | 100.0000% |
| `submission_mix2_r480.csv` | inferred | 2026-08-10 | 0.0025767425 | 3,217,458 | 100.0000% |
| `submission_mkt_moderate.csv` | inferred | 2026-08-13 | 0.0039867107 | 3,217,458 | 100.0000% |
| `submission_mkt_shrunk.csv` | inferred | 2026-08-13 | 0.003997751 | 3,217,458 | 100.0000% |
| `submission_mkt_shrunk_slowfast.csv` | ledger_text | 2026-08-17 | 0.0041150085 | 3,217,458 | 100.0000% |
| `submission_mktwe_s116.csv` | inferred | 2026-08-13 | 0.0039673997 | 3,217,458 | 100.0000% |
| `submission_phasebal_r480_s116.csv` | inferred | 2026-08-10 | 0.0026330806 | 3,217,458 | 100.0000% |
| `submission_r320_s116.csv` | sha256 | 2026-08-10 | 0.0025651 | 3,217,458 | 100.0000% |
| `submission_r480_s116.csv` | sha256 | 2026-08-10 | 0.0025821304 | 3,217,458 | 100.0000% |
| `submission_r960_pb_hist_mktwe.csv` | inferred | 2026-08-13 | 0.0037609312 | 3,217,458 | 100.0000% |
| `submission_replace_r80_s116.csv` | sha256 | 2026-08-09 | 0.0023682898 | 3,217,458 | 100.0000% |
| `submission_replace_s116.csv` | sha256 | 2026-08-09 | 0.0024872338 | 3,217,458 | 100.0000% |
| `submission_slowfast_runner.csv` | inferred | 2026-08-18 | 0.0041150085 | 3,217,458 | 100.0000% |
| `submission_xs_moderate.csv` | ledger_text | 2026-08-14 | 0.0039260128 | 3,217,458 | 100.0000% |
| `submission_xs_shrunk.csv` | inferred | 2026-08-13 | 0.0035771492 | 3,217,458 | 100.0000% |

## 只剩指纹（CSV 已删）

| 文件 | 公榜分数 | 模型 | 能否靠重跑补回 |
|---|---:|---|:--:|
| `baseline_submission.csv` | 0.00119088 | baseline_v1 | 重跑可补 |
| `submission.csv` | 0.00151886 | legacy_a2e6 | ❌ 模型已不在仓库 |
| `submission_scale05.csv` | 0.00128602 | legacy_a2e6 | ❌ 模型已不在仓库 |
| `submission_scale05_8dp.csv` | 0.00128602 | legacy_a2e6 | ❌ 模型已不在仓库 |
| `submission_v3_s113.csv` | 0.00186805 | legacy_a2e6 | ❌ 模型已不在仓库 |
| `submission_v3_s113_8dp.csv` | 0.00186805 | legacy_a2e6 | ❌ 模型已不在仓库 |
| `submission_v2_s080.csv` | 0.00150852 | legacy_a5e5 | ❌ 模型已不在仓库 |
| `submission_v2_s080_8dp.csv` | 0.00150852 | legacy_a5e5 | ❌ 模型已不在仓库 |
| `submission_v2_s120.csv` | 0.0011693833 | legacy_a5e5 | ❌ 模型已不在仓库 |
| `submission_strict_scale113.csv` | 0.00187232 | strict_ridge | 重跑可补 |
| `submission_strict_scale092.csv` | 0.001805154 | strict_ridge | 重跑可补 |
| `submission_hybrid_base0856.csv` | None | v3_hybrid_w050 | 重跑可补 |
| `submission_hybrid_scale090.csv` | 0.0021381 | v3_hybrid_w050 | 重跑可补 |
| `submission_hybrid_scale130.csv` | 0.0022857726 | v3_hybrid_w050 | 重跑可补 |

## 8/23 待办

1. `--labels <回补数据>` 重跑本脚本；**先看「复现」那一列全绿**，再看任何拆解；
2. `inferred` 归属靠复算分数落在指派的 ledger 行上来验证；
3. 复现通过后按时期/分块/资产拆解，出「本地 Δ% → 公榜 Δ%」实测斜率。

