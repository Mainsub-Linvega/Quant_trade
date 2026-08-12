# Responder 结构分析（`responder_analysis`)

> 这是同期标签结构报告，不是可部署提分证据。

- 行数：13,227,692
- time_id 截面数：888,315
- 确定性 Spearman 样本：661,385 行
- 聚类数：24
- PCA 达到 90% 累计解释方差：10 个成分

## 与 target 同期关系最强的列

| responder | Pearson | Spearman | market | cross-section | IC | cluster |
|---|---:|---:|---:|---:|---:|---:|
| responder_03 | 0.8169 | 0.8080 | 0.8540 | 0.7384 | 0.6826 | 23 |
| responder_28 | 0.6935 | 0.6780 | 0.7033 | 0.6699 | 0.6495 | 18 |
| responder_02 | 0.5691 | 0.5531 | 0.6073 | 0.4937 | 0.4504 | 22 |
| responder_29 | 0.5554 | 0.5370 | 0.5639 | 0.5347 | 0.5141 | 20 |
| responder_18 | 0.4348 | 0.6272 | 0.5056 | 0.3223 | 0.4670 | 6 |
| responder_19 | 0.3963 | 0.5846 | 0.4614 | 0.2941 | 0.4334 | 6 |
| responder_04 | 0.3938 | 0.3897 | 0.4158 | 0.3457 | 0.3297 | 17 |
| responder_11 | 0.3935 | 0.6319 | 0.4366 | 0.2938 | 0.4794 | 8 |
| responder_30 | 0.3905 | 0.3736 | 0.3986 | 0.3710 | 0.3536 | 16 |
| responder_17 | 0.3861 | 0.5338 | 0.4539 | 0.2783 | 0.3990 | 5 |
| responder_10 | 0.3613 | 0.5400 | 0.4094 | 0.2582 | 0.4102 | 11 |
| responder_01 | 0.3583 | 0.3433 | 0.3920 | 0.2996 | 0.2711 | 21 |

## 稳定族群

| cluster | representative | members |
|---:|---|---|
| 1 | responder_37 | responder_31, responder_32, responder_33, responder_34, responder_35, responder_36, responder_37, responder_38, responder_39 |
| 2 | responder_41 | responder_40, responder_41 |
| 3 | responder_42 | responder_42 |
| 4 | responder_16 | responder_15, responder_16 |
| 5 | responder_17 | responder_17 |
| 6 | responder_18 | responder_18, responder_19 |
| 7 | responder_20 | responder_20 |
| 8 | responder_11 | responder_11, responder_12 |
| 9 | responder_13 | responder_13 |
| 10 | responder_09 | responder_08, responder_09 |
| 11 | responder_10 | responder_10 |
| 12 | responder_14 | responder_07, responder_14, responder_21, responder_22, responder_23, responder_24 |
| 13 | responder_46 | responder_25, responder_26, responder_27, responder_44, responder_45, responder_46 |
| 14 | responder_43 | responder_43 |
| 15 | responder_06 | responder_06 |
| 16 | responder_30 | responder_30 |
| 17 | responder_04 | responder_04 |
| 18 | responder_28 | responder_28 |
| 19 | responder_05 | responder_05 |
| 20 | responder_29 | responder_29 |
| 21 | responder_01 | responder_01 |
| 22 | responder_02 | responder_02 |
| 23 | responder_03 | responder_03 |
| 24 | responder_00 | responder_00 |

## 裁决

阶段 A 只冻结族群与候选代表；是否可预测、是否能补 target 残差由阶段 B/C 决定。
