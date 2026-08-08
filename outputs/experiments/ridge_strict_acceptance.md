# Strict Ridge 验收结果

结论：`accept_strict_solver_candidate`。正式模型未覆盖。

| 门禁 | 结果 |
|---|---:|
| base_grid | PASS |
| half_offset_grid | PASS |
| thread_reproducibility | PASS |
| train_inference_consistency | PASS |
| sequential_inference | PASS |
| configuration_match_except_solver | PASS |
| candidate_converged | PASS |

- base-grid pooled Δ：+1.866e-06
- half-offset pooled Δ：+1.431e-06
- 线程预测漂移：4.805e-05 → 4.470e-07（改善 107.5×）
- 候选训练/推理最大差：1.118e-07
- 完整顺序推理：3,217,458 行 / 214,538 次调用，候选耗时 87.00s，非法预测 0，clip 0 行。
