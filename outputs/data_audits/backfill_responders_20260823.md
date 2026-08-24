# 8/23 回补包 responder 列核查（`check_backfill_responders`）

审计来源：`outputs/data_audits/data_release_20260823.json`

train 分区共 12 个；本次 added/modified **3** 个

| 文件 | 列数 | responder | target | weight |
|---|---:|---:|:--:|:--:|
| `train_partition_009.parquet` | 375 | **47** | ✅ | ✅ |
| `train_partition_010.parquet` | 375 | **47** | ✅ | ✅ |
| `train_partition_011.parquet` | 375 | **47** | ✅ | ✅ |

## 判定：`backfill_has_responders`

回补的 train 文件**全部带 responder 列** ⟹ **触发 2026-08-22 收口的 responder 四项 `REJECTED` 的统一重开条件**，按**原规格**各复验一次（不得借机改设计）。

需按原规格复验的产物：
`responder_stage_c_fill`, `responder_selection_probe`, `nn_capacity_ladder_respsel`, `responder_reaudit_20260814`
