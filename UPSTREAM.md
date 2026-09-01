# 主办方原文与数据（不随本仓库分发）

本仓库是**个人参赛记录**，与主办方无关。以下内容的版权归 2026 量化交易研究大赛主办方，
本仓库不予转载，因此克隆下来的目录树里**没有**它们：

| 路径 | 内容 | 本仓库是否依赖 |
|---|---|---|
| `data/` | 比赛数据（约 20 GB parquet + `manifest.json` + `sample_submission.csv`） | 训练与实验全部依赖 |
| `docs/` | `competition_description.md`、`data_description.md` | 仅被代码注释引用 |
| `examples/` | 主办方示例策略（`random_strategy`/`linear_window_strategy`/`lightgbm_baseline`/`data_io`） | 不依赖 |
| `timeseries_api/` | 官方顺序推理 runner（`runner.py`、`run_timeseries_api.py` 等 5 个文件） | 交付验证脚本依赖 |

`docs/`、`examples/`、`timeseries_api/` 曾经在版本控制里（截至提交 `182348a`），
2026-09-01 公开发布整改时移出跟踪并加入 `.gitignore`；历史提交中仍可检索到。

## 版本口径

数据与文档来自两个发布包，本仓库通过文件级审计产物记录其身份（这些审计 JSON **入库**）：

| 发布包 | manifest `version` | train | test | 审计产物 |
|---|---|---|---|---|
| 主公开包 | `public_release_20260630` | 9 文件 / 13,227,692 行 / `time_id` 0–888,479 | 3 文件 / 3,217,458 行 / 888,480–1,105,919 | `outputs/data_audits/data_release_20260812.json` |
| 8/23 标签回补包 | 同上 + 3 个新 train 分区 | 12 文件 / 16,445,150 行 / 0–1,105,919 | 不变 | `outputs/data_audits/data_release_20260823.json` |

回补包做的事：把公榜窗口（`time_id` 888,480–1,105,919）的标签回填 —— 3,217,458 行与
`data/test` 逐行 `row_id` 相同、323 个特征逐 bit 相同，新增 `weight` / `target` / `responder_00..46`。
证据见 `outputs/data_audits/backfill_responders_20260823.md`。

数据规模（`data/manifest.json`）：15 个匿名标的、323 个特征、47 个 responder。

## 如何放回

从主办方处取得发布包后，按原始目录名放回仓库根目录：

```text
Quant_trade/
├── data/              # 主公开包的 data/（train/ test/ manifest.json sample_submission.csv）
├── docs/
├── examples/
└── timeseries_api/
```

放回后建议先做一次文件级审计，确认拿到的包与本仓库结论所依据的是同一份：

```bash
.venv/bin/python scripts/audit_data_release.py \
  --data-root data \
  --output outputs/data_audits/my_release.json \
  --baseline outputs/data_audits/data_release_20260812.json
```

`comparison.changed` 为 false，说明拿到的与本仓库结论所依据的是逐文件同一份。

回补包不放进 `data/`，而是通过环境变量指给派生数据根的构建脚本：

```bash
export QUANT_BACKFILL_ROOT=/path/to/public_release_20260823/data
.venv/bin/python scripts/build_extended_data_root.py --roles extended_full --execute
```

（生产模型 `strategies/v3_hybrid/model/` 训练所用的根就是 `extended_full`，
其成员 sha256、训练段边界与行数记在 `outputs/data_roots/extended_full/root_identity.json`。）

## 缺失时会发生什么

- 缺 `timeseries_api/`：`scripts/verify_delivery_runtime.py`、`scripts/measure_harness_memory.py`、
  `scripts/validate_ridge_candidate.py` 在导入时抛出指向本文件的 `SystemExit`，不是裸 `ImportError`。
- 缺 `data/`：所有训练、OOF 与实验脚本都跑不了；`strategies/v3_hybrid/model/` 里的
  模型权重与全部 `outputs/**` 产物仍可离线阅读与校验。
- 缺 `docs/`：只影响追溯代码注释里的行号引用（如 `verify_delivery_runtime.py` 引的评测环境硬约束
  4 核 / 12 GB）。不影响任何执行路径。
