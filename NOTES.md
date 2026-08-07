# NOTES.md — 会过期的工作笔记

（规则类内容在 CLAUDE.md；优先级在 ROADMAP.md。本文件随时改写，不保证常新。）

## 仓库结构（2026-07-26 重构后）

```
data/                    # 主办方数据，20G，只读，gitignore
docs/                    # 主办方赛题/数据说明，只读
examples/                # 主办方示例，只读
timeseries_api/          # 主办方本地 runner，只读（vendoring：入库以便 8/23 更新时 git diff）

src/                     # 离线专用公共库（提交包不含）
  metric.py              #   加权零均值 R² —— 全项目唯一实现
  io.py                  #   FEATURE_COLUMNS / train_files / load_time_sample
  validation.py          #   partition_folds；P0 的 rolling_time_folds 占位

strategies/v1_ridge/     # 自包含：整个目录 = 提交包内容
  features.py            #   ★ 预处理+推理唯一实现，train.py 和 main.py 都 import 它
  train.py               #   训练（可 import src/；main.py 不可以）
  main.py                #   交付推理件，只依赖 numpy + 同目录 features.py
  model/baseline_model.json

experiments/             # 离线实验脚本 + 台账
  walk_forward.py        #   训练窗口对比
  walk_forward_history.py#   因果历史特征验证（依赖先跑 walk_forward.py）
  history_features.py    #   因果 lag/rolling 特征构造（未接入主模型）
  ledger.csv             #   ★ 提交台账：本地分 vs 公榜分校准

scripts/
  check_consistency.py   #   ★ 断言 train 与 main 预测逐元素一致（改口径必跑）
  make_submission.py     #   只读打包 main.py+features.py+model/ → zip

outputs/                 # 生成产物，csv/zip gitignore；experiments/ 小结入库
```

## 数据事实（从 parquet/manifest 核实）

| 字段 | 训练 | 测试 | 说明 |
|---|---|---|---|
| row_id, time_id, asset_id | ✅ | ✅ | 索引；15 个 asset；分区顺序 = 时间顺序 |
| feature_000..322（323 个） | ✅ | ✅ | 唯一可用输入，匿名 |
| weight | ✅ | ❌ | 主办方给定，仅训练/验证用 |
| responder_00..46（47 个） | ✅ | ❌ | 未来构造，绝不可当输入（=泄露），只能当辅助目标 |
| target | ✅ | ❌ | 预测目标 |

- 训练 1322 万行 / 9 分区；测试 322 万行 / 3 分区。float32 + zstd。
- E_w[y²] = 1.1757，sd(y) = 1.0784。
- 推理端约束：4 核 / 12GB / 无 GPU / 无网络；超时该 time_id 置 0。

## 常用命令

```bash
# 训练（生成 strategies/v1_ridge/model/baseline_model.json）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python strategies/v1_ridge/train.py

# 训练/推理口径一致性（改预处理后必跑）
.venv/bin/python scripts/check_consistency.py

# 时序验证
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward_history.py

# 本地全量顺序推理（≈21.7 万次 predict 调用）
.venv/bin/python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir strategies/v1_ridge --output outputs/submission.csv

# 私榜提交包
.venv/bin/python scripts/make_submission.py --strategy v1_ridge
```

## 当前模型参数（v1, 2026-08-07 更新）

```
算法              加权 Ridge（lsqr, alpha=2e6），400 设计列 = 200 原始 + 200 截面去均值
prediction_scale  auto     ← 验证期闭式最优 a*=Σwyf/Σwf²（≈0.64）
prediction_clip   0.5
feature_count     200/323  ← P1 测试 323 反而微降 -5.2e-6，保持 200
train_partitions  4        ← walk_forward 推荐 4，已统一
sample_modulo     5        ← 意外的好处：可能去掉了大半 target 窗口重叠，先别改
intercept         ~0.004   ← P1 测试置 0 无影响，保持原值
NaN 预处理        nanquantile ← 修复了 NaN→0 污染统计量的 bug
```

## P1 拆分测试结果（2026-08-07, 基线=window4+nanquantile, score=0.00089825）

| 配置 | validation_score | vs 基线 | 结论 |
|---|---|---|---|
| 基线（200特征, scale=0.5, 真截距） | 0.00089825 | — | — |
| 只改截距=0 | 0.00089825 | ±0 | 无影响 |
| 只改 323 特征 | 0.00089307 | -5.2e-6 | 拖后腿 |
| 只改 auto-scale | 0.00094467 | +4.6e-5 | 有效，保留 |

## 时间线

- 8/23 公榜截止 + 标签回补（主办方重发包 → 对 docs/examples/timeseries_api 跑 git diff）
- 8/31 策略文件提交截止（私榜策略共 10 次）
- 9 月 实盘评估；10 月 答辩（交付物最好逐行自己重写，验收标准：复现 0.00119）
