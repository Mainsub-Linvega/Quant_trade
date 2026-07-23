# Robust Ridge Cross-Section Baseline

这个目录是一条可以直接训练、验证、生成公榜提交并打包私榜策略的 baseline。

模型只使用当前可见的 `feature_*`：

- 最近三个训练分区，按完整 `time_id` 抽样；
- 训练期分位数裁剪和中位数/IQR 标准化；
- 按加权单变量相关性选择 200 个特征；
- 原始特征加当前时点截面偏离特征；
- 加权 Ridge、保守预测缩放和限幅；
- 严格按时间前推的最后分区验证。

从项目根目录训练：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  examples/baseline_strategy/train.py
```

比较 2/3/4/6 个训练窗口的三折时间验证：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  examples/baseline_strategy/walk_forward.py
```

结果会写入 `outputs/experiments/walk_forward_windows.{json,csv,md}`。

验证严格因果的每标的 lag/rolling 特征：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  examples/baseline_strategy/walk_forward_history.py
```

生成公榜 CSV：

```bash
.venv/bin/python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir examples/baseline_strategy \
  --output outputs/baseline_submission.csv
```

本地完整测试会进行约 21.7 万次 API 调用。不要自行设置过紧的
`--per-step-timeout-seconds` 配合 `zero_remaining`；偶发的系统调度抖动可能让后续预测全部被置零。

私榜提交 ZIP 中必须让 `main.py` 位于压缩包根目录，同时包含 `model/baseline_model.json`。
推理只依赖 NumPy/Pandas，不依赖 sklearn。
