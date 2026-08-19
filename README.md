# 2026 量化交易研究大赛公开发布包

本目录是面向参赛者的公开 release。数据、文档和示例代码均已匿名化。

## 目录结构

```text
data/
  manifest.json
  train/train_partition_*.parquet
  test/test_partition_*.parquet
  sample_submission.csv
docs/
  competition_description.md
  data_description.md
examples/
  data_io/
  random_strategy/
  linear_window_strategy/
timeseries_api/
  runner.py
  run_timeseries_api.py
  example_main.py
  main.py
  README.md
```

## 本地 Time-Series API 验证

```bash
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir timeseries_api \
  --output /tmp/example_submission.csv
```

## 结构信号严格 OOF 筛选（研究用途）

以下命令不会生成提交文件，也不会修改生产模型。它复用生产等效 OOF，分别检查集合级 market、
截面 rank/tail 与按当前残差重新选列的 XS 信号：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  experiments/structural_signal_screen.py \
  --label structural_signal_screen_1s160
```

结果先按 1 seed × 160 rounds 门禁筛选；只有通过的臂才允许升级到 3 seeds × 480 rounds 确认。

完整真实时间流上的隔离 temporal family（研究用途）：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  experiments/temporal_multiscale.py \
  --arms baseline x1_rank x2_change_rank f_lags f_changes f_volatility f_trend \
  --sample-modulo 5 --sampling phase_balanced --train-window 78960 \
  --lgbm-rounds 160 --lgbm-seeds 1 \
  --label temporal_change_families_1s160
```
