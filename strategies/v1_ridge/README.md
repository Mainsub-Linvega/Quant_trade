# v1_ridge — Robust Ridge Cross-Section Baseline

本目录自包含：**整个目录（去掉 train.py）= 私榜提交包内容**。

模型只使用当前可见的 `feature_*`：

- 最近三个训练分区，按完整 `time_id` 抽样；
- 训练期分位数裁剪和中位数/IQR 标准化；
- 按加权单变量相关性选择 200 个特征；
- 原始特征加当前时点截面偏离特征；
- 加权 Ridge、保守预测缩放和限幅；
- 严格按时间前推的最后分区验证。

## 文件分工

- `features.py` — 预处理与线性推理的**唯一实现**，train.py 和 main.py 都 import 它。
  改任何口径只改这里，改完必跑 `scripts/check_consistency.py`。
- `train.py` — 离线训练，可依赖仓库根的 `src/`（sklearn 也只在这里用）。
- `main.py` — 交付推理件，只依赖 numpy + 同目录 `features.py`，**绝不 import src/**。
- `model/baseline_model.json` — 训练产物（入库，用于复现公榜 0.00119088）。

## 命令（均从项目根目录执行）

训练：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python strategies/v1_ridge/train.py
```

训练/推理口径一致性检查：

```bash
.venv/bin/python scripts/check_consistency.py
```

时序验证（脚本在 `experiments/`，结果写 `outputs/experiments/`）：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/walk_forward_history.py
```

生成公榜 CSV：

```bash
.venv/bin/python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir strategies/v1_ridge \
  --output outputs/baseline_submission.csv
```

本地完整测试会进行约 21.7 万次 API 调用。不要自行设置过紧的
`--per-step-timeout-seconds` 配合 `zero_remaining`；偶发的系统调度抖动可能让后续预测全部被置零。

私榜提交包：

```bash
.venv/bin/python scripts/make_submission.py --strategy v1_ridge
```

生成的 zip 内 `main.py` 位于根目录，含 `features.py` 与 `model/baseline_model.json`；
推理只依赖 NumPy/Pandas，不依赖 sklearn。
