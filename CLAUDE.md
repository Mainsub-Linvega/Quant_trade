# CLAUDE.md — 2026 量化交易研究大赛 · AI 协作作战手册

本文件是每个 AI 会话的操作总纲。开始任何工作前，先读这里，再动手。
（人类协作者：`docs/` 是主办方原始赛题/数据说明，本文件是在其之上的**工作约定**。）

---

## 1. 任务本质（一句话）

用 **323 个匿名特征**，为 **15 个匿名标的** 的每个时点预测一个连续 `target`；
预测越接近真值（按隐藏 `weight` 加权）分越高；**全程严禁使用任何未来信息**。

- 评分指标：**加权零均值 R²** = `1 − Σwᵢ(yᵢ−ŷᵢ)² / Σwᵢyᵢ²`
- 基准：全预测 0 得 **0 分**。模型必须显著优于全 0 才有正分。
- `target` = 当前时点之后某未来窗口内的"风险调整表现"。正负号=方向，绝对值=强度。

## 2. 数据事实（已从 parquet / manifest 核实，勿凭空假设）

| 字段 | 训练集 | 测试集 | 说明 |
|---|---|---|---|
| `row_id, time_id, asset_id` | ✅ | ✅ | 索引。time_id 越大越靠后；asset_id 共 15 个 |
| `feature_000..322`（323 个） | ✅ | ✅ | 唯一可用输入。匿名，无语义，只能靠统计挖掘 |
| `weight` | ✅ | ❌ | 主办方给定的一列，**不是你算的**。仅训练/验证用 |
| `responder_00..46`（47 个） | ✅ | ❌ | 未来构造的辅助目标。**绝不可当输入特征**（=未来泄露）；只能当辅助训练目标 |
| `target` | ✅ | ❌ | 要预测的答案 |

- 规模：训练 1322 万行 / 9 分区；测试 322 万行 / 3 分区。float32，zstd 压缩。
- 分区按连续 time_id 切分；**分区顺序 = 时间顺序**，验证必须尊重它。

## 3. 铁律（违反即无效或泄露，不可协商）

1. **无未来信息**：任何预处理统计（分位数、均值、标准化参数、特征选择）只能在**训练期**拟合，禁止用验证/测试期的分布、缺失模式、responder、target 反推当前预测。
2. **responder / weight / target 不进测试期输入**——测试时它们根本不存在。
3. **推理端约束**：官方评测 4 核 / 12GB 内存 / 无 GPU / 无网络。模型要轻、`predict()` 要快，否则超时→该 time_id 预测被置 0。
4. **交付形态**：私榜提交 zip 内 `main.py` 必须在根目录，`Model` 类 `__init__` 一次、按递增 time_id 循环 `predict(test)`，返回长度=len(test) 的一维有限浮点数组，顺序同输入行。推理只依赖 NumPy/Pandas，不依赖 sklearn。
5. **靠本地验证，不靠公榜试错**：公榜每天最多 5 次有效提交，私榜策略共 10 次。改动好坏一律以本地 `walk_forward` 的加权 R² 为准。

## 4. 仓库结构

```
data/                          # 20G，已 gitignore，勿入库
examples/
  baseline_strategy/           # ★ 当前主力策略（v1 = Ridge 截面 baseline）
    train.py                   # 训练 → model/baseline_model.json
    main.py                    # Model 类（最终交付推理逻辑）
    walk_forward.py            # 多窗口时序验证
    walk_forward_history.py    # 因果 lag/rolling 特征的验证
    history_features.py        # 因果历史特征（★已备好但尚未接入主模型）
    model/baseline_model.json  # 训练产物（已入库）
  {data_io,random_strategy,linear_window_strategy}/   # 参考示例
timeseries_api/                # 官方本地顺序推理 runner
docs/                          # 赛题与数据说明（主办方原文）
outputs/experiments/           # walk_forward 小结（入库）；大 csv/zip 已忽略
```

## 5. 版本管理约定

- 每一版策略 = 一次里程碑，用 tag 标记：`git tag -a baseline-vN -m "..."`。当前起点 `baseline-v1`。
- 较大改动在新分支上做（如 `baseline-v2`），本地验证达标后再合回 `main` 并打 tag。
- **不入库**：`data/`、`.venv/`、`outputs/*.csv`、`outputs/*.zip`、`__pycache__/`、`nvim.log`（见 `.gitignore`）。
- 提交信息用中文，结尾附 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- 未经用户明确要求，不 push、不打包提交、不改 `docs/`。

## 6. 常用命令

```bash
# 训练（生成 model/baseline_model.json）——限核避免抢占
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python examples/baseline_strategy/train.py

# 时序验证：多训练窗口对比加权 R²（结果写 outputs/experiments/）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python examples/baseline_strategy/walk_forward.py

# 验证因果 lag/rolling 特征
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python examples/baseline_strategy/walk_forward_history.py

# 生成公榜提交（约 21.7 万次 API 调用，勿设过紧的 per-step 超时）
.venv/bin/python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir examples/baseline_strategy \
  --output outputs/submission.csv
```

## 7. 每次改进的标准循环（AI 每个会话遵循）

1. **明确目标**：本次要验证的假设是什么（新特征？调参？新模型？）。
2. **在分支上改**：动 `train.py` / 新增特征模块；同步保证 `main.py` 推理逻辑与训练口径一致（预处理、特征顺序必须完全对应）。
3. **本地验证**：跑 `walk_forward*.py`，对比改动前后的加权 R²。**只认时序验证集的稳定提升**，训练集提升无意义。
4. **一致性自检**：确认没有未来泄露（统计只在训练期拟合）、`predict()` 返回形状/顺序正确、推理够快。
5. **汇报**：把加权 R² 前后对比、代价（耗时/内存/复杂度）如实告诉用户，给出保留或回滚建议。
6. **定版**：用户认可后再 commit + `git tag baseline-vN`；必要时生成 submission。

## 8. 改进方向储备（按性价比）

- **接入 `history_features.py` 的因果 lag/rolling 特征**（已验证、未接主模型）——最省事的下一步。
- 特征交互 / 降维（PCA）；用 LightGBM 看特征重要性再回喂线性模型。
- 把 47 个 responder 当**辅助目标**做多任务或两段式（先预测 responder 再组合）。
- 按 asset 分组建模；滚动重新标准化以适应非平稳。
- 预测后处理：缩放 + 限幅（baseline 已用 `prediction_scale/clip`），保守缩放常能提升加权 R²。

## 9. 交给人类自己跑时

完整循环就是 §6 的四步：`train → walk_forward → run_timeseries_api → git commit/tag`。
判断标准始终是 `walk_forward` 的加权 R²；每天公榜 5 次、私榜 10 次的额度要省着用。
