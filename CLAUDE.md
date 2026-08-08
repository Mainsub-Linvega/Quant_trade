# CLAUDE.md

## 绝对禁止

1. 不删除、不移动、不重命名任何文件。需要清理时列出清单交给我执行。
2. 不 git commit / push / tag。
3. 不修改 `docs/` `examples/` `timeseries_api/`（主办方原文，只读）。
4. 不生成公榜提交，不打包 zip（打包走 `scripts/make_submission.py`，由我执行）。
5. AI 不得往「已核实的事实」里加条目。

## 已核实的事实

（只写我本人亲手验证过的，注明日期。）

- 2026-07-24 市场共同分量占 target 方差 73%（只在 p008 测过，待在 9 分区上复验）
- 2026-07-24 分区级加权均值符号随机（5 正 4 负），漂移不可预测
- 2026-07-24 train.py 特征选择与预处理均在训练段拟合，无泄漏
- 2026-08-08 share 的 73% 此前只在最后一个分区算过；9 个分区全算了一遍，
范围 0.626~0.732，均值 68.4%±3.4%，p008=0.732 恰是最高的那个
（证据 outputs/experiments/mt_diagnostics.md）
- 2026-08-08 上面「符号随机」那条复现无误，但它被错误解读成「m_t 不可预测」。
符号随机说的是分区级长期漂移没有方向，不等于每个 time_id 的 m_t 不可预测。
实测 m_t 弱可预测：样本外 R²≈0.0018（8/10 折为正），而它占 target 方差 72%，
两者相乘 → 当前 85% 的分数来自这一块
（证据 outputs/experiments/mt_predictability.md、ab_meandev.md）

## 当前状态

- 公榜 0.00151886（8/07 提交，scale=0.6424）；公榜第一 0.00520002
- 本地同配置 0.00152539（滚动 10 折）—— 本地框架与公榜偏差 0.4%~4%，详见 NOTES.md
- 本地最优配置（**尚未上榜**）：200 特征 / alpha=5e5 / scale=0.759 → 0.00175839
- 提交台账见 `experiments/ledger.csv`；实验产物见 `outputs/experiments/`

## 评分

Score = 1 − Σw(y−ŷ)² / Σwy²。全零预测得 0 分。
唯一实现在 `src/metric.py`，不得另写。

## 工作方式

- 每次改动前，先说明要验证的假设是什么，等我确认。
- 改完跑本地验证，把改动前后的数字都报给我。
- 改动涉及预处理 / 推理口径时，必须跑 `scripts/check_consistency.py`。
- 不确定的事情问我，不要自己假设，不要自己补全。

## 伤疤清单

- 2026-07-23 AI 把自己的策略写进 examples/（主办方目录），导致后续误判文件归属。
- 2026-07-23 walk_forward 报告写 Accepted: True，但特征从未接入 train.py，
  报告结论与实际代码无机械联系。
- 2026-07-26 发现主办方 `README.md` 与 `examples/linear_window_strategy/train.py`
  曾被 AI 写入过（mtime 07-22，非发布包解压时间 07-01），原始发布包已丢失，
  改动内容无法核验。README 已尽量还原；linear_window train.py 保持现状，
  待重新下载发布包或 8/23 重发包后 diff 核验。
