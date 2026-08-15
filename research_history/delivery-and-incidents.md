# 推理、晋级、打包与工程事故

本文件保存会直接影响交付正确性的工程经验。当前交付动作以 [`../ROADMAP.md`](../ROADMAP.md) 为准。

## 1. 提交包约束塑造了策略结构

评测包只包含策略目录 Python 文件和 `model/`，不能依赖仓库 `src/`。因此生产策略使用同目录模块，
并保留纯 NumPy 推理后端作为 LightGBM 兼容兜底。

这一约束是合理的，但也造成仓库内多个策略都有 `main.py`、`train.py`、`features.py` 等同名模块。
跨策略工具若依赖 `sys.path` 顺序，容易命中 `sys.modules` 中已有的错误模块。

长期规则：跨策略加载一律按文件路径和唯一模块名；测试需要覆盖同一进程连续加载多个策略。

## 2. LightGBM 文本模型的 NumPy 兜底

评测环境 LightGBM 版本未知。仅仅 `import lightgbm` 成功，不能证明它会正确读取 v4 文本模型。
生产模型因此：

1. 无条件解析一份 `NumpyForest`；
2. 尝试加载 LightGBM Booster；
3. 用固定种子合成输入逐元素对拍；
4. 自动模式下不一致则退到 NumPy；显式 LightGBM 模式则直接失败。

解析器锁定的关键语义包括：

- 数值分裂使用 `<=`；
- 缺失值类型和默认方向；
- 分类特征的整数截断和 bitset；
- leaf value 已包含 shrinkage；
- threshold 按 float64 解析和比较。

测试同时覆盖合成批次、真实测试分区、非法模型文本、重复资产和越界资产。

## 3. `INCIDENT`：训练与推理数学等价但不逐位等价

训练端和推理端曾分别使用两个数学等价的截面去均值函数。由于 float32 求和顺序不同，输入相差
一个 ULP；树在阈值附近因此翻到另一叶子，输出差被放大到约 2.85e-03，影响约 0.52% 行。

修复：LGBM 分支训练和推理共用同一个 `cross_sectional_deviation` 实现；Ridge 保持原线性路径。
一致性误差随后降到约 4e-08。

长期规则：对阶跃模型，“数学等价”不构成推理一致性证据，必须逐预测对拍。

## 4. 端到端推理性能：树不是唯一成本

旧测量只计时 `booster.predict`，低估了完整 runner。实际每次只有约 15 行，小批 pandas 取列、
稳健变换、截面去均值和历史状态维护占据主要成本。

2026-08-13 的等价优化包括：

- 整帧一次转 float32，再按缓存位置切列；不支持时永久退回 `.loc`；
- `AssetHistory.transform_online` 使用定槽环形缓冲；
- 探测后使用 `validate_features=False`，兼容旧 LightGBM；
- 旧候选在新代码上逐位不变。

单森林 history 候选的局部测量从约 2.205 ms/call 降到约 0.979 ms/call；加入第二森林后的
`mkt_we` 全量 `predict_total` 记录为 6.23 分钟。当前 `mkt_shrunk` 的最终全量 wall-clock 仍应在
打包前重新测量。

NumPy 双森林兜底估计约 15 分钟，是独立风险，不能用 LightGBM 主路径耗时代替。

## 5. `INCIDENT`：公榜 CSV 与候选 meta 不是同一模型

### 现象

历史公榜高分 CSV 由临时副本通过 `blend_weight=1.0`、`prediction_scale=1.16` 生成；训练候选目录的
meta 仍保存本地占位值 `blend_weight=0.5`、`prediction_scale=0.856`。旧 promotion 会覆盖 scale，
却没有校验 blend weight。

如果直接转正，代码和模型文件都能正常运行，烟测也会通过，但提交的是另一模型。

### 证据

留档 CSV 重算：

- blend 1.0 与已交 CSV 的最大差约 5e-09，只是 CSV 舍入底噪；
- blend 0.5 与已交 CSV 最大差约 0.121，相关性也明显下降。

### 修复

- `PUBLIC_BASELINE` 明确完整生产结构；
- promotion 新增 blend、market、权重和模型数量校验；
- packaging 在烟测前检查 meta；
- staging 可复用但必须核对来源候选；
- 激活采用 incoming + 原子替换 + backup；
- smoke test 使用非退化随机特征和连续 time_id；
- 对 blend/market lambda 做敏感性断言。

长期规则：模型身份是“代码 + meta + 所有模型文件”，不能只看策略名字或预测是否有限。

## 6. Promotion 流程

当前流程：

```text
candidate
→ stage_candidate（复制并重写允许的后处理项）
→ validate_meta
→ NumpyForest / LightGBM 自检
→ 结构敏感性检查
→ 写 promotion_manifest
→ 用户显式 activate
→ production backup + atomic replace
→ 再验证 production
```

2026-08-13 的生产 staging：

- 来源：`outputs/candidates/v3_hybrid_mkt_shrunk`；
- staging：`outputs/promotions/v3_hybrid_s1.16_w1_3seed`；
- production：`strategies/v3_hybrid/model/`；
- 双后端最大差：`2.498001805406602e-16`；
- blend 和 market lambda 敏感性检查均非零；
- 当前生产文件 hash 与 staging manifest 一致。

### 当前生产 hash

| 文件 | SHA-256 |
|---|---|
| `baseline_model.json` | `54dc6afba78b16cb47ef06f1392901690b4161d93c37ce0357cda2cdf31ed2fd` |
| `hybrid_meta.json` | `777bbeaad1ccc6460670a8b7354d1fb9b739e81824a676938b611361f1016d20` |
| `lgbm_seed2026.txt` | `582a1e93117b1b1105b06899bdbab76e6c4d4f58643ff40101905953b4866168` |
| `lgbm_seed2027.txt` | `9bdc8a5300a607c710199b0836a97b049d8543001128a1e3f65bdf84cfdcb195` |
| `lgbm_seed2028.txt` | `5f57f744f48b352cb3841d05d070c74ac1d838f5fb349c7ba9cac8904b01557c` |
| `lgbm_market_seed2026.txt` | `c055f119b5539380c8e7b929f7bee0fe540657ae2e54e7c67708bef2ea86552c` |
| `lgbm_market_seed2027.txt` | `591e13692eb283a69710050484545bf4deae56f6b88b2207fe08e8f0ff87458a` |
| `lgbm_market_seed2028.txt` | `91dcef3a63fb6b8818d7757dedef8046e07d037023b12fce7dcae84401bce4f9` |

Hash 是 2026-08-13 的身份记录；未来合法转正后应更新本表，而不是继续称其为当前值。

## 7. 私榜打包

`scripts/make_submission.py` 只读复制策略 Python 文件（排除训练模块）和模型目录，随后：

1. 校验生产 meta；
2. 运行非退化 smoke test；
3. 排除缓存、pyc 和 promotion manifest；
4. 生成 zip。

用户负责执行打包；AI 不生成 zip。打包后还应运行 `scripts/audit_submission_zip.py`，检查：

- 必需文件；
- 禁止文件（尤其 `train.py`）；
- 模型大小和数量；
- 可导入性；
- 包内模型 meta；
- 路径布局。

## 8. Runner 超时语义

本地 `timeseries_api/runner.py` 在 `predict()` 返回后才比较 elapsed time，因此是“事后判超时”，
不是可中断的硬超时。超时步骤可能已经推进模型内部历史状态，然后输出被替换为零。

主办方文件只读，本项目不修改它；最终风险验收应在独立进程外层设置 wall-clock 限制，并明确区分：

- runner 的 `predict_total_seconds`；
- 完整进程 wall clock；
- 模型初始化时间；
- 单步尾延迟；
- LightGBM 主路径与 NumPy 兜底。

## 9. 数据更新审计

主办方原始目录只读。2026-08-12 已保存数据快照：

- `outputs/data_audits/data_release_20260812.json`

收到新包时先比较文件、行数、schema 和 SHA-256。只有 audit 明确指出 train split 变化，
`scripts/retrain_extended.py` 才允许进入执行路径。该门禁防止把“文件日期变化”误当成“新增训练数据”。
