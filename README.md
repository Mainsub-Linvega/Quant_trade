# Quant_trade — 2026 量化交易研究大赛参赛记录

一个低信噪比多标的时序预测赛题的**完整决策链**：从第一个 Ridge 基线到最终私榜交付件，
连同每一次否决的口径、每一次事故的根因一起留档。

赛题：给定 15 个匿名标的、323 个匿名特征的多标的时序数据，按 `time_id` **顺序**推理，
预测风险调整目标 `target`，指标是加权零均值 R²。约束：4 核 / 12 GB / 无网络 /
总耗时有硬预算，超时的 `time_id` 一律按预测 0 计。

这个仓库特别的地方不在模型，而在于它把**整条决策链连同失败一起留了下来**：35 条实验台账、
478 份实验产物、每一次否决的口径与证据、以及 19 条从真实事故里长出来的「长期伤疤规则」。

## 结果

数值一律以产物为准，本节只给指针（避免同一个数字在多处手工维护）：

| | 值 | 出处 |
|---|---|---|
| 首个公榜基线（Ridge 截面，2026-07-23） | 0.00119088 | [`experiments/ledger.csv`](experiments/ledger.csv) |
| 公榜最好（2026-08-21，long512） | 0.0041833953 | 同上 |
| 交付模型 | `v3_hybrid` / extended_full 重训 | [`strategies/v3_hybrid/model/hybrid_meta.json`](strategies/v3_hybrid/model/hybrid_meta.json) |
| 交付包身份 | zip sha256 `d934d246…` | `outputs/experiments/audit_submission_20260829_main_uploadday.json` |
| 评测机全量实测 | wall clock 15.54 min / 3,217,458 行 / 0 超时 | `outputs/cloud/delivery_cloud_20260829_unpinned.json` |

⚠️ 最终交付模型**没有公榜分数**：它在公榜停更后用回补标签重训（训练段 3,289,030 行、
`time_id` 0–1,105,919），采纳依据是密封期裁决而非榜分，理由与门禁全写在 ROADMAP §P-D。

## 模型是什么

`v3_hybrid` = 冻结的截面 Ridge + 两片 LightGBM 森林的加性结构：

- **截面块**：480 轮 × 3 种子，441 列设计（200 原始特征 ‖ 截面去均值 ‖ 40 个 history 列 ‖ 长窗 512 的滚动均值与偏离），带权拟合。
- **市场块**：480 轮 × 3 种子，逐 `time_id` 预测市场共同分量（该分量占 target 方差约 68%），与 Ridge 的市场估计按 λ=0.5 混合，**不带权**。
- **slow/fast**：逐资产对自身预测做因果滚动均值（窗口 2000 个真实 `time_id` 步），线性重组。
- 后处理：`prediction_scale = 1.16`、`clip ±0.5`。

全部超参、每一项的采纳证据与否决理由都写在 `hybrid_meta.json` 的 `*_note` 字段里。
推理端有 LightGBM 主路径与纯 NumPy 树遍历兜底（`lgbm_numpy.py`），两条路都过交付验证。

## 仓库地图

```text
strategies/           v1_ridge → v2_lgbm → v3_hybrid（交付基线）→ v4_mlp（未采纳）
  v3_hybrid/model/    榜上那套权重（入库，可离线校验 sha256）
src/                  指标、切分、IO、OOF 缓存、产物哈希
experiments/          92 个研究脚本 + ledger.csv（35 条台账）+ INDEX.md（逐脚本索引）
scripts/              打包、晋级、交付验证、数据审计、云端同步
tests/                39 个测试文件；多数是事故后装上去的回归门禁
outputs/              全部实验产物、promotion manifest、数据审计、交付验证读数
research_history/     按主题归档的旧结论（含 SUPERSEDED 与 REJECTED）
```

顶层还有六份文档，分工见下表；不确定某个名词是什么意思时，先查 [`GLOSSARY.md`](GLOSSARY.md)。

## 文档导航

| 文件 | 读它是为了 |
|---|---|
| [`GLOSSARY.md`](GLOSSARY.md) | **先读这个**：编号体系（`P7`/`D4.5`）、信息标签、评价口径、模型部件、流程黑话，一页说清 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | **这套结构为什么长成这样**：逐块来历、scale 沿哪条轴分区才成立、被关闭的方向、交付层的每一道门禁 |
| [`CLAUDE.md`](CLAUDE.md) | 人与 AI 的协作契约、权限边界，以及 **19 条长期伤疤规则** |
| [`ROADMAP.md`](ROADMAP.md) | 收官时的状态、每个课题的结论与证据路径 |
| [`NOTES.md`](NOTES.md) | 研究日志：每个实验当时为什么这么做 |
| [`RUNBOOK_8_23.md`](RUNBOOK_8_23.md) | 8/23–8/31 收官执行序与上传日卡片 |
| [`research_history/`](research_history/README.md) | 主题史：模型演进、特征、验证标定、交付事故（ROADMAP 的已结案课题也归档在这里） |
| [`experiments/INDEX.md`](experiments/INDEX.md) | 92 个研究脚本逐个一行：在问什么、结论、产物、有没有被叙述引用 |

想只看一件事的话，看 `CLAUDE.md` §8 第 17 条：`main.py` 从不设 `num_threads`，而四次
全量交付验证的「4 线程」全部来自命令行前缀 `OMP_NUM_THREADS=4` —— 那个开关**不随提交包走**。
评测机 128 核可见、4 核配额，lightgbm 默认起 128 线程挤 4 核，端到端实测慢 **74.50×**
（当天第一个信号是只量截面森林的微基准 73.27×，端到端补上市场森林后量级更糟），
折算 20.98 h vs 总预算 2.98 h ⟹ 会让 **86%** 的 `time_id` 被填 0。
四次验证全都没抓到，因为它们量的正是那个外部开关。

## 跑起来

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

不需要数据。只放代码的克隆上是 **285 passed / 6 skipped**；补齐主办方 `timeseries_api/`
后是 **370 passed** —— 差的那些是交付验证门禁，依赖官方 runner，
由 [`tests/conftest.py`](tests/conftest.py) 在缺目录时跳过并打印说明。

训练、OOF 与交付验证需要主办方的 `data/` 与 `timeseries_api/`，见 [`UPSTREAM.md`](UPSTREAM.md)。
⚠️ 不要用 `strategies/v3_hybrid/requirements.txt` 装环境 —— 那是**评测机 conda 环境的全量
freeze**（约 200 个包），随提交包交付、作为「榜上那次推理跑在什么依赖面上」的证据，不是安装清单。

```bash
# 复现交付验证（需要 data/ 与 timeseries_api/）
.venv/bin/python scripts/verify_delivery_runtime.py --model-dir strategies/v3_hybrid/model
```

## 已知的粗糙处

- `outputs/**` 里的留档产物含开发机绝对路径（`/home/mainsub/...`）。它们是当时口径的落盘证据，
  与 sha256 证据链绑定，因此**未做事后清洗**。
- 云端同步脚本 `scripts/cloud_sync.py` 需要 `$JHUB_BASE` 与 `$JHUB_TOKEN`，两者都是本机配置，不入库。
- `strategies/v4_mlp/` 是被否决的方向，保留是为了留证（见 ROADMAP 的 REJECTED 条目）。

## 数据、版权与许可

本仓库是**个人参赛记录**，与主办方无关。比赛数据、赛题文档、示例代码与官方评测 runner
版权归主办方，**不随本仓库分发** —— 获取方式见 [`UPSTREAM.md`](UPSTREAM.md)。

作者本人编写的代码、文档与训练所得的模型权重依 Apache-2.0 授权（[`LICENSE`](LICENSE)）。
许可**不覆盖**主办方的数据、文档、示例代码与官方 runner —— 范围说明见 [`NOTICE`](NOTICE)。
