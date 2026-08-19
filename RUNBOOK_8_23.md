# RUNBOOK：8/23 回补数据 → 8/31 私榜提交

> **这份文件的目的**：8/23 当天不需要再做任何设计决策。所有命令、判据、失败回退都已写死并
> 用现有数据干跑验证过（2026-08-18）。剩 8 天，别把其中 2 天用来写代码。
>
> 状态口径以 [`ROADMAP.md`](ROADMAP.md) 为准；本文件只写「怎么做」。

## 0. 时间与额度

| 事件 | 日期 | 说明 |
|---|---|---|
| 公榜停更 + 标签回补 | 8/23 | 发布扩展训练数据，供最终模型训练与**本地验证** |
| 策略提交截止 | 8/31 | 关闭提交入口 |
| 实盘评估 | 9/1–9/30 | **私榜是前向实盘**，不是历史留出集 |

- 公榜每天 5 次（`docs/competition_description.md:192`），8/23 后作废。
- 私榜共 10 次，**至少留 3 次余量**。
- ⚠️ 8/23 之后**没有外部尺子**。本项目已三次本地↔公榜量反（CLAUDE.md §8.1）⟹
  D0 的「修尺子」是后面一切比较的前提，不能跳。

## 1. 已知的数据规模（2026-08-18 审计实测）

```text
train  9 分区  13,227,692 行  time_id 0–888,479
test   3 分区   3,217,458 行  time_id 888,480–1,105,919
```

⟹ 回补后训练期延长 **+24.5% 的 time_id**，而且是**最靠近 9 月实盘期**的那一段。
⚠️ 这与 P4「数据饱和」不冲突：P4 测的是往窗口**前端**加旧数据（+1.08%、2/5 折、CI 跨 0），
这里是往**后端**加新数据 —— 不同的问题，此前无法测。

---

## D0（8/23）：审计 + 修尺子 —— 最优先，不可跳过

### D0.1 只读核验主办方原文

```bash
git diff --stat docs/ examples/ timeseries_api/     # 必须为空或已知变更
```

### D0.2 数据审计（工具已于 08-18 干跑验证）

```bash
.venv/bin/python scripts/audit_data_release.py \
    --baseline outputs/data_audits/data_release_20260818.json \
    --output   outputs/data_audits/data_release_20260823.json
```

判据：`comparison.changed == true` 且 `splits.train` 有 `added/modified/row_delta`。
- **train 未变** ⟹ 停止一切重训（ROADMAP P1 动作 3）。只做 D0.3 修尺子。
- **train 已变** ⟹ 继续 D1。

⚠️ `retrain_extended.py` 会自己复核这一条，audit 不达标它直接拒绝运行（已验证）。

### D0.3 ⭐ 修尺子：离线复算每一份历史公榜提交

```bash
.venv/bin/python experiments/public_replay.py --labels <回补数据目录> --force
```

**先看「复现」那一列**：离线分数必须复现主办方公布的分数（CSV 8 位小数，预期 ~1e-8）。
**对不上就先修复算器，不要解释现象** —— 否则后面所有拆解都是在解释 bug。

已就位的材料（08-18 干跑确认）：

| | 数量 |
|---|---:|
| 盘上 CSV（含逐行预测值） | **21** |
| 其中 sha256 硬校验归属 | 4 |
| ledger 原文归属 | 2 |
| 按模型名推断归属（**靠复算分数验证**） | 15 |
| ledger 有公榜分数但无逐行 CSV | 10 |
| 只剩指纹、CSV 已删 | 14（其中 **8 份模型已不在仓库 ⟹ 永久不可复算**）|

21 份覆盖公榜 0.0015→0.0041 的整个 v3 时代，row_id 连接 100%、0 非有限值。

复现通过后再做拆解，回答 `NOTES.md` §6 未解问题第一条：
- 按时期分桶 ⟹ 公榜期内部的分数漂移；
- 「本地 OOF Δ%」对「公榜 Δ%」的**实测斜率**（迁移率 0.51×~2.3× 到底怎么来的）；
- 哪一折最像公榜期 ⟹ 私榜期该给哪一折加权。

**产出**：一条可写进 ROADMAP 的结论：**8/23–8/31 期间本地增量该打几折**。

---

## D1：固定结构重训（只有 train 确实变了才做）

```bash
.venv/bin/python scripts/retrain_extended.py \
    --audit outputs/data_audits/data_release_20260823.json        # 默认 dry-run，先看命令计划
.venv/bin/python scripts/retrain_extended.py \
    --audit outputs/data_audits/data_release_20260823.json --execute
```

写入 `outputs/candidates/v3_hybrid_extended_fixed/`，**不碰生产**。

### ⚠️ D1 的两个已知坑（08-18 干跑发现）

1. **重训候选的 meta 不含 `slow_fast_*` 三个键** —— `retrain_extended.py` 的命令计划里没有
   任何 slow/fast 参数。这三个键是公榜 0.0041150085 与 0.0039977510 的**全部差别**，
   而 `main.py:222` 是 `PredictionTrail(...) if window else None` ⟹ **缺键会静默关掉
   slow/fast、不报错**。08-18 已给三道门禁补上检查，所以转正时会被**明确拦下**（设计如此）。
   两条正确做法，**必须二选一并写进 ledger**：
   - (a) 用新 OOF **重新标定** slow/fast 的两个 relative（`experiments/v3_slow_variance.py`），
     这是原规格复验，扩展数据后本来就该重算；
   - (b) 沿用当前三个键的值，把它们写进候选 meta。
   **不要**用 `--off-baseline` 蒙混过去 —— 那是给「有意不带 slow/fast」准备的出口。
2. `--prediction-scale 1.16` 是公榜两点法标定的。扩展数据重训后这个值是否还成立，
   **只能靠 D0.3 校准后的本地尺子判断**（公榜已停更）。不搜网格，按
   `outputs/experiments/joint_recalibration_plan.json` 里冻结的格子来。

## D2：用**校准后的尺子**比较重训 vs 当前生产

⚠️ **不要拿 `outputs/cache/v3_production_oof_phasebal_prodwindow_exact.npz` 当配对基准。**
它的时间戳是 2026-08-14 11:12，而 `experiments/v3_production_oof.py` 的**首次提交**是
08-15 11:18 ⟹ 它出自**已不存在的代码版本**，与当前脚本输出差
`max|Δ(market_ridge)| = 3.37e-05`（约折均 peak 的 2.4%）。基准必须用当前代码现跑。

```bash
# 现跑基准（约 11 分钟 @ 8 线程）
.venv/bin/python experiments/v3_production_oof.py --label oof_baseline_20260823 --num-threads 8
# 比较（通用配对比较器，含 6 道门禁 + 配对 block bootstrap + 检出下限）
.venv/bin/python experiments/feature_screen_compare.py \
    --cache-dir outputs/cache --baseline oof_baseline_20260823 \
    --arms extended=<重训臂的 cache label> --label extended_vs_current --min-relative-gain 0.03
```

门槛：折均>0、≥4/5 折、去最好折>0、相对≥3%、`2ΔA>ΔB`、配对 CI 下界>0、超过检出下限。
⚠️ 1% 那档没有牙 —— 1s160/5 折的检出下限实测是基准 peak 的 **6.1%**，
3s480 是 **8.7%**（`v3_recency_expanding_ladder_1s160.md`）。

## D2.5：full-resolution 只作条件复验

2026-08-19 本地结果已钉死：固定生产 200 特征、短真实窗的 1,182,292-row 双森林 160 轮
resource smoke 可以在约 11.5GB RSS 完成，但它使用全局生产 stats/features、跳过 Ridge，且明确
`oof_valid=false`。保持生产真实训练跨度时需要约 5.92m train rows，当前 30GB/无 swap 机器会 OOM。

因此 8/23 后：

- 默认**不在本地**运行正式 full-resolution 多折 OOF；
- 只有完成 chunked design writer，或准备好 64GB+ CPU 服务器，才运行 `sample_modulo=1`；
- full-resolution 与 sampled 必须共用相同真实 train/valid 边界；
- fixed-production smoke 不能进入 D2 候选比较；正式 OOF 必须 fold-local 拟合 stats/选列/history；
- 若资源条件不满足，跳过本项，不影响扩展数据固定结构重训。

## D3：V4-R 压缩 market regime 原规格复验

唯一保留「扩展数据复验资格」的 V4 结构项（08-12：+1.34%、4/5 折，机制干净但未过 +3%）。
**原规格复验，不重开 T1/T2/T3 和 MLP 搜索**（那些已全否）。

## D4：转正门禁全链

```bash
.venv/bin/python -m pytest -q                     # 基线 73 passed / 18 subtests
.venv/bin/python scripts/check_consistency.py --strategy v3_hybrid --backend lightgbm
.venv/bin/python scripts/check_consistency.py --strategy v3_hybrid --backend numpy
.venv/bin/python scripts/promote_v3_candidate.py --candidate outputs/candidates/<候选>
# 4 核全量，两条路径（08-18 新增；走官方 runner 的 run_loaded_model，不写任何 CSV）
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/verify_delivery_runtime.py \
    --backend lightgbm --threads 4 --label delivery_runtime_lightgbm_4t_<日期>
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/verify_delivery_runtime.py \
    --backend numpy-fallback --threads 4 --label delivery_runtime_numpy_4t_<日期>
```

当前基线（2026-08-18 实测，4 核）：

| | LightGBM | NumPy 兜底 |
|---|---:|---:|
| `predict_total` | 5.26 分钟 | 10.94 分钟（2.08×）|
| wall clock | 6.20 分钟 | 11.81 分钟 |
| 超时 / 非有限值 / 触 clip | 0 / 0 / 0 | 0 / 0 / 0 |

兜底是**单核 100%**（纯 numpy 树遍历不并行）⟹ 4 核评测机不会更慢。
新候选若列数变多（例如选列宽度改成 323），**耗时必须重测** —— scar §5：
小批量推理中取列可能比树本身更贵。

## D5：用户打包 + 审计（**只能由用户执行**，CLAUDE.md §1.4）

```bash
.venv/bin/python scripts/make_submission.py --strategy v3_hybrid
.venv/bin/python scripts/audit_submission_zip.py \
    outputs/v3_hybrid_submission_<YYYYMMDD>.zip --expect-public-baseline
```

`--expect-public-baseline` 是 08-18 新增的，会拿 `PUBLIC_BASELINE` **全表**核对
（含 slow/fast 三键）。**打私榜包务必带上。**

⚠️ **现存 `outputs/v3_hybrid_submission_20260813.zip` 是 slow/fast 转正前的旧模型** ——
旧审计八项全 PASS，加 `--expect-public-baseline` 才被拦下。**不要拿它提交。**

## D6+：缓冲

- 私榜留 **≥3 次**余量。
- 若 D1–D3 全部不过门槛：**维持当前生产 `v3_hybrid_slowfast` 原样提交**。
  拒绝改动不需要额外证据；空动作就是保持现状。

---

## 失败回退

| 情形 | 动作 |
|---|---|
| D0.2 显示 train 未变 | 停止重训，只做 D0.3；生产维持不变 |
| D0.3 复算对不上公布分数 | **先修复算器**，不解释现象；修不好就不用本地尺子做任何采纳决策 |
| D1 重训候选被 slow/fast 门禁拦下 | 按 D1 坑 1 的 (a) 或 (b) 处理，写进 ledger；不用 `--off-baseline` 绕 |
| D4 耗时或双后端对拍不过 | 回滚到 `v3_hybrid_slowfast`，`outputs/promotions/backups/` 有备份 |
| 任何一步时间不够 | 直接跳到 D5，交当前生产模型 |
