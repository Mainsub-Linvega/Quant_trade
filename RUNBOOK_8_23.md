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
- 私榜共 10 次，**且「最终采用最新提交版本」**（`docs/competition_description.md:201` 原文）。
  ⚠️ **不是 best-of-10** ⟹ **8/31 最后一次上传的那份就是最终答案**。
  「留 3 次余量」的正确含义是**给上传失败/包损坏留重试**，不是「交一组分散候选让主办方挑」。
  推论：任何实验性变体**都不能最后上传**；收尾顺序见 D6。
- ⚠️ 8/23 之后**没有外部尺子**。本项目已三次本地↔公榜量反（CLAUDE.md §8.1）⟹
  D0 的「修尺子」是后面一切比较的前提，不能跳。
- ⭐ **2026-08-20 订正上一行的措辞**：8/23 之后不是「没有外部尺子」，而是**尺子从「盲、5 枪/天」
  换成「不盲、无限次」** —— 回补的就是公榜期的标签，那 3,217,458 行从此可以在本地反复打分。
  ⟹ 新的风险不是没尺子，是**有一把可以无限次拟合的尺子**。因此 8/23 之前预注册了**密封段**：
  公榜期最后 **60,000 real time_id** 封存不训练，只当测试集（ROADMAP P10、下面 D0.4 / D4.5）。
  ⚠️ 那段数据**只能用一次** —— 当训练数据或当干净测试集，用来训了就不能再当测试。

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

### D0.2b ⭐ 回补包里有没有 `responder_*` 列（2026-08-22 新增，约 1 秒）

```bash
.venv/bin/python scripts/check_backfill_responders.py \
    --audit  outputs/data_audits/data_release_20260823.json \
    --output outputs/data_audits/backfill_responders_20260823.json
```

**为什么必须当天核**：`docs/data_description.md:173` 只说「发布**标签回补数据**，该部分数据将
作为**扩展训练数据**使用」，Timeline（`competition_description.md:213`）也只写「发布扩展训练
数据」—— **原文从未逐字列出回补包的字段清单**。窄读（标签 = `target`+`weight`）与宽读
（扩展训练数据 = 含 responder）都成立，**只能实测**。

而 2026-08-22 收口的 responder 四项 `REJECTED` **共用同一条重开条件**，答案直接决定
8/23–8/31 要不要重开一条已经关掉的线：

| 判定 | 含义 | 动作 |
|---|---|---|
| `train_unchanged` | train split 没变 | responder 轴维持关闭（D0.2 主判据也会拦下重训）|
| `backfill_has_responders` | 回补文件**全部**带 responder | ⭐ **触发重开** ⟹ 走 D3.5 |
| `backfill_has_no_responders` | 回补文件**全都不**带 | responder 轴维持关闭，8/31 前不再碰 |
| `backfill_schema_inconsistent` | 部分带部分不带 | ⚠️ **退出码 1，停下来查原因**，不要按任一分支走 |

⚠️ 本脚本**不重新扫 parquet**，只读 D0.2 已落盘的审计 JSON（`audit_data_release.py:92`
逐文件存了 `columns`，且 `columns` 本身就在 `file_identity` 的比较键里 `:133`）⟹
它与 D0.2 是同一份真值，不会出现两处口径分家。
四支判定都有回归用例（`tests/test_check_backfill_responders.py`）。

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

### D0.4 ⭐ 标定密封期尺子的检出下限（约 40 分钟，Tier 1）

```bash
# 六个候选各出一次预测（每个约 6 分钟；不落任何提交格式 CSV）
for SPEC in production_slowfast:strategies/v3_hybrid \
            mkt_shrunk:outputs/candidates/v3_hybrid_mkt_shrunk \
            mktwe:outputs/candidates/v3_hybrid_r480_pb_hist_mktwe \
            asset_adapter:outputs/candidates/v3_asset_cross_3s480_shrink500 \
            r960:outputs/candidates/v3_hybrid_r960_pb_hist_mktwe \
            xs_shrunk:outputs/candidates/v3_hybrid_xs_shrunk; do
  .venv/bin/python experiments/sealed_period_eval.py \
      --candidate "${SPEC#*:}" --label "seal_${SPEC%%:*}"
done

.venv/bin/python experiments/sealed_period_eval.py \
    --baseline seal_production_slowfast \
    --arms mkt_shrunk=seal_mkt_shrunk mktwe=seal_mktwe \
           asset_adapter=seal_asset_adapter r960=seal_r960 xs_shrunk=seal_xs_shrunk \
    --labels <回补数据目录> --label sealed_tier1_calibration
```

六个候选**都有已知公榜真值**（0.0041150085 / 0.0039977510 / 0.0039673997 / 0.0039908352 /
0.0037609312 / 0.0035771492）。这一步的产出**不是找收益，是读出这把尺子的块级方差**⟹
`--detection-floor` 该填多少。⚠️ 在它填出来之前，任何裁决的第七道门判 `PENDING_CALIBRATION`，
**不是自动通过**。

⭐ 顺带回答一个悬案：`asset_adapter` 在 OOF 上 +1.99%、在公榜上 −0.17% —— 密封期站哪边。

**⚠️ 这一步不需要重训，也不消耗任何私榜/公榜额度。** 若时间紧只能做一件事，做这个而不是 D3。

## D1：固定结构重训（只有 train 确实变了才做）

```bash
.venv/bin/python scripts/retrain_extended.py \
    --audit outputs/data_audits/data_release_20260823.json        # 默认 dry-run，先看命令计划
.venv/bin/python scripts/retrain_extended.py \
    --audit outputs/data_audits/data_release_20260823.json --execute
```

写入 `outputs/candidates/v3_hybrid_extended_fixed/`，**不碰生产**。

⚠️ **决策期重训的训练段必须止于 time_id `1,045,889`**（密封段起点 1,045,920 减 30 个 embargo）。
训进密封段就等于把测试集喂给了模型，D2 之后的一切比较全部作废 —— 而且**不会报错**。
最终交付件的全量重训在 **D4.5**，不在这里。

### ⚠️ D1 的坑（坑 1 / 坑 3 已于 08-19 接线，当天不需要临场决策）

1. ✅ **重训候选的 meta 不含 `slow_fast_*` 三个键** —— `train.py` 的 CLI 里根本没有这个概念，
   所以任何重训候选都必定缺键；而 `main.py:222` 是 `PredictionTrail(...) if window else None`
   ⟹ **缺键会静默关掉 slow/fast、不报错**（这三个键是公榜 0.0041150085 与 0.0039977510 的
   **全部差别**）。**08-19 已给 `promote_v3_candidate` 加上生产者**，两条路各是一条命令，
   仍然**必须二选一并写进 ledger**：
   - (b) 沿用当前标定 = **什么都不传**，staging 自动按 `PUBLIC_BASELINE` 写入；
   - (a) 用新 OOF 重标定（原规格复验，扩展数据后本来就该重算）：
     先 `experiments/v3_slow_variance.py` 算出两个 relative，再

     ```bash
     .venv/bin/python scripts/promote_v3_candidate.py --candidate <候选> \
         --slow-fast-slow-relative <新值> --slow-fast-fast-relative <新值> --off-baseline
     ```

     这里的 `--off-baseline` 是**有意偏离公榜标定**的正当用法（偏离要按下去），
     不是绕过缺键检查。缺键那条路已经不存在了。
2. `--prediction-scale 1.16` 是公榜两点法标定的。扩展数据重训后这个值是否还成立，
   **只能靠 D0.3 校准后的本地尺子判断**（公榜已停更）。不搜网格，按
   `outputs/experiments/joint_recalibration_plan.json` 里冻结的格子来。
3. ✅ **命令计划此前不复现当前生产架构**（08-19 复查发现，原 runbook 没有这一条）：
   缺 `--weighted-cross-section`、`--market-model`、`--market-lambda`、`--market-spec`、
   `--market-min-data-scale`。前两个是 `store_true`，不传 = `False` ——
   **不是「用默认值」，是另一个模型**：没有行级市场森林、截面块不带权，
   等于退回 08-11 那版架构（公榜 0.0032523499，比生产低 21.99%）。
   现在这些项由 `production_structure()` 从生产 `hybrid_meta.json` 派生，并与 `PUBLIC_BASELINE`
   逐键对拍，对不上就**拒绝生成计划**。
   ⟹ **dry-run 的输出里现在有一段 `production_structure`，先看它再 `--execute`。**

## D2：用**校准后的尺子**比较重训 vs 当前生产

⚠️ **配对基准必须用当前代码现跑。** `experiments/v3_production_oof.py` 的**首次提交**是
2026-08-15 11:18，**早于它的 OOF 缓存都出自已不存在的代码版本**：

| 缓存 | 产出 | 状态 |
|---|---|---|
| `..._phasebal_prodwindow_exact.npz` | 08-14 11:12 | **隔离**（实测差 `max\|Δ(market_ridge)\| = 3.37e-05`，约折均 peak 的 2.4%，与被测效应同量级）|
| `..._phasebal_prodwindow.npz` | 08-14 10:56 | **隔离**（08-20 复查新增：13 数组、无 checkpoint，与上一行**签名完全一致**，08-18 那次漏点名了）|
| `..._confirm_3s480_phasebal_prodwindow.npz` | 08-14 12:52 | 未隔离但**未经现跑复验**：19 数组含 checkpoint，结构与已入库脚本一致 |
| `..._1s160_prodwindow_20260818.npz` | 08-18 15:21 | ✅ **唯一确认由当前代码产出** |

隔离由 `src/oof_cache.assert_reproducible_cache` 强制执行（`load_oof_bundle` 里也调了），
指到隔离名单上的文件会**当场报错并给出现跑命令**——不是靠记性。
⚠️ 改名封存后的 `.STALE-DO-NOT-USE` 路径同样在名单里：**别把名字改回去绕过它**。

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

## D3.5：responder 原规格复验 —— **只在 D0.2b 判 `backfill_has_responders` 时才做**

⚠️⚠️ **默认不做。** responder 这条线于 2026-08-22 走完并**由证据关闭**：Stage C 补测覆盖了
全部 24 族 / 47 列（28 格无一过门禁）、选列判据在前置测量处结案、P9 范围项 ③ 也已否掉。
只有回补包**确实带来新的 responder 列**才回到这一步。

**纪律（比做不做更重要）**：

- **按原规格复验，不得借机改设计** —— 沿用各自的预注册文件，不换臂、不调阈值、不加梯子成员：

  ```bash
  .venv/bin/python experiments/responder_stage_c_fill.py --label responder_stage_c_fill_0823
  .venv/bin/python experiments/responder_selection_probe.py --label responder_selection_probe_0823
  ```

- **顺序在 D3 之后、D4 之前。** 两个脚本对训练窗的依赖不同，别一刀切：
  - `responder_stage_c_fill` 读的是**固定缓存**（`responder_oof_*.npz` + v3 基准 OOF cache），
    重训与训练窗都影响不到它 ⟹ 什么时候跑都一样；
  - `responder_selection_probe` 是在 fold 上**现算**选列的（`TRAIN_WINDOW = 78_960`）⟹
    若 ROADMAP P2-R 的 recency 阶梯改了生产训练窗，**必须先定下窗口再跑它**，
    否则量出来的重合度不是新生产结构上的重合度。
    ⚠️ P2-R 只在 ROADMAP 里，**本文件没有把它排成 D 步骤**。
- **过门槛也只能进 P10 Tier 2**，由密封期尺子裁决；**不建候选模型、不碰生产、不花提交额度**。
- ⚠️ 时间不够就**直接跳过这一步** —— 它的期望值是「关严」不是「涨分」，
  而 8/31 的交付件不依赖它。

**证据入口**：`outputs/experiments/responder_stage_c_fill.md`、
`responder_selection_probe.md`、`nn_capacity_ladder_respsel.md`、
`responder_reaudit_20260814.md`（母条件原文在 :93-100）。

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

## D4.5 ⭐ 最终交付件：决定拍完之后才用 100% 数据重训

到这一步，所有采纳/拒绝的决定都已经由密封期做出。**现在**（而不是更早）把密封段放回训练集，
用 `0 – 1,105,919` 全量（+24.5%）按**已经定下来的结构**重训一次，作为最终交付件。

⚠️ **风险要认**：这一份训练在**没有任何评估覆盖过**的数据上。缓解是它与刚被密封期验过的是同一
结构，且 D4 覆盖机械正确性（meta 结构、双后端对拍、一致性、耗时）。

**三层回退，都有落盘产物**：

| 情形 | 交哪一份 |
|---|---|
| D4.5 全部门禁通过 | 全量重训件 |
| D4.5 任一门禁不过 | **决策期那份**（训练止于 1,045,889，被密封期评过分） |
| 决策期那份也不过 | **当前生产 `v3_hybrid_slowfast`**（`outputs/v3_hybrid_submission_20260819.zip`） |

⟹ 任何时候都有一份可交的东西；**时间不够就直接跳到 D5 交当前生产**。

## D5：用户打包 + 审计（**只能由用户执行**，CLAUDE.md §1.4）

```bash
.venv/bin/python scripts/make_submission.py --strategy v3_hybrid
.venv/bin/python scripts/audit_submission_zip.py \
    outputs/v3_hybrid_submission_<YYYYMMDD>.zip --expect-public-baseline \
    --output outputs/experiments/submission_audit_v3_hybrid_<YYYYMMDD>.json
```

`--expect-public-baseline` 是 08-18 新增的，会拿 `PUBLIC_BASELINE` **全表**核对
（含 slow/fast 三键）。**打私榜包务必带上。**

⚠️ **`--output` 不要省** —— 不带它脚本只打印不落盘，等于「审过了但盘上没有证据」。
08-18 那次就是这样：包是对的，却没有任何可追溯的审计记录。
审计 JSON 放 `outputs/experiments/`（该目录入库），`outputs/private_submissions/` 是 gitignore 的。

⚠️ 08-19 起审计还会核**包内容身份**（`no_unexpected_modules`）：包里的 `.py` 必须恰好是
`make_submission.SUBMISSION_MODULES` 声明的那几个。现存 `20260818.zip` 会因为多带一个
研究模块 `temporal.py` 判 FAIL —— 那是**当时的打包口径**造成的，模型身份本身零漂移
（`public_baseline_drift == []`），重打一次即可。

⚠️ slow/fast 转正**前**的旧模型包已于 08-19 改名封存为
`outputs/v3_hybrid_submission_20260813.PRE-SLOWFAST.zip`（旧审计八项全 PASS，
加 `--expect-public-baseline` 才被拦下）。**改名就是防呆措施本身——不要改回去、不要提交它。**
当前唯一通过全部门禁的包是 `outputs/v3_hybrid_submission_20260819.zip`
（审计记录 `outputs/experiments/submission_audit_v3_hybrid_20260819.json`）。

## D6+：缓冲与收尾顺序

- 私榜留 **≥3 次**余量（**重试用**，不是候选组合用 —— 见 §0）。
- 若 D1–D3 全部不过门槛：**维持当前生产 `v3_hybrid_slowfast` 原样提交**。
  拒绝改动不需要额外证据；空动作就是保持现状。

### ⚠️ 8/31 收尾顺序（因为「最新提交版本生效」，顺序本身就是纪律）

1. 先确定**要交哪一份**，并对它跑
   `audit_submission_zip.py --expect-public-baseline --output <落盘路径>`；
2. 审计 `passed: true` 之后**才**上传；
3. **上传完不要再上传任何东西** —— 之后的每一次上传都会覆盖它成为最终答案；
4. 若必须重传（包损坏、网络失败），重传的**必须是同一份 zip**，并再核一次 sha256。

⟹ 不存在「最后交个实验版试试」这种操作。想试的东西在 8/23 之前用公榜试完。

---

## 失败回退

| 情形 | 动作 |
|---|---|
| D0.2 显示 train 未变 | 停止重训，只做 D0.3；生产维持不变 |
| D0.2b 判 `backfill_schema_inconsistent` | **退出码 1，停下来查原因**（部分回补文件带 responder、部分不带 ⟹ 要么理解错了包结构、要么包本身有问题）。**不要按任一分支走**，也不要跳过 —— 它同时说明 D0.2 的 schema 比较可能也在读一个不一致的东西 |
| D0.2b 判 `backfill_has_responders` | 记进 ledger，**排在 D3 之后**做 D3.5 原规格复验；时间不够就跳过（期望值是关严不是涨分）|
| D0.3 复算对不上公布分数 | **先修复算器**，不解释现象；修不好就不用本地尺子做任何采纳决策 |
| D1 重训候选被 slow/fast 门禁拦下 | 按 D1 坑 1 的 (a) 或 (b) 处理，写进 ledger；不用 `--off-baseline` 绕 |
| D4 耗时或双后端对拍不过 | 回滚到 `v3_hybrid_slowfast`，`outputs/promotions/backups/` 有备份 |
| 任何一步时间不够 | 直接跳到 D5，交当前生产模型 |
