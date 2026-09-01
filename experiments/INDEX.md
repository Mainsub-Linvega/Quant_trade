# experiments/INDEX.md — 92 个研究脚本的索引

`experiments/` 下全是**一次性研究脚本**：跑一次、把读数落进 `outputs/experiments/`、
结论记进 [`ledger.csv`](ledger.csv) 或 ROADMAP，然后就不再动了。它们**不是库**，
生产推理路径一个都不 import（那条链是 `strategies/v3_hybrid/`，见 [`../README.md`](../README.md)）。

**这份索引是机器生成的**（`scripts` 之外的一次性工具），每一列都来自产物而非事后回忆：

| 列 | 来源 |
|---|---|
| 在问什么 | 脚本模块 docstring 的第一段 |
| 结论 | 产物 JSON 的 `verdict`/`status` 字段，或产物 Markdown 里的结论标签 |
| 产物 | `outputs/experiments/` 下的同名文件数 |
| 叙述 | ✓ = 现行文档提到过；`旧` = 只在 08-13 重构前的冻结快照里出现；— = 哪里都没有 |

⚠️ **「叙述 —」的 35 个脚本是这个仓库最容易失忆的部分**：
它们跑过、多数也有产物，但**现行文档与冻结快照里都没有任何一段叙述**
解释过为什么跑、结论算不算数。要判断它们，只能直接读 docstring 与产物。
另有 **15** 个标 `旧` —— 只在 08-13 文档重构前的 `source_snapshots/` 里被提过，
现行叙述已经不再提它们。

标签含义（`REJECTED` / `RESULT` / `INCONCLUSIVE` …）见 [`../GLOSSARY.md`](../GLOSSARY.md) §2。

---

| 脚本 | 在问什么 | 结论 | 产物 | 叙述 |
|---|---|---|---:|:---:|
| `ab_decomposition.py` | 把「本地为什么和公榜不一致」分解到具体的项上。 | — | 2 | ✓ |
| `asset_blend_check.py` | per-asset ridge 能不能**叠加**到生产截面块上（而不是替换它）？ | — | 2 | — |
| `asset_grouping_diagnostic.py` | 资产分组只读诊断：15 个资产里有没有天然的分群结构？不训练任何模型。 | — | 2 | ✓ |
| `asset_loading_diagnostic.py` | per-asset 完整载荷诊断：15 个资产该不该有各自的线性系数？ | — | 2 | — |
| `combo_market_weight.py` | 2A × 2C 组合臂：行级市场模型 与 带权训练 能不能叠加？ | `PASS` | 2 | — |
| `component_optimum.py` | 三分量 `f = c_m·m̂ + c_r·ê_ridge + c_l·ê_lgbm` 的最优配比 —— 闭式解。 | — | 4 | — |
| `conditional_blend.py` | A4：条件性组合器 —— `m̂` 与 `ê` 的相对权重随状态量变化，值不值得做。 | — | 2 | — |
| `feature_screen_compare.py` | 选列筛子对比：把 323→200 的线性单变量筛子拆掉，树会不会变好？ | — | — | ✓ |
| `function_class_probe.py` | 函数类探针：在**与生产截面块逐列相同**的输入上，非树的函数类能拿到多少？ | `FAIL` | 3 | ✓ |
| `history_features.py` | *（无 docstring）* | — | — | 旧 |
| `history_gap.py` | 为什么本地把 A1′ 低估了 2.3 倍？—— 四个口径臂的汇总。 | — | 2 | — |
| `history_peak.py` | A1：每资产历史特征，在**尺度无关的 peak 口径**下复现 07-23 那个 +2.5%。 | `PASS` | 10 | 旧 |
| `horizon_auxiliary_cache_probe.py` | horizon auxiliary 的准入筛：responder_00 / responder_02 能不能补强 v3 的 target 残差？ | — | 2 | ✓ |
| `joint_recalibration.py` | Pre-registered joint recalibration matrix for the post-refresh window. | — | 1 | ✓ |
| `lgbm_blend.py` | 把 LightGBM 的截面分量装进**完整预测**，整体量一次。 | `INCONCLUSIVE` | 4 | 旧 |
| `lgbm_inference_cost.py` | 🚨 阻塞项：LightGBM 在评测环境里推理跑得完吗？ | — | 2 | — |
| `lgbm_market_row.py` | 2A：行级 LGBM 直接打 `y` —— 给市场分量 `m̂` 找第二个来源。 | `PASS` | 4 | — |
| `lgbm_mt.py` | LightGBM 能不能在择时块 m_t 上打赢岭回归的线性天花板？（ROADMAP #4 的主攻） | `INCONCLUSIVE` | 4 | ✓ |
| `lgbm_nested_check.py` | 轮数是不是「同族嵌套」—— 一次训练能不能白送多个 `num_iteration`。 | — | 1 | 旧 |
| `lgbm_speed.py` | LightGBM 训练提速基准 —— 在**真实设计矩阵**上量，不靠猜。 | — | — | — |
| `lgbm_weight_select.py` | 2C：两个口径口子 —— LGBM 训练权重 / 截面块的选列分母。 | `PASS` | 2 | — |
| `lgbm_xs.py` | LightGBM 能不能吃下截面块 e = y − m？（ROADMAP #4 原定的主攻方向） | `INCONCLUSIVE` | 2 | — |
| `local_public_lb.py` | 本地公榜 —— 对训练段止于 888,479 的候选，在公榜窗口上无限次精确打分。 | — | — | ✓ |
| `long_history_probe.py` | 长历史窗探针：逐 asset 的 64~4096 个观测里，还有没有生产模型没用到的信息？ | `REJECTED` | 3 | ✓ |
| `long_window_confirm.py` | 长窗 w512 的**确认档**（3 种子 × 480 轮）—— 筛选档结果在生产强度下还成立吗？ | `PASS_BUT_BELOW_DETECTION_FLOOR` | 3 | ✓ |
| `long_window_ladder.py` | 长窗列数阶梯：把 240 列拆成三个 80 列的单窗口臂，看有没有哪一档由负转正。 | `PASS` | 3 | ✓ |
| `market_dev_reuse.py` | 2E：市场模型的**偏差部分**被白扔了 —— 捡回来当第二个 ê | — | 2 | — |
| `market_direct_recheck.py` | 同口径复测：在**今天的口径**下，直接回归 `m_t` 还是不是输给今天的两分量 `m̂`？ | — | 2 | — |
| `market_features.py` | 2D：市场模型该看什么特征？—— 列集合 × 显式截面均值 的 2×2 | — | 2 | — |
| `market_model.py` | 市场模型（`m̂`）能不能重建得更好 —— **在可交付口径下**重测一遍。 | — | 4 | ✓ |
| `market_submission.py` | 从一份已有的提交 CSV 里抽出**市场分量 `m̂`**，出一份 `c_m·m̂ + c_l·ê` 的提交 CSV。 | — | — | — |
| `mt_diagnostics.py` | P0 前置诊断:m_t(每 time_id 加权截面均值)的自相关衰减形状 + 市场共同分量占比稳定性。 | — | 2 | 旧 |
| `mt_lagged.py` | 滞后的特征截面均值对 m_t 有没有增量预测力？（ROADMAP #3 的诊断） | — | 2 | ✓ |
| `mt_predictability.py` | P2-2：市场共同分量 m_t 能不能被预测？—— 决定 68% 那块蛋糕吃不吃得到。 | — | 2 | 旧 |
| `multitask_mlp.py` | 多任务辅助监督：共享 trunk 的 MLP，responder 只在**训练时**提供梯度。 | — | 18 | ✓ |
| `nn_capacity_ladder.py` | NN 独立能力阶梯：把「MLP 只有树的 20%」从**预算事实**变成**能力事实**。 | `REJECTED` | 6 | ✓ |
| `peer_leadlag.py` | 2G：跨资产 lead-lag + ①类分量配比重验 | — | 4 | — |
| `phase_diagnostic.py` | 诊断：sample_modulo 从 10 换到 5 后分数腰斩，是评估集变难还是训练集变差？ | — | 2 | 旧 |
| `public_csv_fingerprints.py` | 把历史公榜 CSV 压成指纹存档 —— 删掉那 1.1 GB 之前必须先跑这个。 | — | 1 | 旧 |
| `public_replay.py` | 公榜分数离线复算器 —— 8/23 回补标签一到就能修尺子。 | — | 4 | ✓ |
| `responder_analysis.py` | Stage A: reproducible, streaming structural analysis of responder labels. | — | 2 | ✓ |
| `responder_family_grid.py` | responder 族群表：47 列其实是一张「维度 × 窗口」的网格。 | — | 2 | — |
| `responder_nonlinear_reaudit.py` | Responder 重新审计：严格 OOF 预测 → 固定过去窗口上的非线性二层门禁。 | — | 2 | — |
| `responder_predictability.py` | Stage B: strict rolling out-of-sample feature-to-responder predictability. | — | 6 | 旧 |
| `responder_reconstruction.py` | 补测并落盘：错位 responder 能重建出多少 target？（NOTES 2026-08-18 的「决定性重建测试」） | — | 2 | ✓ |
| `responder_residual_increment.py` | Stage C: test whether Stage-B responder predictions add target residual signal. | — | 6 | 旧 |
| `responder_selection_probe.py` | 用 responder 的窗口梯子给 feature 打分选列 —— 先量重合度，再决定要不要跑 OOF。 | — | 6 | ✓ |
| `responder_stage_c_fill.py` | Stage C 补测：把 `multi_member_family` 挡掉的 14 个格子填满。 | `PASS` | 6 | ✓ |
| `responder_targets.py` | A0：`responder_*` 换训练目标 —— 47 列全扫 + 两层折判决 + multi 臂。 | `PASS` | 10 | 旧 |
| `responder_window_atlas.py` | responder 窗口图谱：47 个 responder 各自的预测窗口有多长？ | — | 2 | ✓ |
| `ridge_data_ladder.py` | 岭回归到底缺不缺数据？—— 训练窗阶梯的逐折配对测量。 | — | 2 | 旧 |
| `ridge_phase_sampling.py` | 同预算比较周期采样与全相位平衡采样的 Ridge 泛化。 | — | 4 | 旧 |
| `ridge_reproducibility.py` | 检查 Ridge 在不同 BLAS 线程数下是否得到同一模型。 | — | — | — |
| `ridge_strict_acceptance.py` | 汇总 Strict Ridge 的机械验收门禁。 | `accept_strict_solver_candidate` | 2 | — |
| `robustness_probe.py` | 稳健性诊断器 —— 把「分数」拆成「分数从哪来、有多容易塌」。 | — | — | ✓ |
| `scale_transfer.py` | 私榜交付的 `prediction_scale` 取多少 —— 把「本地最优 vs 公榜最优」的账算清楚。 | `押后到 8/23 标签回补；退路是调和平均。判据见 md。` | 2 | ✓ |
| `sealed_period_eval.py` | 密封期尺子：把公榜期尾段封存成本地测试集，让 8/23 之后的决定可测。 | — | — | ✓ |
| `selection_criterion_probe.py` | 选列准则探针：history 那 40 列「选的标准」与「用的方式」对不上，换掉会怎样？ | `REJECTED` | 3 | ✓ |
| `slow_fast_csv.py` | 把 slow/fast 分离施加到一份已有的公榜 CSV 上 —— 纯后处理，不重训、不碰模型产物。 | — | — | ✓ |
| `slow_fast_vertex.py` | slow/fast 直线上的抛物线顶点标定 —— ①类后处理，公榜是正确的尺子。 | `不改交付，私榜维持 t=1` | 2 | ✓ |
| `structural_signal_screen.py` | Strict OOF screening for genuinely new structural signal representations. | `PASS` | 10 | ✓ |
| `target_mlp.py` | Target-only two-head MLP screen against cached v3 OOF predictions. | — | 4 | — |
| `target_mlp_oracle_blend.py` | 对 `target_mlp_screen` 的重新分析：等权 −54.49% 到底否证了什么。 | — | 2 | — |
| `temporal_multiscale.py` | V4-T: pre-registered multi-scale per-asset temporal states for the LGBM cross-section block. | — | 2 | ✓ |
| `thread_default_probe.py` | 线程数默认值探针 —— 回答「`num_threads` 不设（-1）在评测机上会不会劣化」。 | — | 1 | ✓ |
| `v3_asset_adapter_candidate.py` | Build a local v3 candidate with OOF-fitted per-asset cross-section scales. | — | — | ✓ |
| `v3_block_ceiling.py` | ①：market / cross 两个块各自还剩多少 —— 纯 OOF 后处理，不训练、不写模型。 | — | 2 | — |
| `v3_confirm_summary.py` | Machine-judge the 3-seed/480-round confirmation and select the final local arm. | — | — | — |
| `v3_fullres_resource_smoke.py` | Memory-safe local full-resolution v3 resource smoke. | `ok` | 2 | ✓ |
| `v3_fullres_slow_probe.py` | P1 口径核对：慢分量在**全分辨率**下还成立吗？ | — | 8 | — |
| `v3_fullres_slow_probe_summary.py` | 把多个 fold 的全分辨率探针窗合并成一个估计。 | — | 2 | — |
| `v3_lowrank_cross_residual.py` | Frozen low-rank cross residual adapters using PCA factors × asset exposures. | — | 2 | — |
| `v3_market_round_scan.py` | Screen independent shrunk-market LightGBM checkpoints with the XS OOF block fixed. | `PASS` | 2 | ✓ |
| `v3_phase_scale.py` | ③：最优 scale 该不该分 phase —— OOF 上解析求解，零公榜配额。 | — | 4 | ✓ |
| `v3_production_oof.py` | 严格 OOF：复现当前 v3_hybrid 生产架构并保存逐行组件预测。 | — | 12 | ✓ |
| `v3_raw_cross_state_adapter.py` | Screen causal raw-feature cross-sectional dispersion as a residual adapter state. | — | 2 | — |
| `v3_recency_ladder.py` | P4：训练窗阶梯（recency）—— 未来更像最近的训练窗，还是完整历史窗？ | — | 2 | — |
| `v3_residual_adapters.py` | Frozen residual adapters trained on the earliest OOF fold and evaluated later. | `PASS` | 22 | ✓ |
| `v3_residual_atlas.py` | 从严格 v3 OOF 生成 market/cross residual atlas。 | — | 4 | ✓ |
| `v3_residual_signal_search.py` | Frozen OOF search for low-capacity, inference-available cross residual adapters. | `PASS` | 10 | — |
| `v3_slow_variance.py` | P1：预测里的「死方差」—— 慢分量该不该降权。纯 OOF 后处理，不训练、不写模型。 | `PASS` | 4 | ✓ |
| `v3_sparse_asset_feature_residual.py` | Sparse asset×feature residual interaction screening on strict v3 OOF. | — | 2 | — |
| `v3_temporal_smoothing.py` | ②：时间平滑值不值得做 —— OOF 上闭式求解，外加一次窄列全分辨率扫描。 | — | 2 | — |
| `variant_submission.py` | 出一份 v3_hybrid 变体的公榜 CSV，**不碰生产模型产物**。 | — | — | ✓ |
| `walk_forward.py` | *（无 docstring）* | — | 7 | 旧 |
| `walk_forward_history.py` | *（无 docstring）* | — | 2 | 旧 |
| `walk_forward_rolling.py` | P0 细粒度验证：按 time_id 滚动切 fold + embargo。 | — | 2 | ✓ |
| `xs_loss_stack.py` | 2F：截面块的三条没碰过的③类 —— 岭回归残差堆叠 / 稳健损失 / 标签裁尾 | — | 2 | — |
| `xs_market_state_probe.py` | 截面块市场态交互探针：喂 market_pred_t 给 XS 树，能不能学出 asset×market 交互？ | `REJECTED` | 4 | ✓ |
| `xs_peer_deployable_probe.py` | peer 对轴收口：把 oracle 的 `peer_e_lag1` 换成**可部署**量之后，那 +3.29% 还剩多少？ | `INCONCLUSIVE_NO_DETECTION_POWER` | 2 | ✓ |
| `xs_peer_pair_confirm.py` | 截面块窄 peer 对确认档（3 种子×480 轮）——把筛选档的低于检出下限的结果测清楚。 | `REJECTED` | 2 | — |
| `xs_peer_pair_probe.py` | 截面块窄 peer 对探针：喂 3 对资产的滞后共动信息，能不能补上树看不到的部分？ | `REJECTED` | 2 | — |

---

## 统计

- 脚本 **92** 个，合计 31,228 行
- 有产物的 **80** 个；无产物 **12** 个（多为被更好的臂取代、或只作冒烟）
- 现行文档引用的 **42** 个；只在冻结快照里出现的 **15** 个；**哪里都没被提过的 35 个**
- `ledger.csv` 有 35 条记录 —— 它记的是**进过生产候选比较**的改动，
  不是每个脚本一行，所以覆盖数远小于脚本数，这是设计如此。
