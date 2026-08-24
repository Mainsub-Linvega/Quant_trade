# 交付运行时验证 —— `lightgbm` @ 4 线程

> P0 交付闭环：在钉死线程数下走官方 runner 全量推理，把此前只写在 ROADMAP 的耗时数字变成可审计产物

## 环境

- 声明线程数 **4**；`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`；机器 32 核
- lightgbm：`4.3.0`；numpy `2.5.1`；Python 3.13.5
- 对照：官方评测环境 **4 核 / 12.0 GB**（`docs/competition_description.md:158-159`）

## 资源

| 项 | 值 |
|---|---:|
| **峰值 RSS** | **11.51 GB** |
| 上限 | 12.0 GB |
| 余量线（20% 余量）| 9.60 GB |
| 占用率 | 95.9% |

## 模型身份

- 与 promotion manifest 逐文件 sha256 比对：**True**（8 个文件）
- manifest 来源：`/home/mainsub/Documents/Quant_trade/outputs/promotions/v3_hybrid_long512/promotion_manifest.json`（`--manifest auto`）
- 公榜基线偏离：无
- meta 身份：{"blend_weight": 1.0, "num_iteration": 480, "prediction_scale": 1.16, "prediction_clip": 0.5, "market_lambda": 0.5, "history_window": 5, "sample_modulo": 5, "sampling": "phase_balanced", "cross_section_weighted": true, "slow_fast_window": 2000, "slow_fast_slow_relative": 0.387609649122807, "slow_fast_fast_relative": 1.0801809210526316, "long_window": 512, "n_lgbm_models": 3, "n_market_models": 3, "n_features": 200, "n_history_positions": 40}

## 计时

| 项 | 值 |
|---|---:|
| model_init | 0.38 s |
| **predict_total** | **5.97 分钟** |
| wall clock | 6.96 分钟 |
| predict 调用次数 | 214,538 |
| 单步最大 | 0.701 s |
| 单步平均 | 1.67 ms |
| 超时次数 | 0 |

## 预测健康度

| 项 | 值 |
|---|---:|
| 行数 | 3,217,458（期望 3,217,458）|
| 非有限值 | 0 |
| max\|pred\| | 0.402099（clip 0.5）|
| 触 clip 行数 | 0 |

## 门禁

- ✅ `model_matches_promotion_manifest`
- ✅ `model_matches_public_baseline`
- ✅ `peak_rss_under_limit`
- ❌ `peak_rss_has_headroom`
- ✅ `backend_as_requested`
- ✅ `row_count_correct`
- ✅ `zero_non_finite`
- ✅ `zero_clip_rows`
- ✅ `zero_timeouts`
- ✅ `not_aborted`
- ✅ `predict_calls_expected`
- ✅ `no_error_messages`

## 判定：❌ FAIL

> 调 run_loaded_model 而非 run_strategy ⟹ 全程不写任何 CSV；提交文件只能由用户执行 scripts/make_submission.py 生成

