# 交付运行时验证 —— `numpy-fallback` @ 4 线程

> P0 交付闭环：在钉死线程数下走官方 runner 全量推理，把此前只写在 ROADMAP 的耗时数字变成可审计产物

## 环境

- 声明线程数 **4 线程**；`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`；机器 32 核
- lightgbm：`shimmed (ImportError)`；numpy `2.5.1`；Python 3.13.5
- 对照：官方评测环境 **4 核 / 12.0 GB**（`docs/competition_description.md:158-159`）

## 资源

| 项 | 值 |
|---|---:|
| **峰值 RSS** | **11.52 GB** |
| 上限 | 12.0 GB |
| 余量线（20% 余量）| 9.60 GB |
| 占用率 | 96.0% |

## 模型身份

- 与 promotion manifest 逐文件 sha256 比对：**True**（8 个文件）
- manifest 来源：`/home/mainsub/Documents/Quant_trade/outputs/promotions/v3_hybrid_extended_full_20260824/promotion_manifest.json`（`--manifest auto`）
- 公榜基线偏离：无
- meta 身份：{"blend_weight": 1.0, "num_iteration": 480, "prediction_scale": 1.16, "prediction_clip": 0.5, "market_lambda": 0.5, "history_window": 5, "sample_modulo": 5, "sampling": "phase_balanced", "cross_section_weighted": true, "slow_fast_window": 2000, "slow_fast_slow_relative": 0.387609649122807, "slow_fast_fast_relative": 1.0801809210526316, "long_window": 512, "n_lgbm_models": 3, "n_market_models": 3, "n_features": 200, "n_history_positions": 40}

## 计时

| 项 | 值 |
|---|---:|
| model_init | 0.36 s |
| **predict_total** | **10.52 分钟** |
| wall clock | 11.36 分钟 |
| predict 调用次数 | 214,538 |
| 单步最大 | 0.716 s |
| 单步平均 | 2.94 ms |
| 超时次数 | 0 |

## 运行时间限制（`docs/competition_description.md:161,166-172`）

> 预算按官方公式取 `a = b = 0` 的**下限**（两者都 ≥ 0），判定比真实评测更严。

| 项 | 实测 | 限额 | 占比 |
|---|---:|---:|---:|
| `model_init_seconds` | 0.36 s | 180 s | 0.2% |
| `total_seconds` | 678.5 s | 10,726.9 s | 6.3% |
| `mean_predict_seconds` | 2.94 ms | 50 ms | 5.9% |

- 本次实际开闸值：10,726.9 s；`aborted_after_timeout` = **False**
- ⚠️ 超总闸的后果是**剩余全部 `time_id` 填 0**（`timeseries_api/runner.py:180-198`），是悬崖不是线性损失。

## 预测健康度

| 项 | 值 |
|---|---:|
| 行数 | 3,217,458（期望 3,217,458）|
| 非有限值 | 0 |
| max\|pred\| | 0.426071（clip 0.5）|
| 触 clip 行数 | 0 |

## 门禁

- ✅ `zip_audit_passed`
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
- ✅ `model_init_under_limit`
- ✅ `total_under_budget`
- ✅ `mean_predict_under_budget`
- ✅ `no_error_messages`

## 判定：❌ FAIL

> 调 run_loaded_model 而非 run_strategy ⟹ 全程不写任何 CSV；提交文件只能由用户执行 scripts/make_submission.py 生成

