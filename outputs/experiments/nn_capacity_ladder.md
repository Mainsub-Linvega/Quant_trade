# NN 独立能力阶梯（`nn_capacity_ladder`）

预注册：`/home/mainsub/Documents/Quant_trade/outputs/experiments/nn_capacity_ladder_plan.json`（sha256 `cfdd43bc5c8a2bf9…`） ⟹ 判据先于结果落盘，可核验。

## 曲线：独立 MLP peak / 基准 peak

| max_iter | target_only | multitask | 较好者 | 相对上一档 | 耗时 |
|---:|---:|---:|---:|---:|---:|
| 12 | 17.4% | 20.3% | **20.3%** | — | 2.1 min |
| 50 | 26.2% | 28.8% | **28.8%** | +42.0% | 4.7 min |
| 150 | 6.5% | 5.7% | **6.5%** | -77.3% | 10.4 min |
| 400 | 1.4% | 1.2% | **1.4%** | -78.3% | 26.3 min |

基准 peak（生产 3s480，fold 0）= 0.00105595

⭐ **自检通过**：12 档与 08-19 的最大偏差 0.00e+00 < 容差 0.001 ⟹ 环境与数据一致，曲线可解读。

## 判定

```text
最好一档        max_iter=50  （multitask）  28.8%
门槛            50%
末档相对前一档  -78.3%   （≥ +5% 视为仍在爬）
```

## **REJECTED**

曲线在 <50% 处掉头 ⟹ **sklearn MLPRegressor + 生产特征表示 + 这套预算**下不具竞争力。⚠️ 这**不是**「NN 不行」—— 见适用范围三条。

## ⚠️ 适用范围

本阶梯测的是**一个特定 NN 配方**，不是「NN 这个模型族」。三处对 NN 不利且本轮未动：
特征按 `|corr(feature, e)|` 选的 top-200（为线性/树挑的判据）、`asset_id` 是 15 维
one-hot 而非 embedding、sklearn `MLPRegressor` 无学习率调度 / 无 LayerNorm / 无 dropout。
⟹ 上面三条正是 v5（8/31 之后）要处理的东西；本阶梯为它定范围，不替它下结论。
