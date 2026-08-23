"""资产分组只读诊断：15 个资产里有没有天然的分群结构？不训练任何模型。

## 为什么现在问

`asset_loading_diagnostic`（08-18）已经测过"给每个资产单独一套线性系数"这个极端版本
（等于 15 个组、每组 1 个资产）：异质性真实存在（+95.8%、5/5 折），但 per-asset 线性模型
本身比生产 LGBM 的 `asset_id` categorical 处理**低 50.5%**——因为树在每次分裂时已经对
15 个类别值做自适应最优二分组，且共享叶子结构自带跨资产部分池化，比独立拟合 15 个模型
更省数据。

粗粒度分组（比如 3~5 群）不受这条负结果直接约束，因为它解决的是 per-asset 那次真正输掉
的原因——**数据饥饿**：每个资产单独一个模型，能用的行数只有全量的 1/15；分组把多个资产的
时间序列合并训练，每组的有效样本量远大于单资产。但分组要有意义，前提是这 15 个资产之间
**真的存在自然分群**（不是均匀连续谱），否则分组只是任意切一刀，等价于又在重复 per-asset
那次已经输给树的机制。

本诊断只回答"有没有分群结构"，不训练、不下结论、不建候选，几分钟跑完：

1. 逐资产 IC 贡献（生产 e_lgbm 相对截面残差 e 的加权 IC）——树当前对每个资产做得好不好；
2. 逐资产 e（原始截面残差）两两相关矩阵——资产之间原始行为是否共动；
3. 逐资产"未解释残差"（e − e_lgbm）两两相关矩阵——树**没学到**的部分是否在资产间共享，
   这是分组能不能补的关键信号：若未解释残差在某几个资产间高度相关，说明存在树目前没利用的
   跨资产共享结构；若接近独立噪声，分组大概率也捞不到东西；
4. 对 (2)(3) 两个相关矩阵做层次聚类，看有没有清晰的分群边界。

数据来源：生产 OOF cache（`function_class_probe.CACHE_PATH`），5 折验证段拼接、
覆盖全部采样 time_id，不需要重新训练或重新扫 parquet。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from function_class_probe import CACHE_PATH, N_ASSETS, weighted_ic  # noqa: E402
from src.oof_cache import assert_reproducible_cache  # noqa: E402


def main() -> None:
    assert_reproducible_cache(CACHE_PATH)
    cache = np.load(CACHE_PATH)
    # ⚠️ cache 里混了 fold=-1 的行（非 OOF 验证段，e_lgbm 全 NaN，1,183,798 行）——
    # 只留 5 个真实验证折，否则逐资产统计量会被 NaN 污染成全零/全 NaN。
    keep = cache["fold"] >= 0
    time_id = cache["time_id"][keep]
    asset_id = cache["asset_id"][keep]
    target = cache["target"][keep]
    weight = cache["weight"][keep]
    e_lgbm = cache["e_lgbm"][keep]
    assert np.isfinite(e_lgbm).all(), "过滤后仍有非有限 e_lgbm，需要先查清楚再往下算"

    order = np.argsort(time_id, kind="stable")
    time_id, asset_id = time_id[order], asset_id[order]
    target, weight, e_lgbm = target[order], weight[order], e_lgbm[order]
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(np.float64)
    group_mean = np.repeat(np.add.reduceat(target, starts) / counts, counts.astype(int))
    e = target - group_mean
    residual = e - e_lgbm

    print(f"{len(time_id):,} 行 / {len(starts):,} 个采样 time_id / {N_ASSETS} 资产\n")

    # ---- 1. 逐资产 IC 贡献
    print("### 逐资产：生产 e_lgbm 对 e 的加权 IC（越高＝树目前做得越好）\n")
    rows = []
    for aid in range(N_ASSETS):
        mask = asset_id == aid
        ic, a, b, d = weighted_ic(e[mask], e_lgbm[mask], weight[mask])
        var_e = float(np.average(e[mask] ** 2, weights=np.maximum(weight[mask], 0.0)))
        var_resid = float(np.average(residual[mask] ** 2, weights=np.maximum(weight[mask], 0.0)))
        rows.append({"asset": aid, "n": int(mask.sum()), "ic": ic,
                    "var_e": var_e, "var_residual": var_resid,
                    "explained_frac": 1.0 - var_resid / var_e if var_e > 0 else float("nan")})
    per_asset = pd.DataFrame(rows).set_index("asset")
    print(per_asset.to_string(float_format=lambda v: f"{v:+.5f}" if abs(v) < 10 else f"{v:.1f}"))
    print(f"\nIC 展布：max/min = {per_asset['ic'].max() / per_asset['ic'].min():.2f}x "
          f"（max={per_asset['ic'].max():+.5f} @asset{per_asset['ic'].idxmax()}, "
          f"min={per_asset['ic'].min():+.5f} @asset{per_asset['ic'].idxmin()}）")

    # ---- 2/3. 两个相关矩阵：需要按 (time_id, asset_id) 对齐成宽表
    frame = pd.DataFrame({"time_id": time_id, "asset_id": asset_id, "e": e, "residual": residual})
    wide_e = frame.pivot_table(index="time_id", columns="asset_id", values="e")
    wide_resid = frame.pivot_table(index="time_id", columns="asset_id", values="residual")
    print(f"\n宽表覆盖率：e {wide_e.notna().mean().mean():.3f}，"
          f"residual {wide_resid.notna().mean().mean():.3f}（1.0＝每个采样 time_id 15 资产全在）")

    corr_e = wide_e.corr()
    corr_resid = wide_resid.corr()

    def summarize_corr(name: str, corr: pd.DataFrame) -> None:
        off_diag = corr.to_numpy()[~np.eye(N_ASSETS, dtype=bool)]
        print(f"\n### {name} 两两相关：均值 {off_diag.mean():+.4f}，"
              f"std {off_diag.std():.4f}，[{off_diag.min():+.4f}, {off_diag.max():+.4f}]")
        print(corr.round(3).to_string())

    summarize_corr("原始截面残差 e", corr_e)
    summarize_corr("未解释残差 e−e_lgbm", corr_resid)

    # ---- 4. 层次聚类
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    def cluster(corr: pd.DataFrame, name: str) -> None:
        dist = 1.0 - corr.to_numpy()
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="average")
        print(f"\n### {name} 层次聚类（average linkage，1−corr 距离）")
        for k in (2, 3, 4, 5):
            labels = fcluster(link, t=k, criterion="maxclust")
            groups: dict[int, list[int]] = {}
            for aid, lab in zip(corr.index, labels):
                groups.setdefault(int(lab), []).append(int(aid))
            print(f"  k={k}: " + " | ".join(str(sorted(v)) for v in groups.values()))

    cluster(corr_e, "原始截面残差 e")
    cluster(corr_resid, "未解释残差 e−e_lgbm")

    print("\n本诊断只读，未训练任何模型，不构成裁决。")


if __name__ == "__main__":
    main()
