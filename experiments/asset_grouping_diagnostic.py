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
5.（2026-08-23 新增）**oracle vs 可部署**：把「搭档的真实 e」换成「模型自身对搭档的预测
   `ê`」之后，驱动相关还剩多少 —— 这一项直接给 ROADMAP §5 留的重新开放条件一个数字。

⚠️ **2026-08-23 订正一处弱论证**：此前把「`e` 与残差两个相关矩阵几乎逐位相同」读成
「生产模型完全没碰这部分结构」。那个不动其实是**算术必然** —— 模型只解释约 0.4% 的方差
（本文件第 1 项的 `explained_frac` 自己就印着），拿掉 0.4% 看不出相关矩阵变化。
该结论**另有独立支撑**：`asset_id` 是 categorical 分裂，树看不到「另一个资产这一刻在干什么」，
那是代码事实、不依赖这两个矩阵。⟹ 结论成立，论证换掉。

⚠️ 本诊断只读、不训练、不构成裁决。产物落 `outputs/experiments/asset_grouping_diagnostic.{json,md}`
（2026-08-23 补上；此前只 print，ROADMAP §5 因此标着「无产物文件」）。

数据来源：生产 OOF cache（`function_class_probe.CACHE_PATH`），5 折验证段拼接、
覆盖全部采样 time_id，不需要重新训练或重新扫 parquet。
"""

from __future__ import annotations

import argparse
import json
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
from xs_peer_pair_probe import PAIRS  # noqa: E402

OUTPUT_DIR = _REPO_ROOT / "outputs" / "experiments"
LABEL = "asset_grouping_diagnostic"

# ROADMAP §5 记录的三对当期相关。新版若算不出同样的数，说明缓存或口径被换过 —— 当场失败，
# 不要让一份「看起来正常」的产物覆盖掉旧结论。
RECORDED_PAIR_CORR = {(0, 6): 0.183, (2, 14): 0.125, (1, 13): 0.119}
PAIR_CORR_TOL = 0.005


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--label", default=LABEL)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and json_path.exists():
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")
    report: dict[str, object] = {"experiment": LABEL, "readonly": True,
                                 "cache": str(CACHE_PATH)}

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
    report["rows"] = int(len(time_id))
    report["groups"] = int(len(starts))
    report["per_asset"] = {int(k): {kk: float(vv) for kk, vv in v.items()}
                           for k, v in per_asset.to_dict("index").items()}
    print(f"\nIC 展布：max/min = {per_asset['ic'].max() / per_asset['ic'].min():.2f}x "
          f"（max={per_asset['ic'].max():+.5f} @asset{per_asset['ic'].idxmax()}, "
          f"min={per_asset['ic'].min():+.5f} @asset{per_asset['ic'].idxmin()}）")

    # ---- 2/3. 两个相关矩阵：需要按 (time_id, asset_id) 对齐成宽表
    frame = pd.DataFrame({"time_id": time_id, "asset_id": asset_id, "e": e, "residual": residual})
    frame_ehat = pd.DataFrame({"time_id": time_id, "asset_id": asset_id, "ehat": e_lgbm})
    wide_e = frame.pivot_table(index="time_id", columns="asset_id", values="e")
    wide_resid = frame.pivot_table(index="time_id", columns="asset_id", values="residual")
    print(f"\n宽表覆盖率：e {wide_e.notna().mean().mean():.3f}，"
          f"residual {wide_resid.notna().mean().mean():.3f}（1.0＝每个采样 time_id 15 资产全在）")

    corr_e = wide_e.corr()
    corr_resid = wide_resid.corr()

    corr_summary: dict[str, dict[str, float]] = {}

    def summarize_corr(name: str, corr: pd.DataFrame, key: str) -> None:
        off_diag = corr.to_numpy()[~np.eye(N_ASSETS, dtype=bool)]
        print(f"\n### {name} 两两相关：均值 {off_diag.mean():+.4f}，"
              f"std {off_diag.std():.4f}，[{off_diag.min():+.4f}, {off_diag.max():+.4f}]")
        print(corr.round(3).to_string())
        corr_summary[key] = {"mean": float(off_diag.mean()), "std": float(off_diag.std()),
                             "min": float(off_diag.min()), "max": float(off_diag.max())}

    summarize_corr("原始截面残差 e", corr_e, "e")
    summarize_corr("未解释残差 e−e_lgbm", corr_resid, "residual")
    report["offdiag_summary"] = corr_summary
    # 零和约束把均值机械压到 −1/(N−1)；真正的信号是**偏离这条基线**的对子，不是相关本身。
    report["zero_sum_baseline"] = {"theoretical": -1.0 / (N_ASSETS - 1),
                                   "observed_mean": corr_summary["e"]["mean"]}
    report["pair_correlations"] = {
        f"{i}-{j}": {"corr": float(corr_e.loc[i, j]),
                     "above_baseline": float(corr_e.loc[i, j] - corr_summary["e"]["mean"])}
        for (i, j) in RECORDED_PAIR_CORR}
    # 与 ROADMAP §5 记录的三对数对拍 —— 对不上说明缓存/口径被换过，当场失败
    for (i, j), expected in RECORDED_PAIR_CORR.items():
        got = float(corr_e.loc[i, j])
        if abs(got - expected) > PAIR_CORR_TOL:
            raise SystemExit(f"对 ({i},{j}) 当期相关 {got:+.4f} 与 ROADMAP 记录的 "
                             f"{expected:+.3f} 差超过 {PAIR_CORR_TOL} ⟹ 先查缓存与口径")

    # ---- 4. 层次聚类
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    clusters: dict[str, dict[int, list[list[int]]]] = {}

    def cluster(corr: pd.DataFrame, name: str, key: str) -> None:
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
            clusters.setdefault(key, {})[k] = [sorted(v) for v in groups.values()]

    cluster(corr_e, "原始截面残差 e", "e")
    cluster(corr_resid, "未解释残差 e−e_lgbm", "residual")
    report["clusters"] = clusters

    # ---- 5. oracle vs 可部署：换成模型自身预测之后，驱动相关还剩多少
    #
    # 这一项直接回答 ROADMAP §5 给 peer 对轴留的重新开放条件。三个口径：
    #   ① oracle 滞后    corr(e_i(t),  e_j(t−1))   —— xs_peer_pair_probe 实际用的量
    #   ② 可部署 滞后    corr(e_i(t), ê_j(t−1))    —— 推理端真拿得到的量
    #   ③ 可部署 当期    corr(e_i(t), ê_j(t))      —— 同一次 predict 内两阶段
    print("\n### 5. oracle vs 可部署：驱动相关（i 的当期 e 对上不同版本的搭档量）\n")
    wide_ehat = frame_ehat.pivot_table(index="time_id", columns="asset_id", values="ehat")
    driver = []
    for i, j in PAIRS.items():
        ei = wide_e[i].to_numpy(); ej = wide_e[j].to_numpy(); hj = wide_ehat[j].to_numpy()
        m = np.isfinite(ei[1:]) & np.isfinite(ej[:-1]) & np.isfinite(hj[:-1])
        m0 = np.isfinite(ei) & np.isfinite(hj)
        driver.append({
            "target": int(i), "partner": int(j),
            "oracle_lag": float(np.corrcoef(ei[1:][m], ej[:-1][m])[0, 1]),
            "deployable_lag": float(np.corrcoef(ei[1:][m], hj[:-1][m])[0, 1]),
            "deployable_now": float(np.corrcoef(ei[m0], hj[m0])[0, 1])})
    driver_df = pd.DataFrame(driver).set_index(["target", "partner"])
    print(driver_df.to_string(float_format=lambda v: f"{v:+.5f}"))
    abs_mean = driver_df.abs().mean()
    signs = (driver_df > 0).sum()
    print(f"\n|均值|  oracle {abs_mean['oracle_lag']:.5f}  "
          f"可部署滞后 {abs_mean['deployable_lag']:.5f}"
          f"（存活 {abs_mean['deployable_lag']/abs_mean['oracle_lag']:.1%}）  "
          f"可部署当期 {abs_mean['deployable_now']:.5f}"
          f"（存活 {abs_mean['deployable_now']/abs_mean['oracle_lag']:.1%}）")
    print(f"同号数 {signs['oracle_lag']}/6 → {signs['deployable_lag']}/6 → "
          f"{signs['deployable_now']}/6")
    explain = {int(a): float(np.corrcoef(
        wide_e[a][wide_e[a].notna() & wide_ehat[a].notna()],
        wide_ehat[a][wide_e[a].notna() & wide_ehat[a].notna()])[0, 1])
        for a in range(N_ASSETS)}
    print(f"根因：corr(e_j, ê_j) 逐资产只有 "
          f"{min(explain.values()):.3f}~{max(explain.values()):.3f} "
          f"⟹ ê_j 是 e_j 的极弱代理")
    report["driver_correlations"] = {f"{d['target']}<-{d['partner']}": d for d in driver}
    report["driver_abs_mean"] = {k: float(v) for k, v in abs_mean.items()}
    report["driver_sign_agreement"] = {k: int(v) for k, v in signs.items()}
    report["model_explains_partner"] = explain
    report["reopening_condition_verdict"] = (
        "载体只剩 %.1f%%（滞后）/ %.1f%%（当期）幅度，同号从 6/6 掉到 %d/6 和 %d/6 ⟹ "
        "ROADMAP §5 那条重新开放条件已定价。裁决见 xs_peer_deployable_probe。"
        % (100 * abs_mean["deployable_lag"] / abs_mean["oracle_lag"],
           100 * abs_mean["deployable_now"] / abs_mean["oracle_lag"],
           signs["deployable_lag"], signs["deployable_now"]))

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    md = [f"# 资产分组只读诊断（`{LABEL}`）", "",
          "> 只读、不训练、不构成裁决。5 项：逐资产 IC / `e` 相关 / 未解释残差相关 / "
          "层次聚类 / oracle vs 可部署。", "",
          f"评估 {report['rows']:,} 行 / {report['groups']:,} 个采样 time_id / {N_ASSETS} 资产。", "",
          "## 零和基线与三对偏离", "",
          f"逐 time_id 零和约束把两两相关均值机械压到 −1/{N_ASSETS - 1} = "
          f"{-1.0/(N_ASSETS-1):+.4f}（实测 {corr_summary['e']['mean']:+.4f}）；"
          "真正的信号是**偏离这条基线**的对子。", "",
          "| 对 | 当期 corr | 偏离基线 |", "|---|--:|--:|"]
    for key, v in report["pair_correlations"].items():
        md.append(f"| ({key.replace('-', ',')}) | {v['corr']:+.4f} | {v['above_baseline']:+.4f} |")
    md += ["", "## oracle vs 可部署（ROADMAP §5 重新开放条件的定价）", "",
           "| i←j | ① oracle 滞后 `e_j` | ② 可部署 滞后 `ê_j` | ③ 可部署 当期 `ê_j` |",
           "|---|--:|--:|--:|"]
    for d in driver:
        md.append(f"| {d['target']}←{d['partner']} | {d['oracle_lag']:+.5f} | "
                  f"{d['deployable_lag']:+.5f} | {d['deployable_now']:+.5f} |")
    md += [f"| **\\|均值\\|** | **{abs_mean['oracle_lag']:.5f}** | "
           f"**{abs_mean['deployable_lag']:.5f}**"
           f"（存活 {abs_mean['deployable_lag']/abs_mean['oracle_lag']:.1%}）| "
           f"**{abs_mean['deployable_now']:.5f}**"
           f"（存活 {abs_mean['deployable_now']/abs_mean['oracle_lag']:.1%}）|",
           f"| **同号数** | **{signs['oracle_lag']}/6** | **{signs['deployable_lag']}/6** | "
           f"**{signs['deployable_now']}/6** |", "",
           f"根因：`corr(e_j, ê_j)` 逐资产只有 "
           f"{min(explain.values()):.3f}~{max(explain.values()):.3f}"
           f" ⟹ `ê_j` 是 `e_j` 的极弱代理。", "",
           f"> {report['reopening_condition_verdict']}", ""]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n本诊断只读，未训练任何模型，不构成裁决。\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
