"""P0 前置诊断:m_t(每 time_id 加权截面均值)的自相关衰减形状 + 市场共同分量占比稳定性。

- 假设 1:自相关线性衰减到 0(target 窗口重叠,H≈6);备择:指数长尾(状态持续)
- 假设 2:mean(m_t²)/E_w[y²] 在 9 个分区上都在 70% 上下

口径与 HANDOFF §4 的命令一致(原始 weight,不截断负值),p008 应复现:
share≈0.732、ac1≈0.830、sd(m_t)≈0.9277。

用法:.venv/bin/python experiments/mt_diagnostics.py
输出:outputs/experiments/mt_diagnostics.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io import train_files

MAX_LAG = 12
H_THEORY = 6


def partition_mt(path: Path) -> tuple[pd.Series, float, float]:
    """返回 (m_t 序列按 time_id 升序, E_w[y²], mean_w(y))。分批读并按 time_id 聚合。"""
    acc: pd.DataFrame | None = None
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=400_000, columns=["time_id", "weight", "target"]
    ):
        frame = batch.to_pandas()
        frame["wy"] = frame["weight"] * frame["target"]
        frame["wy2"] = frame["weight"] * frame["target"] ** 2
        part = frame.groupby("time_id")[["weight", "wy", "wy2"]].sum()
        acc = part if acc is None else acc.add(part, fill_value=0.0)
    acc = acc.sort_index()
    m_t = acc["wy"] / acc["weight"]
    total_w = float(acc["weight"].sum())
    e_y2 = float(acc["wy2"].sum() / total_w)
    mean_y = float(acc["wy"].sum() / total_w)
    return m_t, e_y2, mean_y


def autocorr_profile(m_t: pd.Series, max_lag: int) -> list[float]:
    return [float(m_t.autocorr(k)) for k in range(1, max_lag + 1)]


def pooled_autocorr(series_list: list[pd.Series], max_lag: int) -> list[float]:
    """把 9 个分区的 (m_t, m_{t+k}) 配对合并后算相关;跨分区边界的配对不产生。"""
    out = []
    for k in range(1, max_lag + 1):
        a_parts, b_parts = [], []
        for m in series_list:
            v = m.to_numpy()
            if len(v) > k:
                a_parts.append(v[:-k])
                b_parts.append(v[k:])
        a = np.concatenate(a_parts)
        b = np.concatenate(b_parts)
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def main() -> None:
    files = train_files(_REPO_ROOT / "data")
    per_partition = []
    series_list: list[pd.Series] = []
    for index, path in enumerate(files):
        m_t, e_y2, mean_y = partition_mt(path)
        share = float((m_t**2).mean() / e_y2)
        row = {
            "partition": f"p{index:03d}",
            "n_time_ids": int(len(m_t)),
            "mean_w_y": mean_y,
            "sd_mt": float(m_t.std()),
            "share": share,
            "ac": autocorr_profile(m_t, MAX_LAG),
        }
        per_partition.append(row)
        series_list.append(m_t)
        print(
            f"{row['partition']}: time_ids={row['n_time_ids']:,} share={share:.3f} "
            f"ac1={row['ac'][0]:.3f} sd={row['sd_mt']:.4f} mean_w(y)={mean_y:+.5f}",
            flush=True,
        )

    pooled = pooled_autocorr(series_list, MAX_LAG)
    ac1 = pooled[0]
    ma_theory = [max(0.0, (H_THEORY - k) / H_THEORY) for k in range(1, MAX_LAG + 1)]
    exp_theory = [ac1**k for k in range(1, MAX_LAG + 1)]

    payload = {
        "definition": "m_t = per-time_id weighted cross-sectional mean of target (raw weights)",
        "share_definition": "mean(m_t^2) / E_w[y^2] per partition",
        "partitions": per_partition,
        "pooled_autocorr": pooled,
        "ma_overlap_theory_H6": ma_theory,
        "exponential_theory_ac1^k": exp_theory,
        "share_min": min(r["share"] for r in per_partition),
        "share_max": max(r["share"] for r in per_partition),
        "mean_w_y_signs": "".join("+" if r["mean_w_y"] > 0 else "-" for r in per_partition),
    }
    out_dir = _REPO_ROOT / "outputs" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mt_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# m_t 诊断:共同分量占比 + 自相关衰减形状",
        "",
        "## 每分区(share = mean(m_t²)/E_w[y²],原始 weight 口径)",
        "",
        "| 分区 | time_ids | share | ac1 | sd(m_t) | mean_w(y) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_partition:
        lines.append(
            f"| {r['partition']} | {r['n_time_ids']:,} | {r['share']:.3f} | "
            f"{r['ac'][0]:.3f} | {r['sd_mt']:.4f} | {r['mean_w_y']:+.5f} |"
        )
    lines += [
        "",
        f"share 范围:{payload['share_min']:.3f} ~ {payload['share_max']:.3f};",
        f"mean_w(y) 符号序列:{payload['mean_w_y_signs']}",
        "",
        "## 自相关衰减(9 分区内配对合并;跨分区不配对)",
        "",
        "| lag | 实测 | MA 重叠理论(H=6) | 指数(ac1^k) |",
        "|---:|---:|---:|---:|",
    ]
    for k in range(MAX_LAG):
        lines.append(f"| {k+1} | {pooled[k]:.3f} | {ma_theory[k]:.3f} | {exp_theory[k]:.3f} |")
    lines += [
        "",
        "观察(结论由人判):实测列更接近哪一列——线性归零(窗口重叠)还是指数长尾(状态持续)。",
    ]
    (out_dir / "mt_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "mt_diagnostics.md")}, indent=2))


if __name__ == "__main__":
    main()
