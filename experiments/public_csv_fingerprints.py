"""把历史公榜 CSV 压成指纹存档 —— 删掉那 1.1 GB 之前必须先跑这个。

## 为什么这些 CSV 不是「没用的旧文件」

它们是**「预测向量 ↔ 公榜分数」的唯一物证**。2026-08-09 就是靠
`submission_strict_scale113.csv` + `submission_hybrid_base0856.csv` 两份，
在**不花一次额度**的情况下解出了整条 `blend_weight` 曲线（见 NOTES）。
而且 legacy 求解器（`e2bec9a9`）与 alpha=5e5 那两个模型**已经不在仓库里**，
CSV 删了就永远重建不出来。

## 存什么

关键在于**不能只存标量**。上面那套推理用到的是**两个模型预测之间的交叉矩**
`Σfᵢfⱼ/n` —— 单文件的 `mean(f²)` 给不出来。所以这里连**两两 Gram 矩阵**一起存：

    有了 Gram 矩阵，任何线性组合 g = Σcᵢfᵢ 的 Σg²/n = cᵀGc 都还能算出来，
    也就是**任何线性族的 B 都还能算**。

1.1 GB → 约 1 KB，做那类分析所需的信息一点不丢。
（丢掉的是逐行的值本身，那只有「重新提交同一份 CSV」才需要 —— 而模型在库里，重跑即可。）

## 自检

`MANIFEST` 里「同一个模型、不同 scale」的那些文件，除以各自的 scale 之后**必须相等**。
脚本会实测这一点：对不上就说明 MANIFEST 的归属写错了，直接报错而不是写出一份错的存档。

## ⚠️ 存档是**累积**的（2026-08-10 修）

本脚本只能对**当前还在**的 CSV 求 Gram。原来的写法直接覆盖 json ——
于是「删了 CSV → 再跑一次归档」会把之前存的全冲掉，正好毁掉这份存档存在的理由。
08-10 就这么踩了一次（14 份被压成 4 份，靠 git 捞回来的）。
现在 `merge_with_archive()` 把已删 CSV 的记录原样搬过来，
**两份从未同时在场的 CSV 之间那一格是 `null`（不可知），不是 0** ——
写 0 会让下游默默算出错的 B。

用法：.venv/bin/python experiments/public_csv_fingerprints.py
输出：outputs/experiments/public_csv_fingerprints.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.artifact import sha256_file

OUTPUTS = _REPO_ROOT / "outputs"
EXPERIMENTS = OUTPUTS / "experiments"

# 每份 CSV 的归属。model 相同 = 同一组系数，只差 prediction_scale。
# public 是 experiments/ledger.csv 里对应那一行的公榜分；None 表示这份没交过。
# ⚠️ certain=False 的两条是按文件 mtime 与 ledger 日期对上的，没有更硬的凭据；
#    脚本的同模型一致性自检会顺带验证它们（对不上就会报错）。
MANIFEST: list[dict[str, Any]] = [
    {"file": "baseline_submission.csv", "model": "baseline_v1",
     "scale": 0.5, "public": 0.00119088, "certain": False,
     "note": "ledger 2026-07-23 首个 baseline；按 mtime 07-22 对上"},

    {"file": "submission.csv", "model": "legacy_a2e6",
     "scale": 0.6424065227113341, "public": 0.00151886, "certain": False,
     "note": "ledger 2026-08-07 baseline-v2；按 mtime 08-07 对上"},
    {"file": "submission_scale05.csv", "model": "legacy_a2e6",
     "scale": 0.5, "public": 0.00128602, "certain": True,
     "note": "ledger 2026-08-08 单变量隔离 scale 0.6424→0.5"},
    {"file": "submission_scale05_8dp.csv", "model": "legacy_a2e6",
     "scale": 0.5, "public": 0.00128602, "certain": True, "note": "上一份的 8 位小数版"},
    {"file": "submission_v3_s113.csv", "model": "legacy_a2e6",
     "scale": 1.13, "public": 0.00186805, "certain": True,
     "note": "ledger 2026-08-08 alpha 回滚 2e6 + scale 1.13。⚠️ 文件名里的 v3 不是 v3_hybrid"},
    {"file": "submission_v3_s113_8dp.csv", "model": "legacy_a2e6",
     "scale": 1.13, "public": 0.00186805, "certain": True, "note": "上一份的 8 位小数版"},

    {"file": "submission_v2_s080.csv", "model": "legacy_a5e5",
     "scale": 0.8, "public": 0.00150852, "certain": True,
     "note": "ledger 2026-08-08 alpha 2e6→5e5 + scale 0.8"},
    {"file": "submission_v2_s080_8dp.csv", "model": "legacy_a5e5",
     "scale": 0.8, "public": 0.00150852, "certain": True, "note": "上一份的 8 位小数版"},
    {"file": "submission_v2_s120.csv", "model": "legacy_a5e5",
     "scale": 1.2, "public": 0.0011693833, "certain": True,
     "note": "ledger 2026-08-08 同模型 scale 0.8→1.2（CSV 按比例缩放）"},

    {"file": "submission_strict_scale113.csv", "model": "strict_ridge",
     "scale": 1.13, "public": 0.00187232, "certain": True,
     "note": "ledger 2026-08-08 严格求解器 c23a8cfb"},
    {"file": "submission_strict_scale092.csv", "model": "strict_ridge",
     "scale": 0.92, "public": 0.0018051540, "certain": True,
     "note": "ledger 2026-08-09 同模型第二点"},

    {"file": "submission_hybrid_base0856.csv", "model": "v3_hybrid_w050",
     "scale": 0.856, "public": None, "certain": True,
     "note": "runner 原始产物（全精度），下面两份都是从它缩放来的"},
    {"file": "submission_hybrid_scale090.csv", "model": "v3_hybrid_w050",
     "scale": 0.90, "public": 0.00213810, "certain": True, "note": "ledger 2026-08-09 第一点"},
    {"file": "submission_hybrid_scale130.csv", "model": "v3_hybrid_w050",
     "scale": 1.30, "public": 0.0022857726, "certain": True, "note": "ledger 2026-08-09 第二点"},

    # ── replace 系列（blend_weight 1.0）。⚠️ **每个轮数是一个独立的 model 标签** ──
    # 它们是嵌套的（480 棵的前 k 棵与 k 轮模型逐棵相同，lgbm_nested_check 验过），
    # 但**预测向量不同** ⟹ 不能共用一个 model 标签，否则同模型一致性自检会误报。
    # 嵌套关系体现在 Gram 里：corr(f160, f480) = 0.972，不是 1。
    {"file": "submission_replace_r80_s116.csv", "model": "v3_hybrid_replace_r080",
     "scale": 1.16, "public": 0.0023682898, "certain": True,
     "note": "ledger 2026-08-09 免训练减轮数（只用前 80 棵）"},
    {"file": "submission_replace_s116.csv", "model": "v3_hybrid_replace_r160",
     "scale": 1.16, "public": 0.0024872338, "certain": True,
     "note": "ledger 2026-08-09 replace 生产轮数。⚠️ B(160)=0.00179844 是**真值**，"
             "整套无权 B 代理都拿它标定"},
    {"file": "submission_r320_s116.csv", "model": "v3_hybrid_replace_r320",
     "scale": 1.16, "public": 0.00256510, "certain": True,
     "note": "ledger 2026-08-10 轮数梯子，出自 outputs/candidates/v3_hybrid_r480 的前 320 棵"},
    {"file": "submission_r480_s116.csv", "model": "v3_hybrid_replace_r480",
     "scale": 1.16, "public": 0.0025821304, "certain": True,
     "note": "ledger 2026-08-10 当前最好成绩；峰值 0.00258931 @ scale 1.1020"},
]

# 8 位小数版相对全精度版的舍入量级，同模型一致性自检按它定门限
EIGHT_DP_TOLERANCE = 1e-7


def merge_with_archive(
    output: Path, fresh_names: list[str], fresh: np.ndarray,
    entries: list[dict[str, Any]], checks: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """把这次算出来的 Gram **并进**上一版存档，而不是覆盖它。

    ## 为什么必须这样

    这份存档的**全部意义**就是「CSV 删了之后还能算」。而本脚本只能对**当前还在**的
    CSV 求 Gram —— 直接覆盖等于：删了 CSV → 跑一次归档 → 把之前存的全冲掉。
    2026-08-10 就这么踩了一次（14 份被压成 4 份，靠 git 捞回来的）。

    ## 不可知的元素必须是 null，不能是 0

    两份从未同时在场的 CSV，它们的内积**谁也算不出来**（需要两条完整向量）。
    写 0 会让下游默默算出错的 B —— 那正是这套方法最不该出错的地方。
    """
    if not output.exists():
        return fresh_names, fresh, entries, checks

    archive = json.loads(output.read_text(encoding="utf-8"))
    old_names: list[str] = archive.get("gram_names", [])
    old_gram = np.array([[np.nan if v is None else v for v in row]
                         for row in archive.get("gram", [])], dtype=np.float64)

    names = old_names + [n for n in fresh_names if n not in old_names]
    index = {name: i for i, name in enumerate(names)}
    merged = np.full((len(names), len(names)), np.nan, dtype=np.float64)

    def paste(block_names: list[str], block: np.ndarray) -> None:
        rows = [index[name] for name in block_names]
        merged[np.ix_(rows, rows)] = block

    if len(old_names):
        paste(old_names, old_gram)
    paste(fresh_names, fresh)          # 这次现算的覆盖旧值（同一份 CSV 结果应一致）

    carried = [name for name in old_names if name not in fresh_names]
    unknown = int(np.isnan(merged).sum())
    print(f"\n并入上一版存档：沿用 {len(carried)} 份已删 CSV 的记录"
          f"（{', '.join(carried) if carried else '无'}）")
    print(f"  Gram {len(names)}×{len(names)}，其中 {unknown} 个元素不可知（null）"
          f" —— 那些是从未同时在场的两份之间的内积")

    known = {entry["file"] for entry in entries}
    entries = entries + [e for e in archive.get("files", []) if e["file"] not in known]
    seen = {(c["a"], c["b"]) for c in checks}
    checks = checks + [c for c in archive.get("same_model_checks", [])
                       if (c["a"], c["b"]) not in seen]
    return names, merged, entries, checks


def main() -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    reference_ids: np.ndarray | None = None
    entries: list[dict[str, Any]] = []
    raw_columns: list[np.ndarray] = []          # 各文件除以 scale 之后的「原始预测」

    for record in MANIFEST:
        path = OUTPUTS / record["file"]
        if not path.exists():
            print(f"跳过 {record['file']}：文件不存在", flush=True)
            continue
        frame = pd.read_csv(path)
        row_ids = frame["row_id"].to_numpy(dtype=np.int64)
        values = frame["target"].to_numpy(dtype=np.float64)
        aligned = True
        if reference_ids is None:
            reference_ids = row_ids
        elif len(row_ids) != len(reference_ids) or not np.array_equal(row_ids, reference_ids):
            aligned = False
            print(f"⚠️ {record['file']} 的 row_id 与其它文件对不齐，不进 Gram 矩阵", flush=True)

        raw = values / record["scale"]
        entry = {
            **record,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": int(len(values)),
            "row_id_aligned": aligned,
            "mean": float(values.mean()),
            "mean_square": float((values * values).mean()),
            "std": float(values.std()),
            "abs_max": float(np.abs(values).max()),
            "raw_mean_square": float((raw * raw).mean()),
            "raw_abs_max": float(np.abs(raw).max()),
            "n_nonfinite": int((~np.isfinite(values)).sum()),
        }
        entries.append(entry)
        raw_columns.append(raw if aligned else None)
        print(f"{record['file']:<40} n={entry['rows']:,}  raw_rms={entry['raw_mean_square']**0.5:.6f}",
              flush=True)

    # ---- 自检：同一个模型的各份，除以 scale 之后必须相等
    checks: list[dict[str, Any]] = []
    by_model: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        if raw_columns[index] is not None:
            by_model.setdefault(entry["model"], []).append(index)
    for model, indices in by_model.items():
        base = indices[0]
        for other in indices[1:]:
            difference = float(np.abs(raw_columns[base] - raw_columns[other]).max())
            checks.append({"model": model, "a": entries[base]["file"],
                           "b": entries[other]["file"], "max_abs_diff_raw": difference,
                           "passed": difference <= EIGHT_DP_TOLERANCE / min(
                               entries[base]["scale"], entries[other]["scale"])})
            status = "✅" if checks[-1]["passed"] else "❌"
            print(f"{status} 同模型 {model}: {entries[base]['file']} vs {entries[other]['file']} "
                  f"→ max|Δraw| = {difference:.2e}", flush=True)
    failed = [check for check in checks if not check["passed"]]
    if failed:
        raise SystemExit(
            f"同模型一致性自检未过（{len(failed)} 组）—— MANIFEST 里的模型归属或 scale 写错了，"
            "先修好再存档，别写出一份错的")

    # ---- Gram 矩阵：Σfᵢfⱼ/n，用**除过 scale 的原始预测**
    usable = [index for index, column in enumerate(raw_columns) if column is not None]
    matrix = np.column_stack([raw_columns[index] for index in usable])
    fresh = (matrix.T @ matrix) / len(matrix)
    fresh_names = [entries[index]["file"] for index in usable]

    output = EXPERIMENTS / "public_csv_fingerprints.json"
    names, gram, entries, checks = merge_with_archive(
        output, fresh_names, fresh, entries, checks)

    payload = {
        "why": "删 CSV 前把「预测向量↔公榜分数」的信息榨出来；Gram 矩阵让任何线性组合的 B 仍可算",
        "identity": "对 g = Σcᵢ·rawᵢ，有 Σg²/n = cᵀGc；B = (Σw·g²/Σw·y²) 的无权近似 ∝ 它",
        "unweighted_b_note":
            "无权近似实测在 B 的**比值**上误差约 1.6%（用 strict_ridge 与 v3_hybrid 的真值标定过）",
        "merge_note":
            "本文件是**累积**的：CSV 已删的那些从上一版存档原样搬过来。"
            "两份从未同时在场的 CSV，它们之间的 Gram 元素是 null（不可知），不是 0",
        "rows": int(len(matrix)),
        "files": entries,
        "same_model_checks": checks,
        "gram_basis": "raw = target / prediction_scale",
        "gram_names": names,
        "gram": [[None if not np.isfinite(v) else float(v) for v in row] for row in gram],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(entries)} 份 CSV → {output}（{output.stat().st_size/1024:.1f} KB）")
    # 只统计**这次真的在场**的那些 —— 沿用的记录对应的文件早就不在了
    present = sum(e["bytes"] for e in entries if (OUTPUTS / e["file"]).exists())
    print(f"当前在场的 CSV 合计 {present/1e9:.2f} GB，已榨干，可以删了")


if __name__ == "__main__":
    main()
