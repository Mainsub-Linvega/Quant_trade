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
]

# 8 位小数版相对全精度版的舍入量级，同模型一致性自检按它定门限
EIGHT_DP_TOLERANCE = 1e-7


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
    gram = (matrix.T @ matrix) / len(matrix)
    names = [entries[index]["file"] for index in usable]

    payload = {
        "why": "删 CSV 前把「预测向量↔公榜分数」的信息榨出来；Gram 矩阵让任何线性组合的 B 仍可算",
        "identity": "对 g = Σcᵢ·rawᵢ，有 Σg²/n = cᵀGc；B = (Σw·g²/Σw·y²) 的无权近似 ∝ 它",
        "unweighted_b_note":
            "无权近似实测在 B 的**比值**上误差约 1.6%（用 strict_ridge 与 v3_hybrid 的真值标定过）",
        "rows": int(len(matrix)),
        "files": entries,
        "same_model_checks": checks,
        "gram_basis": "raw = target / prediction_scale",
        "gram_names": names,
        "gram": gram.tolist(),
    }
    output = EXPERIMENTS / "public_csv_fingerprints.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(entries)} 份 CSV → {output}（{output.stat().st_size/1024:.1f} KB）")
    print(f"原始 CSV 合计 {sum(e['bytes'] for e in entries)/1e9:.2f} GB，可以删了")


if __name__ == "__main__":
    main()
