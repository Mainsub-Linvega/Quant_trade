"""轮数是不是「同族嵌套」—— 一次训练能不能白送多个 `num_iteration`。

## 为什么需要这个检查

8/09 测 80 轮是免训练的：生产模型文件本来就有 160 棵树，`predict(num_iteration=80)`
只用前 80 棵。要往上走（320 / 480）就必须重训，于是出现一个问题：

**新训的 480 轮模型，它的前 160 棵树是不是就是生产那 160 棵？**

如果是，一次训练白送 160/240/320/480 四个轮数，而且 `A(160)` 已经有公榜精确值，
整族的比较是干净的；如果不是（数据加载变了、`min_data_in_leaf` 依赖行数所以跟着变了、
或者 bagging 序列与总轮数有关），那么「480 轮的前 160 棵 = 160 轮模型」不成立，
轮数比较里混进了「换了个模型」的成分，8/09 那条 −4.5% 的结论也不能直接外推。

LightGBM 的 boosting 是可加的，且 `deterministic=True` + 固定
`seed/bagging_seed/feature_fraction_seed` 下第 k 棵树只依赖前 k−1 棵 ——
**理论上应该逐棵相同**。但这是需要机械验证的事情，不是可以写在报告里的推测
（伤疤清单 #2：报告结论与实际代码没有机械联系）。

## 怎么验

1. 两边的 `hybrid_meta.json` 必须报同一套选列、同一套预处理统计量、
   同一个 `min_data_in_leaf` —— 有一项不同，树就不在同一个空间里，后面免谈
2. 固定种子造合成设计矩阵，两边各预测一次，`max|Δ|` 必须落在浮点噪声地板内

   ⚠️ 门限取 `1e-10`，与 `strategies/v3_hybrid/main.py` 的开机对拍同一个常数、同一个理由：
   两条路径**只该差求和顺序**（480 棵树的模型与 160 棵树的模型，predict 内部的
   分块/线程划分不同 → 160 个 float64 叶子值的累加顺序不同，实测 ~1e-18）；
   而真翻了一个分裂，输出会跳一个叶子值（~1e-3）。1e-10 在两者中间隔着
   八个数量级，怎么定都不会误判。**卡「恰好 0」是错的** —— 它把浮点求和顺序
   当成了模型不同。

用法：
    .venv/bin/python experiments/lgbm_nested_check.py \\
        --candidate outputs/candidates/v3_hybrid_r480
输出：outputs/experiments/lgbm_nested_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE_SEED = 20260810
PROBE_ROWS = 4_000
# 与 main.py 的 _BACKEND_SELFCHECK_ATOL 同值同理由：浮点求和顺序 ~1e-18，
# 翻一个分裂 ~1e-3，门限取在中间。
NESTING_ATOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that a longer retrain nests the production model.")
    parser.add_argument("--production", default=str(_REPO_ROOT / "strategies" / "v3_hybrid" / "model"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=str(_REPO_ROOT / "outputs" / "experiments" / "lgbm_nested_check.json"))
    return parser.parse_args()


def load_meta(model_dir: Path) -> dict:
    return json.loads((model_dir / "hybrid_meta.json").read_text(encoding="utf-8"))


def compare_meta(production: dict, candidate: dict) -> dict[str, object]:
    """预处理与选列必须逐位相同，否则两组树根本不在同一个特征空间里。"""
    checks: dict[str, object] = {
        "lgbm_features_identical": production["lgbm_features"] == candidate["lgbm_features"],
        "min_data_in_leaf_identical":
            production["lgbm_params"]["min_data_in_leaf"] == candidate["lgbm_params"]["min_data_in_leaf"],
        "train_rows_identical": production["train_rows"] == candidate["train_rows"],
        "sampling_identical": (production["sample_modulo"], production["sampling"])
                              == (candidate["sample_modulo"], candidate["sampling"]),
        "model_files_identical": production["lgbm_model_files"] == candidate["lgbm_model_files"],
    }
    for name in ("lower", "upper", "center", "scale"):
        checks[f"{name}_max_abs_diff"] = float(np.max(np.abs(
            np.asarray(production[name], dtype=np.float64) - np.asarray(candidate[name], dtype=np.float64))))
    return checks


def main() -> None:
    args = parse_args()
    import lightgbm as lgb

    production_dir, candidate_dir = Path(args.production), Path(args.candidate)
    production_meta, candidate_meta = load_meta(production_dir), load_meta(candidate_dir)
    meta_checks = compare_meta(production_meta, candidate_meta)
    print("meta 对拍：")
    for name, value in meta_checks.items():
        print(f"  {name}: {value}")

    width = len(production_meta["lgbm_features"]) + 1        # + asset_id（分类列，放最后）
    rng = np.random.default_rng(PROBE_SEED)
    probe = np.empty((PROBE_ROWS, width), dtype=np.float32)
    # 设计矩阵那 200 列是截面 deviation（已稳健标准化），量级约 N(0,1)、被裁到 ±10；
    # 探针照这个分布造，才会打到真实用到的那些分裂阈值上。
    probe[:, :-1] = np.clip(rng.normal(0.0, 1.0, size=(PROBE_ROWS, width - 1)), -10.0, 10.0)
    probe[:, -1] = rng.integers(0, 15, size=PROBE_ROWS)       # asset_id 0..14

    rounds = int(production_meta["num_iteration"])
    per_seed: list[dict[str, object]] = []
    worst = 0.0
    for name in production_meta["lgbm_model_files"]:
        old = lgb.Booster(model_file=str(production_dir / name))
        new = lgb.Booster(model_file=str(candidate_dir / name))
        difference = float(np.max(np.abs(
            old.predict(probe, num_iteration=rounds) - new.predict(probe, num_iteration=rounds))))
        worst = max(worst, difference)
        per_seed.append({
            "model": name,
            "production_trees": int(old.num_trees()),
            "candidate_trees": int(new.num_trees()),
            "max_abs_diff_at_num_iteration": difference,
        })
        print(f"  {name}: 生产 {old.num_trees()} 棵 / 候选 {new.num_trees()} 棵，"
              f"前 {rounds} 棵预测 max|Δ| = {difference:.3e}")

    nested = worst <= NESTING_ATOL \
        and all(bool(v) for k, v in meta_checks.items() if k.endswith("identical")) \
        and all(v == 0.0 for k, v in meta_checks.items() if k.endswith("max_abs_diff"))

    payload = {
        "question": f"候选模型的前 {rounds} 棵树是否逐棵等于生产模型？",
        "why": "成立才能一次训练白送多个 num_iteration，且 A(160) 的公榜精确值才能继续用",
        "production": str(production_dir), "candidate": str(candidate_dir),
        "probe": {"seed": PROBE_SEED, "rows": PROBE_ROWS, "width": width},
        "meta_checks": meta_checks,
        "per_seed": per_seed,
        "max_abs_diff": worst,
        "nesting_atol": NESTING_ATOL,
        "nested": bool(nested),
        "candidate_num_iteration": int(candidate_meta["num_iteration"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n嵌套成立：{nested}（max|Δ| = {worst:.3e}）→ {output}")
    if not nested:
        print("⚠️ 不成立 —— 轮数比较不是同族的，别把 8/09 那条 −4.5% 直接外推")


if __name__ == "__main__":
    main()
