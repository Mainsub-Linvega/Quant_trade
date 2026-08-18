"""选列筛子对比：把 323→200 的线性单变量筛子拆掉，树会不会变好？

## 问题

生产训练在 **LightGBM 前面**叠了线性单变量筛子（`strategies/v3_hybrid/train.py:329/346`，
`experiments/v3_production_oof.py` 同构）：

```text
xs_selected       = top-200 by |corr(feature, e)|   → 截面 LGBM 块 **和** 市场 LGBM 块
history_positions = top-40  within xs_selected      → history 块
```

判据是 `strategies/v1_ridge/train.py:86-108` 的 |加权 Pearson 相关| 排序 ⟹
**123 个原始特征从未进过任何模型**。而这一刀是任意的：partition_008 上实测
200th=0.00299、201st=0.00295，落差只有 1.33%，没有断崖。

边际相关是线性、单变量的，对「单独看没用、交互才有用」和「只在某些 regime/资产下有用」
的特征完全失明 —— 而这正是 GBM 存在的理由。此前 `ab_featsweep` 测过 feat323，但那是
**线性 Ridge 时代**（+4.4%、7/10 折、未晋级），对树是另一个问题。

## 臂（每个都是单变量改动）

| 臂 | 唯一变化 | 设计列 XS / market |
|---|---|---|
| `xs323` | 截面块看 323 | 484 / 561 |
| `mkt323` | 市场块看 323 | 361 / 807 |
| `both323` | 两块都看 323（**组合臂**，只作梯度诊断） | 484 / 807 |

⭐ history 块自动不变：`history_positions` 是 `xs_selected` 内的 top-40，而 top-200 的
top-40 == 全 323 的 top-40 ⟹ 换宽度后 history 原始列名逐位相同。本脚本**断言**这一点。

## ⚠️ 基准必须现跑，不能用 outputs/cache 里那份 `*_exact`

`v3_production_oof_phasebal_prodwindow_exact.npz` 的时间戳是 2026-08-14 11:12，而
`experiments/v3_production_oof.py` 的**首次提交**是 08-15 11:18 ⟹ 它是由**已不存在的
代码版本**产出的，与当前脚本输出差 `max|Δ(market_ridge)| = 3.37e-05`（约折均 peak 的 2.4%）。
拿它当配对基准 = 跨两个代码版本比较。本实验的基准由当前代码现跑。

用法：
    .venv/bin/python experiments/feature_screen_compare.py \
        --cache-dir <dir> --baseline arm_baseline \
        --arms xs323=arm_xs323 mkt323=arm_mkt323 both323=arm_both323
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# 复用 recency 阶梯里已经验证过的配对/矩/峰值实现，不另写一份（伤疤规则 §3）
from v3_recency_ladder import load, moment_rows, peak_of  # noqa: E402
from market_model import sign_test_p  # noqa: E402
from src.metric import scale_invariant_score  # noqa: E402

MIN_POSITIVE_FOLDS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--baseline", required=True, help="基准 cache 的 label（不含 .npz）")
    p.add_argument("--arms", nargs="+", required=True, metavar="NAME=LABEL")
    p.add_argument("--report-dir", default=None,
                   help="各臂 JSON 报告所在目录（用于核对 history 列不变）；缺省 = cache-dir")
    p.add_argument("--min-relative-gain", type=float, default=0.03,
                   help="相对增益门槛。默认 3% —— 08-14 responder 重新开放条件的原值；"
                        "本项目 1s160/5 折的检出下限是基准 peak 的 6.1%%，1%% 那档没有牙")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="feature_screen_1s160")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def history_names(report_dir: Path, label: str) -> list[list[str]] | None:
    path = report_dir / f"{label}.json"
    if not path.exists():
        return None
    arch = json.loads(path.read_text(encoding="utf-8")).get("architecture", {})
    return arch.get("history_names_per_fold")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    report_dir = Path(args.report_dir) if args.report_dir else cache_dir
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    arms: dict[str, str] = {}
    for item in args.arms:
        if "=" not in item:
            raise SystemExit(f"--arms 需要 NAME=LABEL 形式，收到 {item!r}")
        name, label = item.split("=", 1)
        arms[name] = label

    base = load(args.baseline, cache_dir)
    loaded = {n: load(l, cache_dir) for n, l in arms.items()}

    # ---- 配对前提：验证行逐位相同，否则停
    for name, data in loaded.items():
        for key in ("time_id", "asset_id", "fold", "target", "weight"):
            if not np.array_equal(data[key], base[key]):
                raise AssertionError(f"臂 {name} 的 `{key}` 与基准不一致 ⟹ 配对比较无效")
    print(f"配对前提已核实：{len(loaded)} 个臂与基准逐位同行（{len(base['target']):,} 行）",
          flush=True)

    # ---- 单变量前提：history 原始列名必须与基准相同
    base_hist = history_names(report_dir, args.baseline)
    hist_check: dict[str, Any] = {}
    for name, label in arms.items():
        arm_hist = history_names(report_dir, label)
        if base_hist is None or arm_hist is None:
            hist_check[name] = "报告缺 history_names_per_fold，无法核对"
            continue
        same = arm_hist == base_hist
        hist_check[name] = bool(same)
        if not same:
            raise AssertionError(
                f"臂 {name} 的 history 列与基准不同 ⟹ 不是单变量改动，结论会把两件事混在一起")
    print(f"单变量前提：history 列与基准一致 = {hist_check}", flush=True)

    starts = np.r_[0, np.flatnonzero(base["time_id"][1:] != base["time_id"][:-1]) + 1]
    counts = np.diff(np.r_[starts, len(base["time_id"])]).astype(np.int64)
    n_groups = len(starts)
    gidx = np.repeat(np.arange(n_groups), counts)
    group_fold = base["fold"][starts]
    folds = sorted(np.unique(group_fold))

    base_rows = moment_rows(base, gidx, n_groups)
    arm_rows = {n: moment_rows(d, gidx, n_groups) for n, d in loaded.items()}
    reference = scale_invariant_score(base["target"], base["raw"], base["weight"])
    if abs(peak_of(base_rows.sum(axis=0))["peak"] - float(reference["peak"])) > 1e-12:
        raise AssertionError("矩法 peak 与 src.metric 对不上")

    def per_fold(rows: np.ndarray) -> dict[int, dict[str, float]]:
        return {int(f): peak_of(rows[group_fold == f].sum(axis=0)) for f in folds}

    base_fold = per_fold(base_rows)
    base_peaks = np.array([base_fold[int(f)]["peak"] for f in folds])

    rng = np.random.default_rng(args.boot_seed)
    n_blocks = int(np.ceil(n_groups / args.block_size))
    pre_base = np.vstack([np.zeros(3), np.cumsum(base_rows, axis=0)])
    pre_arm = {n: np.vstack([np.zeros(3), np.cumsum(r, axis=0)]) for n, r in arm_rows.items()}
    blocks = [rng.integers(0, max(n_groups - args.block_size, 0) + 1, size=n_blocks)
              for _ in range(args.n_boot)]

    results: dict[str, Any] = {}
    for name, rows in arm_rows.items():
        af = per_fold(rows)
        peaks = np.array([af[int(f)]["peak"] for f in folds])
        delta = peaks - base_peaks
        drop = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
        pos = int((delta > 0).sum())
        dA = np.mean([af[int(f)]["A"] - base_fold[int(f)]["A"] for f in folds])
        dB = np.mean([af[int(f)]["B"] - base_fold[int(f)]["B"] for f in folds])
        rel_A = float(dA / np.mean([base_fold[int(f)]["A"] for f in folds]))
        rel_B = float(dB / np.mean([base_fold[int(f)]["B"] for f in folds]))

        samples = np.empty(args.n_boot)
        for i, st in enumerate(blocks):
            sp = np.minimum(st + args.block_size, n_groups)
            b = peak_of((pre_base[sp] - pre_base[st]).sum(axis=0))["peak"]
            a = peak_of((pre_arm[name][sp] - pre_arm[name][st]).sum(axis=0))["peak"]
            samples[i] = a - b
        ci = {k: float(np.nanpercentile(samples, q))
              for k, q in (("p2.5", 2.5), ("p50", 50.0), ("p97.5", 97.5))}
        floor = float(abs(ci["p97.5"] - ci["p2.5"]) / 2.0)
        rel = float(delta.mean() / base_peaks.mean())

        checks = {
            "1_mean_delta_positive": bool(delta.mean() > 0),
            "2_at_least_4_of_5_folds_positive": bool(pos >= MIN_POSITIVE_FOLDS),
            "3_survives_drop_best_fold": bool(drop.mean() > 0),
            f"4_relative_gain_at_least_{args.min_relative_gain:g}":
                bool(rel >= args.min_relative_gain),
            "5_two_delta_A_exceeds_delta_B": bool(2.0 * rel_A > rel_B),
            "6_paired_bootstrap_ci_lower_bound_positive": bool(ci["p2.5"] > 0),
            "7_exceeds_detection_floor": bool(delta.mean() > floor),
        }
        results[name] = {
            "arm_cache": arms[name],
            "history_identical_to_baseline": hist_check.get(name),
            "per_fold_delta": [float(v) for v in delta],
            "mean_delta": float(delta.mean()), "relative": rel,
            "mean_delta_drop_best": float(drop.mean()),
            "relative_drop_best": float(drop.mean() / base_peaks.mean()),
            "positive_folds": pos, "n_folds": len(folds),
            "sign_test_p": sign_test_p(pos, len(folds)),
            "mechanism": {"relative_delta_A": rel_A, "relative_delta_B": rel_B},
            "paired_bootstrap": ci, "detection_floor": floor,
            "checks": checks, "pass": all(checks.values()),
        }
        print(f"  {name:10s} Δ折均 {delta.mean():+.3e}（{rel*100:+.2f}%）正折 {pos}/{len(folds)}"
              f" 去最好折 {drop.mean():+.3e} 检出下限 {floor:.2e}"
              f" CI [{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] "
              f"{'PASS' if all(checks.values()) else 'FAIL'}", flush=True)

    passed = [n for n, r in results.items() if r["pass"]]
    payload = {
        "experiment": "feature_screen_compare",
        "question": "把 323→200 的线性单变量筛子拆掉，树会不会变好？",
        "why_never_tested": ("ab_featsweep 测过 feat323 但那是线性 Ridge 时代（+4.4%、7/10、"
                             "未晋级）；joint_recalibration_plan 里 Ridge 有 200/323 两档、"
                             "LGBM 那 9 格只变容量和轮数，没有 feature_count 档"),
        "cut_is_arbitrary": ("partition_008 实测 |corr(feature,e)| 200th=0.00299 / "
                             "201st=0.00295，落差 1.33%，无断崖"),
        "baseline_note": ("基准由**当前代码**现跑。outputs/cache 里的 "
                          "v3_production_oof_phasebal_prodwindow_exact.npz 时间戳 08-14 11:12，"
                          "早于该脚本首次提交 08-15 11:18 ⟹ 出自已不存在的代码版本，"
                          "与当前输出差 max|Δ(market_ridge)|=3.37e-05，不可用作配对基准"),
        "single_variable_guarantee": ("history 列名与基准逐折相同（已断言）—— top-200 的 top-40 "
                                      "== 全 323 的 top-40，所以换宽度不动 history 块"),
        "gates": {"min_relative_gain": args.min_relative_gain,
                  "min_positive_folds": MIN_POSITIVE_FOLDS,
                  "note": "1s160/5 折的检出下限历史实测为基准 peak 的 6.1%，故门槛取 3% 并"
                          "另加第 7 道「超过本次实测检出下限」"},
        "baseline": {"cache": args.baseline,
                     "per_fold_peak": {str(f): base_fold[int(f)]["peak"] for f in folds},
                     "mean_peak": float(base_peaks.mean())},
        "arms": results,
        "verdict": {"passed": passed,
                    "decision": "ENTER_STAGE3" if passed else "REJECT"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = ["# 选列筛子：323 vs 200（截面块 / 市场块分开测）", "",
             f"> {payload['question']}", "",
             f"基准折均 peak **{base_peaks.mean():.8f}**；{len(base['target']):,} 行验证数据。", "",
             f"> **单变量保证**：{payload['single_variable_guarantee']}", "",
             f"> ⚠️ **基准来源**：{payload['baseline_note']}", "",
             "| 臂 | Δ折均 | 相对 | 正折 | 去最好折 | ΔA | ΔB | 检出下限 | 配对 CI | 判定 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|:--:|"]
    for name, r in results.items():
        ci = r["paired_bootstrap"]
        lines.append(
            f"| `{name}` | {r['mean_delta']:+.3e} | {r['relative']*100:+.2f}% | "
            f"{r['positive_folds']}/{r['n_folds']} | {r['mean_delta_drop_best']:+.3e} | "
            f"{r['mechanism']['relative_delta_A']*100:+.2f}% | "
            f"{r['mechanism']['relative_delta_B']*100:+.2f}% | {r['detection_floor']:.2e} | "
            f"[{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] | {'✅' if r['pass'] else '❌'} |")
    lines += ["", "## 逐臂门槛", ""]
    for name, r in results.items():
        lines.append(f"### `{name}`（{'✅ 通过' if r['pass'] else '❌ 不通过'}）")
        lines += [f"- {'✅' if v else '❌'} {k}" for k, v in r["checks"].items()]
        lines.append("")
    lines += [f"## 裁决：{payload['verdict']['decision']}", "",
              ("通过臂：" + ", ".join(passed)) if passed else "没有臂通过预注册门禁。", "",
              f"> {payload['why_never_tested']}", ""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n裁决：{payload['verdict']['decision']}\nwrote {json_path}\nwrote {md_path}",
          flush=True)


if __name__ == "__main__":
    main()
