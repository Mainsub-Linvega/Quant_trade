"""per-asset ridge 能不能**叠加**到生产截面块上（而不是替换它）？

## 为什么要问

`asset_loading_diagnostic` 的结论是：线性模型内资产异质性很大（per-asset ridge 比共享
ridge 高 +95.8%、5/5 折），但 per-asset ridge 仍比**生产 LGBM 截面块低 50.5%** ——
因为生产截面块是树、`asset_id` 本来就是它的 categorical 特征。

「单独更弱」不等于「叠加无用」：只要它与 LGBM 足够去相关，组合仍可能加分。
⚠️ 但 CLAUDE.md 长期伤疤 §6 写死了：**「低相关」不代表适合集成，弱模型可能只是噪声，
必须看组合后的目标指标** —— 所以本脚本只以**配对 peak 增量**判决，相关系数仅作诊断。

## 设计

- 口径对齐现有 OOF（5 折 / modulo 5 `phase_balanced` / 训练窗 78,960 / embargo 6），
  预处理与选列只在训练折内拟合；ridge 预测按 time_id 投影成无权零均值，
  与生产截面块 `e_lgbm` 同口径（`main.py` 里也是 `e_lgbm -= e_lgbm.mean()`）。
- **基准 = 生产 `e_lgbm` 单独**（在同一批验证行上解出的最优单 scale），不是共享 ridge。
- 候选 = 两系数组合 `c1·e_lgbm + c2·ridge`，系数按**扩展窗口**只用 fold 0..k−1 拟合。
- 两条 ridge 臂：`per_asset`（κ 阶梯）与 `shared`（**对照** —— 若共享 ridge 也能叠加，
  那说明是「ridge 补树」而不是「per-asset 结构补树」）。

判据：折均 >0、≥3/4 折为正、去最好折 >0、相对 ≥1%、配对 bootstrap CI 下界 >0。

用法：OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python experiments/asset_blend_check.py
输出：outputs/experiments/<label>.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "strategies" / "v1_ridge"),
              str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from features import apply_robust_transform, cross_sectional_deviation  # noqa: E402
from lgbm_xs import load_rows  # noqa: E402
from market_model import sign_test_p  # noqa: E402
from src.validation import rolling_time_folds  # noqa: E402
from train import robust_transform_fit, select_features  # noqa: E402

FEATURE_COUNT, RIDGE_ALPHA = 200, 2_000_000.0
SAMPLE_MODULO, SAMPLING = 5, "phase_balanced"
TRAIN_WINDOW, EMBARGO, N_FOLDS = 78_960, 6, 5
KAPPAS = [0.0, 1.0, 10.0, 100.0]
OOF = _REPO_ROOT / "outputs" / "cache" / "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data"))
    p.add_argument("--oof", default=str(OOF))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="asset_blend_check")
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def zero_mean_per_time(values: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(time_id)]).astype(np.float64)
    return values - np.repeat(np.add.reduceat(values, starts) / counts, counts.astype(int))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{args.label}.json", out_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {json_path}; use --force to overwrite")

    started = time.perf_counter()
    data = load_rows(Path(args.data_root), SAMPLE_MODULO, SAMPLING)
    features, target = data["features"], data["target"].astype(np.float64)
    weight = np.maximum(data["weight"].astype(np.float64), 0.0)
    time_ids, asset_ids = data["time_id"], data["asset_id"]
    folds = rolling_time_folds(np.unique(time_ids), N_FOLDS, TRAIN_WINDOW, EMBARGO)
    assets = np.unique(asset_ids)

    with np.load(args.oof, allow_pickle=False) as d:
        if not (np.array_equal(d["time_id"], time_ids) and np.array_equal(d["asset_id"], asset_ids)):
            raise AssertionError("OOF cache 与 load_rows 的行序不一致 ⟹ 无法逐行对齐")
        e_lgbm_all = d["e_lgbm"].astype(np.float64)
        fold_all = d["fold"].astype(np.int16)
    print(f"{len(target):,} 行 / {len(assets)} 资产 / {len(folds)} 折；行序与 OOF cache 对齐 ✅",
          flush=True)

    arms = ["shared"] + [f"per_asset_k{k:g}" for k in KAPPAS]
    ridge_pred = {name: np.full(len(target), np.nan) for name in arms}

    for index, (train_ids, valid_ids) in enumerate(folds):
        tr, va = np.isin(time_ids, train_ids), np.isin(time_ids, valid_ids)
        transformed, stats = robust_transform_fit(features[tr].copy())
        valid_t = features[va].copy()
        apply_robust_transform(valid_t, stats["lower"], stats["upper"], stats["center"], stats["scale"])
        y_tr, w_tr, tid_tr, aid_tr = target[tr], weight[tr], time_ids[tr], asset_ids[tr]
        tid_va, aid_va = time_ids[va], asset_ids[va]

        s = np.r_[0, np.flatnonzero(tid_tr[1:] != tid_tr[:-1]) + 1]
        c = np.diff(np.r_[s, len(tid_tr)]).astype(np.float64)
        e_tr = y_tr - np.repeat(np.add.reduceat(y_tr, s) / c, c.astype(int))
        sel = select_features(transformed, e_tr, np.ones_like(e_tr), FEATURE_COUNT)
        X_tr = cross_sectional_deviation(transformed[:, sel].copy(), tid_tr).astype(np.float64)
        X_va = cross_sectional_deviation(valid_t[:, sel].copy(), tid_va).astype(np.float64)
        del transformed, valid_t; gc.collect()

        alpha = RIDGE_ALPHA * len(train_ids) / TRAIN_WINDOW
        ridge = np.eye(X_tr.shape[1]) * alpha
        Xw = X_tr * w_tr[:, None]
        G_all, b_all = Xw.T @ X_tr, Xw.T @ e_tr
        beta_shared = np.linalg.solve(G_all + ridge, b_all)
        ridge_pred["shared"][va] = zero_mean_per_time(X_va @ beta_shared, tid_va)

        per = {}
        for a in assets:
            m = aid_tr == a
            Xa = X_tr[m] * w_tr[m][:, None]
            per[int(a)] = (Xa.T @ X_tr[m], Xa.T @ e_tr[m])
        pg, pb = G_all / len(assets), b_all / len(assets)
        for k in KAPPAS:
            pred = np.empty(int(va.sum()))
            for a in assets:
                Ga, ba = per[int(a)]
                beta = np.linalg.solve(Ga + k * pg + ridge, ba + k * pb)
                mm = aid_va == a
                pred[mm] = X_va[mm] @ beta
            ridge_pred[f"per_asset_k{k:g}"][va] = zero_mean_per_time(pred, tid_va)
        print(f"  fold {index} 完成 [{time.perf_counter()-started:.0f}s]", flush=True)
        del X_tr, X_va, per; gc.collect()

    keep = fold_all >= 0
    y, w, tid, fo = target[keep], weight[keep], time_ids[keep], fold_all[keep]
    lgbm = e_lgbm_all[keep]
    starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]
    gidx = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(tid)]).astype(int))
    n_groups, group_fold = len(starts), fo[starts]
    fold_list = sorted(np.unique(group_fold))

    def moments(p1, p2):
        """逐 time_id 的 (D, y·p1, y·p2, p1², p1p2, p2²)。"""
        cols = [w*y*y, w*y*p1, w*y*p2, w*p1*p1, w*p1*p2, w*p2*p2]
        return np.column_stack([np.bincount(gidx, weights=cc, minlength=n_groups) for cc in cols])

    def solve2(t):
        G = np.array([[t[3], t[4]], [t[4], t[5]]]); v = np.array([t[1], t[2]])
        return np.linalg.solve(G, v)

    def score(t, coef):
        G = np.array([[t[3], t[4]], [t[4], t[5]]]); v = np.array([t[1], t[2]])
        return float((2*coef@v - coef@G@coef) / t[0])

    def base_score(t):
        a, b = t[1]/t[0], t[3]/t[0]
        return a*a/b if b > 0 else 0.0

    rng = np.random.default_rng(args.boot_seed)
    nb = int(np.ceil(n_groups / args.block_size))
    blocks = [rng.integers(0, max(n_groups-args.block_size, 0)+1, size=nb) for _ in range(args.n_boot)]
    results: dict[str, Any] = {}

    for name in arms:
        r = ridge_pred[name][keep]
        rows = moments(lgbm, r)
        per_fold = {int(f): rows[group_fold == f].sum(axis=0) for f in fold_list}
        deltas, bases = [], []
        for i, f in enumerate(fold_list):
            if i == 0:
                continue
            tr_tot = np.sum([per_fold[g] for g in fold_list[:i]], axis=0)
            coef = solve2(tr_tot)
            ev = per_fold[int(f)]
            b, cand = base_score(ev), score(ev, coef)
            deltas.append(cand - b); bases.append(b)
        deltas, bases = np.array(deltas), np.array(bases)
        drop = np.delete(deltas, int(np.argmax(deltas))) if len(deltas) > 1 else deltas
        pos = int((deltas > 0).sum())
        # 配对 bootstrap：系数冻结，只对评估折的 time_id 重采样
        frozen = {int(f): solve2(np.sum([per_fold[g] for g in fold_list[:i]], axis=0))
                  for i, f in enumerate(fold_list) if i > 0}
        prefix = np.vstack([np.zeros(6), np.cumsum(rows, axis=0)])
        samples = []
        for st in blocks:
            sp = np.minimum(st + args.block_size, n_groups)
            tot = (prefix[sp] - prefix[st]).sum(axis=0)
            coef = frozen[fold_list[-1]]
            samples.append(score(tot, coef) - base_score(tot))
        boot = np.percentile(samples, [2.5, 50, 97.5])
        checks = {
            "1_mean_delta_positive": bool(deltas.mean() > 0),
            "2_at_least_3_of_4_folds_positive": bool(pos >= 3),
            "3_survives_drop_best_fold": bool(drop.mean() > 0),
            "4_relative_gain_at_least_1pct": bool(deltas.mean()/bases.mean() >= 0.01),
            "5_paired_bootstrap_ci_lower_bound_positive": bool(boot[0] > 0),
        }
        corr = float(np.corrcoef(lgbm, r)[0, 1])
        results[name] = {"mean_delta": float(deltas.mean()), "relative": float(deltas.mean()/bases.mean()),
                         "mean_delta_drop_best": float(drop.mean()),
                         "positive_folds": pos, "n_folds": len(deltas),
                         "sign_test_p": sign_test_p(pos, len(deltas)),
                         "corr_with_lgbm": corr,
                         "paired_bootstrap": {"p2.5": boot[0], "p50": boot[1], "p97.5": boot[2]},
                         "checks": checks, "pass": all(checks.values())}
        print(f"  {name:16s} Δ折均 {deltas.mean():+.3e}（{deltas.mean()/bases.mean()*100:+.2f}%）"
              f" 正折 {pos}/{len(deltas)} corr(lgbm)={corr:+.3f} "
              f"{'PASS' if all(checks.values()) else 'FAIL'}", flush=True)

    payload = {"experiment": "asset_blend_check",
               "question": "per-asset ridge 能不能叠加到生产截面块上（而不是替换）？",
               "baseline": "生产 e_lgbm 单独（同批验证行上的最优单 scale）",
               "scar_note": "CLAUDE.md §8.6：低相关≠适合集成 ⟹ 只以配对 peak 增量判决，"
                            "corr 仅作诊断",
               "control_note": "`shared` 臂是对照：若它也能叠加，说明是「ridge 补树」"
                               "而不是「per-asset 结构补树」",
               "arms": results, "elapsed_seconds": time.perf_counter()-started}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    lines = ["# per-asset ridge 能否叠加到生产截面块", "",
             f"基准：{payload['baseline']}。", "",
             f"> {payload['scar_note']}", "", f"> {payload['control_note']}", "",
             "| 臂 | Δ折均 | 相对 | 正折 | 去最好折 | corr(与 e_lgbm) | 配对 CI | 判定 |",
             "|---|---:|---:|---:|---:|---:|---|:--:|"]
    for name in arms:
        r = results[name]; ci = r["paired_bootstrap"]
        lines.append(f"| `{name}` | {r['mean_delta']:+.3e} | {r['relative']*100:+.2f}% | "
                     f"{r['positive_folds']}/{r['n_folds']} | {r['mean_delta_drop_best']:+.3e} | "
                     f"{r['corr_with_lgbm']:+.3f} | [{ci['p2.5']:+.2e}, {ci['p97.5']:+.2e}] | "
                     f"{'✅' if r['pass'] else '❌'} |")
    passed = [n for n in arms if results[n]["pass"]]
    lines += ["", f"## 判定：{'✅ ' + ', '.join(passed) if passed else '❌ 全部不通过'}", ""]
    md_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"wrote {json_path}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
