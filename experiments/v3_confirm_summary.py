"""Machine-judge the 3-seed/480-round confirmation and select the final local arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.metric import scale_invariant_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", default=str(_REPO_ROOT / "outputs" / "cache" /
                                         "v3_production_oof_confirm_3s480_phasebal_prodwindow.npz"))
    p.add_argument("--adapter", default=str(_REPO_ROOT / "outputs" / "experiments" /
                                             "v3_residual_adapters_confirm_3s480_shrink100.json"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "experiments"))
    p.add_argument("--label", default="v3_confirm_3s480_decision")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def paired(candidate: np.ndarray, baseline: np.ndarray, need_positive: int) -> dict[str, Any]:
    delta = candidate - baseline
    drop = np.delete(delta, int(np.argmax(delta))) if len(delta) > 1 else delta
    base = float(baseline.mean())
    out = {"baseline_mean": base, "candidate_mean": float(candidate.mean()),
           "relative_gain": float(delta.mean() / base),
           "positive_folds": int((delta > 0).sum()), "n_folds": int(len(delta)),
           "drop_best_relative_gain": float(drop.mean() / base),
           "per_fold_delta": [float(v) for v in delta]}
    out["pass"] = bool(out["relative_gain"] >= 0.01 and out["positive_folds"] >= need_positive
                       and out["drop_best_relative_gain"] > 0)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.label}.json"; md_path = output_dir / f"{args.label}.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit("output exists; use --force")
    with np.load(args.oof, allow_pickle=False) as d:
        valid = d["fold"] >= 0
        y, w, fold = d["target"][valid], np.maximum(d["weight"][valid], 0), d["fold"][valid]
        p160 = d["prediction_raw_checkpoint_160"][valid]
        p480 = d["prediction_raw_checkpoint_480"][valid]
    market160, market480 = [], []
    for current in range(5):
        mask = fold == current
        market160.append(scale_invariant_score(y[mask], p160[mask], w[mask])["peak"])
        market480.append(scale_invariant_score(y[mask], p480[mask], w[mask])["peak"])
    market_gate = paired(np.asarray(market480), np.asarray(market160), 4)
    adapter_payload = json.loads(Path(args.adapter).read_text(encoding="utf-8"))
    adapter_peak = dict(adapter_payload["summary"]["cross_asset"]["peak"])
    adapter_fixed = dict(adapter_payload["summary"]["cross_asset"]["frozen_scale_score"])
    adapter_pass = bool(adapter_peak["relative_gain"] >= 0.01
                        and adapter_peak["positive_folds"] >= 3
                        and adapter_peak["relative_gain_drop_best"] > 0
                        and adapter_fixed["mean_delta"] > 0
                        and adapter_fixed["positive_folds"] >= 3)
    payload = {"experiment": "v3_confirm_3s480", "oof": args.oof,
               "market_480_vs_160": market_gate,
               "asset_adapter_vs_market480": {"peak": adapter_peak,
                                                "frozen_scale_score": adapter_fixed,
                                                "pass": adapter_pass},
               "decision": "market480_plus_asset_adapter" if market_gate["pass"] and adapter_pass
                           else "retain_current_or_investigate",
               "submission_generated": False}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v3 3-seed × 480 confirmation", "",
             f"Decision: **{payload['decision']}**", "",
             "| Gate | Gain | Positive | Drop-best | Result |", "|---|---:|---:|---:|:---:|",
             f"| market480 vs market160 | {market_gate['relative_gain']*100:+.2f}% | "
             f"{market_gate['positive_folds']}/{market_gate['n_folds']} | "
             f"{market_gate['drop_best_relative_gain']*100:+.2f}% | {'PASS' if market_gate['pass'] else 'FAIL'} |",
             f"| asset adapter vs market480 | {adapter_peak['relative_gain']*100:+.2f}% | "
             f"{adapter_peak['positive_folds']}/{adapter_peak['n_folds']} | "
             f"{adapter_peak['relative_gain_drop_best']*100:+.2f}% | {'PASS' if adapter_pass else 'FAIL'} |",
             "", f"Frozen-scale adapter delta: `{adapter_fixed['mean_delta']:+.8f}`.",
             "", "No public submission CSV was generated."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
