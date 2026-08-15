# v3 market rounds + asset adapter confirmation — 2026-08-14

## Scope

This phase followed the exact strict OOF baseline and tested two remaining high-value changes:

1. independent shrunk-market LightGBM checkpoint rounds;
2. per-asset cross-sectional calibration on the XS LightGBM residual.

No public submission CSV was generated and the production model directory was not modified.

## Phase A: market checkpoint screening

Configuration:

```text
modulo5 / phase_balanced
train_window=78,960
embargo=6
5 rolling folds
XS OOF fixed from v3 exact 1-seed/160 cache
market shrunk spec, 1 seed, max 480 rounds
```

| market rounds | mean Peak | vs 160 | positive folds | drop-best | gate |
|---:|---:|---:|---:|---:|:---:|
| 160 | 0.00140840 | 0.00% | 0/5 | 0.00% | baseline |
| 240 | 0.00140686 | -0.11% | 3/5 | -0.49% | FAIL |
| 320 | 0.00141763 | +0.66% | 3/5 | -0.10% | FAIL |
| 400 | 0.00143195 | +1.67% | 3/5 | +0.67% | FAIL |
| 480 | 0.00143689 | +2.02% | 3/5 | +0.81% | FAIL |

The mean curve rises, but only three of five folds are positive. The pre-registered 4/5-fold gate therefore rejects choosing a new market checkpoint from the screening run. The result is not a reason to lower capacity: it is a reason to confirm with the production 3-seed setting.

Artifact: `outputs/experiments/v3_market_round_scan_phasebal_prodwindow.{json,md}`.

## Phase B: production-capacity confirmation

Configuration:

```text
5 rolling folds
XS: weighted loose, 3 seeds × 480 rounds
market: unweighted shrunk, 3 seeds × 480 rounds
market checkpoints retained: 160 and 480
```

Artifact: `outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz`.

Market 480 versus market 160, with XS fixed at 480:

```text
mean Peak gain: +2.91%
positive folds: 4/5
drop-best gain: +2.05%
pooled Peak: 0.00155137
```

This passes the confirmation gate. The final market setting remains 480, which is already the current production round count; no market-round metadata change is needed for the final candidate.

## Asset adapter confirmation

The adapter was trained only on fold 0 and evaluated frozen on folds 1–4:

```text
asset shrink: 100 in the single-fold gate
```

Results against market480 + XS480:

```text
mean Peak gain: +1.99%
positive folds: 3/4
drop-best gain: +1.34%
frozen-scale absolute score delta: +0.00007401
```

The adapter passes both Peak and frozen-scale gates. Prediction-only market experts remain rejected:

```text
linear market expert: -38.37% Peak, 1/4 positive
nonlinear market expert: -26.69% Peak, 0/4 positive
```

Machine decision: `market480_plus_asset_adapter`.

Artifact: `outputs/experiments/v3_confirm_3s480_decision.{json,md}`.

## Candidate

Created:

```text
outputs/candidates/v3_asset_cross_3s480_shrink500/
```

The candidate uses the existing production forests (XS and market both 480 rounds) and adds only OOF-fitted `asset_cross_scales`. The final scale fit uses all five strict OOF folds with shrink 500, preserving the per-fold shrink strength used in the gate.

Consistency checks:

```text
LightGBM backend max|train-infer| = 3.961e-09
NumPy backend max|train-infer|    = 3.961e-09
market_num_iteration=160 smoke    = 1.801e-09
```

## Decision

Keep the candidate local-only. It is ready for a later one-shot public submission if the user explicitly chooses to spend an upload quota, but no leaderboard CSV has been produced in this phase.
