# v3 strict OOF residual atlas — 2026-08-14

## Question

After the responder re-audit rejected responder stacking, where does the current public model still fail under a production-equivalent, leakage-safe validation protocol?

## Exact screening protocol

- sampling: `sample_modulo=5`, `phase_balanced`;
- folds: 5 rolling folds, `train_window=78,960`, embargo 6;
- architecture: Ridge market + weighted loose XS LightGBM + unweighted shrunk market LightGBM;
- history: 40 selected features, window 5;
- blend: cross block fully replaced by LightGBM (`blend_weight=1.0`), market lambda 0.5;
- screening capacity: 1 seed × 160 rounds;
- strictness: preprocessing, feature selection and models are fitted independently inside each fold.

Canonical artifacts:

- `outputs/cache/v3_production_oof_phasebal_prodwindow_exact.npz`
- `outputs/experiments/v3_production_oof_phasebal_prodwindow_exact.{json,md}`
- `outputs/experiments/v3_residual_atlas_phasebal_prodwindow_exact.{json,md}`

The earlier artifact without the `_exact` suffix used the loose XS spec for the market forest and is superseded. It remains on disk only to preserve experiment history.

## Screening result

Pooled strict OOF:

| component | score | peak |
|---|---:|---:|
| full raw prediction | 0.00130233 | 0.00135167 |
| market blend | 0.00059263 | 0.00065156 |
| Ridge market | 0.00059214 | 0.00059924 |
| shrunk LGBM market | 0.00033317 | 0.00061462 |
| XS LightGBM | 0.00083833 | 0.00084673 |

The first fold is much weaker than later folds, so pooled-only conclusions remain unsafe. Public scale `1.16` is not locally optimal in this OOF (`raw optimal scale ≈ 0.84`), confirming that local scale must not overwrite leaderboard calibration.

## Frozen second-stage gate

`experiments/v3_residual_adapters.py` trains adapters only on the earliest OOF fold and freezes them for folds 1–4.

- conditional linear market expert: **rejected**, peak −48.52%, 0/4 folds;
- nonlinear market expert: **rejected**, peak −32.05%, 0/4 folds;
- per-asset XS scale, shrink 100 per fold: **passed**, peak +3.33%, 4/4 folds, drop-best +3.10%; frozen-scale score +0.00009069.

Rotating the meta-training fold kept the per-asset adapter positive. With meta folds 0–4, peak gains were approximately +3.42%, +3.80%, +3.00%, +2.69%, and +0.65%, each positive on all four held-out folds. The effect is therefore small-to-moderate but unusually stable.

## Candidate

A local-only candidate was created at:

`outputs/candidates/v3_asset_cross_shrink500/`

The 5-fold OOF fit uses shrink 500, preserving the screening strength of shrink 100 per fold. Only `hybrid_meta.json` gains `asset_cross_scales`; all six production forests and the Ridge artifact are copied unchanged. After scaling, the XS prediction is projected back to zero mean per `time_id`, so it cannot alter the market component.

Consistency check:

`outputs/experiments/v3_asset_cross_shrink500_consistency.json`

Result: online vs offline `max|Δ| = 3.961e-09`.

## Decision

- Do not pursue conditional market experts from current prediction-only regime features.
- Keep the asset XS adapter as the first promoted research candidate, but do not generate a public CSV yet.
- The likely gain is incremental, not enough alone to move 0.00399775 to 0.0050.
- Next high-value experiment is an exact market-round checkpoint scan (160/240/320/400/480) with the XS block fixed, followed by 3-seed confirmation only if the screening curve is stable.
