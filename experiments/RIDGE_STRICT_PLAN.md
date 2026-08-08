# Strict Ridge Solver Validation Plan

## Hypothesis

Tightening LSQR from `tol=1e-4, max_iter=100` to `tol=1e-8, max_iter=2000`
reduces BLAS-thread-dependent prediction drift without a measurable out-of-sample loss.

## Controls

- 200 selected features
- alpha 2,000,000
- prediction scale 1.13
- prediction clip 0.5
- periodic time sampling, modulo 5
- four chronological training partitions

Only the LSQR stopping condition changes.

## Validation

1. Run paired rolling A/B on the base grid and the equal-length half-offset grid.
2. Require pooled delta on both grids to be at least -2.16e-5.
3. Require identical selected features and no fit reaching `max_iter`.
4. Compare one-thread and four-thread fits; require max prediction drift below 1e-6.
5. Train the strict model only into `outputs/candidates/v1_ridge_strict/`.
6. Require train/inference parity below 1e-6 on 500 real time IDs.
7. Run the full sequential API without writing a public submission file and record timing/error statistics.

The production model is not overwritten automatically.
