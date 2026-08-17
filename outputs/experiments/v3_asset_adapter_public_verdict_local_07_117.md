# v3 asset adapter public verdict - fork local 0.7/1.17

Decision: **REJECTED_PUBLIC**

| Model | Public score |
|---|---:|
| `market_lambda=0.7 / blend_weight=1.17` | **0.00407075** |
| same model + fold-0 asset adapter (`shrink=100`) | 0.0039613753 |

Absolute delta: `-0.0001093747`; relative delta: `-2.68684%`.

The strict OOF gate passed, but the public result did not transfer. Runner integrity checks passed:
3,217,458 rows, exact row alignment, 0 timeouts, 0 non-finite values, and 0 clipped rows.

The fork therefore restores and keeps `market_lambda=0.7`, `blend_weight=1.17`,
`prediction_scale=1.16`, with no `asset_cross_scales`. Production artifacts were not modified.
