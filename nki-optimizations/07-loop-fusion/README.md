# Loop Fusion

## Applicability

≥2 loops over same dim; HBM intermediate between them; `spill_save_bytes > 0`

## When to skip

Cross-tile data dependencies between loops

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
