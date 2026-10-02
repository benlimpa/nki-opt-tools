# Inline Function Fusion

## Applicability

Separate functions share a tile loop; HBM intermediate between them; `spill_save_bytes > 0`

## When to skip

Functions compute on different dimensions/tiles

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
