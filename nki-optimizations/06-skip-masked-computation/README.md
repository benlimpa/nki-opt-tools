# Skip Masked Tiles

## Applicability

Some tiles are all-padding/invalid; loop iterates all tiles including padding

## When to skip

All tiles contain valid data

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
