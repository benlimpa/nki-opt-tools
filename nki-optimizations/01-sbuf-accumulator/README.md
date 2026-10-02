# SBUF Accumulator

## Applicability

Loop accumulates into HBM buffer (`private_hbm`); `spill_save_bytes > 0`

## When to skip

Accumulator too large for SBUF

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
