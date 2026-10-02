# Full Compute-DMA Pipeline

## Applicability

Load + compute + store all significant; `active_time% < 85%`; SBUF fits 2×input + 2×output

## When to skip

Output store is negligible

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
