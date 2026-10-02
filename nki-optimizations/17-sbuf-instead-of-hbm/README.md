# SBUF Intermediate (not HBM)

## Applicability

Two-loop pass: loop1 writes intermediate to HBM; loop2 reads it; `spill_save_bytes > 0`

## When to skip

Intermediate too large for SBUF

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
