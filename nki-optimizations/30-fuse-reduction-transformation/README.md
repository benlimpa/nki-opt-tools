# Fuse Reduction + Transform

## Applicability

Two-pass: pass1 reduces to HBM (per-row scalar); pass2 reads reduction + input

## When to skip

Reduction output needed by other kernels

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
