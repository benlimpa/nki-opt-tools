# PSUM Fused Bias Eviction

## Applicability

Scalar scale/bias applied right after PSUM eviction as separate instruction; compute-bound

## When to skip

Kernel is DMA-bound (gain negligible)

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
