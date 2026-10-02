# Fuse Dependent Scalar Ops

## Applicability

Consecutive `tensor_scalar` chain fused via two-op form; kernel is compute-bound

## When to skip

Kernel is DMA-bound (gain < 5%)

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
