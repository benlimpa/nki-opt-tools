# Specialized Instructions

## Applicability

Explicit Python loop over free dim for elementwise op; `nisa.activation` or `tensor_scalar` can replace it

## When to skip

Operation isn't expressible as a fused instruction

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
