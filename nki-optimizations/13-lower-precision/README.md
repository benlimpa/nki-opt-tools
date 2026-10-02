# Lower Input Precision

## Applicability

nc_matmul inputs are float32; bf16 input precision acceptable; DMA-bound

## When to skip

Output precision requires float32 inputs

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
