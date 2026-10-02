# Lower Precision SBUF Buffers

## Applicability

SBUF buffers are float32 for bf16 input data; compute-bound; large SBUF footprint

## When to skip

Compute requires float32 intermediate precision

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
