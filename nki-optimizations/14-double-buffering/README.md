# Double Buffering

## Applicability

Sequential load→compute loop; DMA load is significant; SBUF fits 2× tile

## When to skip

Loop has <2 iterations; SBUF too tight

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
