# PSUM Accumulation Group

## Applicability

K-tile loop uses `sequential_range` + manual VectorE add; high VectorE% in matmul

## When to skip

Already using `affine_range` for K-tiles

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
