# Parallelize 1D Scan

## Applicability

Serial cumsum / prefix scan over large 1D array; underutilizing all 128 partitions

## When to skip

Operation has non-associative dependencies

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
