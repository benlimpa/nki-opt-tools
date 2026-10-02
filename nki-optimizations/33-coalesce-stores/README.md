# Coalesce Stores

## Applicability

Many small SBUF→HBM stores to adjacent addresses; high DMA packet count for stores

## When to skip

Stores already contiguous and large

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
