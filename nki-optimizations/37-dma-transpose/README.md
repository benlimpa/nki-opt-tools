# DMA Transpose

## Applicability

`nc_transpose` used to transpose a tile loaded from HBM

## When to skip

Tile is generated in SBUF (no HBM load to intercept)

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
