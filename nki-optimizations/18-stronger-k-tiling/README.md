# Stronger K-Tiling

## Applicability

`TILE_K < 128`; K > 128; rhs same across M-tiles; many nc_matmul calls per M-tile

## When to skip

Already TILE_K=128

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
