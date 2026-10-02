# nc_matmul Max Moving Tile

## Applicability

`TILE_N < 512` with `nc_matmul`; N is 512-aligned

## When to skip

Already using TILE_N=512

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
