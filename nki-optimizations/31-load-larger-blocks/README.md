# Load Larger Free-Dim Blocks

## Applicability

`TILE_N << N`; full tile fits in SBUF; compiler NOT already caching (check `hbm_read_bytes > 1×`)

## When to skip

Compiler already loads data only once (verify!)

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
