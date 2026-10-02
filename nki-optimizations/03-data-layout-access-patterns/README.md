# Full Partition Utilization

## Applicability

`TILE_P < 128`; tensor partition dim ≥ 128

## When to skip

Already TILE_P=128

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
