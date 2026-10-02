# SPMD Launch Grid

## Applicability

Kernel is *not* launched with an SPMD grid (`nl.num_programs(0) == 1`); outer loop iterations are independent (no cross-iteration dependencies / running scan / cross-tile accumulator); outer dim divisible by candidate grid sizes; running on multi-NC hardware (sweep `G ∈ {1, 2}` and pick best)

## When to skip

Kernel already launched with an SPMD grid (`kernel[G](...)` with G > 1); cross-iteration dependencies; outer dim too small to shard; single-NC target

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
