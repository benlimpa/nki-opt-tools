# Fuse Producer-Consumer Loops

## Applicability

Loop A writes HBM; Loop B reads it; same iteration bounds; no other consumers of HBM output

## When to skip

Other code needs the HBM intermediate

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
