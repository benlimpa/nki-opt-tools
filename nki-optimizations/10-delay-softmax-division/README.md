# Delay Softmax Division

## Applicability

Softmax stores exp tiles to HBM between reduction and normalization passes

## When to skip

Reduction and normalization can't be separated

Compare `before.py` and `after.py` for the transformation where present. Validate full outputs and measure on the target SDK and hardware. Experimental profiles and timing results are not distributed.
