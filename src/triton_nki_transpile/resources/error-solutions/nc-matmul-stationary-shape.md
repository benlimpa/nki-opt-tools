# `nc_matmul` Broadcast: Stationary Shape Must Be `(1, TILE_M)`, Not `(TILE_M, 1)`

## Problem

To broadcast a `(1, N)` vector to `(TILE_M, N)`, the ones vector passed as `stationary`
must have shape `(partition=1, free=TILE_M)`. The Tensor Engine transposes `stationary`
before the multiply, so `(1, TILE_M).T = (TILE_M, 1)` outer-products with the moving
`(1, N)` tile to produce the desired `(TILE_M, N)` result.

Using `(TILE_M, 1)` directly (without the leading partition-dim-1) causes a compiler error.

```python
# BUG: causes neuronx-cc compiler error
ones_col = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(dst=ones_col, value=1.0)
nisa.nc_matmul(dst=psum, stationary=ones_col, moving=vec_1xN)
```

**Symptom:** `[INTERNAL_ERROR] [NCC_INLA001] Unhandled exception with message: argument not found`

## Fix

Allocate the ones vector with **partition dim = 1, free dim = TILE_M**.

```python
# FIX: partition dim = 1, free dim = TILE_M
ones_row = nl.ndarray((1, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(dst=ones_row, value=1.0)
nisa.nc_matmul(dst=psum, stationary=ones_row, moving=vec_1xN)
```

The NKI Tensor Engine requires `stationary.shape[0]` (the partition dim) to be 1 when
performing a broadcast multiply. The free dimension carries the row count.
