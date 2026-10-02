# PSUM Accumulation: `nc_matmul` Adds Into Destination — Never Reuse Across Calls

## Problem

`nisa.nc_matmul` **adds** into the destination PSUM buffer. It does **not** overwrite it.
Reusing the same PSUM buffer across multiple `nc_matmul` calls causes the results to
accumulate, so every output after the first contains the sum of all prior computations.

```python
# BUG: b_exp will contain mask + weight + bias
tmp_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

nisa.nc_matmul(dst=tmp_psum, stationary=ones_row, moving=m_tile)   # tmp_psum = mask
nisa.activation(dst=m_exp, op=nl.copy, data=tmp_psum)

nisa.nc_matmul(dst=tmp_psum, stationary=ones_row, moving=w_tile)   # tmp_psum = mask + weight  ← WRONG
nisa.activation(dst=w_exp, op=nl.copy, data=tmp_psum)

nisa.nc_matmul(dst=tmp_psum, stationary=ones_row, moving=b_tile)   # tmp_psum = mask + weight + bias  ← WRONG
nisa.activation(dst=b_exp, op=nl.copy, data=tmp_psum)
```

**Symptom:** Numerically wrong results where later outputs contain the accumulated sum of
all prior `nc_matmul` writes to that PSUM buffer.

## Fix

Allocate a **separate PSUM buffer for each `nc_matmul` call**.

```python
# FIX: separate PSUM buffer per nc_matmul
m_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)
nisa.nc_matmul(dst=m_psum, stationary=ones_row, moving=m_tile)
nisa.activation(dst=m_exp, op=nl.copy, data=m_psum)

w_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)
nisa.nc_matmul(dst=w_psum, stationary=ones_row, moving=w_tile)
nisa.activation(dst=w_exp, op=nl.copy, data=w_psum)

b_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)
nisa.nc_matmul(dst=b_psum, stationary=ones_row, moving=b_tile)
nisa.activation(dst=b_exp, op=nl.copy, data=b_psum)
```
