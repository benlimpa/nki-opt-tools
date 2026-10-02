# `Value is not modifiable` / `mutation not supported` — NKI tiles cannot be written via Python assignment

## Error messages

```
Value is not modifiable
```

```
mutation not supported
```

The `mutation not supported` variant appears specifically when attempting tile
item-assignment inside an `nl.affine_range` loop body.

## What it means

NKI tiles — both SBUF (`nl.sbuf`) and HBM (`nl.shared_hbm`) — cannot be
written using Python's item-assignment operator (`tile[...] = value` or
`tile[i] = value`).  All writes to NKI tiles must go through NKI ISA
operations (`nisa.*`).

This error fires at **tracing time** (kernel compilation), not at runtime.

## Common buggy patterns

### Pattern 0: Item-assignment inside `nl.affine_range` loop

```python
idx_tile = nl.ndarray((1, vocab_size), dtype=nl.int32, buffer=nl.sbuf)
for i in nl.affine_range(vocab_size):
    idx_tile[0, i] = i    # ERROR: mutation not supported
```

This pattern is commonly attempted when trying to fill an index tile with
`[0, 1, 2, ..., N-1]`. Use `nisa.iota` instead:

```python
idx_tile = nl.ndarray((1, vocab_size), dtype=nl.float32, buffer=nl.sbuf)
nisa.iota(dst=idx_tile, pattern=[[1, vocab_size]], offset=0, channel_multiplier=0)
```

### Pattern 1: Scalar or Python-value write to a pre-declared SBUF tile

```python
result = nl.ndarray((4, 1), nl.float32, buffer=nl.sbuf)
result[0, 0] = 1.0          # ERROR: Value is not modifiable
result[...] = some_nki_val  # ERROR: Value is not modifiable
```

This often appears when a developer tries to "store" a computed result back
into a pre-allocated tile using Python assignment:

```python
row_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
row_sum[...] = nisa.tensor_reduce(op=nl.add, data=x, axis=1)  # ERROR
```

### Pattern 2: Direct scalar-index write to an HBM output tile

```python
out = nl.ndarray((4, 8), nl.float32, buffer=nl.shared_hbm)
out[0, 0] = 1.0  # ERROR: Value is not modifiable
out[i]    = v    # ERROR: Value is not modifiable
```

## Fix

### Fix for Pattern 1 — use `nisa.tensor_reduce` with `dst=`

Pass the pre-allocated tile as the `dst=` argument to the ISA operation:

```python
result = nl.ndarray((4, 1), nl.float32, buffer=nl.sbuf)
# CORRECT: ISA op writes into the dst tile
nisa.tensor_reduce(dst=result, op=nl.add, data=x, axis=1)
```

Other ISA ops that accept `dst=` follow the same pattern: `nisa.activation`,
`nisa.tensor_scalar`, etc.

### Fix for Pattern 2 — compute in SBUF, then `nisa.dma_copy` to HBM

Never write to `nl.shared_hbm` tiles directly.  Compute the result into an
SBUF tile first, then DMA it to the HBM output:

```python
out    = nl.ndarray((4, 8), nl.float32, buffer=nl.shared_hbm)
sbuf   = nl.ndarray((4, 8), nl.float32, buffer=nl.sbuf)
# ... fill sbuf via ISA operations ...
nisa.dma_copy(dst=out[0:4, 0:8], src=sbuf)  # CORRECT
```

## Root cause

NKI's tracing mechanism does not implement `__setitem__` on tile objects.
Any attempt to use Python's `=` operator (which calls `__setitem__`) raises
this error at trace time.  Only NKI ISA operations from `nki.isa` (`nisa.*`)
are permitted to write tile memory.

## Key rule

> **Never write `tile[...] = expr` or `tile[i] = expr` in an NKI kernel.**
> Use `nisa.tensor_reduce`, `nisa.tensor_scalar`, `nisa.activation`,
> `nisa.dma_copy`, etc. to write tile contents.
