# `unsupported comparison between 'access' and 'access'/'int'` — no Python operators on NKI tiles

## Error messages

```
unsupported comparison between 'access' and 'access'
unsupported comparison between 'access' and 'int'
```

## What it means

During NKI kernel tracing, tensor elements are represented as **`access` objects**, not Python
scalars. Python's comparison operators (`>`, `<`, `>=`, `<=`, `==`, `!=`) are not implemented
on `access` objects and therefore cannot be used inside a `@nki.jit` kernel.

The same restriction applies to `if tile_elem:`, `while tile_elem < N:`, and any other
Python control flow that inspects a tile value.

## Common buggy patterns

### Pattern 1 — comparing a loaded tile element with a Python scalar

```python
for row in nl.affine_range(M):
    val = nisa.dma_copy(...)
    # ERROR: 'access' vs 'int' — cannot branch on tile content at trace time
    if val[0, 0] > threshold:
        ...
```

### Pattern 2 — comparing two tile elements against each other

```python
cur_val  = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
best_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
...
# ERROR: 'access' vs 'access'
if cur_val[0, 0] > best_val[0, 0]:
    best_val[...] = cur_val  # also wrong (value-not-modifiable)
```

This pattern is triggered when the LLM tries to implement argmax / conditional accumulation
using Python control-flow instead of ISA-level select/reduce operations.

## Fix

Use NKI ISA operations that produce predicate tiles and select results on the device:

### Element-wise comparison → predicate tile

```python
pred = nl.ndarray((1, N), dtype=nl.bool_, buffer=nl.sbuf)
nisa.tensor_tensor(dst=pred, data1=a_tile, data2=b_tile, op=nl.greater)
# pred[i] = (a_tile[i] > b_tile[i])
```

Supported comparison ops in `nisa.tensor_tensor`: `nl.greater`, `nl.less`, `nl.equal`,
`nl.greater_equal`, `nl.less_equal`.

### Conditional selection / masked update — use `nisa.select_reduce`

```python
best_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(best_val, -float("inf"))

for c in nl.sequential_range(num_chunks):
    chunk_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=chunk_max, op=nl.maximum, data=chunk_tile, axis=1, keepdims=True)

    # Predicate: is this chunk's max > running best?
    better = nl.ndarray((1, 1), dtype=nl.bool_, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=better, data1=chunk_max, data2=best_val, op=nl.greater)

    # Select: update best_val only where better == True
    updated = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.select_reduce(
        dst=updated,
        predicate=better,
        on_true=chunk_max,
        on_false=best_val,
        reduce_res=best_val,
        reduce_cmd=nisa.reduce_cmd.idle,
        reduce_op=nl.maximum,
    )
```

### Running maximum over all rows — use `nisa.tensor_reduce` directly

If you only need the max value (not argmax), `nisa.tensor_reduce(op=nl.maximum)` handles
the full reduction without any per-element comparisons:

```python
max_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_reduce(dst=max_val, op=nl.maximum, data=row_tile, axis=1, keepdims=True)
```

## Key rule

> **Never use Python comparison operators (`>`, `<`, `==`, …) on any NKI tile or tile element
> inside a `@nki.jit` kernel.**  Use `nisa.tensor_tensor` with a comparison `op` to produce
> a predicate tile, then use `nisa.select_reduce` or `nisa.tensor_scalar` to act on it.

## Observed in benchmark logs

- `cascaded_max_20260401_062417`: `cur_val > local_best_val[0, 0]` in a running-max loop →
  "unsupported comparison between 'access' and 'access'".  The LLM initially tried
  `nisa.select_reduce` but then used `np.int32(-2147483647)` as a sentinel (see
  `numpy-types-in-kernel.md`), causing a follow-on error.
- `crossentropy_20260401_012830`: `if (label_tile[0] >= start) and (label_tile[0] < end)` →
  "unsupported comparison between 'access' and 'int'".
