# `nisa.tensor_tensor` — "Expect AP same number of elements" shape mismatch

## Error message

```
[INTERNAL_ERROR] [NCC_INLA001] Unhandled exception:
=== BIR verification failed ===
Reason: Expect AP same number of elements  [[256,1],[1,256]]
Memory Location: {kernel._fn-0:pred.68}@SB<0,0>(1x256)
```

Occurs during **neuronx-cc compilation** (after tracing succeeds).

## What it means

`nisa.tensor_tensor` requires both input tiles (`data1` and `data2`) to have the
same total number of elements. When the access patterns report different element
counts — for example `(1, 256)` vs `(1, 1)` — the compiler raises this BIR
verification failure.

This typically happens when trying to compare or operate on a per-row vector
(`(1, vocab)`) against a scalar-shaped tile (`(1, 1)`), such as a loaded label
index.

## Common buggy pattern

```python
idx_tile   = nl.ndarray((1, vocab_size), dtype=nl.float32, buffer=nl.sbuf)
label_fp   = nl.ndarray((1, 1),          dtype=nl.float32, buffer=nl.sbuf)

# ERROR: (1, vocab_size) vs (1, 1) — element counts differ
pred = nl.ndarray((1, vocab_size), dtype=nl.bool_, buffer=nl.sbuf)
nisa.tensor_tensor(dst=pred, data1=idx_tile, data2=label_fp, op=nl.equal)
```

`idx_tile` has `vocab_size` elements, `label_fp` has 1 element. The compiler
rejects this mismatch.

## Fix

Ensure both operands to `tensor_tensor` have exactly the same shape before
calling the instruction. For a scalar operand used against a full row, use
`nisa.tensor_scalar` (not `tensor_tensor`) which is explicitly designed for
broadcasting a scalar value against every element of a tile:

```python
# CORRECT — tensor_scalar broadcasts the scalar operand across data
diff = nl.ndarray((1, vocab_size), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_scalar(
    dst=diff,
    data=idx_tile,          # (1, vocab_size)
    op0=nl.subtract,
    operand0=label_fp,      # (1, 1) scalar — OK here
)
```

If you genuinely need `tensor_tensor` (e.g. element-wise comparison of two
same-length rows), both tiles must have matching shapes:

```python
# Both tiles must be (1, vocab_size)
label_row = nl.ndarray((1, vocab_size), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_scalar(dst=label_row, data=idx_tile, op0=nl.add, operand0=0.0)  # or broadcast differently

pred = nl.ndarray((1, vocab_size), dtype=nl.bool_, buffer=nl.sbuf)
nisa.tensor_tensor(dst=pred, data1=idx_tile, data2=label_row, op=nl.equal)  # OK: same shape
```

## `tensor_scalar` vs `tensor_tensor`

| Instruction | Use when |
|---|---|
| `nisa.tensor_scalar` | One operand is a tile, the other is a scalar/`(1,1)` value. Broadcast semantics. |
| `nisa.tensor_tensor` | Both operands are the **same-shaped** tiles. Element-wise operation. |

## Key rule

> **Before calling `nisa.tensor_tensor`, verify both input tiles are the same
> shape.** If one operand is scalar or `(1, 1)`, use `nisa.tensor_scalar` instead.

## Observed in benchmark logs

- `crossentropy_20260413_072530` (cycle 6): `nisa.tensor_tensor(data1=pred_tile
  (1,vocab), data2=label_fp (1,1), op=nl.equal)` → "Expect AP same number of
  elements [[256,1],[1,256]]". The (1,1) label tile needed to be broadcast to
  (1,vocab) first via `tensor_scalar` before a same-shape `tensor_tensor` comparison.
