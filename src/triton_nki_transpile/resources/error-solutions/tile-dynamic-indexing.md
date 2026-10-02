# `expecting 'integer', got 'access'` / `cannot negate values of this type` — no dynamic tile values as indices

## Error messages

```
/tmp/kernel.py:49:
    picked_logit = row_tile[0, label_val]
                   ^-- expecting 'integer', got 'access'
```

```
/tmp/kernel.py:55:
    offset=-label_val,
           ^-- cannot negate values of this type
```

## What it means

NKI tile element indices and `nisa.affine_select`'s `offset` parameter must be
**compile-time Python integers**, not values derived from tile memory at runtime.
When `label_val` is obtained by indexing an SBUF tile (`label_tile[0, 0]`), it
becomes an NKI `access` object — a symbolic trace-time handle — which cannot be
used as:

- A Python subscript index into another tile (`row_tile[0, label_val]`)
- A negated integer (`-label_val`)
- The `offset=` argument of `nisa.affine_select`

These all raise tracing errors because NKI's compiler cannot resolve the value at
compile time.

## Common buggy patterns

### Pattern 1 — direct tile subscript with runtime index

```python
label_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.dma_copy(dst=label_tile, src=labels[row:row+1])

label_val = label_tile[0, 0]          # 'access' object, NOT a Python int
picked = row_tile[0, label_val]       # ERROR: expecting 'integer', got 'access'
```

### Pattern 2 — negate an access object for affine_select offset

```python
label_val = label_tile[0, 0]
nisa.affine_select(
    dst=dst,
    pattern=[[1, vocab_size], [1, 1]],
    offset=-label_val,                # ERROR: cannot negate values of this type
    ...
)
```

### Pattern 3 — affine_select with a tile-derived offset (no negation)

```python
label_val = label_tile[0, 0]
nisa.affine_select(dst=dst, ..., offset=label_val, ...)
# ERROR: expecting 'integer', got 'access'
```

All three patterns fail because `affine_select`'s `offset` must be a Python `int`
literal or expression — it cannot be a runtime-loaded tile value.

## Fix — use `.ap()` indirect load for dynamic gather

NKI supports data-dependent (gather) loads through the **indirect access pattern**
(`.ap()`) on HBM tensors. This is the only supported way to load
`logits[row, label]` when `label` is a runtime value stored in SBUF.

```python
# target_tile holds the dynamic index, shape (1, 1), dtype int32, in SBUF
target_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.dma_copy(dst=target_tile, src=targets_hbm[position:position+1, :])

# Indirect gather: load logits_hbm[row, target_tile[0,0]] into dst
dst_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.dma_copy(
    dst=dst_tile,
    src=logits_hbm.ap(
        pattern=[[vocab_size, 1], [1, 1]],
        offset=row * vocab_size,          # base offset: start of this row
        scalar_offset=target_tile.ap(
            pattern=[[1, 1], [1, 1]],
            offset=0,
        ),
        indirect_dim=1,                   # the dynamic index selects along dim 1
    ),
)
# dst_tile[0, 0] now holds logits_hbm[row, target]
```

`indirect_dim=1` tells the DMA engine to use `scalar_offset` to select within
dimension 1 (the vocab/column dimension). The `offset` anchors the base row.

## Cross-entropy label selection — full working example

The most common use case is computing `logits[position, label[position]]` for
cross-entropy. The reference pattern (from the NKI cross-entropy kernel):

```python
# batch_targets: (positions_per_batch, 1), int32, sbuf — labels loaded earlier
# batch_target_logits: (positions_per_batch, 1), dtype, sbuf — output

for position_idx in range(actual_batch_size):
    absolute_position = shard_offset + batch_start_idx + position_idx
    nisa.dma_copy(
        dst=batch_target_logits[position_idx : position_idx + 1, :],
        src=logits_hbm.ap(
            pattern=[[vocab_size, 1], [1, 1]],
            offset=absolute_position * vocab_size,   # row base
            scalar_offset=batch_targets.ap(
                pattern=[[1, 1], [1, 1]],
                offset=position_idx,                 # pick this row's label
            ),
            indirect_dim=1,
        ),
    )
```

Note that the loop index `position_idx` is a **Python int** (from `range()`), so
`offset=position_idx` is valid. The label itself (`batch_targets[position_idx, 0]`)
is accessed via the SBUF `.ap()` — this is the indirection.

## Why "iota + float equality" is unreliable

A common fallback is to build a one-hot mask via `nisa.iota` and float equality:

```python
# Build idx_tile = [0.0, 1.0, 2.0, ..., vocab-1] then subtract label
nisa.iota(dst=idx_tile, ...)
nisa.tensor_scalar(dst=diff, data=idx_tile, op0=nl.subtract, operand0=label_fp)
nisa.tensor_tensor(dst=pred, data1=diff, data2=zero_row, op=nl.equal)
```

This compiles but produces **wrong results**: floating-point equality (`== 0.0`)
is unreliable when comparing the result of `float(index) - float(label)` because
the subtraction can introduce rounding errors at large index values, causing the
exact-zero test to fail. Additionally, this approach is less efficient than a
direct gather. **Use `.ap()` instead.**

## Key rule

> **Never use a tile-derived value as a Python index or as `affine_select`'s `offset`.**
> For gather-style loads (selecting element `[row, label]` from an HBM tensor where
> `label` is a runtime value), use `tensor.ap(..., scalar_offset=label_tile.ap(...),
> indirect_dim=N)` inside `nisa.dma_copy`.

## Observed in benchmark logs

- `crossentropy_20260413_072530` (cycles 3, 4, 8): `row_tile[0, label_val]` →
  "expecting 'integer', got 'access'"; `offset=-label_val` in `affine_select` →
  "cannot negate values of this type"; same `affine_select` pattern after regression
  in cycle 8. Cycles 5–7 tried iota+float-equality which compiled but gave 75%
  numerically wrong results.
