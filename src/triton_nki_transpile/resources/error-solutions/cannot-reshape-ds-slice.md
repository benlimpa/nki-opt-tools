# `cannot reshape a complex access pattern` — do not call `.reshape()` on a sliced HBM tensor

## Error message

```
^-- cannot reshape a complex access pattern
```

## What it means

NKI's compiler represents sliced tensor accesses (range slices, `nl.ds()` dynamic
slices, multi-dimensional index expressions) as **complex access patterns**.
Calling `.reshape()` on such a pattern is not supported — the compiler cannot fold
the reshape into the slice descriptor and raises this error at **tracing time**.

## Patterns and fixes

**The key insight:** if the HBM tensor already has the shape the tile expects,
no reshape is needed.  Restructure how HBM tensors are shaped so that slices
naturally match tile shapes.

### Pattern 1: 1-D slice reshaped to 2-D in `dma_copy`

```python
# in_hbm has shape (N,)
tile = nl.ndarray((1, N), nl.float32, buffer=nl.sbuf)
nisa.dma_copy(dst=tile, src=in_hbm[0:N].reshape((1, N)))  # ERROR
```

**Fix:** shape the input as `(1, N)` instead of `(N,)`:

```python
# Pass/receive in_hbm with shape (1, N) rather than (N,)
tile = nl.ndarray((1, N), nl.float32, buffer=nl.sbuf)
nisa.dma_copy(dst=tile, src=in_hbm[0:1, 0:N])  # CORRECT: slice shape = tile shape
```

### Pattern 2: `nl.ds()` dynamic slice followed by `.reshape()`

```python
# in_hbm has shape (N,); start is a runtime value
nisa.dma_copy(dst=tile, src=in_hbm[nl.ds(start, N)].reshape((1, N)))  # ERROR
```

**Fix:** shape the input as `(ROWS, N)` and use a row index:

```python
# in_hbm has shape (ROWS, N)
tile = nl.ndarray((1, N), nl.float32, buffer=nl.sbuf)
row = start  # or compute the row index at trace time / via nl.affine_range
nisa.dma_copy(dst=tile, src=in_hbm[row:row+1, 0:N])  # CORRECT
```

### Pattern 3: Row slice of a 2-D tensor reshaped to a different 2-D shape

```python
# x2 has shape (M, N)
tile = nl.ndarray((N, 1), nl.float32, buffer=nl.sbuf)
nisa.dma_copy(dst=tile, src=x2[row, 0:N].reshape((N, 1)))  # ERROR
```

**Fix:** loop over rows and slice each one as `(1, N)`:

```python
for row in nl.affine_range(M):
    tile = nl.ndarray((1, N), nl.float32, buffer=nl.sbuf)
    # Slice is now (1, N); matches the tile — no reshape needed
    nisa.dma_copy(dst=tile, src=x2[row:row+1, 0:N])  # CORRECT
```

### Pattern 4: Multi-dimensional range slice with reshape

```python
# Q has shape (B, S, H, D)
nisa.dma_copy(dst=q1, src=Q[b:b+1, s:s+1, h:h+1, 0:D].reshape((1, D)))  # ERROR
```

**Fix:** flatten leading dims before passing to the kernel:

```python
# Reshape Q from (B, S, H, D) to (B*S*H, D) before passing to the kernel
# Then inside the kernel:
nisa.dma_copy(dst=q1, src=Q_flat[flat_idx:flat_idx+1, 0:D])  # CORRECT
```

## Root cause

NKI tiles must always be 2-D (partition dimension × free dimension).  When
loading from HBM, the compiler needs to emit a DMA descriptor that matches
the tile layout exactly.  A slice already encodes address-offset logic; adding
a reshape on top creates a transformation the DMA engine cannot express in a
single descriptor.  The fix is to arrange data in HBM so slices have the
desired shape without needing an extra reshape step.

## Key rule

> **Never call `.reshape()` on a sliced HBM tensor.**
> Shape your HBM tensors so that a plain slice (e.g. `t[row:row+1, 0:N]`)
> already has the dimensions your SBUF tile expects.
