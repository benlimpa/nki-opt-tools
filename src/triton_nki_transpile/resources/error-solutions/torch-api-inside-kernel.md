# torch API inside an NKI kernel

## Error message

```
Error(s) during tracing:

/tmp/tmpXXXXXX.py:20:
        t = torch.tensor(tile)
            ^-- torch.tensor not found!
```

Or for other torch symbols:

```
        result = torch.max(tile, dim=1, keepdim=True).values
                 ^-- torch.max not found!
```

## What it means

NKI kernels run inside a **restricted tracing environment**. Only symbols explicitly
provided by the NKI runtime (`nl.*`, `nisa.*`, `nki.*`, standard Python builtins) are
visible. The entire `torch` namespace — `torch.tensor`, `torch.max`, `torch.sort`,
`torch.clamp`, `torch.cumsum`, etc. — is not re-exported and appears undefined to the
tracer, which raises `<symbol> not found!` immediately at trace time.

## Patterns and fixes

### Pattern 1: Converting a tile to a PyTorch tensor to call `torch.max`

```python
@nki.jit(platform_target="trn1")
def buggy_torch_max(in_hbm):
    tile = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:TILE_M, 0:TILE_N])
    # BUG: torch.tensor and torch.max are not available here
    t = torch.tensor(tile)
    result = torch.max(t, dim=1, keepdim=True).values
    ...
```

**Fix:** Use `nisa.tensor_reduce` with `op=nl.maximum` to compute the row-wise max
entirely inside the NKI world — no tensor conversion needed:

```python
@nki.jit(platform_target="trn1")
def fixed_nisa_reduce_max(in_hbm):
    tile = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:TILE_M, 0:TILE_N])
    dst = nl.ndarray((TILE_M, 1), nl.float32, buffer=nl.sbuf)
    # CORRECT: bare-int axis, no torch conversion
    nisa.tensor_reduce(dst=dst, op=nl.maximum, data=tile, axis=1, keepdims=True)
    ...
```

### Pattern 2: Using `torch.clamp` for elementwise clamping

```python
@nki.jit(platform_target="trn1")
def buggy_torch_clamp(in_hbm):
    tile = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:TILE_M, 0:TILE_N])
    # BUG: torch.clamp is not available inside nki.jit
    clamped = torch.clamp(tile, min=0.0)
    ...
```

**Fix:** Use `nisa.tensor_scalar` with `op0=nl.maximum` and a scalar operand.
`max(x, 0)` is equivalent to `clamp(min=0)` / ReLU:

```python
@nki.jit(platform_target="trn1")
def fixed_nisa_tensor_scalar_clamp(in_hbm):
    tile = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:TILE_M, 0:TILE_N])
    dst = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
    # CORRECT: max(x, 0) == ReLU == clamp(min=0)
    nisa.tensor_scalar(dst=dst, data=tile, op0=nl.maximum, operand0=0.0)
    ...
```

### Pattern 3: Using `torch.sort` for sorting inside a kernel

```python
@nki.jit(platform_target="trn1")
def buggy_torch_sort(in_hbm):
    tile = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:1, 0:TILE_N])
    # BUG: torch.sort is not available inside nki.jit
    sorted_x, _ = torch.sort(torch.tensor(tile), dim=-1, descending=True)
    ...
```

**Fix:** Use `nisa.sort` if available, or restructure the algorithm to avoid
in-kernel sorting. Many algorithms that use sorting can be rewritten using
`nisa.tensor_reduce` for max/sum and `nisa.iota` for index generation:

```python
@nki.jit(platform_target="trn1")
def fixed_using_nisa_sort(in_hbm):
    tile = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:1, 0:TILE_N])
    dst = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    # CORRECT: nisa.sort operates on SBUF tiles directly
    nisa.sort(dst=dst, src=tile, axis=1, ascending=False)
    ...
```

### Pattern 4: Using `torch.cumsum` for a prefix scan

```python
@nki.jit(platform_target="trn1")
def buggy_torch_cumsum(in_hbm):
    tile = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:1, 0:TILE_N])
    # BUG: torch.cumsum is not available inside nki.jit
    result = torch.cumsum(torch.tensor(tile).float(), dim=-1)
    ...
```

**Fix:** Use `nisa.tensor_tensor_scan` for prefix scans:

```python
@nki.jit(platform_target="trn1")
def fixed_tensor_tensor_scan(in_hbm):
    tile = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm[0:1, 0:TILE_N])
    zeros = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    nisa.memset(zeros, 0.0)
    dst = nl.ndarray((1, TILE_N), nl.float32, buffer=nl.sbuf)
    # CORRECT: tensor_tensor_scan computes the prefix sum in hardware
    nisa.tensor_tensor_scan(dst=dst, data0=tile, data1=zeros,
                            initial=0.0, op0=nl.add, op1=nl.add)
    ...
```

## Root cause

NKI kernels are **traced** (not interpreted) at call time. The tracer evaluates Python
code in a sandboxed namespace that only contains NKI symbols. Any name lookup that falls
outside this namespace — including the entire `torch` module — fails with `not found!`
because the object literally does not exist in the tracing context.

This is by design: NKI kernels compile to native Trainium/Inferentia instructions
and cannot call PyTorch's CPU or GPU compute paths.

## Key rule

Never call `torch.*` functions inside an `@nki.jit` kernel. Replace every torch operation
with its `nisa.*` or `nl.*` counterpart:

| torch operation | NKI equivalent |
|---|---|
| `torch.max(x, dim=d)` | `nisa.tensor_reduce(op=nl.maximum, axis=d)` |
| `torch.sum(x, dim=d)` | `nisa.tensor_reduce(op=nl.add, axis=d)` |
| `torch.clamp(x, min=v)` | `nisa.tensor_scalar(op0=nl.maximum, operand0=v)` |
| `torch.abs(x)` | `nisa.activation(op=np.abs)` |
| `torch.exp(x)` | `nisa.activation(op=nl.exp)` |
| `torch.cumsum(x, dim)` | `nisa.tensor_tensor_scan(op0=nl.add, op1=nl.add)` |
| `torch.sort(x, dim)` | `nisa.sort(src, axis)` |
| `x + scalar` | `nisa.tensor_scalar(op0=nl.add, operand0=scalar)` |
| `x * scalar` | `nisa.tensor_scalar(op0=nl.multiply, operand0=scalar)` |
