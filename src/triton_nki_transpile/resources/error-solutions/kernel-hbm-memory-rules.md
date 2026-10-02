# Kernel HBM memory rules — `shared_hbm` only for returned outputs, not intermediates

## Error messages

```
Shared memory is only supported on trn2, but inst__kernel_...:<tensor>_set is using Shared memory on an unsupported target
```

and the follow-on error that appears after the incorrect repair:

```
IO tensor must be shared_hbm tensor, but found TensorName(..., memory=private_hbm, ...)
```

## What it means

NKI kernels have three memory locations for tensors declared inside the kernel body:

| Buffer type | Where | When to use |
|---|---|---|
| `nl.sbuf` | On-chip SBUF | All intermediate / scratch tiles. Default for compute. |
| `nl.private_hbm` | Off-chip HBM, private per partition | Intermediate results that are too large for SBUF but are NOT outputs. |
| `nl.shared_hbm` | Off-chip HBM, shared across partitions | **The kernel's output tensors that are returned to the caller.** |

The constraint on trn1:

> **`nl.shared_hbm` may only be used for tensors that are returned from the kernel as outputs.**
> Any tensor declared `buffer=nl.shared_hbm` that is not in the kernel's return tuple is treated as
> an intermediate and will fail with "Shared memory is only supported on trn2".

Conversely:

> **Every tensor in the kernel's return tuple must use `buffer=nl.shared_hbm`.**
> Returning an `nl.sbuf` or `nl.private_hbm` tile fails with "IO tensor must be shared_hbm tensor".

## The common incorrect repair chain

When the LLM sees the "Shared memory is only supported on trn2" error it often incorrectly
proposes one of two workarounds, each of which creates a new error:

### Incorrect repair 1 — pass the output tensor as a function parameter

```python
# WRONG: wrapper allocates output and passes it into kernel
def wrapper(x, ...):
    y = torch.zeros(x.shape, device=xla_device)
    _kernel(x, y, ...)   # y is a pre-allocated HBM tensor
    return y

@nki.jit(platform_target="trn1")
def _kernel(x, y, ...):
    ...
    nisa.dma_copy(dst=y[row:row+1, :], src=y_tile)  # ERROR: mutation not supported
```

HBM tensors passed as *input parameters* are read-only. Writing to them produces:
```
mutation not supported
```

### Incorrect repair 2 — use `private_hbm` for the output

```python
@nki.jit(platform_target="trn1")
def _kernel(x, ...):
    y = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.private_hbm)  # WRONG for output
    ...
    return y  # ERROR: IO tensor must be shared_hbm tensor
```

`nl.private_hbm` is valid for non-returned tensors but the compiler requires returned
tensors to be `shared_hbm`.

## Correct pattern

```python
@nki.jit(platform_target="trn1")
def _kernel(x, ...):
    # Returned output: MUST be shared_hbm
    y = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    # Intermediate scratch that would be too big for SBUF: use private_hbm
    scratch = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)

    # Small working tiles: always sbuf
    x_tile   = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
    y_tile   = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
    mean_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)

    for row in nl.affine_range(M):
        nisa.dma_copy(dst=x_tile, src=x[row:row+1, :])
        # ... compute ...
        nisa.dma_copy(dst=y[row:row+1, :], src=y_tile)

    return y   # shared_hbm tensor — OK on trn1
```

## Key rules

1. Every tensor listed in `return` → `buffer=nl.shared_hbm`.
2. Every intermediate scratch tile → `buffer=nl.sbuf` (preferred) or `buffer=nl.private_hbm`.
3. Never declare an intermediate-only tensor as `shared_hbm` on trn1.
4. Never write into a tensor that was passed as an input parameter — allocate a fresh `shared_hbm` tile instead.

## Observed in benchmark logs

- `layernorm_20260330_035304`: `Mean` and `Rstd` declared `shared_hbm` but not returned →
  "Shared memory is only supported on trn2"; incorrect repair passed `Y` as parameter →
  "mutation not supported"; second incorrect repair used `private_hbm` for `Y` →
  "IO tensor must be shared_hbm tensor".
- `rms_20260329_063202`: same `shared_hbm` on intermediate `r` tensor, same chain.
