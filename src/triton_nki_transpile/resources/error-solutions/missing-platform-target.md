# Missing `platform_target` in `@nki.jit`

## Error message

```
ValueError: Could not identify a target platform. Set the environment variable
NEURON_PLATFORM_TARGET_OVERRIDE to one of trn1|gen2 | trn2|gen3 | trn3|gen4
```

## What it means

The NKI compiler must know which Trainium/Inferentia hardware generation to target
before it can emit binary instructions. This is specified via the `platform_target`
keyword argument to `@nki.jit`. When it is absent — and the environment variable
`NEURON_PLATFORM_TARGET_OVERRIDE` is also not set — the compiler raises a
`ValueError` immediately at kernel call time.

## Patterns and fixes

### Pattern 1: `@nki.jit` with no arguments

```python
# BUG: decorator applied without any arguments — platform_target is absent
@nki.jit
def my_kernel(in_hbm):
    out = nl.ndarray(in_hbm.shape, nl.float32, buffer=nl.shared_hbm)
    tile = nl.ndarray(in_hbm.shape, nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm)
    nisa.dma_copy(dst=out, src=tile)
    return out
```

**Fix:** Call the decorator with `platform_target` as a keyword argument:

```python
# CORRECT: platform_target tells the compiler which chip to target
@nki.jit(platform_target="trn1")
def my_kernel(in_hbm):
    out = nl.ndarray(in_hbm.shape, nl.float32, buffer=nl.shared_hbm)
    tile = nl.ndarray(in_hbm.shape, nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=in_hbm)
    nisa.dma_copy(dst=out, src=tile)
    return out
```

### Pattern 2: `@nki.jit()` with empty parentheses

```python
# BUG: empty parentheses still omit platform_target
@nki.jit()
def my_kernel(in_hbm):
    ...
```

**Fix:** same as above — add `platform_target="trn1"` (or `"trn2"` for Trainium 2):

```python
@nki.jit(platform_target="trn1")
def my_kernel(in_hbm):
    ...
```

## Root cause

`@nki.jit` accepts an optional `platform_target` string. Without it, the compiler
cannot select the appropriate instruction set, memory layout, or DMA constraints for
the target chip family. The error fires at the first kernel invocation, not at
import time, because NKI uses tracing (lazy compilation).

## Supported platform_target values

| Value | Hardware |
|---|---|
| `"trn1"` | AWS Trainium (first generation) |
| `"trn2"` | AWS Trainium 2 |

The environment variable `NEURON_PLATFORM_TARGET_OVERRIDE` can serve as a fallback
when `platform_target` is absent, but relying on it makes kernels less portable.
Always specify `platform_target` explicitly in the decorator.
