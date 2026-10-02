# `unsupported dtype 'int64'` — NKI does not support 64-bit integer tensors

## Error message

```
Error(s) during tracing0:0:unsupported dtype 'int64'
```

## What it means

NKI's tracing and compilation stack does not support `torch.int64` (64-bit signed integer)
tensors as kernel arguments.  This error fires at trace time when a tensor with dtype
`torch.int64` is passed to a `@nki.jit` function.

Common sources of `int64` tensors:
- `torch.randint` returns `torch.int64` by default.
- PyTorch loss functions (cross-entropy, etc.) accept label tensors as `torch.long` (= int64).
- Slicing/indexing torch tensors can produce int64 index tensors.

## Supported integer dtypes in NKI

| Dtype | Supported |
|---|---|
| `torch.int8` / `nl.int8` | Yes |
| `torch.int16` / `nl.int16` | Yes |
| `torch.int32` / `nl.int32` | Yes |
| `torch.int64` | **No** |

## Common buggy pattern

```python
# wrapper — labels arrive as torch.int64
def cross_entropy(logits, labels):
    device = torch_xla.device()
    logits = logits.to(device, dtype=torch.float32)
    labels = labels.to(device)   # BUG: still int64!
    return _kernel(logits, labels)

@nki.jit(platform_target="trn1")
def _kernel(logits, labels):
    # trace fails immediately: unsupported dtype 'int64'
    label_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=label_tile, src=labels[row:row+1])
```

## Fix

Cast integer tensors to `torch.int32` in the **wrapper** before passing them to the kernel.

```python
def cross_entropy(logits, labels):
    device = torch_xla.device()
    logits = logits.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.int32)   # CORRECT: cast to int32
    return _kernel(logits, labels)

@nki.jit(platform_target="trn1")
def _kernel(logits, labels):
    # labels is now int32 — tracing succeeds
    label_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=label_tile, src=labels[row:row+1])
```

## Key rule

> **Always cast index/label tensors to `torch.int32` in the wrapper before calling a
> `@nki.jit` kernel.**  `torch.int64` (= `torch.long`) is not a supported NKI dtype.

## Observed in benchmark logs

- `crossentropy_20260401_012830`: labels passed as `torch.int64` (default for
  `torch.randint`) → "unsupported dtype 'int64'" at trace entry.  Correct repair: cast
  labels to `torch.int32` in the wrapper.
