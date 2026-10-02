# `nisa.tensor_reduce` — "not a valid dim" when axis is a list or tuple

## Error

```
nisa.tensor_reduce(dst=c_tile, op=nl.maximum, data=sbuf_row, axis=[1], keepdims=True)
^-- not a valid dim
```

Occurs at **tracing time** (before any compilation step).

## Cause

`nisa.tensor_reduce` requires the `axis` argument to be a **bare integer**.
Passing a list (`axis=[1]`), a tuple (`axis=(1,)`), or a multi-element sequence
(`axis=[1, 2]`, `axis=(1, 2)`) all raise "not a valid dim", even though the NKI
documentation says the field takes "a list of integer(s)".

Confirmed failing forms:

| Call form        | Result          |
|-----------------|-----------------|
| `axis=1`        | ✅ works         |
| `axis=[1]`      | ❌ not a valid dim |
| `axis=(1,)`     | ❌ not a valid dim |
| `axis=[1, 2]`   | ❌ not a valid dim |
| `axis=(1, 2)`   | ❌ not a valid dim |

## Fix

Always pass a bare integer:

```python
# WRONG — causes "not a valid dim" (list, tuple, or multi-element sequence)
nisa.tensor_reduce(dst=dst, op=nl.maximum, data=tile, axis=[1], keepdims=True)
nisa.tensor_reduce(dst=dst, op=nl.add,     data=tile, axis=(1,), keepdims=True)

# CORRECT — bare integer
nisa.tensor_reduce(dst=dst, op=nl.maximum, data=tile, axis=1, keepdims=True)
nisa.tensor_reduce(dst=dst, op=nl.add,     data=tile, axis=1, keepdims=True)
```

For multi-axis reductions, chain multiple `nisa.tensor_reduce` calls, each with a
single bare integer axis.

## Notes

- Applies to all reduction operators: `nl.maximum`, `nl.add`, `nl.multiply`, etc.
- `keepdims=True` works correctly together with a bare int axis.
- The NKI documentation example text says "list of integer(s)" but the runtime
  **only accepts a bare `int`** — sequences of any kind are rejected.

## Test

`src/triton_nki_transpile/tests/test_tensor_reduce_axis.py`
