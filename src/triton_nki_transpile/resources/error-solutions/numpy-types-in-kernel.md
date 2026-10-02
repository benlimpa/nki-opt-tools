# `name numpy.int32 not found` — NumPy typed constants are not visible inside `@nki.jit`

## Error message

```
Error(s) during tracing
/tmp/tmpXXX.py:N:
    on_false=np.int32(-2147483647),
             ^-- name numpy.int32 not found
```

or similarly:

```
name numpy.float32 not found
name numpy.bool_ not found
```

## What it means

Inside a `@nki.jit` kernel the NKI tracer operates in a restricted namespace.
NumPy's type constructor functions (`np.int32`, `np.float32`, `np.int64`, `np.bool_`, etc.)
are **not accessible** in that namespace, so any attempt to call them at trace time raises
a "name not found" error.

This typically appears when the LLM tries to pass a typed sentinel value to an NKI ISA op:

```python
nisa.select_reduce(
    ...,
    on_false=np.int32(-2147483647),   # ERROR: np.int32 not in scope
)
```

## Related follow-on error — integer overflow

When the LLM works around the above error by using the Python `int` literal `2147483648`
(i.e. 2^31), a second error fires:

```
Error(s) during specialize:
<unknown>:-1:0 integer value (2147483648) is too large
```

Because NKI's int32 range is `[−2 147 483 648, 2 147 483 647]`, the value `2147483648` does
not fit and the compiler rejects it.  Use `2147483647` (2^31 − 1) as the maximum positive
int32 sentinel.

## Fix

Replace every `np.<dtype>(value)` with a plain Python literal of the appropriate type.
The NKI tracer accepts bare Python `int` and `float` literals and converts them to the
expected tile dtype automatically.

```python
# WRONG — numpy constructor not visible in NKI tracing context
nisa.select_reduce(
    ...,
    on_false=np.int32(-2147483647),
)

# CORRECT — use a plain Python int literal
nisa.select_reduce(
    ...,
    on_false=-2147483647,
)
```

```python
# WRONG
nisa.memset(tile, np.float32(0.0))

# CORRECT
nisa.memset(tile, 0.0)
```

## Why numpy types are unavailable

The NKI tracer serialises the kernel body for hardware compilation and only admits a
restricted set of Python builtins.  The `numpy` module is not imported into the tracing
namespace, even if the outer Python file contains `import numpy as np`.

## Key rules

1. Never use `np.<dtype>(value)` inside a `@nki.jit` function.
2. Use plain Python `int` or `float` literals for scalar constants in ISA op arguments.
3. The maximum safe positive int32 sentinel is `2147483647` (not `2147483648`).
4. The minimum safe (most negative) int32 sentinel is `-2147483648`.

## Observed in benchmark logs

- `cascaded_max_20260401_062417`: After correctly switching from Python comparisons to
  `nisa.select_reduce`, the LLM used `np.int32(-2147483647)` as the `on_false` sentinel →
  "name numpy.int32 not found".  The next attempt used `2147483648` (off by one) →
  "integer value (2147483648) is too large".  Correct fix: use the bare literal
  `-2147483647` or `2147483647`.
