# `nisa.activation` with `op=nl.copy` Ignores the Bias Argument

## Problem

When calling `nisa.activation` with `op=nl.copy`, the `bias` argument is silently
ignored — the output equals the input data unchanged, as if no bias was provided.

This bug does **not** affect other activation functions: `op=nl.relu`, for example,
applies the bias correctly.

```python
# BUG: bias is silently ignored when op=nl.copy
nisa.activation(dst=out, op=nl.copy, data=x_tile, bias=neg_mean)
# out == x_tile  (bias was never added)

# Works correctly with other ops:
nisa.activation(dst=out, op=nl.relu, data=x_tile, bias=neg_mean)
# out == relu(x_tile + neg_mean)  ✓
```

**Symptom:** Output is numerically wrong — the bias has no effect. For example, in a
layernorm kernel the centering step (`x - mean`) produces the wrong result because
`nisa.activation(op=nl.copy, bias=neg_mean)` leaves the data unmodified.

## Fix

Use `nisa.tensor_scalar` instead. It correctly applies an element-wise operation
between a `(P, N)` tensor and a `(P, 1)` per-partition value, and also broadcasts
across the free dimension.

```python
# FIX: tensor_scalar correctly applies the bias
nisa.tensor_scalar(dst=out, data=x_tile, op0=nl.subtract, operand0=mean_vec)
```

### Safe patterns for per-partition scalar operations

| Goal | Safe API | Avoid |
|---|---|---|
| Copy `(P,N)` + add `(P,1)` bias | `nisa.tensor_scalar(op0=nl.add, operand0=...)` | `nisa.activation(op=nl.copy, bias=tensor)` |
| Subtract `(P,1)` from `(P,N)` | `nisa.tensor_scalar(op0=nl.subtract, operand0=...)` | `nisa.activation(op=nl.copy, bias=-tensor)` |
| Multiply `(P,N)` by `(P,1)` | `nisa.tensor_scalar(op0=nl.multiply, operand0=...)` | `nisa.activation(scale=tensor)` |
