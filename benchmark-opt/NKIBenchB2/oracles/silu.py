"""Oracle for silu kernel: y = x / (1 + exp(-x))."""

import numpy as np
import torch
import torch.testing


def _ref(x):
    return x / (1 + np.exp(-x))


def _make_inputs(M, N, seed=42):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(0, 1, (M, N)).astype(np.float32))
    return x


def run_test(wrapper):
    for M, N in [(128, 512), (256, 1024)]:
        x = _make_inputs(M, N)
        result = wrapper(x)
        ref = torch.tensor(_ref(x.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, N=7168
    x = _make_inputs(4096, 7168)
    wrapper(x)
    print("Bench setup passed")


# (M, N) — small / medium (matches bench()) / large
_BENCH3_CASES = [
    (2048, 7168),
    (4096, 7168),
    (8192, 7168),
]


def bench3(wrapper, idx):
    M, N = _BENCH3_CASES[idx]
    x = _make_inputs(M, N)
    wrapper(x)
    print("Bench setup passed")
