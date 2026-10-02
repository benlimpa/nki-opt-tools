"""Oracle for lora kernel: x @ w + (x @ a) @ b."""

import numpy as np
import torch
import torch.testing


def _ref(x, w, a, b):
    y1 = np.matmul(x, w)
    y2 = np.matmul(np.matmul(x, a), b)
    return y1 + y2


def _make_inputs(M, K, N, R, seed=42):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(0, 1, (M, K)).astype(np.float32))
    w = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    a = torch.tensor(np.random.normal(0, 1, (K, R)).astype(np.float32))
    b = torch.tensor(np.random.normal(0, 1, (R, N)).astype(np.float32))
    return x, w, a, b


def run_test(wrapper):
    # M%128=0, K%128=0, N%512=0, R%128=0
    for M, K, N, R in [(128, 128, 512, 128), (256, 256, 512, 128)]:
        x, w, a, b = _make_inputs(M, K, N, R)
        result = wrapper(x, w, a, b)
        ref = torch.tensor(_ref(x.numpy(), w.numpy(), a.numpy(), b.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, N=12288, K=5120, R=128
    x, w, a, b = _make_inputs(4096, 5120, 12288, 128)
    wrapper(x, w, a, b)
    print("Bench setup passed")


# (M, K, N, R) — small / medium (matches bench()) / large
# Constraints: M%128=0, K%128=0, N%512=0, R%128=0.
# R is fixed at 128: the kernel hardcodes R_TILE=128, so any R>128 overflows
# the state buffer (`Allocated memory out of bound` from neuronx-cc). Scale
# the largest case via M instead — the kernel is M-tiled and handles M growth.
_BENCH3_CASES = [
    (2048, 5120, 12288, 128),
    (4096, 5120, 12288, 128),
    (5120, 5120, 12288, 128),
]


def bench3(wrapper, idx):
    M, K, N, R = _BENCH3_CASES[idx]
    x, w, a, b = _make_inputs(M, K, N, R)
    wrapper(x, w, a, b)
    print("Bench setup passed")
