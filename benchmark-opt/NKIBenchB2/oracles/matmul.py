"""Oracle for matmul kernel: C = lhs @ rhs."""

import numpy as np
import torch
import torch.testing


def _ref(lhs, rhs):
    return np.matmul(lhs, rhs)


def _make_inputs(M, K, N, seed=42):
    np.random.seed(seed)
    lhs = torch.tensor(np.random.normal(0, 1, (M, K)).astype(np.float32))
    rhs = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    return lhs, rhs


def run_test(wrapper):
    # M%128=0, K%128=0, N%512=0
    for M, K, N in [(128, 128, 512), (256, 256, 512)]:
        lhs, rhs = _make_inputs(M, K, N)
        result = wrapper(lhs, rhs)
        ref = torch.tensor(_ref(lhs.numpy(), rhs.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, N=12288, K=5120
    lhs, rhs = _make_inputs(4096, 5120, 12288)
    wrapper(lhs, rhs)
    print("Bench setup passed")


# (M, K, N) — small / medium (matches bench()) / large
# Constraints: M%128=0, K%128=0, N%512=0
_BENCH3_CASES = [
    (2048, 5120, 12288),
    (4096, 5120, 12288),
    (4096, 5120, 16384),
]


def bench3(wrapper, idx):
    M, K, N = _BENCH3_CASES[idx]
    lhs, rhs = _make_inputs(M, K, N)
    wrapper(lhs, rhs)
    print("Bench setup passed")
