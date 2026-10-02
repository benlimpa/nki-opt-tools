"""Oracle for bmm kernel: C[b] = lhs[b] @ rhs[b]."""

import numpy as np
import torch
import torch.testing


def _ref(lhs, rhs):
    return np.matmul(lhs, rhs)


def _make_inputs(B, M, K, N, seed=42):
    np.random.seed(seed)
    lhs = torch.tensor(np.random.normal(0, 1, (B, M, K)).astype(np.float32))
    rhs = torch.tensor(np.random.normal(0, 1, (B, K, N)).astype(np.float32))
    return lhs, rhs


def run_test(wrapper):
    # M%128=0, N%512=0; K can be partial (handled by ceil)
    for B, M, K, N in [(2, 128, 64, 512), (4, 256, 128, 512)]:
        lhs, rhs = _make_inputs(B, M, K, N)
        result = wrapper(lhs, rhs)
        ref = torch.tensor(_ref(lhs.numpy(), rhs.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: B=16, M=4096, K=64, N=4096
    lhs, rhs = _make_inputs(16, 4096, 64, 4096)
    wrapper(lhs, rhs)
    print("Bench setup passed")


# (B, M, K, N) — small / medium (matches bench()) / large
# Constraints: M%128=0, N%512=0
_BENCH3_CASES = [
    (8, 4096, 64, 4096),
    (16, 4096, 64, 4096),
    (32, 4096, 64, 4096),
]


def bench3(wrapper, idx):
    B, M, K, N = _BENCH3_CASES[idx]
    lhs, rhs = _make_inputs(B, M, K, N)
    wrapper(lhs, rhs)
    print("Bench setup passed")
