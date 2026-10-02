"""Oracle for transpose_matmul kernel: C = lhs_t.T @ rhs."""

import numpy as np
import torch
import torch.testing


def _ref(lhs_t, rhs):
    return np.matmul(lhs_t.T, rhs)


def _make_inputs(K, M, N, seed=42):
    np.random.seed(seed)
    lhs_t = torch.tensor(np.random.normal(0, 1, (K, M)).astype(np.float32))
    rhs = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    return lhs_t, rhs


def run_test(wrapper):
    # M%128=0, K%128=0; N_TILE=128 handles partial N
    for K, M, N in [(128, 128, 256), (256, 256, 384)]:
        lhs_t, rhs = _make_inputs(K, M, N)
        result = wrapper(lhs_t, rhs)
        ref = torch.tensor(_ref(lhs_t.numpy(), rhs.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, K=2048, N=10944
    lhs_t, rhs = _make_inputs(2048, 4096, 10944)
    wrapper(lhs_t, rhs)
    print("Bench setup passed")


# (K, M, N) — small / medium (matches bench()) / large
# Constraints: M%128=0, K%128=0; N can be partial
# M is capped at 6144 for the largest case: at M=8192 the kernel compiles fine
# but `neuron-profile view` times out post-capture (NEFF too large for the
# view tool to process within the 600s profiler timeout). 6144 keeps the case
# meaningfully larger than the medium one without tripping the profiler.
_BENCH3_CASES = [
    (2048, 2048, 10944),
    (2048, 4096, 10944),
    (2048, 6144, 10944),
]


def bench3(wrapper, idx):
    K, M, N = _BENCH3_CASES[idx]
    lhs_t, rhs = _make_inputs(K, M, N)
    wrapper(lhs_t, rhs)
    print("Bench setup passed")
