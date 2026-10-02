"""Oracle for rmsnorm_matmul kernel: (x / rms(x)) @ w."""

import numpy as np
import torch
import torch.testing


def _ref(x, w):
    K = x.shape[1]
    sq_sum = np.sum(x ** 2, axis=1, keepdims=True) / K
    rms = np.sqrt(sq_sum)
    x_norm = x / rms
    return np.matmul(x_norm, w)


def _make_inputs(M, K, N, seed=42):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(0, 1, (M, K)).astype(np.float32))
    w = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    return x, w


def run_test(wrapper):
    # M%128=0, K%128=0, N%512=0
    for M, K, N in [(128, 128, 512), (256, 256, 512)]:
        x, w = _make_inputs(M, K, N)
        result = wrapper(x, w)
        ref = torch.tensor(_ref(x.numpy(), w.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, N=2048, K=1024
    x, w = _make_inputs(4096, 1024, 2048)
    wrapper(x, w)
    print("Bench setup passed")


# (M, K, N) — small / medium (matches bench()) / large
# Constraints: M%128=0, K%128=0, N%512=0
_BENCH3_CASES = [
    (2048, 1024, 2048),
    (4096, 1024, 2048),
    (4096, 2048, 2048),
]


def bench3(wrapper, idx):
    M, K, N = _BENCH3_CASES[idx]
    x, w = _make_inputs(M, K, N)
    wrapper(x, w)
    print("Bench setup passed")
