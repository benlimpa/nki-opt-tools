"""Oracle for matmul_add_rmsnorm kernel: RMSNorm(x @ w + z) * g."""

import numpy as np
import torch
import torch.testing

EPS = 1e-5


def _ref(x, w, eps, z, g):
    y = np.matmul(x, w) + z
    N = y.shape[-1]
    sq_mean = np.sum(y ** 2, axis=-1, keepdims=True) / N
    rms_inv = 1.0 / np.sqrt(sq_mean + eps)
    return y * rms_inv * g


def _make_inputs(M, K, N, seed=42):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(0, 1, (M, K)).astype(np.float32))
    w = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    eps = EPS
    z = torch.tensor(np.random.normal(0, 1, (M, N)).astype(np.float32))
    g = torch.tensor(np.random.normal(0, 1, (N,)).astype(np.float32))
    return x, w, eps, z, g


def run_test(wrapper):
    # M%128=0, K%128=0, N%512=0
    for M, K, N in [(128, 128, 512), (256, 256, 512)]:
        x, w, eps, z, g = _make_inputs(M, K, N)
        result = wrapper(x, w, eps, z, g)
        ref = torch.tensor(_ref(x.numpy(), w.numpy(), eps, z.numpy(), g.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, N=2048, K=2048
    x, w, eps, z, g = _make_inputs(4096, 2048, 2048)
    wrapper(x, w, eps, z, g)
    print("Bench setup passed")


# (M, K, N) — small / medium (matches bench()) / large
# Constraints: M%128=0, K%128=0, N%512=0
_BENCH3_CASES = [
    (2048, 2048, 2048),
    (4096, 2048, 2048),
    (4096, 4096, 2048),
]


def bench3(wrapper, idx):
    M, K, N = _BENCH3_CASES[idx]
    x, w, eps, z, g = _make_inputs(M, K, N)
    wrapper(x, w, eps, z, g)
    print("Bench setup passed")
