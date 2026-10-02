"""Oracle for mamba kernel: SSM selective scan."""

import numpy as np
import torch
import torch.testing


def _ref(delta, u, a, b, c):
    C, M = delta.shape
    S = a.shape[1]
    deltaA = np.exp(delta[:, None, :] * a[:, :, None])  # (C, S, M)
    deltaB_u = delta[:, None, :] * b[None, :, :] * u[:, None, :]  # (C, S, M)
    scan_res = np.zeros((C, S, M), dtype=np.float32)
    for i in range(M):
        prev = scan_res[:, :, i - 1] if i > 0 else 0
        scan_res[:, :, i] = deltaA[:, :, i] * prev + deltaB_u[:, :, i]
    return np.sum(c[None, :, :] * scan_res, axis=1)  # (C, M)


def _make_inputs(C, M, S, seed=42):
    np.random.seed(seed)
    delta = torch.tensor(np.random.normal(0, 0.05, (C, M)).astype(np.float32))
    u = torch.tensor(np.random.normal(0, 0.05, (C, M)).astype(np.float32))
    a = torch.tensor(np.random.normal(0, 0.05, (C, S)).astype(np.float32))
    b = torch.tensor(np.random.normal(0, 0.05, (S, M)).astype(np.float32))
    c = torch.tensor(np.random.normal(0, 0.05, (S, M)).astype(np.float32))
    return delta, u, a, b, c


def run_test(wrapper):
    # M%1024=0, C%128=0
    for C, M, S in [(128, 1024, 4), (128, 2048, 8)]:
        delta, u, a, b, c = _make_inputs(C, M, S)
        result = wrapper(delta, u, a, b, c)
        ref = torch.tensor(_ref(delta.numpy(), u.numpy(), a.numpy(), b.numpy(), c.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=7168, C=256, S=16
    delta, u, a, b, c = _make_inputs(256, 7168, 16)
    wrapper(delta, u, a, b, c)
    print("Bench setup passed")


# (C, M, S) — small / medium (matches bench()) / large
# Constraints: M%1024=0, C%128=0
_BENCH3_CASES = [
    (256, 4096, 16),
    (256, 7168, 16),
    (256, 8192, 16),
]


def bench3(wrapper, idx):
    C, M, S = _BENCH3_CASES[idx]
    delta, u, a, b, c = _make_inputs(C, M, S)
    wrapper(delta, u, a, b, c)
    print("Bench setup passed")
