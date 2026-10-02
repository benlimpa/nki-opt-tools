"""Oracle for swiglu kernel: (silu(x @ w_gate) * (x @ w_up)) @ w_down."""

import numpy as np
import torch
import torch.testing


def _ref(x, w_up, w_down, w_gate):
    up = np.matmul(x, w_up)
    gate = np.matmul(x, w_gate)
    silu_gate = gate / (1 + np.exp(-gate))
    return np.matmul(silu_gate * up, w_down)


def _make_inputs(M, K, N, seed=42):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(0, 1, (M, K)).astype(np.float32))
    w_up = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    w_down = torch.tensor(np.random.normal(0, 1, (N, K)).astype(np.float32))
    w_gate = torch.tensor(np.random.normal(0, 1, (K, N)).astype(np.float32))
    return x, w_up, w_down, w_gate


def run_test(wrapper):
    # M%128=0, K%128=0, N%512=0; K_out=K must also be %512=0 (w_down output)
    for M, K, N in [(128, 512, 512), (256, 512, 512)]:
        x, w_up, w_down, w_gate = _make_inputs(M, K, N)
        result = wrapper(x, w_up, w_down, w_gate)
        ref = torch.tensor(_ref(x.numpy(), w_up.numpy(), w_down.numpy(), w_gate.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=4096, N=3072, K=1024
    x, w_up, w_down, w_gate = _make_inputs(4096, 1024, 3072)
    wrapper(x, w_up, w_down, w_gate)
    print("Bench setup passed")


# (M, K, N) — small / medium (matches bench()) / large
# Constraints: M%128=0, K%128=0 (and K%512=0 for w_down), N%512=0
_BENCH3_CASES = [
    (2048, 1024, 3072),
    (4096, 1024, 3072),
    (4096, 2048, 3072),
]


def bench3(wrapper, idx):
    M, K, N = _BENCH3_CASES[idx]
    x, w_up, w_down, w_gate = _make_inputs(M, K, N)
    wrapper(x, w_up, w_down, w_gate)
    print("Bench setup passed")
