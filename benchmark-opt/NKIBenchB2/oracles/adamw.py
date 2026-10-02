"""Oracle for adamw kernel: AdamW update step."""

import numpy as np
import torch
import torch.testing


def _ref(theta, g, m, v):
    theta_t = theta - 1e-5 * theta
    m_t = 0.9 * m + 0.1 * g
    v_t = 0.999 * v + 0.001 * g * g
    v_hat = v_t * 1000
    return theta_t - 0.01 * m_t / (np.sqrt(v_hat) + 1e-8)


def _make_inputs(M, N, seed=42):
    np.random.seed(seed)
    theta = torch.tensor(np.random.normal(0, 1, (M, N)).astype(np.float32))
    g = torch.tensor(np.random.normal(0, 1, (M, N)).astype(np.float32))
    m = torch.tensor(np.random.normal(0, 1, (M, N)).astype(np.float32))
    v = torch.tensor(np.abs(np.random.normal(0, 1, (M, N))).astype(np.float32))
    return theta, g, m, v


def run_test(wrapper):
    for M, N in [(128, 1024), (256, 1024)]:
        theta, g, m, v = _make_inputs(M, N)
        result = wrapper(theta, g, m, v)
        ref = torch.tensor(_ref(theta.numpy(), g.numpy(), m.numpy(), v.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: M=10944, N=2048
    theta, g, m, v = _make_inputs(10944, 2048)
    wrapper(theta, g, m, v)
    print("Bench setup passed")


# (M, N) — small / medium (matches bench()) / large
_BENCH3_CASES = [
    (5472, 2048),
    (10944, 2048),
    (16416, 2048),
]


def bench3(wrapper, idx):
    M, N = _BENCH3_CASES[idx]
    theta, g, m, v = _make_inputs(M, N)
    wrapper(theta, g, m, v)
    print("Bench setup passed")
