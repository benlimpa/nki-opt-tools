"""Oracle for rope_single_freq_apply kernel: RoPE rotation."""

import numpy as np
import torch
import torch.testing


def _ref(x, freqs_cos, freqs_sin):
    half_d = x.shape[0] // 2
    x0 = x[:half_d, :]
    x1 = x[half_d:, :]
    out0 = x0 * freqs_cos - x1 * freqs_sin
    out1 = x0 * freqs_sin + x1 * freqs_cos
    return np.concatenate([out0, out1], axis=0)


def _make_inputs(D, S, seed=42):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(0, 1, (D, S)).astype(np.float32))
    freqs_cos = torch.tensor(np.random.normal(0, 1, (D // 2, S)).astype(np.float32))
    freqs_sin = torch.tensor(np.random.normal(0, 1, (D // 2, S)).astype(np.float32))
    return x, freqs_cos, freqs_sin


def run_test(wrapper):
    # S must be divisible by S_TILE=512
    for D, S in [(128, 512), (128, 1024)]:
        x, cos, sin = _make_inputs(D, S)
        result = wrapper(x, cos, sin)
        ref = torch.tensor(_ref(x.numpy(), cos.numpy(), sin.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: B=1, H=64, N=4096, D=128 → S=B*H*N=262144
    D, S = 128, 1 * 64 * 4096
    x, cos, sin = _make_inputs(D, S)
    wrapper(x, cos, sin)
    print("Bench setup passed")


# (D, S) — small / medium (matches bench()) / large
# Constraints: S%512=0 (S_TILE=512)
_BENCH3_CASES = [
    (128, 131072),
    (128, 262144),
    (128, 393216),
]


def bench3(wrapper, idx):
    D, S = _BENCH3_CASES[idx]
    x, cos, sin = _make_inputs(D, S)
    wrapper(x, cos, sin)
    print("Bench setup passed")
