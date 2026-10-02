"""Oracle for gqa_full kernel: GQA = softmax(Q @ K.T / sqrt(D)) @ V."""

import math
import numpy as np
import torch
import torch.testing


def _ref(q, k, v):
    B, N, QH, D = q.shape
    KH = k.shape[2]
    n_rep = QH // KH

    xk = np.repeat(k, n_rep, axis=2)  # (B, N, QH, D)
    xv = np.repeat(v, n_rep, axis=2)  # (B, N, QH, D)
    xq = q.transpose(0, 2, 1, 3)      # (B, QH, N, D)
    xk = xk.transpose(0, 2, 1, 3)     # (B, QH, N, D)
    xv = xv.transpose(0, 2, 1, 3)     # (B, QH, N, D)

    scale = 1.0 / math.sqrt(D)
    attn = (xq @ xk.transpose(0, 1, 3, 2)) * scale  # (B, QH, N, N)
    exp_attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
    attn_w = exp_attn / np.sum(exp_attn, axis=-1, keepdims=True)
    return attn_w @ xv  # (B, QH, N, D)


def _make_inputs(B, N, QH, KH, D, seed=42):
    np.random.seed(seed)
    q = torch.tensor(np.random.normal(0, 0.1, (B, N, QH, D)).astype(np.float32))
    k = torch.tensor(np.random.normal(0, 0.1, (B, N, KH, D)).astype(np.float32))
    v = torch.tensor(np.random.normal(0, 0.1, (B, N, KH, D)).astype(np.float32))
    return q, k, v


def run_test(wrapper):
    # N%128=0, D=128 (fixed), QH%KH=0
    for B, N, QH, KH, D in [(1, 128, 2, 1, 128), (1, 256, 4, 2, 128)]:
        q, k, v = _make_inputs(B, N, QH, KH, D)
        result = wrapper(q, k, v)
        ref = torch.tensor(_ref(q.numpy(), k.numpy(), v.numpy()))
        torch.testing.assert_close(result.float(), ref.float(), atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    # Benchmark shape: B=1, N=4096, QH=16, KH=8, D=128
    q, k, v = _make_inputs(1, 4096, 16, 8, 128)
    wrapper(q, k, v)
    print("Bench setup passed")


# (B, N, QH, KH, D) — small / medium / large (matches bench())
# Constraints: N%128=0, D=128 (fixed), QH%KH=0
# N is capped at 4096: the attention matrix (B, QH, N, N) is O(N^2) on device
# and at N=8192 with QH=16 (4 GB matrix) compilation reliably OOM'd the host
# before the watchdog could react, hard-freezing the instance.
_BENCH3_CASES = [
    (1, 1024, 16, 8, 128),
    (1, 2048, 16, 8, 128),
    (1, 4096, 16, 8, 128),
]


def bench3(wrapper, idx):
    B, N, QH, KH, D = _BENCH3_CASES[idx]
    q, k, v = _make_inputs(B, N, QH, KH, D)
    wrapper(q, k, v)
    print("Bench setup passed")
