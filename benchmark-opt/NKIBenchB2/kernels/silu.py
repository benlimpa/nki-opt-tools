"""SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x) = x / (1 + exp(-x)).

Source: NKIBench/kernels/silu_M4096_N7168_0.py (Beta 1 → Beta 2 migration).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

P = 128       # partition dimension
N_TILE = 512  # free dimension tile size


@nki.jit
def _silu_kernel(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    for i_p in nl.affine_range(M // P):
        p_start = i_p * P
        for i_f in nl.affine_range(N // N_TILE):
            f_start = i_f * N_TILE

            x_tile = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(x_tile, x[p_start:p_start + P, f_start:f_start + N_TILE])

            # exp(-x)
            neg_exp = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(neg_exp, nl.exp, x_tile, scale=-1.0)

            # 1 + exp(-x)
            denom = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(denom, neg_exp, nl.add, 1.0)

            # sigmoid(x) = 1 / (1 + exp(-x))
            sig = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(sig, denom)

            # silu = x * sigmoid(x)
            result = nl.ndarray((P, N_TILE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(result, x_tile, sig, nl.multiply)

            nisa.dma_copy(out[p_start:p_start + P, f_start:f_start + N_TILE], result)

    return out


def silu(x):
    """SiLU activation. Input: CPU float32 tensor of shape (M, N)."""
    device = xm.xla_device()
    result = _silu_kernel(x.to(device))
    return result.cpu()
