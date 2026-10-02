"""LoRA adapter: out = x @ w + (x @ a) @ b.

Three passes:
  1. y_w (M,N) = x (M,K) @ w (K,N)
  2. x_a (M,R) = x (M,K) @ a (K,R)
  3. out (M,N) = x_a (M,R) @ b (R,N) + y_w (M,N)

R=128 fits in a single K tile for the final matmul.
Source: NKIBench/kernels/lora_M4096_N12288_K5120_R128_0.py (Beta 1 → Beta 2 migration).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

M_TILE = 128
K_TILE = 128
N_TILE = 512


@nki.jit
def _matmul_kernel(lhs_T, rhs):
    """lhs_T (K,M) × rhs (K,N) → out (M,N) via nc_matmul."""
    K, M = lhs_T.shape
    _K, N = rhs.shape
    out = nl.ndarray((M, N), dtype=lhs_T.dtype, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE
            psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(K // K_TILE):
                k_start = i_k * K_TILE
                stat = nl.ndarray((K_TILE, M_TILE), dtype=lhs_T.dtype, buffer=nl.sbuf)
                nisa.dma_copy(stat, lhs_T[k_start:k_start + K_TILE, m_start:m_start + M_TILE])
                mov = nl.ndarray((K_TILE, N_TILE), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(mov, rhs[k_start:k_start + K_TILE, n_start:n_start + N_TILE])
                nisa.nc_matmul(psum, stat, mov)

            result_sb = nl.ndarray((M_TILE, N_TILE), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(result_sb, psum)
            nisa.dma_copy(out[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result_sb)

    return out


@nki.jit
def _lora_combine_kernel(x_a_T, b, y_w):
    """x_a_T (R,M) × b (R,N) + y_w (M,N) → out (M,N). R=K_TILE (single K tile)."""
    R, M = x_a_T.shape
    _R, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE

            psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)
            stat = nl.ndarray((R, M_TILE), dtype=x_a_T.dtype, buffer=nl.sbuf)
            nisa.dma_copy(stat, x_a_T[0:R, m_start:m_start + M_TILE])
            mov = nl.ndarray((R, N_TILE), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(mov, b[0:R, n_start:n_start + N_TILE])
            nisa.nc_matmul(psum, stat, mov)

            result_sb = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(result_sb, psum)
            y_w_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(y_w_tile, y_w[m_start:m_start + M_TILE, n_start:n_start + N_TILE])
            nisa.tensor_tensor(result_sb, result_sb, y_w_tile, nl.add)
            nisa.dma_copy(out[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result_sb)

    return out


def lora(x, w, a, b):
    """x @ w + (x @ a) @ b. x:(M,K), w:(K,N), a:(K,R), b:(R,N). CPU float32 inputs.

    x @ a (R=128 output) is computed via PyTorch since R < N_TILE=512.
    The two large matmuls (x@w and x_a@b with N=12288) run on NKI hardware.
    """
    device = xm.xla_device()
    x_T = x.t().contiguous()
    y_w = _matmul_kernel(x_T.to(device), w.to(device))
    # x @ a: R=128 < N_TILE=512, use PyTorch for this small-N matmul
    x_a = torch.matmul(x, a)  # (M, R) on CPU
    x_a_T = x_a.t().contiguous()  # (R, M)
    result = _lora_combine_kernel(x_a_T.to(device), b.to(device), y_w)
    return result.cpu()
