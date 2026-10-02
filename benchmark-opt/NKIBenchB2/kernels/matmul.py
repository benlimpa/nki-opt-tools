"""Standard matrix multiplication: C = lhs @ rhs.

Source: NKIBench/kernels/matmul_M4096_N12288_K5120_0.py (Beta 1 → Beta 2 migration).

nc_matmul computes stationary.T @ moving, so lhs is pre-transposed in the wrapper
(lhs_T has shape (K, M)) and passed as stationary.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

P = 128       # partition dim = M tile and K tile (K tile for accumulation)
N_TILE = 512  # output free dimension tile


@nki.jit
def _matmul_kernel(lhs_T, rhs):
    # lhs_T: (K, M), rhs: (K, N)
    K, M = lhs_T.shape
    _K, N = rhs.shape

    out = nl.ndarray((M, N), dtype=lhs_T.dtype, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // P):
        m_start = i_m * P
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE

            psum = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(K // P):
                k_start = i_k * P

                # stationary: (K_TILE, M_TILE) — par=K, free=M
                stat = nl.ndarray((P, P), dtype=lhs_T.dtype, buffer=nl.sbuf)
                nisa.dma_copy(stat, lhs_T[k_start:k_start + P, m_start:m_start + P])

                # moving: (K_TILE, N_TILE) — par=K, free=N
                mov = nl.ndarray((P, N_TILE), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(mov, rhs[k_start:k_start + P, n_start:n_start + N_TILE])

                # psum += stat.T @ mov  →  (M_TILE, N_TILE)
                nisa.nc_matmul(psum, stat, mov)

            result_sb = nl.ndarray((P, N_TILE), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(result_sb, psum)
            nisa.dma_copy(out[m_start:m_start + P, n_start:n_start + N_TILE], result_sb)

    return out


def matmul(lhs, rhs):
    """Matrix multiply lhs (M, K) @ rhs (K, N). Inputs: CPU float32 tensors."""
    device = xm.xla_device()
    # nc_matmul computes stationary.T @ moving; pass lhs.T as stationary
    lhs_T = lhs.T.contiguous()
    result = _matmul_kernel(lhs_T.to(device), rhs.to(device))
    return result.cpu()
