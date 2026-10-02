"""Batched matrix multiplication: C[b] = lhs[b] @ rhs[b].

Source: NKIBench/kernels/bmm_B16_M4096_K64_N4096_0.py (Beta 1 → Beta 2 migration).
K=64 fits in a single K tile (K_TILE = min(K, 128)).
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

M_TILE = 128  # output partition dimension tile
N_TILE = 512  # output free dimension tile
K_TILE = 128  # contraction dimension tile (par_dim of stat/mov)


@nki.jit
def _bmm_kernel(lhs_T, rhs):
    # lhs_T: (B, K, M), rhs: (B, K, N)
    B, K, M = lhs_T.shape
    _B, _K, N = rhs.shape

    out = nl.ndarray((B, M, N), dtype=lhs_T.dtype, buffer=nl.shared_hbm)

    n_k_tiles = math.ceil(K / K_TILE)

    for i_b in nl.affine_range(B):
        for i_m in nl.affine_range(M // M_TILE):
            m_start = i_m * M_TILE
            for i_n in nl.affine_range(N // N_TILE):
                n_start = i_n * N_TILE

                psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)

                for i_k in nl.affine_range(n_k_tiles):
                    k_start = i_k * K_TILE
                    k_end = min(k_start + K_TILE, K)
                    k_size = k_end - k_start

                    stat = nl.ndarray((K_TILE, M_TILE), dtype=lhs_T.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        stat[0:k_size, 0:M_TILE],
                        lhs_T[i_b, k_start:k_end, m_start:m_start + M_TILE],
                    )

                    mov = nl.ndarray((K_TILE, N_TILE), dtype=rhs.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        mov[0:k_size, 0:N_TILE],
                        rhs[i_b, k_start:k_end, n_start:n_start + N_TILE],
                    )

                    nisa.nc_matmul(psum, stat[0:k_size, 0:M_TILE], mov[0:k_size, 0:N_TILE])

                result_sb = nl.ndarray((M_TILE, N_TILE), dtype=out.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(result_sb, psum)
                nisa.dma_copy(out[i_b, m_start:m_start + M_TILE, n_start:n_start + N_TILE], result_sb)

    return out


def bmm(lhs, rhs):
    """Batched matmul lhs (B,M,K) @ rhs (B,K,N). Inputs: CPU float32 tensors."""
    device = xm.xla_device()
    # nc_matmul needs (B, K, M) as stationary; transpose lhs
    lhs_T = lhs.transpose(1, 2).contiguous()
    result = _bmm_kernel(lhs_T.to(device), rhs.to(device))
    return result.cpu()
