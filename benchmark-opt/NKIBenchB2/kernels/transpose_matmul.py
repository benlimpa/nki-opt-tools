"""Transposed matrix multiplication: C = lhs_t.T @ rhs.

lhs_t is stored as (K, M) in HBM (already transposed), rhs is (K, N).
N=10944 is not divisible by N_TILE=512 (last tile partial); handled with min-slicing.
Source: NKIBench/kernels/transpose_matmul_M4096_K2048_N10944_0.py (Beta 1 → Beta 2 migration).
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

P = 128       # partition dim (M tile and K tile for accumulation)
N_TILE = 128  # output free dimension tile (use 128 to handle partial N easily)


@nki.jit
def _transpose_matmul_kernel(lhs_T, rhs):
    # lhs_T: (K, M), rhs: (K, N)
    K, M = lhs_T.shape
    _K, N = rhs.shape

    out = nl.ndarray((M, N), dtype=lhs_T.dtype, buffer=nl.shared_hbm)

    n_n_tiles = math.ceil(N / N_TILE)

    for i_m in nl.affine_range(M // P):
        m_start = i_m * P

        for i_n in nl.affine_range(n_n_tiles):
            n_start = i_n * N_TILE
            n_end = min(n_start + N_TILE, N)
            n_size = n_end - n_start

            psum = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(K // P):
                k_start = i_k * P

                stat = nl.ndarray((P, P), dtype=lhs_T.dtype, buffer=nl.sbuf)
                nisa.dma_copy(stat, lhs_T[k_start:k_start + P, m_start:m_start + P])

                mov = nl.ndarray((P, N_TILE), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    mov[0:P, 0:n_size],
                    rhs[k_start:k_start + P, n_start:n_end],
                )

                nisa.nc_matmul(psum[0:P, 0:n_size], stat, mov[0:P, 0:n_size])

            result_sb = nl.ndarray((P, N_TILE), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(result_sb[0:P, 0:n_size], psum[0:P, 0:n_size])
            nisa.dma_copy(out[m_start:m_start + P, n_start:n_end], result_sb[0:P, 0:n_size])

    return out


def transpose_matmul(lhs_t, rhs):
    """Compute lhs_t.T @ rhs. lhs_t is (K, M), rhs is (K, N). CPU float32 inputs."""
    device = xm.xla_device()
    result = _transpose_matmul_kernel(lhs_t.to(device), rhs.to(device))
    return result.cpu()
