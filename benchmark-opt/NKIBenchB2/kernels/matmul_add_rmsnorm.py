"""Matmul + Add + RMSNorm: RMSNorm((x @ w) + z) * g.

Two passes:
  1. _matmul_add_kernel: x (M,K) @ w (K,N) + z (M,N) → y (M,N) on HBM.
  2. _rmsnorm_kernel: y (M,N) → out (M,N) normalized by RMS over N and scaled by g (N,).

Source: NKIBench/kernels/matmul_add_rmsnorm_M4096_N2048_K2048_0.py (Beta 1 → Beta 2 migration).
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
EPS = 1e-5


@nki.jit
def _matmul_add_kernel(lhs_T, w, z):
    """lhs_T (K,M) × w (K,N) + z (M,N) → y (M,N)."""
    K, M = lhs_T.shape
    _K, N = w.shape
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE
            psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(K // K_TILE):
                k_start = i_k * K_TILE
                stat = nl.ndarray((K_TILE, M_TILE), dtype=lhs_T.dtype, buffer=nl.sbuf)
                nisa.dma_copy(stat, lhs_T[k_start:k_start + K_TILE, m_start:m_start + M_TILE])
                mov = nl.ndarray((K_TILE, N_TILE), dtype=w.dtype, buffer=nl.sbuf)
                nisa.dma_copy(mov, w[k_start:k_start + K_TILE, n_start:n_start + N_TILE])
                nisa.nc_matmul(psum, stat, mov)

            result = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(result, psum)
            z_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(z_tile, z[m_start:m_start + M_TILE, n_start:n_start + N_TILE])
            nisa.tensor_tensor(result, result, z_tile, nl.add)
            nisa.dma_copy(y[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result)

    return y


@nki.jit
def _rmsnorm_scale_kernel(y, g_rep):
    """RMSNorm(y) * g over last axis N. y (M,N), g_rep (M,N) → out (M,N)."""
    M, N = y.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE

        # Pass 1: accumulate sum of squares over N
        sq_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(sq_sum, 0.0)

        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE
            y_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(y_tile, y[m_start:m_start + M_TILE, n_start:n_start + N_TILE])
            sq = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(sq, y_tile, y_tile, nl.multiply)
            chunk_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(chunk_sum, nl.add, sq, axis=1)
            nisa.tensor_tensor(sq_sum, sq_sum, chunk_sum, nl.add)

        eps_buf = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(eps_buf, float(EPS))
        rms_inv = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(rms_inv, nl.rsqrt, sq_sum, bias=eps_buf, scale=float(1.0 / N))

        # Pass 2: normalize, scale by g_rep, store
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE
            y_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(y_tile, y[m_start:m_start + M_TILE, n_start:n_start + N_TILE])

            normed = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(normed, y_tile, nl.multiply, rms_inv)

            g_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(g_tile, g_rep[m_start:m_start + M_TILE, n_start:n_start + N_TILE])

            result = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(result, normed, g_tile, nl.multiply)
            nisa.dma_copy(out[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result)

    return out


def matmul_add_rmsnorm(x, w, eps, z, g):
    """RMSNorm(x @ w + z) * g. x:(M,K), w:(K,N), z:(M,N), g:(N,). CPU float32 inputs.

    g (N,) is broadcast to (M,N) in Python to avoid free-dim broadcast in the NKI kernel.
    """
    device = xm.xla_device()
    M = x.shape[0]
    lhs_T = x.t().contiguous()
    g_rep = g.unsqueeze(0).expand(M, -1).contiguous()  # (M, N)
    y = _matmul_add_kernel(lhs_T.to(device), w.to(device), z.to(device))
    result = _rmsnorm_scale_kernel(y, g_rep.to(device))
    return result.cpu()
