"""RMSNorm followed by matrix multiplication: (x / rms(x)) @ w.

Two-pass implementation using an HBM intermediate for the transposed normalized x:
  1. _rmsnorm_kernel: x (M,K) → x_norm_T (K,M) stored on HBM.
  2. _matmul_t_kernel: x_norm_T (K,M) × w (K,N) → out (M,N) via nc_matmul.

nc_matmul computes stationary.T @ moving; x_norm_T is already in (K,M) layout
for use as stationary.

Source: NKIBench/kernels/rmsnorm_matmul_M4096_N2048_K1024_0.py (Beta 1 → Beta 2 migration).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

M_TILE = 128   # partition dim (rows of x per tile)
K_TILE = 128   # K dimension tile for both RMS reduction and nc_matmul contraction
N_TILE = 512   # free dim tile for matmul output
EPS = 1e-6


@nki.jit
def _rmsnorm_kernel(x):
    """RMS-normalize x (M, K) → x_norm_T (K, M): stores result transposed for nc_matmul."""
    M, K = x.shape
    x_norm_T = nl.ndarray((K, M), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE

        sq_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(sq_sum, 0.0)

        for i_k in nl.affine_range(K // K_TILE):
            k_start = i_k * K_TILE
            x_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(x_tile, x[m_start:m_start + M_TILE, k_start:k_start + K_TILE])
            sq = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(sq, x_tile, x_tile, nl.multiply)
            chunk_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(chunk_sum, nl.add, sq, axis=1)
            nisa.tensor_tensor(sq_sum, sq_sum, chunk_sum, nl.add)

        eps_buf = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(eps_buf, float(EPS))
        rms = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(rms, nl.rsqrt, sq_sum, bias=eps_buf, scale=float(1.0 / K))

        for i_k in nl.affine_range(K // K_TILE):
            k_start = i_k * K_TILE
            x_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(x_tile, x[m_start:m_start + M_TILE, k_start:k_start + K_TILE])

            x_norm = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(x_norm, x_tile, nl.multiply, rms)

            # Transpose (M_TILE, K_TILE) → (K_TILE, M_TILE) via PSUM for nc_matmul layout
            x_norm_T_psum = nl.ndarray((K_TILE, M_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(x_norm_T_psum, x_norm)
            x_norm_T_sb = nl.ndarray((K_TILE, M_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(x_norm_T_sb, x_norm_T_psum)
            nisa.dma_copy(
                x_norm_T[k_start:k_start + K_TILE, m_start:m_start + M_TILE],
                x_norm_T_sb,
            )

    return x_norm_T


@nki.jit
def _matmul_t_kernel(x_norm_T, w):
    """Compute x_norm @ w from x_norm_T (K, M) and w (K, N), returns (M, N)."""
    K, M = x_norm_T.shape
    _K, N = w.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE

            psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(K // K_TILE):
                k_start = i_k * K_TILE
                stat = nl.ndarray((K_TILE, M_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(stat, x_norm_T[k_start:k_start + K_TILE, m_start:m_start + M_TILE])
                mov = nl.ndarray((K_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(mov, w[k_start:k_start + K_TILE, n_start:n_start + N_TILE])
                nisa.nc_matmul(psum, stat, mov)

            result_sb = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(result_sb, psum)
            nisa.dma_copy(out[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result_sb)

    return out


def rmsnorm_matmul(x, w):
    """RMSNorm(x) @ w. x: (M, K), w: (K, N). CPU float32 inputs."""
    device = xm.xla_device()
    x_norm_T = _rmsnorm_kernel(x.to(device))   # (K, M) on XLA device
    result = _matmul_t_kernel(x_norm_T, w.to(device))
    return result.cpu()
