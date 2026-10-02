"""Add + RMSNorm (with gain) + Matmul: ((x+z)/rms(x+z)*g) @ w.

Steps: y = x + z; normalize y by RMS; scale by gain g; matmul with w.
Source: NKIBench/kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py (Beta 1 → Beta 2 migration).
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
def _add_rmsnorm_kernel(x, z):
    """Compute y_norm_T (K, M): (x+z) normalized by RMS, then transposed. g absorbed into w."""
    M, K = x.shape
    y_norm_T = nl.ndarray((K, M), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE

        # Accumulate sum of squares of (x+z) over K
        sq_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(sq_sum, 0.0)

        for i_k in nl.affine_range(K // K_TILE):
            k_start = i_k * K_TILE
            x_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(x_tile, x[m_start:m_start + M_TILE, k_start:k_start + K_TILE])
            z_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(z_tile, z[m_start:m_start + M_TILE, k_start:k_start + K_TILE])

            y_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(y_tile, x_tile, z_tile, nl.add)

            sq = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(sq, y_tile, y_tile, nl.multiply)
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
            z_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(z_tile, z[m_start:m_start + M_TILE, k_start:k_start + K_TILE])

            y_tile = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(y_tile, x_tile, z_tile, nl.add)

            # Normalize: y_norm = y_tile * rms_inv (rms is (M_TILE,1), broadcasts along free dim)
            y_norm = nl.ndarray((M_TILE, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(y_norm, y_tile, nl.multiply, rms)

            # g scaling is absorbed into w in the Python wrapper (w_g = w * g[:,None]),
            # so y_scaled = y_norm here (no separate g multiplication needed in kernel).
            y_scaled = y_norm

            # Transpose (M_TILE, K_TILE) → (K_TILE, M_TILE)
            y_T_psum = nl.ndarray((K_TILE, M_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(y_T_psum, y_scaled)
            y_T_sb = nl.ndarray((K_TILE, M_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(y_T_sb, y_T_psum)
            nisa.dma_copy(y_norm_T[k_start:k_start + K_TILE, m_start:m_start + M_TILE], y_T_sb)

    return y_norm_T


@nki.jit
def _matmul_t_kernel(x_T, w):
    """x_T (K, M) × w (K, N) → out (M, N) via nc_matmul."""
    K, M = x_T.shape
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
                nisa.dma_copy(stat, x_T[k_start:k_start + K_TILE, m_start:m_start + M_TILE])
                mov = nl.ndarray((K_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(mov, w[k_start:k_start + K_TILE, n_start:n_start + N_TILE])
                nisa.nc_matmul(psum, stat, mov)

            result_sb = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(result_sb, psum)
            nisa.dma_copy(out[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result_sb)

    return out


def add_rmsnorm_matmul(x, w, eps, z, g):
    """((x+z)/rms(x+z)*g) @ w. x,z:(M,K), w:(K,N), g:(K,). CPU float32 inputs.

    g scaling is absorbed into w: w_g = w * g.unsqueeze(1), so the kernel computes
    y_norm_T (without explicit g) and then nc_matmul with the g-scaled weights.
    """
    device = xm.xla_device()
    # Absorb g into w to avoid free-dimension broadcast in the kernel
    w_g = w * g.unsqueeze(1)  # (K, N) — each row of w scaled by g[k]
    y_norm_T = _add_rmsnorm_kernel(x.to(device), z.to(device))
    result = _matmul_t_kernel(y_norm_T, w_g.to(device))
    return result.cpu()
