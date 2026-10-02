"""AdamW optimizer parameter update step.

AdamW at t=1, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01.
Source: NKIBench/kernels/adamw_M10944_N2048_0.py (Beta 1 → Beta 2 migration).

M=10944 is not divisible by P=128 (last tile has 64 rows); handled with min-slicing.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

import math

P = 128        # partition dimension
N_TILE = 1024  # free dimension tile size


@nki.jit
def _adamw_kernel(theta, g, m, v):
    M, N = theta.shape
    out = nl.ndarray((M, N), dtype=theta.dtype, buffer=nl.shared_hbm)

    n_p_tiles = math.ceil(M / P)
    n_f_tiles = N // N_TILE

    for i_p in nl.affine_range(n_p_tiles):
        p_start = i_p * P
        p_end = min(p_start + P, M)
        p_size = p_end - p_start

        for i_f in nl.affine_range(n_f_tiles):
            f_start = i_f * N_TILE

            theta_tile = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            g_tile = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            m_tile = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            v_tile = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)

            nisa.dma_copy(
                theta_tile[0:p_size, 0:N_TILE],
                theta[p_start:p_end, f_start:f_start + N_TILE],
            )
            nisa.dma_copy(
                g_tile[0:p_size, 0:N_TILE],
                g[p_start:p_end, f_start:f_start + N_TILE],
            )
            nisa.dma_copy(
                m_tile[0:p_size, 0:N_TILE],
                m[p_start:p_end, f_start:f_start + N_TILE],
            )
            nisa.dma_copy(
                v_tile[0:p_size, 0:N_TILE],
                v[p_start:p_end, f_start:f_start + N_TILE],
            )

            # theta_t = theta - 1e-5 * theta  (weight decay)
            wd = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(wd, theta_tile, nl.multiply, 1e-5)
            theta_t = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(theta_t, theta_tile, wd, nl.subtract)

            # m_t = 0.9 * m + 0.1 * g
            m_scaled = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(m_scaled, m_tile, nl.multiply, 0.9)
            g_scaled = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(g_scaled, g_tile, nl.multiply, 0.1)
            m_t = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(m_t, m_scaled, g_scaled, nl.add)

            # v_t = 0.999 * v + 0.001 * g * g
            v_scaled = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(v_scaled, v_tile, nl.multiply, 0.999)
            g2 = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(g2, g_tile, g_tile, nl.multiply)
            g2_scaled = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(g2_scaled, g2, nl.multiply, 0.001)
            v_t = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(v_t, v_scaled, g2_scaled, nl.add)

            # v_hat = v_t * 1000 = v_t / 0.001 (scale denominator)
            v_hat = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(v_hat, v_t, nl.multiply, 1000.0)

            # sqrt(v_hat)
            sqrt_vhat = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(sqrt_vhat, nl.sqrt, v_hat)

            # sqrt(v_hat) + eps
            denom = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(denom, sqrt_vhat, nl.add, 1e-8)

            # 1 / (sqrt(v_hat) + eps)
            inv_denom = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(inv_denom, denom)

            # lr * m_t = 0.01 * m_t (lr=0.001, but lr/beta1_correction=0.01 for t=1 step)
            lr_mt = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(lr_mt, m_t, nl.multiply, 0.01)

            # update = lr * m_t / (sqrt(v_hat) + eps)
            update = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(update, lr_mt, inv_denom, nl.multiply)

            # new_theta = theta_t - update
            new_theta = nl.ndarray((P, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(new_theta, theta_t, update, nl.subtract)

            nisa.dma_copy(
                out[p_start:p_end, f_start:f_start + N_TILE],
                new_theta[0:p_size, 0:N_TILE],
            )

    return out


def adamw(theta, g, m, v):
    """AdamW update. Inputs: CPU float32 tensors of shape (M, N)."""
    device = xm.xla_device()
    result = _adamw_kernel(
        theta.to(device), g.to(device), m.to(device), v.to(device)
    )
    return result.cpu()
