"""SwiGLU MLP: out = (silu(x @ w_gate) * (x @ w_up)) @ w_down.

Four passes:
  1. up (M,N) = x (M,K) @ w_up (K,N)
  2. gate (M,N) = x (M,K) @ w_gate (K,N)
  3. swiglu (M,N) = silu(gate) * up  [element-wise]
  4. out (M,K) = swiglu (M,N) @ w_down (N,K)

Source: NKIBench/kernels/swiglu_M4096_N3072_K1024_0.py (Beta 1 → Beta 2 migration).
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
def _swiglu_elemwise_kernel(up, gate):
    """Compute silu(gate) * up element-wise. Both (M,N) → out (M,N)."""
    M, N = up.shape
    out = nl.ndarray((M, N), dtype=up.dtype, buffer=nl.shared_hbm)

    for i_m in nl.affine_range(M // M_TILE):
        m_start = i_m * M_TILE
        for i_n in nl.affine_range(N // N_TILE):
            n_start = i_n * N_TILE

            gate_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(gate_tile, gate[m_start:m_start + M_TILE, n_start:n_start + N_TILE])

            # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            neg_exp = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(neg_exp, nl.exp, gate_tile, scale=-1.0)
            denom = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(denom, neg_exp, nl.add, 1.0)
            sig = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(sig, denom)
            silu = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(silu, gate_tile, sig, nl.multiply)

            up_tile = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(up_tile, up[m_start:m_start + M_TILE, n_start:n_start + N_TILE])

            result = nl.ndarray((M_TILE, N_TILE), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(result, silu, up_tile, nl.multiply)
            nisa.dma_copy(out[m_start:m_start + M_TILE, n_start:n_start + N_TILE], result)

    return out


def swiglu(x, w_up, w_down, w_gate):
    """SwiGLU: (silu(x@w_gate) * (x@w_up)) @ w_down.
    x:(M,K), w_up:(K,N), w_down:(N,K), w_gate:(K,N). CPU float32 inputs."""
    device = xm.xla_device()
    x_T = x.t().contiguous()
    up = _matmul_kernel(x_T.to(device), w_up.to(device))
    gate = _matmul_kernel(x_T.to(device), w_gate.to(device))
    sg = _swiglu_elemwise_kernel(up, gate)
    # For final matmul: swiglu (M,N) @ w_down (N,K) — transpose swiglu for nc_matmul
    sg_T = sg.cpu().t().contiguous()
    result = _matmul_kernel(sg_T.to(device), w_down.to(device))
    return result.cpu()
