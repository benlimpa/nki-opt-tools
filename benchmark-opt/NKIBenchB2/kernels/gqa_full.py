"""Grouped Query Attention (GQA): softmax(Q @ K.T / sqrt(D)) @ V.

Flash-attention tiled implementation with online softmax.
Tile sizes: M_TILE=128 (query rows), N_KV_TILE=128 (key/value cols = D), D=128.

Inputs pre-transposed in wrapper:
  q_T  (QH, D, N_q): query in stationary layout for nc_matmul
  k_T  (KH, D, N_k): key in stationary layout for nc_matmul
  v    (KH, N_k, D): value tensor

Output: (QH, N_q, D) — batch dim=1 is squeezed out, added back in wrapper.

Source: NKIBench/kernels/gqa_full_B1_N4096_QH16_KH8_D128_0.py (Beta 1 → Beta 2 migration).
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

M_TILE = 128      # query rows per tile
N_KV_TILE = 128   # key/value cols per tile (= D for the V contraction)
D = 128           # head dimension (fixed)


@nki.jit
def _gqa_kernel(q_T, k_T, v):
    """Flash-attention GQA. Returns out (QH, N_q, D)."""
    QH, _D, N_q = q_T.shape
    KH, _D2, N_k = k_T.shape
    n_rep = QH // KH
    n_q_tiles = N_q // M_TILE
    n_kv_tiles = N_k // N_KV_TILE
    scale = float(1.0 / math.sqrt(D))

    out = nl.ndarray((QH, N_q, D), dtype=nl.float32, buffer=nl.shared_hbm)

    for i_kh in nl.affine_range(KH):
        for i_rep in nl.affine_range(n_rep):
            h = i_kh * n_rep + i_rep

            for i_m in nl.affine_range(n_q_tiles):
                m_start = i_m * M_TILE

                row_max = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(row_max, -3.4028235e38)
                row_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(row_sum, 0.0)
                out_acc = nl.ndarray((M_TILE, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(out_acc, 0.0)

                for i_n in nl.affine_range(n_kv_tiles):
                    n_start = i_n * N_KV_TILE

                    # Attention scores: stat.T @ mov = q_tile @ k_tile.T
                    q_stat = nl.ndarray((D, M_TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(q_stat, q_T[h, 0:D, m_start:m_start + M_TILE])
                    k_mov = nl.ndarray((D, N_KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(k_mov, k_T[i_kh, 0:D, n_start:n_start + N_KV_TILE])
                    attn_psum = nl.ndarray((M_TILE, N_KV_TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(attn_psum, q_stat, k_mov)

                    attn_sb = nl.ndarray((M_TILE, N_KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(attn_sb, attn_psum)
                    nisa.tensor_scalar(attn_sb, attn_sb, nl.multiply, scale)

                    # Online softmax: update max
                    tile_max = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(tile_max, nl.maximum, attn_sb, axis=1)
                    new_max = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(new_max, row_max, tile_max, nl.maximum)

                    neg_new_max = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(neg_new_max, new_max, nl.multiply, -1.0)

                    exp_tile = nl.ndarray((M_TILE, N_KV_TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(exp_tile, nl.exp, attn_sb, bias=neg_new_max)

                    max_diff = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(max_diff, nl.exp, row_max, bias=neg_new_max)

                    exp_tile_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(exp_tile_sum, nl.add, exp_tile, axis=1)
                    scaled_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(scaled_sum, row_sum, nl.multiply, max_diff)
                    nisa.tensor_tensor(row_sum, scaled_sum, exp_tile_sum, nl.add)

                    # Rescale out_acc
                    out_scaled = nl.ndarray((M_TILE, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(out_scaled, out_acc, nl.multiply, max_diff)

                    # exp_tile (M,N_KV) → exp_tile_T (N_KV,M) as stat for nc_matmul
                    exp_tile_T_psum = nl.ndarray((N_KV_TILE, M_TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(exp_tile_T_psum, exp_tile)
                    exp_tile_T_sb = nl.ndarray((N_KV_TILE, M_TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(exp_tile_T_sb, exp_tile_T_psum)

                    v_tile = nl.ndarray((N_KV_TILE, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(v_tile, v[i_kh, n_start:n_start + N_KV_TILE, 0:D])

                    attn_v_psum = nl.ndarray((M_TILE, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(attn_v_psum, exp_tile_T_sb, v_tile)
                    attn_v_sb = nl.ndarray((M_TILE, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(attn_v_sb, attn_v_psum)

                    nisa.tensor_tensor(out_acc, out_scaled, attn_v_sb, nl.add)
                    nisa.tensor_copy(row_max, new_max)

                # Normalize
                inv_sum = nl.ndarray((M_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(inv_sum, row_sum)
                result = nl.ndarray((M_TILE, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(result, out_acc, nl.multiply, inv_sum)
                nisa.dma_copy(out[h, m_start:m_start + M_TILE, 0:D], result)

    return out


def gqa_full(q, k, v):
    """GQA: softmax(Q@K.T/sqrt(D)) @ V. q:(B,N,QH,D), k:(B,N,KH,D), v:(B,N,KH,D). CPU float32."""
    B, N_q, QH, _D = q.shape
    _B, N_k, KH, _D2 = k.shape
    device = xm.xla_device()

    q_T = q.squeeze(0).permute(1, 2, 0).contiguous()   # (QH, D, N_q)
    k_T = k.squeeze(0).permute(1, 2, 0).contiguous()   # (KH, D, N_k)
    v_perm = v.squeeze(0).permute(1, 0, 2).contiguous()  # (KH, N_k, D)

    out = _gqa_kernel(q_T.to(device), k_T.to(device), v_perm.to(device))
    return out.cpu().unsqueeze(0)  # (1, QH, N_q, D)
