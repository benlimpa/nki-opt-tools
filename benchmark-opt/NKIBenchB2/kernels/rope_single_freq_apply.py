"""Rotary Position Embedding (RoPE) with pre-computed cos/sin frequencies.

Layout: x [D, S], cos/sin [D//2, S]. Applies first-half/second-half rotation.
Source: NKIBench/kernels/rope_single_freq_apply_B1_H64_N4096_D128_0.py (Beta 1 → Beta 2 migration).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

S_TILE = 512  # free dimension tile size (S = B*H*N must be divisible by S_TILE)


@nki.jit
def _rope_kernel(x_in, cos, sin):
    D, S = x_in.shape
    half_d = D // 2

    x_out = nl.ndarray((D, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    for i_s in nl.affine_range(S // S_TILE):
        s_start = i_s * S_TILE

        # Load the full D-slice of x for this S tile: (D, S_TILE)
        x_tile = nl.ndarray((D, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(x_tile, x_in[0:D, s_start:s_start + S_TILE])

        # Load cos and sin: (half_d, S_TILE)
        cos_tile = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(cos_tile, cos[0:half_d, s_start:s_start + S_TILE])
        sin_tile = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(sin_tile, sin[0:half_d, s_start:s_start + S_TILE])

        # Split x into even (first half) and odd (second half) over D dimension
        x_even = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(x_even, x_tile[0:half_d, 0:S_TILE])

        x_odd = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(x_odd, x_tile[half_d:D, 0:S_TILE])

        # out_even = x_even * cos - x_odd * sin
        e_cos = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(e_cos, x_even, cos_tile, nl.multiply)
        o_sin = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(o_sin, x_odd, sin_tile, nl.multiply)
        out_even = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(out_even, e_cos, o_sin, nl.subtract)

        # out_odd = x_odd * cos + x_even * sin
        o_cos = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(o_cos, x_odd, cos_tile, nl.multiply)
        e_sin = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(e_sin, x_even, sin_tile, nl.multiply)
        out_odd = nl.ndarray((half_d, S_TILE), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(out_odd, o_cos, e_sin, nl.add)

        # Store output: first half (even), second half (odd)
        nisa.dma_copy(x_out[0:half_d, s_start:s_start + S_TILE], out_even)
        nisa.dma_copy(x_out[half_d:D, s_start:s_start + S_TILE], out_odd)

    return x_out


def rope_single_freq_apply(x, freqs_cos, freqs_sin):
    """Apply RoPE. Inputs: CPU float32 tensors x[D,S], cos[D//2,S], sin[D//2,S]."""
    device = xm.xla_device()
    result = _rope_kernel(x.to(device), freqs_cos.to(device), freqs_sin.to(device))
    return result.cpu()
