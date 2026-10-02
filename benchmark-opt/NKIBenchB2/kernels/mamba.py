"""Mamba SSM: out[c,t] = sum_s( c[s,t] * scan_res[c,s,t] ).

scan_res[c,s,t] = deltaA[c,s,t] * scan_res[c,s,t-1] + deltaBu[c,s,t]
  deltaA[c,s,t]  = exp(delta[c,t] * a[c,s])
  deltaBu[c,s,t] = delta[c,t] * b[s,t] * u[c,t]

NKI kernel performs only the sequential scan (tensor_tensor_scan) over the M dimension.
Broadcasting over the S (state) and C (channel) dimensions is handled in the Python wrapper
via PyTorch operations to avoid free-dimension broadcast limitations in NKI.

Source: NKIBench/kernels/mamba_M7168_C256_S16_0.py (Beta 1 → Beta 2 migration).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla.core.xla_model as xm

C_TILE = 128   # partition dim (channels per tile)
T_TILE = 1024  # seq_len chunk for SBUF (M=7168 = 7 * T_TILE)


@nki.jit
def _mamba_scan_kernel(deltaA_s, deltaBu_s):
    """Sequential scan: out[c,t] = deltaA_s[c,t] * out[c,t-1] + deltaBu_s[c,t].

    deltaA_s, deltaBu_s: (C, M) on HBM, already computed for one state s.
    Returns scan_out (C, M) on HBM.
    """
    C, M = deltaA_s.shape
    scan_out = nl.ndarray((C, M), dtype=nl.float32, buffer=nl.shared_hbm)
    n_t_tiles = M // T_TILE

    for i_c in nl.affine_range(C // C_TILE):
        c_start = i_c * C_TILE

        # Carry state between seq_len chunks; (C_TILE, 1) with 1 element per partition
        state = nl.ndarray((C_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(state, 0.0)

        for i_t in nl.sequential_range(n_t_tiles):
            t_start = i_t * T_TILE

            dA = nl.ndarray((C_TILE, T_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dA, deltaA_s[c_start:c_start + C_TILE, t_start:t_start + T_TILE])
            dBu = nl.ndarray((C_TILE, T_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dBu, deltaBu_s[c_start:c_start + C_TILE, t_start:t_start + T_TILE])

            scan_chunk = nl.ndarray((C_TILE, T_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(scan_chunk, dA, dBu, state, nl.multiply, nl.add)

            # Store chunk result
            nisa.dma_copy(
                scan_out[c_start:c_start + C_TILE, t_start:t_start + T_TILE],
                scan_chunk,
            )
            # Carry last column as state for next chunk
            nisa.dma_copy(
                state,
                scan_out[c_start:c_start + C_TILE, t_start + T_TILE - 1:t_start + T_TILE],
            )

    return scan_out


def mamba(delta, u, a, b, c):
    """Mamba SSM. delta,u:(C,M), a:(C,S), b:(S,M), c:(S,M). CPU float32 inputs."""
    C, M = delta.shape
    S = a.shape[1]
    device = xm.xla_device()

    delta_u = delta * u  # (C, M)
    out = torch.zeros(C, M, dtype=torch.float32)

    for s in range(S):
        a_s = a[:, s:s + 1]              # (C, 1) — broadcasts over M
        b_s = b[s:s + 1, :]             # (1, M) — broadcasts over C
        c_s = c[s:s + 1, :]             # (1, M)

        # Compute per-state operands in PyTorch (handles broadcasting naturally)
        deltaA_s = torch.exp(delta * a_s)    # (C, M)
        deltaBu_s = delta_u * b_s            # (C, M)

        scan_s = _mamba_scan_kernel(
            deltaA_s.to(device), deltaBu_s.to(device)
        ).cpu()

        out += scan_s * c_s  # (C, M) * (1, M) — broadcast in PyTorch

    return out
