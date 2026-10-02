"""
AFTER kernel: multi-level tiling MatMul.
Optimization: Category #32 — Multi-level tiling.

KEY CHANGE: Group the N dimension into multiple tiles that are kept in PSUM.
This allows the K loop to be moved to the outer level relative to the N-tile loop.
Each lhsT[k, 0] tile is loaded exactly ONCE and reused for all N-tiles in the group.

Level 1 (HBM/SBUF): Tile K into 128-element blocks.
Level 2 (PSUM): Group 16 N-tiles (total 2048 elements) in PSUM.

Total loads of lhsT: 16 tiles (vs 256 in baseline).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

M = 128
N = 2048
K = 2048

TILE_M = 128
TILE_N = 128
TILE_K = 128


@nki.jit(platform_target="trn1")
def matmul_multi_level_tiling(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs using multi-level tiling.
    lhsT: [K, M]
    rhs:  [K, N]
    C:    [M, N]
    """
    C = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Use a zero-initialized SBUF tile to reliably zero out PSUM.
    zero_sbuf = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

    # Initialize all PSUM tiles to zero before starting.
    for n_idx in nl.affine_range(N // TILE_N):
        nisa.activation(dst=res_psum[:, n_idx * TILE_N : (n_idx + 1) * TILE_N],
                        data=zero_sbuf,
                        op=nl.copy)

    # Level 1 tiling: K dimension
    for k_idx in nl.affine_range(K // TILE_K):
        # Load lhsT[k, 0] - LOAD ONCE per k_idx!
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k_idx * TILE_K : (k_idx + 1) * TILE_K, 0:TILE_M])

        # Inner loop over N-tiles reuses lhsT_tile
        for n_idx in nl.affine_range(N // TILE_N):
            # Load rhs[k, n]
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile, src=rhs[k_idx * TILE_K : (k_idx + 1) * TILE_K, n_idx * TILE_N : (n_idx + 1) * TILE_N])

            # Matmul into the specific PSUM tile
            nisa.nc_matmul(dst=res_psum[:, n_idx * TILE_N : (n_idx + 1) * TILE_N],
                           stationary=lhsT_tile,
                           moving=rhs_tile)

    # Store all result tiles
    for n_idx in nl.affine_range(N // TILE_N):
        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum[:, n_idx * TILE_N : (n_idx + 1) * TILE_N])
        nisa.dma_copy(dst=C[0:TILE_M, n_idx * TILE_N : (n_idx + 1) * TILE_N], src=res_sbuf)

    return C


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    A_cpu = torch.randn(M, K, dtype=torch.float32)
    B_cpu = torch.randn(K, N, dtype=torch.float32)

    # lhsT must be [K, M] for nc_matmul to contract over K (partition dim)
    lhsT_dev = A_cpu.T.contiguous().to(device)
    rhs_dev = B_cpu.to(device)

    C_dev = matmul_multi_level_tiling(lhsT_dev, rhs_dev)
    xm.mark_step()

    C_cpu = C_dev.cpu()
    ref = A_cpu @ B_cpu

    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
