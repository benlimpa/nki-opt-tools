"""
BEFORE kernel: single-level tiling MatMul.
Baseline: Loads lhsT[k, 0] for every N-tile (N // TILE_N times).

Category: #32 — Multi-level tiling

Concretized Description:
In a [128, 2048, 2048] MatMul (C = lhsT.T @ rhs), the baseline processes one N-tile at a time.
For each of the 16 N-tiles, it must iterate over all 16 K-tiles, loading lhsT each time.
Total loads of lhsT: 16 * 16 = 256 tiles.
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
def matmul_baseline(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs.
    lhsT: [K, M] (Stationary)
    rhs:  [K, N] (Moving)
    C:    [M, N]
    """
    C = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Use a zero-initialized SBUF tile to reliably zero out PSUM
    zero_sbuf = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

    # Process one N-tile at a time
    for n_idx in nl.affine_range(N // TILE_N):
        # Zero out PSUM before accumulation for each N-tile.
        nisa.activation(dst=res_psum, data=zero_sbuf, op=nl.copy)

        for k_idx in nl.affine_range(K // TILE_K):
            # Load lhsT[k, 0] - This is loaded for every n_idx! (16 times per k_idx)
            # lhsT_tile: [TILE_K, TILE_M] -> TILE_K is partition dim
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile, src=lhsT[k_idx * TILE_K : (k_idx + 1) * TILE_K, 0:TILE_M])

            # Load rhs[k, n]
            # rhs_tile: [TILE_K, TILE_N] -> TILE_K is partition dim
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile, src=rhs[k_idx * TILE_K : (k_idx + 1) * TILE_K, n_idx * TILE_N : (n_idx + 1) * TILE_N])

            # Matmul: res = lhsT_tile.T @ rhs_tile
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        # Store result
        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
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

    C_dev = matmul_baseline(lhsT_dev, rhs_dev)
    xm.mark_step()

    C_cpu = C_dev.cpu()
    ref = A_cpu @ B_cpu

    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
