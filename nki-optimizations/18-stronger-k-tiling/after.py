"""
AFTER kernel: C = lhsT.T @ rhs with maximum K-tiling (TILE_K=128).

Optimization: Category #18 — Stronger tiling for contraction / moving-free split

KEY CHANGE: Increase TILE_K from 32 → 128 (the nc_matmul partition maximum).
With TILE_K=128, K=512 is covered in 4 tiles instead of 16, producing 4
nc_matmul calls per M-tile (32 total) instead of 128. Each call processes 4×
more of the K contraction in one TensorE instruction.

Why TILE_K=128 is "moving-free":
  Each nc_matmul operand (stationary + moving) occupies exactly 128 partitions —
  the full SBUF partition width. No further splitting of the contraction
  dimension is possible, so the "moving" tensor slice is loaded exactly once per
  K-tile with no wasted sub-tile reloads.

nc_matmul call count:
  Before: 16 K-tiles × 8 M-tiles = 128 calls
  After:   4 K-tiles × 8 M-tiles =  32 calls  (4× fewer)

HBM traffic is unchanged — same bytes transferred, fewer DMA packets.
The gain comes from amortizing TensorE instruction overhead across 4× larger tiles.
"""

import nki
import nki.isa as nisa
import nki.language as nl

K = 512
M = 1024
N = 512
TILE_M = 128
TILE_K = 128       # AFTER: maximum tile — 4 K-tiles per M-tile


@nki.jit(platform_target="trn1")
def matmul_max_k(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs with TILE_K=128 (maximum K-tiling).

    AFTER: 4 nc_matmul calls per M-tile (32 total) — 4× fewer, 4× larger ops.

    lhsT: [K=512, M=1024] float32, rhs: [K=512, N=512] float32
    Returns: C [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # PSUM accumulates K-partial products; reset each M-tile
        res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            # Load lhsT slice [TILE_K=128, TILE_M=128] — full partition width
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32,
                                   buffer=nl.sbuf)
            nisa.dma_copy(
                dst=lhsT_tile,
                src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                          m * TILE_M:(m + 1) * TILE_M])

            # Load rhs slice [TILE_K=128, N] — moving-free: full partition width
            rhs_tile = nl.ndarray((TILE_K, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=rhs_tile,
                src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:N])

            # Accumulate into PSUM (4 calls per M-tile instead of 16)
            nisa.nc_matmul(dst=res_psum,
                           stationary=lhsT_tile,
                           moving=rhs_tile)

        # Copy PSUM → SBUF → HBM
        res_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(
            dst=result[m * TILE_M:(m + 1) * TILE_M, 0:N],
            src=res_sbuf)

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    lhsT_cpu = torch.randn(K, M, dtype=torch.float32)
    rhs_cpu = torch.randn(K, N, dtype=torch.float32)

    lhsT_dev = lhsT_cpu.to(device)
    rhs_dev = rhs_cpu.to(device)

    result = matmul_max_k(lhsT_dev, rhs_dev)
    xm.mark_step()

    ref = lhsT_cpu.T @ rhs_cpu
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-2), (
        f"Max diff: {(result_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
