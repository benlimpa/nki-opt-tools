"""
BEFORE kernel: single-buffer matmul — DMA and compute serialize per M-tile.

Optimization target: Category #14 — Double buffering

This kernel computes C = lhsT.T @ rhs where lhsT [K=128, M=1024] is loaded
as 8 sequential M-tiles (TILE_M=128). A single SBUF buffer holds the current
M-tile. Because each nc_matmul has a data dependency on the preceding DMA load
of the same buffer, DMA and TensorE cannot overlap:

  Iter 0: [DMA tile0 → buf] [nc_matmul(buf)] [store]
  Iter 1: [DMA tile1 → buf] [nc_matmul(buf)] [store]
  ...     serial chain — TensorE stalls until each DMA completes

The rhs tile [K=128, N=512] is loaded once before the loop (no overhead there).
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
K = 128
N = 512

TILE_M = 128
TILE_K = 128   # = K (single K-tile)
TILE_N = 512   # = N


@nki.jit(platform_target="trn1")
def matmul_single_buffer(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs using a single SBUF buffer per M-tile.

    BEFORE: DMA and nc_matmul serialize — each tile load blocks compute.

    lhsT: [K=128, M=1024] float32
    rhs:  [K=128, N=512]  float32
    Returns: C [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load rhs once — not the bottleneck
    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    # Single buffer for lhsT M-tiles — prevents DMA/TensorE overlap
    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)

    for m in nl.sequential_range(M // TILE_M):
        # DMA must complete before nc_matmul can start (data dependency)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=res_sbuf)

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    lhsT = A.T.contiguous().to(device)
    rhs = B.to(device)

    C = matmul_single_buffer(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
