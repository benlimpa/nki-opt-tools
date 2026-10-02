"""
BEFORE kernel: matmul with fully serial load→compute→store per M-tile.

Optimization target: Category #16 — Pipeline compute and data movement

This kernel computes C = lhsT.T @ rhs, processing lhsT [K=128, M=1024] as
8 sequential M-tiles. Three phases execute serially per tile:

  Per tile:
    [DMA load  64 KB: lhsT tile]
    [TensorE compute: nc_matmul]
    [DMA store 256 KB: result tile]

No overlap between any of the three phases. The 256 KB output store is
4× larger than the 64 KB input load, so output stores dominate DMA time.

Total time ≈ (T_load + T_compute + T_store) × 8 tiles
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
K = 128
N = 512

TILE_M = 128
TILE_K = 128
TILE_N = 512


@nki.jit(platform_target="trn1")
def matmul_serial(lhsT, rhs):
    """
    C = lhsT.T @ rhs — serial load/compute/store per M-tile.

    BEFORE: no pipeline overlap; load, compute, store are sequential.

    lhsT: [K=128, M=1024] float32
    rhs:  [K=128, N=512]  float32
    Returns: C [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load rhs once
    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    # Single input and output buffer — forces serial pipeline
    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
    res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)

    for m in nl.sequential_range(M // TILE_M):
        # Phase 1: load input tile (64 KB) — TensorE stalls
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        # Phase 2: compute
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)

        # Phase 3: store output tile (256 KB) — stalls before next load
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

    C = matmul_serial(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
