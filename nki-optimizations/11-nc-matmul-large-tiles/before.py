"""
BEFORE kernel: nc_matmul with small TILE_N=128 — 4 calls per M-tile.

Optimization target: Category #11 — Perform nc_matmul on large contiguous
blocks within its own affine_range loop.

This kernel computes C = A @ B where A [M, K] and B [K, N]. The inputs are
provided pre-transposed as lhsT [K, M] (stationary) and rhs [K, N] (moving).

The BEFORE version uses TILE_N=128 for the moving tile, which is 1/4 of the
maximum supported size (gemm_moving_fmax=512). This means 4 separate nc_matmul
calls per M-tile are needed to cover the full N=512 columns — 32 nc_matmul
instructions total for 8 M-tiles. Each call processes a [128, 128] moving tile
instead of the hardware-optimal [128, 512].

Cost of under-sizing the moving tile:
  - 4× more nc_matmul instructions (32 vs 8)
  - 4× more PSUM→SBUF evictions
  - 4× more result HBM stores (same total bytes, but more transactions)
  - Lower Tensor Engine throughput per instruction cycle
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024   # rows of A  (8 M-tiles of 128)
K = 128    # contraction dimension (1 K-tile)
N = 512    # columns of B

TILE_M = 128   # stationary free dimension max
TILE_K = 128   # partition dimension max
TILE_N_SMALL = 128  # BEFORE: suboptimal — 1/4 of gemm_moving_fmax=512


@nki.jit(platform_target="trn1")
def matmul_small_n_tiles(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs  (equivalent to A @ B after transpose).

    BEFORE: rhs is loaded in TILE_N_SMALL=128 column chunks, requiring
    N // TILE_N_SMALL = 4 nc_matmul calls per M-tile → 32 calls total.

    lhsT: [K, M]  — A transposed, stationary operand
    rhs:  [K, N]  — B, moving operand
    Returns: C [M, N] in HBM
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # Load stationary tile: lhsT[0:K, m*TILE_M:(m+1)*TILE_M]
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        # SUBOPTIMAL: loop over N in small 128-column chunks — 4 nc_matmul calls
        for n in nl.affine_range(N // TILE_N_SMALL):
            rhs_tile = nl.ndarray((TILE_K, TILE_N_SMALL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[0:TILE_K, n * TILE_N_SMALL:(n + 1) * TILE_N_SMALL])

            # nc_matmul: dst = lhsT_tile.T @ rhs_tile → [TILE_M, TILE_N_SMALL]
            res_psum = nl.ndarray((TILE_M, TILE_N_SMALL), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

            # Evict PSUM → SBUF, then store to HBM
            res_sbuf = nl.ndarray((TILE_M, TILE_N_SMALL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
            nisa.dma_copy(
                dst=result[m * TILE_M:(m + 1) * TILE_M,
                            n * TILE_N_SMALL:(n + 1) * TILE_N_SMALL],
                src=res_sbuf)

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    lhsT = A.T.contiguous().to(device)   # [K, M]
    rhs = B.to(device)                    # [K, N]

    C = matmul_small_n_tiles(lhsT, rhs)
    xm.mark_step()

    ref = (A @ B)
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-4), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
