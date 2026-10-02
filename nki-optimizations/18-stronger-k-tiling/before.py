"""
BEFORE kernel: C = lhsT.T @ rhs with fine K-tiling (TILE_K=32).

Optimization target: Category #18 — Stronger tiling for contraction / moving-free split

This kernel computes C [M=1024, N=512] = lhsT.T @ rhs where:
  lhsT [K=512, M=1024] — left-hand side, already transposed
  rhs  [K=512, N=512]  — right-hand side

K=512 exceeds the nc_matmul stationary limit of 128 partitions, so K must be
tiled. This kernel uses TILE_K=32 — too fine — resulting in 16 nc_matmul calls
per M-tile (128 total across 8 M-tiles). PSUM accumulates across K-tiles.

The loop structure:
  for m in affine_range(8):         # M-tiles
    allocate PSUM [128, 512]
    for k in affine_range(16):      # K-tiles (16 × TILE_K=32 = K=512)
      load lhsT slice [32, 128]
      load rhs  slice [32, 512]
      nc_matmul → accumulate in PSUM
    copy PSUM → SBUF → store result
"""

import nki
import nki.isa as nisa
import nki.language as nl

K = 512
M = 1024
N = 512
TILE_M = 128
TILE_K = 32        # BEFORE: fine tiling — 16 K-tiles per M-tile


@nki.jit(platform_target="trn1")
def matmul_fine_k(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs with TILE_K=32 (fine K-tiling).

    BEFORE: 16 nc_matmul calls per M-tile (128 total) — many small TensorE ops.

    lhsT: [K=512, M=1024] float32, rhs: [K=512, N=512] float32
    Returns: C [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # PSUM accumulates K-partial products; reset each M-tile
        res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            # Load lhsT slice [TILE_K, TILE_M]
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32,
                                   buffer=nl.sbuf)
            nisa.dma_copy(
                dst=lhsT_tile,
                src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                          m * TILE_M:(m + 1) * TILE_M])

            # Load rhs slice [TILE_K, N]
            rhs_tile = nl.ndarray((TILE_K, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=rhs_tile,
                src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:N])

            # Accumulate into PSUM (16 calls per M-tile)
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

    result = matmul_fine_k(lhsT_dev, rhs_dev)
    xm.mark_step()

    ref = lhsT_cpu.T @ rhs_cpu
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-2), (
        f"Max diff: {(result_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
