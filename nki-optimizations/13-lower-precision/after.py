"""
AFTER kernel: matmul with bfloat16 inputs — 2 bytes/element HBM reads.

Optimization: Category #13 — Do operations in lower precision (nl.bfloat16)

KEY CHANGE: lhsT and rhs are stored as bfloat16 in HBM. Each tile load
transfers only 2 bytes per element instead of 4:
  - lhsT [K=512, M=512] bfloat16 = 512 KB
  - rhs  [K=512, N=512] bfloat16 = 512 KB
  Total HBM reads: 1 MB  (vs 2 MB before)

nc_matmul accepts bfloat16 stationary/moving tiles and accumulates the
product in float32 PSUM automatically — no explicit cast needed.
The result is stored as float32 for full output precision.

Before: float32 tiles → 2 MB HBM reads
After:  bfloat16 tiles → 1 MB HBM reads, same float32 PSUM accumulation
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 512
K = 512
N = 512

TILE_M = 128
TILE_K = 128
TILE_N = 512


@nki.jit(platform_target="trn1")
def matmul_bf16(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs with bfloat16 inputs, float32 output.

    lhsT: [K, M] bfloat16
    rhs:  [K, N] bfloat16
    Returns: C [M, N] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            # Load bfloat16 tiles — 2 bytes per element (half the HBM bandwidth)
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            # nc_matmul accepts bfloat16 and accumulates in float32 PSUM automatically
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        # Single PSUM→SBUF eviction after all K-tiles
        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N], src=res_sbuf)

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    # Cast inputs to bfloat16 for HBM storage
    lhsT = A.T.contiguous().to(torch.bfloat16).to(device)
    rhs = B.to(torch.bfloat16).to(device)

    C = matmul_bf16(lhsT, rhs)
    xm.mark_step()

    # Reference computed in float32 from bfloat16-rounded inputs
    A_bf16 = A.to(torch.bfloat16).to(torch.float32)
    B_bf16 = B.to(torch.bfloat16).to(torch.float32)
    ref = A_bf16 @ B_bf16
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-2), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
