"""
BEFORE kernel: matmul with float32 inputs — full 4 bytes/element HBM reads.

Optimization target: Category #13 — Do operations in lower precision (nl.bfloat16)

This kernel computes C = lhsT.T @ rhs where both inputs are stored as float32
in HBM. Each tile load transfers 4 bytes per element:
  - lhsT [K=512, M=512] float32 = 1 MB
  - rhs  [K=512, N=512] float32 = 1 MB
  Total HBM reads: 2 MB

Uses the PSUM accumulation group pattern (affine_range over K-tiles) so the
comparison isolates only the precision effect.
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
def matmul_fp32(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs with float32 inputs.

    lhsT: [K, M] float32
    rhs:  [K, N] float32
    Returns: C [M, N] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            # Load float32 tiles — 4 bytes per element
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            # TensorE: accumulate partial matmul in PSUM
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
    lhsT = A.T.contiguous().to(device)
    rhs = B.to(device)

    C = matmul_fp32(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
