"""
AFTER kernel: K-accumulation grouped in PSUM — one eviction after all K-tiles.

Optimization: Category #12 — Group nc_matmul calls into larger blocks to
maximize Tensor Engine utilization.

KEY CHANGE: all K-tile nc_matmul calls target the SAME PSUM tile inside an
affine_range loop. The NKI compiler forms a PSUM accumulation group: TensorE
accumulates partial results across all K-tiles in PSUM before a single
PSUM→SBUF eviction. No VectorE additions needed.

Before: 4 K-tiles × (nc_matmul → PSUM evict → VectorE add) = 4 evictions + 4 adds
After:  4 K-tiles accumulated in PSUM → 1 eviction + 0 VectorE adds

Per M-tile savings:
  - 3 fewer PSUM→SBUF evictions (ScalarE activation)
  - 4 fewer VectorE tensor_tensor add instructions
Total across 4 M-tiles: 12 fewer evictions + 16 fewer VectorE adds
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
def matmul_psum_accumulate(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs.

    AFTER: all K-tiles accumulate into one PSUM tile within an affine_range
    loop. The compiler forms a PSUM accumulation group — one eviction total
    per M-tile, zero VectorE additions.

    lhsT: [K, M]  stationary (A transposed)
    rhs:  [K, N]  moving     (B)
    Returns: C [M, N] in HBM
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # PSUM accumulator: shared across all K-tile nc_matmul calls
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        # OPTIMAL: affine_range over K-tiles → compiler manages PSUM accumulation
        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            # nc_matmul accumulates into res_psum across all K-tile iterations
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        # SINGLE PSUM→SBUF eviction after all K-tiles are accumulated
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

    C = matmul_psum_accumulate(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
