"""
AFTER kernel: nc_matmul with maximum TILE_N=512 — 1 call per M-tile.

Optimization: Category #11 — Perform nc_matmul on large contiguous blocks
within its own affine_range loop.

KEY CHANGE: rhs is loaded as one contiguous [K, 512] block — the full moving
tile at hardware-maximum width (gemm_moving_fmax=512). A single nc_matmul per
M-tile computes the entire [128, 512] output tile in one instruction.

Before: N // 128 = 4 nc_matmul calls per M-tile → 32 total
After:  N // 512 = 1 nc_matmul call per M-tile  →  8 total

Each nc_matmul instruction now does 4× more useful work. The Tensor Engine
processes the full 512-column row in one pass, reducing instruction overhead
and maximizing throughput per compute cycle.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
K = 128
N = 512

TILE_M = 128
TILE_K = 128
TILE_N_LARGE = 512   # AFTER: maximum moving free dimension = gemm_moving_fmax


@nki.jit(platform_target="trn1")
def matmul_large_n_tile(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs  (equivalent to A @ B after transpose).

    AFTER: rhs is loaded as one contiguous [K, 512] block. A single nc_matmul
    per M-tile covers the full N dimension — 8 calls total.

    lhsT: [K, M]  — A transposed, stationary operand
    rhs:  [K, N]  — B, moving operand
    Returns: C [M, N] in HBM
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load rhs once — full [K, N_LARGE=512] block, loop-invariant across M
    rhs_tile = nl.ndarray((TILE_K, TILE_N_LARGE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N_LARGE])

    for m in nl.affine_range(M // TILE_M):
        # Load stationary tile
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        # OPTIMAL: single nc_matmul with full TILE_N_LARGE=512 moving tile
        # dst = lhsT_tile.T @ rhs_tile → [TILE_M, TILE_N_LARGE]
        res_psum = nl.ndarray((TILE_M, TILE_N_LARGE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        # Evict PSUM → SBUF, then store to HBM
        res_sbuf = nl.ndarray((TILE_M, TILE_N_LARGE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(
            dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N_LARGE],
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

    C = matmul_large_n_tile(lhsT, rhs)
    xm.mark_step()

    ref = (A @ B)
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-4), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
