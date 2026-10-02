"""
BEFORE kernel: K-accumulation via PSUM evict + VectorE add per K-tile.

Optimization target: Category #12 — Group nc_matmul calls into larger blocks
to maximize Tensor Engine utilization.

This kernel computes C = lhsT.T @ rhs where the K dimension (512) is split into
4 tiles of 128. The BEFORE version processes one K-tile at a time in a
sequential_range loop: each nc_matmul evicts its result from PSUM to SBUF
immediately, then a VectorE add accumulates it into a running SBUF sum.

Cost per M-tile (4 M-tiles total, 4 K-tiles each):
  - 4 × nc_matmul instructions         (TensorE)
  - 4 × PSUM→SBUF evictions            (ScalarE / activation copy)
  - 4 × VectorE add to SBUF accumulator (VectorE)
  Total: 16 activations + 16 VectorE adds across 4 M-tiles

This pattern prevents the Tensor Engine from forming a PSUM accumulation
group — every K-tile result is forced through SBUF before the next K-tile
can contribute, serialising what could be a single pipeline stage.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 512    # rows of A  (4 M-tiles of 128)
K = 512    # contraction dimension (4 K-tiles of 128)
N = 512    # columns of B

TILE_M = 128
TILE_K = 128
TILE_N = 512   # max moving tile width


@nki.jit(platform_target="trn1")
def matmul_sbuf_accumulate(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs.

    BEFORE: each K-tile produces a partial PSUM result that is immediately
    evicted to SBUF and VectorE-added to a running accumulator.
    4 K-tiles → 4 PSUM evictions + 4 VectorE adds per M-tile.

    lhsT: [K, M]  stationary (A transposed)
    rhs:  [K, N]  moving     (B)
    Returns: C [M, N] in HBM
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # SBUF accumulator: running partial sum across K-tiles
        acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)

        # SUBOPTIMAL: process one K-tile at a time, evict PSUM after each
        for k in nl.sequential_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            # TensorE: partial matmul for this K-tile
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

            # IMMEDIATE EVICTION: PSUM → SBUF after every K-tile
            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)

            # VectorE add to running accumulator
            new_acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=new_acc, data1=acc, data2=res_sbuf, op=nl.add)
            acc = new_acc

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N], src=acc)

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

    C = matmul_sbuf_accumulate(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
