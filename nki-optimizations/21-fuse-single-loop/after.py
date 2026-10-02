"""
AFTER kernel: Row-wise mean-subtraction over x [M=512, N=1024] tiled by
TILE_M=128, using a SINGLE fused affine_range loop — no HBM intermediate.

Optimization: Category #21 — Fuse into a single loop to avoid
intermediate HBM stores/loads.

KEY CHANGE:
  Before: Loop 1 computes mean → stores to HBM (means_hbm)
           Loop 2 reloads means from HBM, reloads x from HBM, subtracts
           x is read TWICE (once per loop), means round-trip through HBM

  After:  Single loop per tile: load x once, compute mean (keep in SBUF),
           subtract mean from x — all within the same tile iteration.
           No HBM intermediate. x loaded ONCE. Means stay in SBUF.

Expected savings per tile:
  - 1 fewer DMA read  of x       (N * TILE_M * 4 bytes = 512 KB total)
  - 1 fewer DMA write of means   (TILE_M * 1 * 4 bytes =   2 KB total)
  - 1 fewer DMA read  of means   (TILE_M * 1 * 4 bytes =   2 KB total)

Total HBM traffic: reduced by ~50% (x reads halved, means eliminated)
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Same shapes as before kernel
M = 512
N = 1024
TILE_M = 128


@nki.jit(platform_target="trn1")
def mean_subtract_fused(x):
    """
    Compute y[m, n] = x[m, n] - mean(x[m, :]) for each row m.

    AFTER: single fused loop per tile — x loaded once; per-row mean never
    leaves SBUF; no caller-provided intermediate tensor needed.

    x: [M=512, N=1024] float32 in HBM
    Returns: y [M=512, N=1024] float32
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # ---- Load x once ----
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # ---- Compute per-row mean, keep in SBUF ----
        row_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        row_mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_mean, data=row_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        # ---- Subtract mean (stays in SBUF — no HBM round-trip) ----
        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile,
                           op0=nl.subtract, operand0=row_mean)

        nisa.dma_copy(dst=y[m * TILE_M:(m + 1) * TILE_M, 0:N], src=y_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    result = mean_subtract_fused(x_dev)
    xm.mark_step()

    ref = x - x.mean(dim=1, keepdim=True)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-4), (
        f"Max diff: {(result_cpu - ref).abs().max()}")
    print("Correctness check passed.")
