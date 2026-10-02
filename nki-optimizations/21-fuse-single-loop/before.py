"""
BEFORE kernel: Row-wise mean-subtraction over x [M=512, N=1024] tiled by
TILE_M=128, using two separate affine_range loops with an HBM intermediate
for per-row mean values.

Optimization target: Category #21 — Fuse into a single loop to avoid
intermediate HBM stores/loads.

Pattern:
  Loop 1 (over M // TILE_M tiles): compute row mean → store means_hbm[M, 1]
  Loop 2 (over M // TILE_M tiles): load means_hbm[M, 1], subtract from x

The means tensor is M*1*4 bytes = 2 KB.  Because it is produced in loop 1
and consumed in loop 2 (a different affine_range), the compiler must
materialise it through HBM — the SBUF row_mean tile computed in loop 1 is
not live across loops.  This costs:
  - 1 extra HBM write  (loop 1 stores means_hbm)
  - 1 extra HBM read   (loop 2 reloads means_hbm)
  - 1 extra HBM read of x (loop 2 re-reads x since SBUF capacity is limited)
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 512    # total rows
N = 1024   # columns
TILE_M = 128   # partition tile size (≤ 128)


@nki.jit(platform_target="trn1")
def mean_subtract_two_loops(x, means_hbm):
    """
    Compute y[m, n] = x[m, n] - mean(x[m, :]) for each row m.

    BEFORE: two separate loops — loop 1 computes and stores per-row means to
    the caller-provided HBM buffer; loop 2 reloads means and subtracts.

    x:         [M=512, N=1024] float32 in HBM
    means_hbm: [M=512, 1]      float32 in HBM  (intermediate, allocated by caller)
    Returns:   y [M=512, N=1024] float32
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---------- Loop 1: compute per-row mean, write to HBM intermediate ----------
    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Sum across free dimension (N)
        row_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        # Divide by N
        row_mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_mean, data=row_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        # Store means to HBM (necessary intermediate between the two loops)
        nisa.dma_copy(dst=means_hbm[m * TILE_M:(m + 1) * TILE_M, 0:1],
                      src=row_mean)

    # ---------- Loop 2: reload means from HBM, subtract from x ----------
    for m in nl.affine_range(M // TILE_M):
        # Reload x tile (SBUF from loop 1 may no longer be live)
        x_tile2 = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile2, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Reload mean from HBM
        mean_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mean_tile, src=means_hbm[m * TILE_M:(m + 1) * TILE_M, 0:1])

        # Subtract (broadcast mean over N)
        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile2,
                           op0=nl.subtract, operand0=mean_tile)

        nisa.dma_copy(dst=y[m * TILE_M:(m + 1) * TILE_M, 0:N], src=y_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)
    # Allocate HBM intermediate on device
    means_dev = torch.zeros(M, 1, dtype=torch.float32).to(device)

    result = mean_subtract_two_loops(x_dev, means_dev)
    xm.mark_step()

    ref = x - x.mean(dim=1, keepdim=True)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-4), (
        f"Max diff: {(result_cpu - ref).abs().max()}")
    print("Correctness check passed.")
