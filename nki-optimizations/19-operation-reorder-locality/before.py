"""
BEFORE kernel: compute row-wise max AND row-wise sum of X [M=1024, N=512]
in two separate passes — X is read from HBM twice.

Optimization target: Category #19 — Reorder operations to improve locality

Two independent reductions are performed in separate affine_range loops.
Each loop reads all M/TILE_M tiles of X from HBM:

  Pass 1 (max): for each tile, load x_tile → row-reduce(max) → store row_max to HBM
  Pass 2 (sum): for each tile, load x_tile → row-reduce(sum) → store row_sum to HBM

Total HBM reads: 2 × 2 MB = 4 MB (X read twice).
Each tile read, discarded, then read again for the second reduction.

The two passes are order-independent — max does not depend on sum and vice versa —
so they CAN be reordered and co-located on the same tile iteration.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_M = 128


@nki.jit(platform_target="trn1")
def dual_reduce_two_pass(x):
    """
    Compute row-wise max and sum of X in two separate passes.

    BEFORE: X read twice from HBM (4 MB total reads).

    x: [M=1024, N=512] float32
    Returns: (row_max, row_sum) each [M=1024, 1] float32
    """
    row_max = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    row_sum = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    # ── Pass 1: compute row-wise max ─────────────────────────────────────────
    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        max_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_tile, data=x_tile,
                           op=nl.maximum, axis=1, keepdims=True)

        nisa.dma_copy(dst=row_max[m * TILE_M:(m + 1) * TILE_M, 0:1],
                      src=max_tile)

    # ── Pass 2: compute row-wise sum ─────────────────────────────────────────
    # X is reloaded — same data, second read from HBM
    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        sum_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_tile, data=x_tile,
                           op=nl.add, axis=1, keepdims=True)

        nisa.dma_copy(dst=row_sum[m * TILE_M:(m + 1) * TILE_M, 0:1],
                      src=sum_tile)

    return row_max, row_sum


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    row_max, row_sum = dual_reduce_two_pass(x_dev)
    xm.mark_step()

    ref_max = x.max(dim=1, keepdim=True).values
    ref_sum = x.sum(dim=1, keepdim=True)

    row_max_cpu = row_max.cpu()
    row_sum_cpu = row_sum.cpu()
    assert torch.allclose(row_max_cpu, ref_max, atol=1e-4), (
        f"Max diff (max): {(row_max_cpu - ref_max).abs().max()}")
    assert torch.allclose(row_sum_cpu, ref_sum, atol=1e-1), (
        f"Max diff (sum): {(row_sum_cpu - ref_sum).abs().max()}")
    print("Correctness check passed.")
