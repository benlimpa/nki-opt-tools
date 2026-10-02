"""
AFTER kernel: compute row-wise max AND row-wise sum in a single pass —
X is read from HBM once, both reductions execute while the tile is in SBUF.

Optimization: Category #19 — Reorder operations to improve locality

KEY CHANGE: The two independent reductions (max, sum) are reordered from
separate loops into a single affine_range loop. Each x_tile is loaded once
and both nisa.tensor_reduce calls execute on it before it is evicted from SBUF.

  Single pass: load x_tile → reduce(max) → reduce(sum) → store max_tile, sum_tile

Total HBM reads: 1 × 2 MB = 2 MB (X read once).

This works because max and sum are independent — neither depends on the other's
result. Reordering brings the two operations close to the same data access,
eliminating the second HBM read of X.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_M = 128


@nki.jit(platform_target="trn1")
def dual_reduce_one_pass(x):
    """
    Compute row-wise max and sum of X in a single pass.

    AFTER: X read once from HBM (2 MB total reads).
    Both reductions execute on the same SBUF tile — no second HBM load.

    x: [M=1024, N=512] float32
    Returns: (row_max, row_sum) each [M=1024, 1] float32
    """
    row_max = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    row_sum = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    # Single pass: both reductions on the same tile while it is in SBUF
    for m in nl.affine_range(M // TILE_M):
        # Load X once — used for both reductions
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Reduction 1: max — x_tile in SBUF, no reload
        max_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_tile, data=x_tile,
                           op=nl.maximum, axis=1, keepdims=True)

        # Reduction 2: sum — x_tile still in SBUF, no reload
        sum_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_tile, data=x_tile,
                           op=nl.add, axis=1, keepdims=True)

        nisa.dma_copy(dst=row_max[m * TILE_M:(m + 1) * TILE_M, 0:1],
                      src=max_tile)
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

    row_max, row_sum = dual_reduce_one_pass(x_dev)
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
