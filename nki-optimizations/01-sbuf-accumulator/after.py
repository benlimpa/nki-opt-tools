"""
AFTER kernel: Column-wise partial reduction with accumulator kept in SBUF.

Optimization: Category #1 — Eliminate loads/stores, keep data in SBUF/PSUM.

KEY CHANGE: The running accumulator lives in SBUF for the entire loop.
No HBM loads/stores happen inside the loop — only a single final HBM store
after all tiles are processed.

Before: 2*(M//TILE_M) DMA transfers for the accumulator  [+14 round-trips]
After:  1 DMA store of the accumulator at the end        [hoisted out]
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024     # total rows (must be multiple of TILE_M)
N = 512      # columns
TILE_M = 128 # partition size


@nki.jit(platform_target="trn1")
def reduce_sum_sbuf_accumulator(x):
    """
    Compute y[j, k] = sum_{i=0}^{M//TILE_M - 1} x[i*TILE_M + j, k]
    for j in [0, TILE_M), k in [0, N).

    AFTER: accumulator stays in SBUF throughout the loop.
    Only one final HBM store — zero unnecessary DMA round-trips.

    x: [M, N] in HBM
    Returns: y [TILE_M, N] in HBM
    """
    y = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Hoist: initialize accumulator in SBUF once, never touch HBM inside loop ----
    acc_sbuf = nl.zeros((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)

    for i in nl.sequential_range(M // TILE_M):
        # Load input tile from HBM
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        # Add x_tile to accumulator — stays in SBUF, no HBM round-trip
        new_acc = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_acc, data1=acc_sbuf, data2=x_tile, op=nl.add)
        acc_sbuf = new_acc

    # ---- Single final store: accumulator → HBM ----
    nisa.dma_copy(dst=y[0:TILE_M, 0:N], src=acc_sbuf)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = reduce_sum_sbuf_accumulator(x)
    xm.mark_step()

    # Verify correctness
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = torch.zeros(TILE_M, N, dtype=torch.float32)
    for i in range(M // TILE_M):
        ref += x_cpu[i * TILE_M:(i + 1) * TILE_M, :]
    assert torch.allclose(y_cpu, ref, atol=1e-4), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
