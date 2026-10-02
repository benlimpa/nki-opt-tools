"""
BEFORE kernel: Column-wise partial reduction with HBM round-trips for accumulator.

Optimization target: Category #1 — Eliminate loads/stores, keep data in SBUF/PSUM.

This kernel sums [M, N] into [TILE_M, N] by accumulating TILE_M-row tiles.
The BEFORE version stores the running accumulator to HBM after each tile,
then reloads it at the start of the next iteration — generating
2*(M//TILE_M - 1) = 14 extra DMA round-trips of 128*512*4 = 256 KB each.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024     # total rows (must be multiple of TILE_M)
N = 512      # columns
TILE_M = 128 # partition size


@nki.jit(platform_target="trn1")
def reduce_sum_hbm_accumulator(x):
    """
    Compute y[j, k] = sum_{i=0}^{M//TILE_M - 1} x[i*TILE_M + j, k]
    for j in [0, TILE_M), k in [0, N).

    BEFORE: running accumulator is stored to HBM and reloaded each iteration.
    This causes (M // TILE_M - 1) * 2 = 14 unnecessary DMA transfers.

    x: [M, N] in HBM
    Returns: y [TILE_M, N] in HBM
    """
    y = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Initialize accumulator in HBM to zeros
    acc_hbm = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.private_hbm)
    acc_init = nl.zeros((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=acc_hbm[0:TILE_M, 0:N], src=acc_init)

    for i in nl.sequential_range(M // TILE_M):
        # Load input tile from HBM
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        # UNNECESSARY ROUND-TRIP: load accumulator from HBM back into SBUF
        acc_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=acc_sbuf, src=acc_hbm[0:TILE_M, 0:N])

        # Add x_tile to running accumulator
        new_acc = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_acc, data1=acc_sbuf, data2=x_tile, op=nl.add)

        # UNNECESSARY ROUND-TRIP: store updated accumulator back to HBM
        nisa.dma_copy(dst=acc_hbm[0:TILE_M, 0:N], src=new_acc)

    # Copy final accumulator to output
    acc_final = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=acc_final, src=acc_hbm[0:TILE_M, 0:N])
    nisa.dma_copy(dst=y[0:TILE_M, 0:N], src=acc_final)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = reduce_sum_hbm_accumulator(x)
    xm.mark_step()

    # Verify correctness
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    # Expected: sum each TILE_M-row group across the M dimension
    ref = torch.zeros(TILE_M, N, dtype=torch.float32)
    for i in range(M // TILE_M):
        ref += x_cpu[i * TILE_M:(i + 1) * TILE_M, :]
    assert torch.allclose(y_cpu, ref, atol=1e-4), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
