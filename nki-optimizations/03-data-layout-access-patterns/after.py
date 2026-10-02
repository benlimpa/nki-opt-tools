"""
AFTER kernel: Element-wise matrix addition with full-partition tiles.

Optimization target: Category #3 — Improve data layout and access patterns.

What changed vs before.py:
- TILE_P increased from 32 to 128 (the NeuronCore partition count).
- All 128 partitions are active per operation instead of 32.
- Loop iterations reduced from M//32 = 32 to M//128 = 8.
- DMA transactions reduced from 3*32 = 96 to 3*8 = 24.
- Total bytes transferred are identical — only the access granularity changes.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024       # total rows
N = 512        # columns
TILE_P = 128   # FULL: all 128 partitions used per op → 8 iterations


@nki.jit(platform_target="trn1")
def add_full_tiles(a, b):
    """
    Compute c[i*TILE_P:(i+1)*TILE_P, :] = a[...] + b[...] for each tile i.

    AFTER: TILE_P=128 means all 128 partitions active per op, M//128 = 8 iterations,
    and 3*8 = 24 DMA transactions total (vs 96 in the before kernel).

    a: [M, N] in HBM
    b: [M, N] in HBM
    Returns: c [M, N] in HBM
    """
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(M // TILE_P):
        # Load tile from A — full 128 partitions
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        # Load tile from B — full 128 partitions
        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        # Add: all 128 partitions active
        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        # Store result tile
        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    a = torch.randn(M, N, dtype=torch.float32).to(device)
    b = torch.randn(M, N, dtype=torch.float32).to(device)
    c = add_full_tiles(a, b)
    xm.mark_step()

    # Verify correctness
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_cpu = c.cpu()
    ref = a_cpu + b_cpu
    assert torch.allclose(c_cpu, ref, atol=1e-5), f"Max diff: {(c_cpu - ref).abs().max()}"
    print("Correctness check passed.")
