"""
AFTER kernel: double-buffered matmul — DMA prefetch overlaps with TensorE.

Optimization: Category #14 — Double buffering

KEY CHANGE: Two alternating SBUF slots hold consecutive M-tiles. Within each
sequential_range iteration, the DMA load for tile m+1 is issued into slot
(m+1)%2 BEFORE nc_matmul reads slot m%2. Because the two slots are at
different SBUF addresses, DMA and TensorE execute concurrently:

  Prologue: [DMA tile0 → slot0]
  Iter 0:   [DMA tile1 → slot1] + [nc_matmul(slot0)]  ← overlap!
  Iter 1:   [DMA tile2 → slot0] + [nc_matmul(slot1)]  ← overlap!
  ...
  Epilogue: [nc_matmul(slot(NUM_TILES-1)%2)]

The ndarray shape (TILE_K, 2, TILE_M) encodes the two slots as a free
dimension of size 2. Slot s occupies free offsets [s*TILE_M, (s+1)*TILE_M)
within each partition, placing the two slots at separate SBUF addresses.

Before: DMA→TensorE serial per M-tile (DMA stalls TensorE)
After:  DMA for tile m+1 runs in parallel with TensorE on tile m
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
K = 128
N = 512

TILE_M = 128
TILE_K = 128
TILE_N = 512


@nki.jit(platform_target="trn1")
def matmul_double_buffer(lhsT, rhs):
    """
    Compute C = lhsT.T @ rhs using double-buffered SBUF for M-tiles.

    AFTER: DMA prefetch of tile m+1 overlaps with nc_matmul on tile m.

    lhsT: [K=128, M=1024] float32
    rhs:  [K=128, N=512]  float32
    Returns: C [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load rhs once
    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    NUM_TILES = M // TILE_M  # 8

    # Double buffer: two SBUF slots for lhsT M-tiles.
    # Shape (TILE_K, 2, TILE_M): partition=128, slot_idx=2, free=128.
    # slot 0 → free offsets [0,       TILE_M)
    # slot 1 → free offsets [TILE_M, 2*TILE_M)
    # → different SBUF addresses, enabling DMA/TensorE overlap.
    lhsT_bufs = nl.ndarray((TILE_K, 2, TILE_M), dtype=nl.float32, buffer=nl.sbuf)

    # Prologue: pre-fetch tile 0 into slot 0
    nisa.dma_copy(dst=lhsT_bufs[0:TILE_K, 0, 0:TILE_M],
                  src=lhsT[0:TILE_K, 0:TILE_M])

    # Main loop: tiles 0 to NUM_TILES-2
    for m in nl.sequential_range(NUM_TILES - 1):
        # DMA: prefetch tile (m+1) into slot (m+1)%2.
        # No dependency on slot m%2 → runs concurrently with nc_matmul below.
        nisa.dma_copy(dst=lhsT_bufs[0:TILE_K, (m + 1) % 2, 0:TILE_M],
                      src=lhsT[0:TILE_K, (m + 1) * TILE_M:(m + 2) * TILE_M])

        # TensorE: compute on tile m from slot m%2.
        # No dependency on the DMA above (different slot) → can overlap.
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum,
                       stationary=lhsT_bufs[0:TILE_K, m % 2, 0:TILE_M],
                       moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=res_sbuf)

    # Epilogue: compute last tile (already in slot (NUM_TILES-1)%2)
    last = NUM_TILES - 1
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=lhsT_bufs[0:TILE_K, last % 2, 0:TILE_M],
                   moving=rhs_tile)
    res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
    nisa.dma_copy(dst=result[last * TILE_M:(last + 1) * TILE_M, 0:TILE_N],
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

    C = matmul_double_buffer(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
