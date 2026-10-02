"""
AFTER kernel: Only process tiles that contain valid (non-padding) data.

Optimization target: Category #6 — Skip computation when masked out.

What changed vs before.py:
- Loop bound changed from TOTAL_TILES (8) to VALID_TILES (4).
- The 4 padding tiles (i >= VALID_TILES) are skipped entirely — no load,
  no ScalarE, no store for those tiles.
- Eliminates 4 × 3 = 12 DMA transactions and 1 MB of avoidable HBM traffic.
- The valid_tiles count is a compile-time constant (Python int), so the
  compiler emits a loop with exactly 4 iterations rather than 8.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128
TOTAL_TILES = M // TILE_P   # 8 tiles total (padded length)
VALID_TILES = 4              # Only first 4 tiles contain real data


@nki.jit(platform_target="trn1")
def scale_valid_tiles(x):
    """
    Scale only the VALID_TILES=4 real rows by 2.0 — skip all padding tiles.

    AFTER: loop bound is VALID_TILES instead of TOTAL_TILES.
    Padding tiles (i >= VALID_TILES) are never loaded, computed, or stored.
    DMA traffic drops from 8 × (load + store) to 4 × (load + store).

    x: [M, N] in HBM (first VALID_TILES*TILE_P rows are real; rest is padding)
    Returns: y [M, N] in HBM (only first VALID_TILES*TILE_P rows are written)
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # OPTIMIZED: only iterate over the 4 valid tiles — padding tiles are skipped
    for i in nl.sequential_range(VALID_TILES):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result, data=x_tile, op0=nl.multiply, operand0=2.0)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = scale_valid_tiles(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    # Only validate the valid portion (padding region is unwritten)
    ref = x_cpu[:VALID_TILES * TILE_P] * 2.0
    assert torch.allclose(y_cpu[:VALID_TILES * TILE_P], ref, atol=1e-5), \
        f"Max diff: {(y_cpu[:VALID_TILES * TILE_P] - ref).abs().max()}"
    print("Correctness check passed.")
