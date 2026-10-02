"""
BEFORE kernel: Process all tiles regardless of how many contain valid data.

Optimization target: Category #6 — Skip computation when masked out.

This kernel scales a padded tensor (M=1024 rows) by 2.0, but only the first
VALID_TILES * TILE_P rows contain real data — the remaining rows are padding.
The BEFORE version iterates over all M // TILE_P = 8 tiles unconditionally,
performing unnecessary DMA loads + ScalarE compute + DMA stores for 4 padding
tiles that produce output nobody will use.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128
TOTAL_TILES = M // TILE_P   # 8 tiles
VALID_TILES = 4              # Only first 4 tiles contain real data; rest is padding


@nki.jit(platform_target="trn1")
def scale_all_tiles(x):
    """
    Scale all M rows by 2.0 — even the 4 padding tiles beyond VALID_TILES.

    BEFORE: iterates over all TOTAL_TILES=8 tile iterations unconditionally.
    Padding tiles (i >= VALID_TILES) incur 1 load + 1 ScalarE + 1 store each,
    wasting 4 × (TILE_P × N × dtype_size) = 1 MB of avoidable DMA traffic.

    x: [M, N] in HBM (first VALID_TILES*TILE_P rows are real; rest is padding)
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # WASTEFUL: iterates over all 8 tiles, including 4 padding tiles
    for i in nl.sequential_range(TOTAL_TILES):
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
    y = scale_all_tiles(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    # Only validate the valid portion
    ref = x_cpu[:VALID_TILES * TILE_P] * 2.0
    assert torch.allclose(y_cpu[:VALID_TILES * TILE_P], ref, atol=1e-5), \
        f"Max diff: {(y_cpu[:VALID_TILES * TILE_P] - ref).abs().max()}"
    print("Correctness check passed.")
