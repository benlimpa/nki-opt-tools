"""
Sweep variant — opt06 BEFORE: scale all tiles incl. padding.
M=1024, N=512, TILE_P=128, VALID_TILES=2
total_tiles = M // TILE_P = 8 (loop iterates all of them).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128
TOTAL_TILES = 8
VALID_TILES = 2


@nki.jit(platform_target="trn1")
def scale_all_tiles(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(TOTAL_TILES):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result, data=x_tile, op0=nl.multiply, operand0=2.0)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y
