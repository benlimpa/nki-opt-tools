"""
Sweep variant — opt14 AFTER: double-buffered matmul.
M=1024, K=128, N=512, TILE_M=128.
Two alternating SBUF slots; DMA(tile m+1) overlaps nc_matmul(tile m).
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
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    NUM_TILES = M // TILE_M

    lhsT_bufs = nl.ndarray((TILE_K, 2, TILE_M), dtype=nl.float32, buffer=nl.sbuf)

    nisa.dma_copy(dst=lhsT_bufs[0:TILE_K, 0, 0:TILE_M],
                  src=lhsT[0:TILE_K, 0:TILE_M])

    for m in nl.sequential_range(NUM_TILES - 1):
        nisa.dma_copy(dst=lhsT_bufs[0:TILE_K, (m + 1) % 2, 0:TILE_M],
                      src=lhsT[0:TILE_K, (m + 1) * TILE_M:(m + 2) * TILE_M])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum,
                       stationary=lhsT_bufs[0:TILE_K, m % 2, 0:TILE_M],
                       moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=res_sbuf)

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
