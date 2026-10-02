"""
Sweep variant — opt16 AFTER: 3-stage pipeline (input + output ping-pong).
M=512, K=128, N=512, TILE_M=128, dtype=fp32.
Requires M / TILE_M >= 3 (prologue+epilogue fill the pipeline).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 512
K = 128
N = 512
TILE_M = 128
TILE_K = 128
TILE_N = 512
DTYPE = "fp32"
_NL_DTYPE = nl.float32


@nki.jit(platform_target="trn1")
def matmul_pipelined(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=_NL_DTYPE, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=_NL_DTYPE, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    NUM_TILES = M // TILE_M

    input_bufs = nl.ndarray((TILE_K, 2, TILE_M), dtype=_NL_DTYPE, buffer=nl.sbuf)
    output_bufs = nl.ndarray((TILE_M, 2, TILE_N), dtype=_NL_DTYPE, buffer=nl.sbuf)

    # Prologue
    nisa.dma_copy(dst=input_bufs[0:TILE_K, 0, 0:TILE_M],
                  src=lhsT[0:TILE_K, 0:TILE_M])
    nisa.dma_copy(dst=input_bufs[0:TILE_K, 1, 0:TILE_M],
                  src=lhsT[0:TILE_K, TILE_M:2 * TILE_M])
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=input_bufs[0:TILE_K, 0, 0:TILE_M],
                   moving=rhs_tile)
    nisa.activation(dst=output_bufs[0:TILE_M, 0, 0:TILE_N],
                    op=nl.copy, data=res_psum)

    # Steady state
    for m in nl.sequential_range(1, NUM_TILES - 1):
        nisa.dma_copy(dst=input_bufs[0:TILE_K, (m + 1) % 2, 0:TILE_M],
                      src=lhsT[0:TILE_K, (m + 1) * TILE_M:(m + 2) * TILE_M])
        nisa.dma_copy(dst=result[(m - 1) * TILE_M:m * TILE_M, 0:TILE_N],
                      src=output_bufs[0:TILE_M, (m - 1) % 2, 0:TILE_N])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum,
                       stationary=input_bufs[0:TILE_K, m % 2, 0:TILE_M],
                       moving=rhs_tile)
        nisa.activation(dst=output_bufs[0:TILE_M, m % 2, 0:TILE_N],
                        op=nl.copy, data=res_psum)

    # Epilogue
    last = NUM_TILES - 1
    nisa.dma_copy(dst=result[(last - 1) * TILE_M:last * TILE_M, 0:TILE_N],
                  src=output_bufs[0:TILE_M, (last - 1) % 2, 0:TILE_N])
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=input_bufs[0:TILE_K, last % 2, 0:TILE_M],
                   moving=rhs_tile)
    nisa.activation(dst=output_bufs[0:TILE_M, last % 2, 0:TILE_N],
                    op=nl.copy, data=res_psum)
    nisa.dma_copy(dst=result[last * TILE_M:(last + 1) * TILE_M, 0:TILE_N],
                  src=output_bufs[0:TILE_M, last % 2, 0:TILE_N])

    return result
