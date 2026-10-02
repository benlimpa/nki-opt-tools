"""
Sweep variant — opt11 BEFORE: nc_matmul with small TILE_N=256.
M=1024, K=128, N=512 → 2 nc_matmul calls per M-tile.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
K = 128
N = 512
TILE_M = 128
TILE_K = 128
TILE_N = 256


@nki.jit(platform_target="trn1")
def matmul_small_n_tiles(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        for n in nl.affine_range(N // TILE_N):
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[0:TILE_K, n * TILE_N:(n + 1) * TILE_N])

            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
            nisa.dma_copy(
                dst=result[m * TILE_M:(m + 1) * TILE_M,
                            n * TILE_N:(n + 1) * TILE_N],
                src=res_sbuf)

    return result
