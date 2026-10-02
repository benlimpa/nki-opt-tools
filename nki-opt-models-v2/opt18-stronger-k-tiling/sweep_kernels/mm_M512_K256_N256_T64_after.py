"""
Sweep variant — opt18 AFTER: matmul with TILE_K=128.
M=512, K=256, N=256, TILE_M=128, TILE_K=128.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 512
K = 256
N = 256
TILE_M = 128
TILE_K_BEFORE = 64
TILE_K_AFTER = 128
TILE_K = 128


@nki.jit(platform_target="trn1")
def matmul(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                   m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:N])

            nisa.nc_matmul(dst=res_psum,
                           stationary=lhsT_tile,
                           moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:N],
                      src=res_sbuf)

    return result
