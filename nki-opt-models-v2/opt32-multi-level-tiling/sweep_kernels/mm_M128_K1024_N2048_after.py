"""
Sweep variant — opt32 AFTER: K-outer, N-inner. A loaded once per K-tile.
M=128, K=1024, N=2048, TILE_M=128, TILE_N=128, TILE_K=128.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 128
K = 1024
N = 2048
TILE_M = 128
TILE_N = 128
TILE_K = 128


@nki.jit(platform_target="trn1")
def matmul_multi_level_tiling(lhsT, rhs):
    C = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    zero_sbuf = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

    for n_idx in nl.affine_range(N // TILE_N):
        nisa.activation(dst=res_psum[:, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                        data=zero_sbuf, op=nl.copy)

    for k_idx in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k_idx * TILE_K:(k_idx + 1) * TILE_K, 0:TILE_M])

        for n_idx in nl.affine_range(N // TILE_N):
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k_idx * TILE_K:(k_idx + 1) * TILE_K,
                                  n_idx * TILE_N:(n_idx + 1) * TILE_N])

            nisa.nc_matmul(dst=res_psum[:, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                           stationary=lhsT_tile, moving=rhs_tile)

    for n_idx in nl.affine_range(N // TILE_N):
        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy,
                        data=res_psum[:, n_idx * TILE_N:(n_idx + 1) * TILE_N])
        nisa.dma_copy(dst=C[0:TILE_M, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                      src=res_sbuf)

    return C
