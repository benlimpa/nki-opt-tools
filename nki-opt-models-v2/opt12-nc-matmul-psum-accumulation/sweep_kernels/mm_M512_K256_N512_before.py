"""
Sweep variant — opt12 BEFORE: K-loop is sequential_range with SBUF accumulator.
M=512, K=256, N=512, TILE_M=128, TILE_K=128.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 512
K = 256
N = 512
TILE_M = 128
TILE_K = 128
TILE_N = 512


@nki.jit(platform_target="trn1")
def matmul_sbuf_accumulate(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)

        for k in nl.sequential_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)

            new_acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=new_acc, data1=acc, data2=res_sbuf, op=nl.add)
            acc = new_acc

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N], src=acc)

    return result
