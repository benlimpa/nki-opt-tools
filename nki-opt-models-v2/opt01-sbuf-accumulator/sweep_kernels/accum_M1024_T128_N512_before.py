"""
Sweep variant — opt01 BEFORE: column-wise reduction with HBM accumulator.
M=1024, TILE_M=128, N=512 → n_tiles = M / TILE_M = 8.
Generates 16 unnecessary accumulator round-trips of [TILE_M, N] tile.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
TILE_M = 128
N = 512


@nki.jit(platform_target="trn1")
def reduce_sum_hbm_accumulator(x):
    y = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    acc_hbm = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.private_hbm)
    acc_init = nl.zeros((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=acc_hbm[0:TILE_M, 0:N], src=acc_init)

    for i in nl.sequential_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        acc_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=acc_sbuf, src=acc_hbm[0:TILE_M, 0:N])

        new_acc = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_acc, data1=acc_sbuf, data2=x_tile, op=nl.add)

        nisa.dma_copy(dst=acc_hbm[0:TILE_M, 0:N], src=new_acc)

    acc_final = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=acc_final, src=acc_hbm[0:TILE_M, 0:N])
    nisa.dma_copy(dst=y[0:TILE_M, 0:N], src=acc_final)

    return y
