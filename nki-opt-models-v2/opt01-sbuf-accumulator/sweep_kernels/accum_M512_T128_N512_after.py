"""
Sweep variant — opt01 AFTER: column-wise reduction with SBUF accumulator.
M=512, TILE_M=128, N=512 → n_tiles = M / TILE_M = 4.
Single final store; zero unnecessary accumulator round-trips.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 512
TILE_M = 128
N = 512


@nki.jit(platform_target="trn1")
def reduce_sum_sbuf_accumulator(x):
    y = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    acc_sbuf = nl.zeros((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)

    for i in nl.sequential_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        new_acc = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_acc, data1=acc_sbuf, data2=x_tile, op=nl.add)
        acc_sbuf = new_acc

    nisa.dma_copy(dst=y[0:TILE_M, 0:N], src=acc_sbuf)
    return y
