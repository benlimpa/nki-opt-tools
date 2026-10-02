"""
Sweep variant — opt38 BEFORE: element-wise A+B without SPMD launch grid.
M=8192, N=512, n_iters = M / TILE_P = 64; launched as kernel(args).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 8192
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def add_no_spmd(a, b):
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    n_iters = M // TILE_P
    for i in nl.sequential_range(n_iters):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c


def wrapper(a, b):
    return add_no_spmd(a, b)
