"""
Sweep variant — opt38 AFTER: element-wise A+B with SPMD launch grid G=2.
M=2048, N=1024, n_iters = M / TILE_P = 16; launched as kernel[2](args).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 2048
N = 1024
TILE_P = 128
GRID_SIZE = 2


@nki.jit(platform_target="trn1")
def add_spmd(a, b):
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)

    n_iters = M // TILE_P
    iters_per_prg = n_iters // n_prgs
    start = prg_id * iters_per_prg
    end = start + iters_per_prg

    for i in nl.sequential_range(start, end):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c


def wrapper(a, b):
    return add_spmd[GRID_SIZE](a, b)
