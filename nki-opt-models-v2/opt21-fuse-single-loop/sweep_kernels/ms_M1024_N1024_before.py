"""
Sweep variant — opt21 BEFORE: two affine_range loops with private_hbm
intermediate (row means).
M=1024, N=1024, TILE_M=128.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 1024
TILE_M = 128


@nki.jit(platform_target="trn1")
def mean_subtract_two_loops(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    means_hbm = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # Loop 1: compute per-row mean, write to HBM
    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        row_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        row_mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_mean, data=row_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        nisa.dma_copy(dst=means_hbm[m * TILE_M:(m + 1) * TILE_M, 0:1],
                      src=row_mean)

    # Loop 2: reload x and means, subtract
    for m in nl.affine_range(M // TILE_M):
        x_tile2 = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile2, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        mean_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mean_tile,
                      src=means_hbm[m * TILE_M:(m + 1) * TILE_M, 0:1])

        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile2,
                           op0=nl.subtract, operand0=mean_tile)

        nisa.dma_copy(dst=y[m * TILE_M:(m + 1) * TILE_M, 0:N], src=y_tile)

    return y
