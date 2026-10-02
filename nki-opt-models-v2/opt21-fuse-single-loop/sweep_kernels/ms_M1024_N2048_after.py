"""
Sweep variant — opt21 AFTER: single fused affine_range loop, no
private_hbm intermediate.
M=1024, N=2048, TILE_M=128.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 2048
TILE_M = 128


@nki.jit(platform_target="trn1")
def mean_subtract_fused(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        row_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        row_mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_mean, data=row_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile,
                           op0=nl.subtract, operand0=row_mean)

        nisa.dma_copy(dst=y[m * TILE_M:(m + 1) * TILE_M, 0:N], src=y_tile)

    return y
