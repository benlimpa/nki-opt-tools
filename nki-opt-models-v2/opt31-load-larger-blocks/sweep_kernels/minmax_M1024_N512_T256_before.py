"""
Sweep variant — opt31 BEFORE: small TILE_N min-max norm, two-pass with reload.
M=1024, N=512, TILE_P=128, TILE_N=256.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128
TILE_N_BEFORE = 256
TILE_N_AFTER = 512
EPS = 1e-8


@nki.jit(platform_target="trn1")
def minmax_norm(x):
    TILE_N = TILE_N_BEFORE
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):
        x0 = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x0,
                      src=x[m * TILE_P:(m + 1) * TILE_P, 0:TILE_N])

        min_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=min_val, data=x0,
                           op=nl.minimum, axis=1, keepdims=True)
        max_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x0,
                           op=nl.maximum, axis=1, keepdims=True)

        for n in nl.sequential_range(N // TILE_N - 1):
            x_tile = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile,
                          src=x[m * TILE_P:(m + 1) * TILE_P,
                                (n + 1) * TILE_N:(n + 2) * TILE_N])

            tile_min = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_min, data=x_tile,
                               op=nl.minimum, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=min_val, data1=min_val, data2=tile_min,
                               op=nl.minimum)

            tile_max = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=x_tile,
                               op=nl.maximum, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=max_val, data1=max_val, data2=tile_max,
                               op=nl.maximum)

        range_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_val, data=max_val,
                           op0=nl.subtract, operand0=min_val)
        range_eps = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_eps, data=range_val,
                           op0=nl.add, operand0=EPS)
        inv_range = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_range, data=range_eps, op=nl.reciprocal)

        for n in nl.affine_range(N // TILE_N):
            x_tile2 = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile2,
                          src=x[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_N:(n + 1) * TILE_N])

            x_shifted = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_shifted, data=x_tile2,
                               op0=nl.subtract, operand0=min_val)

            y_tile = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y_tile, data=x_shifted,
                               op0=nl.multiply, operand0=inv_range)

            nisa.dma_copy(dst=y[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_N:(n + 1) * TILE_N],
                          src=y_tile)

    return y
