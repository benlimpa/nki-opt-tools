"""
Sweep variant — opt31 AFTER: full free-dim block, single-pass minmax norm.
M=1024, N=512, TILE_P=128, TILE_N=512 (= N).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128
TILE_N_BEFORE = 128
TILE_N_AFTER = 512
EPS = 1e-8


@nki.jit(platform_target="trn1")
def minmax_norm(x):
    TILE_N = TILE_N_AFTER
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):
        x_block = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_block,
                      src=x[m * TILE_P:(m + 1) * TILE_P, 0:N])

        min_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=min_val, data=x_block,
                           op=nl.minimum, axis=1, keepdims=True)
        max_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x_block,
                           op=nl.maximum, axis=1, keepdims=True)

        range_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_val, data=max_val,
                           op0=nl.subtract, operand0=min_val)
        range_eps = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_eps, data=range_val,
                           op0=nl.add, operand0=EPS)
        inv_range = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_range, data=range_eps, op=nl.reciprocal)

        x_shifted = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_shifted, data=x_block,
                           op0=nl.subtract, operand0=min_val)
        y_block = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_block, data=x_shifted,
                           op0=nl.multiply, operand0=inv_range)

        nisa.dma_copy(dst=y[m * TILE_P:(m + 1) * TILE_P, 0:N],
                      src=y_block)

    return y
