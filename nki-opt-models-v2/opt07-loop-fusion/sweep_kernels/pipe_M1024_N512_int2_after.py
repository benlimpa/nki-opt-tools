"""
Sweep variant — opt07 AFTER: 3-loop pipeline,
M=1024, N=512, TILE_P=128, n_intermediates=2.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128
N_OPS = 3

@nki.jit(platform_target="trn1")
def pipeline_fused(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.affine_range(M // TILE_P):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])
        cur = x_tile
        # Op 0: nl.multiply 2.0
        nxt_0 = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nxt_0, data=cur, op0=nl.multiply, operand0=2.0)
        cur = nxt_0
        # Op 1: nl.add 1.0
        nxt_1 = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nxt_1, data=cur, op0=nl.add, operand0=1.0)
        cur = nxt_1
        # Op 2: nl.multiply 1.0
        nxt_2 = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nxt_2, data=cur, op0=nl.multiply, operand0=1.0)
        cur = nxt_2
        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=cur)

    return y
