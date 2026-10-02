"""
Sweep variant — opt35 AFTER (fused nisa.activation, scale-only).
P=128, N=256, OPS=1, NUM_ITERS=50.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N = 256
OPS = 1
NUM_ITERS = 50


@nki.jit(platform_target="trn1")
def tanh_specialized(X, scale, bias):
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.activation(dst=res_tile, op=nl.tanh, data=x_tile, scale=s_tile)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
