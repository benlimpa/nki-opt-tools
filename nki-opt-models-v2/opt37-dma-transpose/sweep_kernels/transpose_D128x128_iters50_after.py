"""
Sweep variant — opt37 AFTER: dma_transpose (DMA Engine) in a loop.
D0=128, D1=1, D2=1, D3=128, NUM_ITERS=50.
"""
import nki
import nki.isa as nisa
import nki.language as nl

D0 = 128
D1 = 1
D2 = 1
D3 = 128
NUM_ITERS = 50


@nki.jit(platform_target="trn1")
def transpose_specialized(X):
    Y = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.shared_hbm)
    y_tile = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.dma_transpose(dst=y_tile, src=X)

    nisa.dma_copy(dst=Y, src=y_tile)
    return Y
