"""
Sweep variant — opt35 BEFORE (manual broadcast loop, scale-only).
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
def tanh_baseline(X, scale, bias):
    # bias is unused in OPS=1 variant but kept in the signature for oracle parity
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        tmp1 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp1[:, j:j+1], data1=x_tile[:, j:j+1],
                               data2=s_tile, op=nl.multiply)

        nisa.activation(dst=res_tile, data=tmp1, op=nl.tanh)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
