"""
Sweep variant — opt35 BEFORE (manual broadcast loop, scale + bias).
P=128, N=512, OPS=2, NUM_ITERS=50.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N = 512
OPS = 2
NUM_ITERS = 50


@nki.jit(platform_target="trn1")
def tanh_baseline(X, scale, bias):
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    b_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=bias)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        tmp1 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp1[:, j:j+1], data1=x_tile[:, j:j+1],
                               data2=s_tile, op=nl.multiply)

        tmp2 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp2[:, j:j+1], data1=tmp1[:, j:j+1],
                               data2=b_tile, op=nl.add)

        nisa.activation(dst=res_tile, data=tmp2, op=nl.tanh)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
