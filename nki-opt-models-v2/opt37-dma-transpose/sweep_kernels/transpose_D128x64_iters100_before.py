"""
Sweep variant — opt37 BEFORE: nc_transpose (TensorE) in a loop.
D0=128, D1=1, D2=1, D3=64, NUM_ITERS=100.
"""
import nki
import nki.isa as nisa
import nki.language as nl

D0 = 128
D1 = 1
D2 = 1
D3 = 64
NUM_ITERS = 100


@nki.jit(platform_target="trn1")
def transpose_baseline(X):
    Y = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((D0, D3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X[:, 0, 0, :])

    y_psum = nl.zeros((D3, D0), dtype=nl.float32, buffer=nl.psum)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.tensor_scalar(dst=y_psum, data=y_psum, op0=nl.multiply, operand0=0.0)
        nisa.nc_transpose(dst=y_psum, data=x_tile)

    y_sbuf = nl.ndarray((D3, D0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=y_sbuf, op=nl.copy, data=y_psum)
    nisa.dma_copy(dst=Y[:, 0, 0, :], src=y_sbuf)
    return Y
