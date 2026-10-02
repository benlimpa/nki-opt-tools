"""
Sweep variant — opt07 BEFORE: 3-loop pipeline,
M=1024, N=1024, TILE_P=128, n_intermediates=2.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 1024
TILE_P = 128
N_OPS = 3

@nki.jit(platform_target="trn1")
def pipeline_separate(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    inter0 = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)
    inter1 = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)

    # Pass 0: nl.multiply 2.0
    for i in nl.affine_range(M // TILE_P):
        t_in = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_in, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])
        t_out = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t_out, data=t_in, op0=nl.multiply, operand0=2.0)
        nisa.dma_copy(dst=inter0[i * TILE_P:(i + 1) * TILE_P, 0:N], src=t_out)

    # Pass 1: nl.add 1.0
    for i in nl.affine_range(M // TILE_P):
        t_in = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_in, src=inter0[i * TILE_P:(i + 1) * TILE_P, 0:N])
        t_out = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t_out, data=t_in, op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=inter1[i * TILE_P:(i + 1) * TILE_P, 0:N], src=t_out)

    # Pass 2: nl.multiply 1.0
    for i in nl.affine_range(M // TILE_P):
        t_in = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t_in, src=inter1[i * TILE_P:(i + 1) * TILE_P, 0:N])
        t_out = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t_out, data=t_in, op0=nl.multiply, operand0=1.0)
        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=t_out)

    return y
