"""
Sweep variant — opt10 AFTER: softmax with delayed division (no exp_hbm).
P=128, N_total=1024, TILE_N=128 → n_tiles=8.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N_total = 1024
TILE_N = 128


@nki.jit(platform_target="trn1")
def softmax_no_intermediate(x):
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    max_val = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=max_val, value=-float("inf"))

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=x_tile, axis=1, keepdims=True)

        new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=max_val, data2=tile_max, op=nl.maximum)
        max_val = new_max

    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=max_val, scale=-1.0)

    exp_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=exp_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        # No store; exp_tile discarded.
        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile, axis=1, keepdims=True)

        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=exp_sum, data2=tile_sum, op=nl.add)
        exp_sum = new_sum

    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=exp_sum)

    for n in nl.affine_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        out_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, op=nl.copy, data=exp_tile, scale=inv_sum)

        nisa.dma_copy(dst=y[0:P, n * TILE_N:(n + 1) * TILE_N], src=out_tile)

    return y
