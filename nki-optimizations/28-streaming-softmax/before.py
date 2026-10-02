"""
BEFORE kernel: Row-wise softmax using the standard 2-pass algorithm.

Optimization target: Category #28 — Streaming softmax with running max and
scaling trick.

This kernel computes softmax(x) over the free (N) dimension for each of P=128
rows, where N_total=2048 and TILE_N=128 (16 tiles per row).

The BEFORE version uses the classical 2-pass approach:
  Pass 1 — max scan:  read all 16 tiles of x, find per-row maximum
  Pass 2 — exp-sum:   read all 16 tiles of x again, compute exp(x - max),
                       accumulate per-row sum
  Pass 3 — normalize: read all 16 tiles of x a third time, recompute exp and
                       multiply by inv_sum, write output

HBM traffic breakdown (before):
  Pass 1 — read x:  16 × [128, 128] × 4 B = 1 MB
  Pass 2 — read x:  16 × [128, 128] × 4 B = 1 MB
  Pass 3 — read x:  16 × [128, 128] × 4 B = 1 MB
           write y: 16 × [128, 128] × 4 B = 1 MB
  Total reads: 3 MB   writes: 1 MB   grand total: 4 MB

The first two passes are both sequential (max and sum accumulate across tiles),
so the compiler cannot fuse them — they require two separate sequential loops.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128        # partition dimension (NeuronCore partitions, <= 128)
N_total = 2048  # free dimension (columns)
TILE_N = 128   # tile width; 16 tiles per row


@nki.jit(platform_target="trn1")
def softmax_2pass(x):
    """
    Row-wise softmax over the N dimension, standard 2-pass algorithm.

    BEFORE: 3 passes over x (max scan + exp-sum scan + normalize pass).
    x:    [P, N_total] in HBM
    Returns: y [P, N_total] in HBM
    """
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Pass 1: row-wise max across all tiles ----
    max_val = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=max_val, value=-float('inf'))

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=x_tile,
                           axis=1, keepdims=True)

        new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=max_val, data2=tile_max,
                           op=nl.maximum)
        max_val = new_max

    # neg_max used as bias in activation: exp(x + neg_max) = exp(x - max)
    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=max_val, scale=-1.0)

    # ---- Pass 2: read x again, compute exp(x - max), accumulate sum ----
    # (x is read a second time — separate sequential pass required because
    #  max was not known until pass 1 completed)
    exp_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=exp_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile,
                           axis=1, keepdims=True)

        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=exp_sum, data2=tile_sum,
                           op=nl.add)
        exp_sum = new_sum

    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=exp_sum)

    # ---- Pass 3: read x a third time, recompute exp, divide, store output ----
    for n in nl.affine_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        out_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, op=nl.copy, data=exp_tile, scale=inv_sum)

        nisa.dma_copy(dst=y[0:P, n * TILE_N:(n + 1) * TILE_N], src=out_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(42)

    x = torch.randn(P, N_total, dtype=torch.float32).to(device)
    y = softmax_2pass(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = torch.softmax(x_cpu, dim=1)

    assert torch.allclose(y_cpu, ref, atol=1e-5), (
        f"Max diff: {(y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
