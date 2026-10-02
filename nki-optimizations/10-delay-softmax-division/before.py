"""
BEFORE kernel: Row-wise softmax with premature HBM intermediate for exp values.

Optimization target: Category #10 — Delay softmax division until after all
reductions are complete.

This kernel computes softmax(x) over the free (N) dimension for each of P rows.
Because the normalization denominator (sum of exp) is not known until all N tiles
have been processed, the CORRECT approach is to delay the division until after
the full sum reduction.

The BEFORE version stores the exp tile to an HBM intermediate buffer during the
reduction loop, then reloads it in a separate normalization pass. This creates an
unnecessary HBM round-trip for the exp values: one extra full-tensor write (256 KB)
and one extra full-tensor read (256 KB) = 512 KB of avoidable HBM traffic.

HBM traffic breakdown (before):
  Pass 1 — max:    read  x       = 256 KB
  Pass 2 — exp:    read  x       = 256 KB
                   write exp_hbm = 256 KB  ← UNNECESSARY
  Pass 3 — norm:   read  exp_hbm = 256 KB  ← UNNECESSARY
                   write y       = 256 KB
  Total: 768 KB reads + 512 KB writes = 1280 KB
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128       # partition dimension (row count, <= 128)
N_total = 512  # free dimension (column count for softmax)
TILE_N = 128   # tile size for N — 4 tiles total


@nki.jit(platform_target="trn1")
def softmax_hbm_intermediate(x):
    """
    Compute row-wise softmax(x) over the N dimension.

    BEFORE: stores exp(x - max) to HBM in pass 2, reloads it in pass 3 to divide.

    x: [P, N_total] in HBM
    Returns: y [P, N_total] in HBM
    """
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Pass 1: row-wise max (sequential — max accumulates across tiles) ----
    max_val = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=max_val, value=-float('inf'))

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=x_tile, axis=1, keepdims=True)

        new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=max_val, data2=tile_max, op=nl.maximum)
        max_val = new_max

    # Precompute neg_max for activation bias: exp(x + neg_max) = exp(x - max)
    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=max_val, scale=-1.0)

    # ---- Pass 2: compute exp(x - max), STORE to HBM intermediate, accumulate sum ----
    exp_hbm = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.private_hbm)
    exp_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=exp_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        # PREMATURE STORE: exp written to HBM even though we'll reload in pass 3
        nisa.dma_copy(dst=exp_hbm[0:P, n * TILE_N:(n + 1) * TILE_N], src=exp_tile)

        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile, axis=1, keepdims=True)

        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=exp_sum, data2=tile_sum, op=nl.add)
        exp_sum = new_sum

    # Compute inv_sum = 1 / exp_sum for normalization
    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=exp_sum)

    # ---- Pass 3: RELOAD exp from HBM, multiply by inv_sum, store output ----
    for n in nl.affine_range(N_total // TILE_N):
        # UNNECESSARY RELOAD: reading back exp values just stored to HBM
        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_tile, src=exp_hbm[0:P, n * TILE_N:(n + 1) * TILE_N])

        # DELAYED DIVISION: divides by full sum (correct) but pays HBM reload cost
        out_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, op=nl.copy, data=exp_tile, scale=inv_sum)

        nisa.dma_copy(dst=y[0:P, n * TILE_N:(n + 1) * TILE_N], src=out_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, N_total, dtype=torch.float32).to(device)
    y = softmax_hbm_intermediate(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = torch.softmax(x_cpu, dim=1)

    assert torch.allclose(y_cpu, ref, atol=1e-5), (
        f"Max diff: {(y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
