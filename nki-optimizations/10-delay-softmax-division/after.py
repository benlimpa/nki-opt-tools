"""
AFTER kernel: Row-wise softmax — division delayed until after full sum reduction,
no HBM intermediate.

Optimization: Category #10 — Delay softmax division until after all reductions
are complete.

KEY CHANGE: exp values are never stored to HBM. Pass 2 reads x and accumulates
the sum of exp without writing anything to HBM. Only after the full sum is known
does pass 3 read x again, recompute exp, divide by the now-complete sum, and store
the output. This eliminates the 256 KB intermediate write and 256 KB reload that
the before version pays.

HBM traffic breakdown (after):
  Pass 1 — max:    read  x = 256 KB
  Pass 2 — sum:    read  x = 256 KB   (no intermediate write)
  Pass 3 — norm:   read  x = 256 KB
                   write y = 256 KB
  Total: 768 KB reads + 256 KB writes = 1024 KB   (saves 256 KB vs before)

Tradeoff: x is read three times instead of two, but each read is 256 KB while the
eliminated intermediate is also 256 KB — the net saving is 256 KB of HBM writes.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N_total = 512
TILE_N = 128


@nki.jit(platform_target="trn1")
def softmax_no_intermediate(x):
    """
    Compute row-wise softmax(x) over the N dimension.

    AFTER: no HBM intermediate for exp values. Division is delayed until the
    full sum is known; pass 3 recomputes exp on-the-fly during normalization.

    x: [P, N_total] in HBM
    Returns: y [P, N_total] in HBM
    """
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Pass 1: row-wise max (same as before) ----
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

    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=max_val, scale=-1.0)

    # ---- Pass 2: compute exp(x - max), accumulate sum — NO intermediate store ----
    exp_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=exp_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        # Accumulate sum only — exp_tile is discarded, NOT written to HBM
        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile, axis=1, keepdims=True)

        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=exp_sum, data2=tile_sum, op=nl.add)
        exp_sum = new_sum

    # Compute inv_sum = 1 / exp_sum — now the FULL sum is available
    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=exp_sum)

    # ---- Pass 3: read x again, recompute exp, divide by full sum, store ----
    # Division is DELAYED until here — after all reductions are complete
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
    torch.manual_seed(0)

    x = torch.randn(P, N_total, dtype=torch.float32).to(device)
    y = softmax_no_intermediate(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = torch.softmax(x_cpu, dim=1)

    assert torch.allclose(y_cpu, ref, atol=1e-5), (
        f"Max diff: {(y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
