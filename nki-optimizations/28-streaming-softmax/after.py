"""
AFTER kernel: Row-wise softmax using streaming online algorithm (1-pass max+sum).

Optimization: Category #28 — Streaming softmax with running max and scaling trick.

KEY CHANGE: The separate max-scan (pass 1) and exp-sum-scan (pass 2) are fused
into a single streaming pass by maintaining a running (max, sum) pair and
rescaling the accumulated sum whenever a new per-tile maximum exceeds the
current running maximum:

    new_max  = max(running_max, tile_max)
    rescaled_sum = running_sum * exp(running_max - new_max)
    running_sum  = rescaled_sum + sum(exp(x_tile - new_max))
    running_max  = new_max

This is the "online softmax" / "safe softmax with running max" algorithm used
in Flash Attention. After one streaming pass, running_max is the true global
per-row maximum and running_sum is the correctly-normalized partition function
Z = sum_i exp(x_i - max). A single normalization pass over x then suffices.

HBM traffic breakdown (after):
  Pass 1 (streaming) — read x: 16 × [128, 128] × 4 B = 1 MB
  Pass 2 (normalize) — read x: 16 × [128, 128] × 4 B = 1 MB
                       write y: 16 × [128, 128] × 4 B = 1 MB
  Total reads: 2 MB   writes: 1 MB   grand total: 3 MB

Saving vs before: 1 MB of HBM reads eliminated (1 full pass over x removed).
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N_total = 2048
TILE_N = 128


@nki.jit(platform_target="trn1")
def softmax_streaming(x):
    """
    Row-wise softmax over the N dimension, streaming online algorithm.

    AFTER: 2 passes over x (1 fused streaming pass + 1 normalize pass).
    x:    [P, N_total] in HBM
    Returns: y [P, N_total] in HBM
    """
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Pass 1: streaming max + sum with running-max rescaling ----
    # State: (running_max, running_sum) — both shape [P, 1] in SBUF
    running_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=running_max, value=-float('inf'))

    running_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=running_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        # Per-tile maximum
        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=x_tile,
                           axis=1, keepdims=True)

        # new running max
        new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=running_max, data2=tile_max,
                           op=nl.maximum)

        # Rescaling factor: exp(running_max - new_max)
        # When running_max == -inf on the first tile this equals exp(-inf) == 0,
        # so the initial running_sum=0 is left unchanged (0 * 0 = 0).
        max_diff = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=max_diff, data1=running_max, data2=new_max,
                           op=nl.subtract)

        scale = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scale, op=nl.exp, data=max_diff)

        scaled_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scaled_sum, data1=running_sum, data2=scale,
                           op=nl.multiply)

        # exp(x_tile - new_max) and its row sum
        neg_new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=neg_new_max, op=nl.copy, data=new_max, scale=-1.0)

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_new_max)

        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile,
                           axis=1, keepdims=True)

        # Update running state
        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=scaled_sum, data2=tile_sum,
                           op=nl.add)

        running_sum = new_sum
        running_max = new_max

    # After pass 1: running_max = global per-row max, running_sum = Z
    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=running_sum)

    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=running_max, scale=-1.0)

    # ---- Pass 2: normalize using final max/sum (x read only a second time) ----
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
    y = softmax_streaming(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = torch.softmax(x_cpu, dim=1)

    assert torch.allclose(y_cpu, ref, atol=1e-5), (
        f"Max diff: {(y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
