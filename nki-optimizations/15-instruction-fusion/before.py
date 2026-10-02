"""
BEFORE kernel: row-wise softmax with separate shift, exp, and sum instructions.

Optimization target: Category #15 — Fuse multiple instructions into one

This kernel computes softmax(x) row-wise over x [M=1024, N=512] processed as
8 tiles of [P=128, N=512]. For each tile, numerically stable softmax requires:

  Step 1: row max         tensor_reduce(max)         → max_val (P, 1)
  Step 2: negate max      activation(copy, scale=-1)  → neg_max (P, 1)
  Step 3: shift x         tensor_scalar(add, neg_max) → x_shifted (P, N)  ← SEPARATE
  Step 4: exp             activation(exp)             → exp_x (P, N)      ← SEPARATE
  Step 5: sum of exp      tensor_reduce(add)          → sum_val (P, 1)    ← SEPARATE
  Step 6: reciprocal      activation(reciprocal)      → inv_sum (P, 1)
  Step 7: normalize       tensor_tensor(multiply)     → out (P, N)

Steps 3, 4, 5 are three separate instructions. In the after kernel these are
fused into a single activation_reduce call.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

M = 1024
N = 512
TILE_M = 128  # = nl.tile_size.pmax


@nki.jit(platform_target="trn1")
def softmax_unfused(x):
    """
    Row-wise softmax over x [M=1024, N=512].

    BEFORE: shift, exp, and sum are three separate instructions per tile.

    x: [M=1024, N=512] float32
    Returns: softmax(x) [M=1024, N=512] float32
    """
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Step 1: row max — reduce free dimension
        max_val = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x_tile, op=nl.maximum,
                           axis=1, keepdims=True)

        # Step 2: negate max
        neg_max = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=neg_max, data=max_val, op=nl.copy, scale=-1.0)

        # Step 3: shift — x - max  (broadcasts (TILE_M,1) across N)
        x_shifted = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_shifted, data=x_tile,
                           op0=nl.add, operand0=neg_max)

        # Step 4: exp
        exp_x = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_x, data=x_shifted, op=nl.exp)

        # Step 5: sum of exp — reduce free dimension
        sum_val = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_val, data=exp_x, op=nl.add,
                           axis=1, keepdims=True)

        # Step 6: reciprocal of sum
        inv_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_sum, data=sum_val, op=nl.reciprocal)

        # Step 7: normalize — inv_sum (P,1) broadcasts across N via tensor_scalar
        out_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out_tile, data=exp_x, op0=nl.multiply, operand0=inv_sum)

        nisa.dma_copy(dst=out[m * TILE_M:(m + 1) * TILE_M, 0:N], src=out_tile)

    return out


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    result = softmax_unfused(x_dev)
    xm.mark_step()

    ref = torch.nn.functional.softmax(x, dim=-1)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-5), (
        f"Max diff: {(result_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
