"""
AFTER kernel: row-wise softmax with activation_reduce fusing shift+exp+sum.

Optimization: Category #15 — Fuse multiple instructions into one

KEY CHANGE: Replace the three separate instructions for shift+exp+sum with a
single nisa.activation_reduce call:

  Before (3 instructions):
    tensor_scalar(add, neg_max)     → x_shifted (P, N)
    activation(exp)                 → exp_x (P, N)
    tensor_reduce(add)              → sum_val (P, 1)

  After (1 instruction):
    activation_reduce(exp, add, bias=neg_max) → exp_x (P, N) + sum_val (P, 1)

activation_reduce computes:
  dst       = op(data * scale + bias)   →  exp(x + neg_max)  = exp(x - max)
  reduce_res = reduce_op(dst along N)   →  sum(exp_x)

This eliminates the x_shifted intermediate SBUF buffer and reduces the
ScalarE instruction count for the core exp+sum path from 3 to 1.

Instruction count per tile:
  Before: tensor_reduce + activation + tensor_scalar + activation + tensor_reduce
          + activation + tensor_tensor = 7
  After:  tensor_reduce + activation + activation_reduce
          + activation + tensor_tensor = 5  (−2)
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_M = 128


@nki.jit(platform_target="trn1")
def softmax_fused(x):
    """
    Row-wise softmax over x [M=1024, N=512].

    AFTER: activation_reduce fuses shift, exp, and sum into one instruction.

    x: [M=1024, N=512] float32
    Returns: softmax(x) [M=1024, N=512] float32
    """
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Step 1: row max (unchanged)
        max_val = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x_tile, op=nl.maximum,
                           axis=1, keepdims=True)

        # Step 2: negate max (unchanged)
        neg_max = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=neg_max, data=max_val, op=nl.copy, scale=-1.0)

        # Steps 3+4+5 FUSED: exp(x + neg_max) and sum(exp_x) in ONE instruction
        # activation_reduce computes:
        #   exp_x = exp(x_tile * 1.0 + neg_max)   [= exp(x - max), numerically stable]
        #   sum_val = sum(exp_x along free dim)
        exp_x = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        sum_val = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=exp_x, op=nl.exp, data=x_tile,
                               reduce_op=nl.add, reduce_res=sum_val,
                               bias=neg_max)

        # Step 6: reciprocal (unchanged)
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

    result = softmax_fused(x_dev)
    xm.mark_step()

    ref = torch.nn.functional.softmax(x, dim=-1)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-5), (
        f"Max diff: {(result_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
