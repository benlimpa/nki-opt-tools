"""
AFTER kernel: y = relu(2.0 * x - 1.0) fusing scale+shift into one tensor_scalar.

Optimization: Category #20 — Fuse dependent operations

KEY CHANGE: nisa.tensor_scalar supports two chained scalar ops in a single
VectorE instruction via its op0/op1 parameters:

  nisa.tensor_scalar(dst, data, op0=multiply, operand0=2.0, op1=add, operand1=-1.0)
  computes: (x * 2.0) + (-1.0) = 2*x - 1   in ONE VectorE instruction

This fuses the two dependent tensor_scalar calls from the before kernel:
  Before: tensor_scalar(multiply) → tensor_scalar(add) → activation(relu)
          VectorE + VectorE + ScalarE  = 3 instructions, 2 SBUF intermediates
  After:  tensor_scalar(multiply+add) → activation(relu)
          VectorE + ScalarE           = 2 instructions, 1 SBUF intermediate

The multiply and add are dependent (add consumes multiply's output), so this
is an example of fusing two dependent operations into one compound instruction.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_M = 128
SCALE = 2.0
BIAS = -1.0


@nki.jit(platform_target="trn1")
def relu_scale_shift_fused(x):
    """
    Compute y = relu(2.0 * x - 1.0) via 2 instructions instead of 3.

    AFTER: tensor_scalar(multiply+add) → activation(relu)
    Scale and shift fused into one VectorE compound instruction.

    x: [M=1024, N=512] float32
    Returns: y [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Fused: (x * 2.0) + (-1.0) in ONE VectorE instruction
        # op0 applied first, then op1 applied to the result
        scaled_shifted = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled_shifted, data=x_tile,
                           op0=nl.multiply, operand0=SCALE,
                           op1=nl.add, operand1=BIAS)

        # ReLU: ScalarE activation (unchanged)
        out_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, data=scaled_shifted, op=nl.relu)

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:N],
                      src=out_tile)

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    result = relu_scale_shift_fused(x_dev)
    xm.mark_step()

    ref = torch.relu(SCALE * x + BIAS)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-4), (
        f"Max diff: {(result_cpu - ref).abs().max()}")
    print("Correctness check passed.")
