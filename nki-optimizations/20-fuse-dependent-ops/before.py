"""
BEFORE kernel: y = relu(2.0 * x - 1.0) over x [M=1024, N=512]
using three separate dependent instructions.

Optimization target: Category #20 — Fuse dependent operations

Each tile iteration issues three instructions with data dependencies:

  1. nisa.tensor_scalar(multiply, 2.0)   → scaled  [VectorE]
  2. nisa.tensor_scalar(add, -1.0)       → shifted [VectorE]
  3. nisa.activation(relu)               → result  [ScalarE]

Operations 1→2→3 form a dependency chain: scaled feeds shifted, which feeds
relu. Each step allocates a separate SBUF buffer for the intermediate result.

The nisa.activation instruction natively supports op(scale * data + bias),
so all three steps can be expressed as one fused call — but that is the after.
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
def relu_scale_shift_separate(x):
    """
    Compute y = relu(2.0 * x - 1.0) via 3 separate dependent instructions.

    BEFORE: VectorE(scale) → VectorE(shift) → ScalarE(relu)
    Two SBUF intermediates allocated per tile.

    x: [M=1024, N=512] float32
    Returns: y [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Step 1: scale by 2.0  [VectorE]
        scaled = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=x_tile,
                           op0=nl.multiply, operand0=SCALE)

        # Step 2: add bias -1.0  [VectorE]
        shifted = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=shifted, data=scaled,
                           op0=nl.add, operand0=BIAS)

        # Step 3: relu  [ScalarE]
        out_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, data=shifted, op=nl.relu)

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

    result = relu_scale_shift_separate(x_dev)
    xm.mark_step()

    ref = torch.relu(SCALE * x + BIAS)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-4), (
        f"Max diff: {(result_cpu - ref).abs().max()}")
    print("Correctness check passed.")
