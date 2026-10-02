"""
AFTER kernel: Scale-then-shift in a single fused loop, intermediate stays in SBUF.

Optimization target: Category #5 — Inline a function so it can be fused.

What changed vs before.py:
- The two separate tile loops are fused into one. The scale and shift operations
  run back-to-back within the same loop iteration.
- The intermediate (scaled) result lives only in SBUF — it is never written to HBM.
- Eliminates 8 intermediate stores and 8 intermediate loads (16 DMA transactions
  and 4 MB of HBM traffic removed).
- Total HBM traffic drops from 4*M*N bytes to 2*M*N bytes (2× reduction).
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def scale_and_shift_fused(x):
    """
    Compute y = (x * 2.0) + 1.0 in a single fused tile loop.

    AFTER: scale and shift are inlined into one loop iteration; intermediate
    result lives in SBUF only. No HBM round-trip between operations.
    HBM traffic = M*N reads + M*N writes (2× less than before).

    x: [M, N] in HBM
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.affine_range(M // TILE_P):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        # Scale — intermediate stays in SBUF, never written to HBM
        scaled = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=x_tile, op0=nl.multiply, operand0=2.0)

        # Shift — consumes SBUF result directly
        shifted = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=shifted, data=scaled, op0=nl.add, operand0=1.0)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=shifted)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = scale_and_shift_fused(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = x_cpu * 2.0 + 1.0
    assert torch.allclose(y_cpu, ref, atol=1e-5), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
