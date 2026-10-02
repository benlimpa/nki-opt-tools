"""
AFTER kernel: Three-step pipeline fused into a single tile loop, no HBM intermediates.

Optimization target: Category #7 — Fuse loops (reordering if necessary).

What changed vs before.py:
- The three separate affine_range loops are fused into one. All three operations
  (scale ×2.0, add +1.0, scale ×0.5) run back-to-back within each tile iteration.
- Intermediates (scaled, added) live only in SBUF — never written to HBM.
- Eliminates 2 × 8 stores + 2 × 8 loads = 32 DMA transactions.
- HBM traffic drops from 3× to 1× (read once, write once per tile).
- All three loops iterated in the same order (i = 0..M//TILE_P), so no reordering
  was needed — the loops were directly fuse-compatible.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def pipeline_fused(x):
    """
    Compute Y = (X * 2.0 + 1.0) * 0.5 in a single fused tile loop.

    AFTER: all three operations share one loop iteration; intermediates stay in SBUF.
    Each tile is loaded once and stored once — minimum possible HBM traffic.

    x: [M, N] in HBM
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.affine_range(M // TILE_P):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        # Op 1: scale by 2.0 — intermediate stays in SBUF
        scaled = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=x_tile, op0=nl.multiply, operand0=2.0)

        # Op 2: add 1.0 — intermediate stays in SBUF
        added = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=added, data=scaled, op0=nl.add, operand0=1.0)

        # Op 3: scale by 0.5 — final result written to HBM once
        result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result, data=added, op0=nl.multiply, operand0=0.5)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = pipeline_fused(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = (x_cpu * 2.0 + 1.0) * 0.5
    assert torch.allclose(y_cpu, ref, atol=1e-5), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
