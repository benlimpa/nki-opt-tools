"""
AFTER kernel: Batched element-wise multiply with W loaded once per tile and reused.

Optimization target: Category #8 — Increase reuse by keeping data in SBUF across
outer loop iterations.

What changed vs before.py:
- Loop order is swapped: tile-outside / batch-inside instead of batch-outside / tile-inside.
- W[i] is loaded once per tile into SBUF, then reused across all BATCH=8 inner iterations.
- W_tile persists in SBUF while the inner loop iterates over batch elements — no eviction
  occurs because the allocator sees no conflicting use of that SBUF region.
- W reload count drops from 32 to 4 (one per tile): saves 28 × 256 KB = 7 MB of HBM reads.
"""

import nki
import nki.isa as nisa
import nki.language as nl

BATCH = 8
M = 512
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def batched_mul_reuse_w(x, w):
    """
    Compute Y[b, i, j] = X[b, i, j] * W[i, j] with loop order (tile, batch).

    AFTER: outer loop is over M tiles; W[i] is loaded once per tile and stays
    in SBUF for all BATCH iterations of the inner loop. Total W-tile loads: 4.

    x: [BATCH, M, N] in HBM
    w: [M, N] in HBM
    Returns: y [BATCH, M, N] in HBM
    """
    y = nl.ndarray((BATCH, M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Outer loop: tiles over M dimension
    for i in nl.sequential_range(M // TILE_P):
        # Load W[i] ONCE — stays in SBUF for all batch iterations below
        w_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tile, src=w[i * TILE_P:(i + 1) * TILE_P, 0:N])

        # Inner loop: batch — w_tile is reused here without reloading
        for b in nl.sequential_range(BATCH):
            x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[b, i * TILE_P:(i + 1) * TILE_P, 0:N])

            result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=result, data1=x_tile, data2=w_tile, op=nl.multiply)

            nisa.dma_copy(dst=y[b, i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(BATCH, M, N, dtype=torch.float32).to(device)
    w = torch.randn(M, N, dtype=torch.float32).to(device)
    y = batched_mul_reuse_w(x, w)
    xm.mark_step()

    x_cpu = x.cpu()
    w_cpu = w.cpu()
    y_cpu = y.cpu()
    ref = x_cpu * w_cpu.unsqueeze(0)
    assert torch.allclose(y_cpu, ref, atol=1e-5), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
