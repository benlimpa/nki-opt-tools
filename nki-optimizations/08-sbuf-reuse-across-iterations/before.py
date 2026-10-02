"""
BEFORE kernel: Batched element-wise multiply with W reloaded every (batch, tile) pair.

Optimization target: Category #8 — Increase reuse by keeping data in SBUF across
outer loop iterations.

This kernel computes Y[b, :, :] = X[b, :, :] * W[:, :] for BATCH=8 batch elements.
The loop order is batch-outside / tile-inside. For each batch element b, the kernel
iterates over all M//TILE_P = 4 tiles and reloads W[i] from HBM on every iteration.
W never changes across batch elements, so every reload is redundant.

W reload count: BATCH × (M // TILE_P) = 8 × 4 = 32 tile loads (only 4 are necessary).
Avoidable W reads: (32 - 4) × TILE_P × N × sizeof(float32) = 28 × 256 KB = 7 MB.
"""

import nki
import nki.isa as nisa
import nki.language as nl

BATCH = 8
M = 512
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def batched_mul_reload_w(x, w):
    """
    Compute Y[b, i, j] = X[b, i, j] * W[i, j] with loop order (batch, tile).

    BEFORE: W is reloaded from HBM for every (batch, tile) pair — 32 total W-tile
    loads instead of the minimum 4. Each of the 28 redundant loads wastes 256 KB
    of HBM bandwidth.

    x: [BATCH, M, N] in HBM
    w: [M, N] in HBM
    Returns: y [BATCH, M, N] in HBM
    """
    y = nl.ndarray((BATCH, M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Outer loop: batch — W is not invariant to this loop from the compiler's
    # perspective because it is loaded inside the body.
    for b in nl.sequential_range(BATCH):
        # Inner loop: tiles over M dimension
        for i in nl.sequential_range(M // TILE_P):
            # REDUNDANT LOAD: W[i] loaded here for every batch element b
            w_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_tile, src=w[i * TILE_P:(i + 1) * TILE_P, 0:N])

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
    y = batched_mul_reload_w(x, w)
    xm.mark_step()

    x_cpu = x.cpu()
    w_cpu = w.cpu()
    y_cpu = y.cpu()
    ref = x_cpu * w_cpu.unsqueeze(0)
    assert torch.allclose(y_cpu, ref, atol=1e-5), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
