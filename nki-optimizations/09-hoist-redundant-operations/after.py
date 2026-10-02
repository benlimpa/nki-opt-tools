"""
AFTER kernel: Load + sigmoid computation hoisted outside tile loop.

Optimization: Category #9 — Hoist redundant operations out of loops.

KEY CHANGES vs before.py:
  1. w is loaded from HBM ONCE before the loop (1 DMA instead of 8).
  2. sigmoid(w) is computed ONCE before the loop (1 ScalarE op instead of 8).
  3. The resulting `gate` tile stays live in SBUF across all 8 tile iterations.
  4. Inside the loop, only x_tile is loaded and multiplied by the cached gate.

Before: 8 w-loads + 8 sigmoid ops  (7 of each are wasted)
After:  1 w-load  + 1 sigmoid op   (zero waste)
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def apply_gate_after(x, w):
    """
    Compute y = x * sigmoid(w), where w is broadcast over M // TILE_P tiles.

    AFTER: loads w and computes sigmoid(w) ONCE outside the loop.
    gate = sigmoid(w) lives in SBUF and is reused across all tile iterations.

    x: [M, N]      — input activation (HBM)
    w: [TILE_P, N] — fixed gate weight (HBM), loop-invariant
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Hoist: load w and compute gate ONCE before the tile loop ----
    w_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_tile, src=w[0:TILE_P, 0:N])

    gate = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate, op=nl.sigmoid, data=w_tile)

    # ---- Main loop: only load x tile; gate is reused from SBUF ----
    for i in nl.affine_range(M // TILE_P):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        y_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_tile, data1=x_tile, data2=gate, op=nl.multiply)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=y_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    w = torch.randn(TILE_P, N, dtype=torch.float32).to(device)

    y = apply_gate_after(x, w)
    xm.mark_step()

    x_cpu = x.cpu()
    w_cpu = w.cpu()
    y_cpu = y.cpu()

    gate_cpu = torch.sigmoid(w_cpu)  # [TILE_P, N]
    ref = torch.zeros(M, N, dtype=torch.float32)
    for i in range(M // TILE_P):
        ref[i * TILE_P:(i + 1) * TILE_P, :] = (
            x_cpu[i * TILE_P:(i + 1) * TILE_P, :] * gate_cpu
        )

    assert torch.allclose(y_cpu, ref, atol=1e-5), (
        f"Max diff: {(y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
