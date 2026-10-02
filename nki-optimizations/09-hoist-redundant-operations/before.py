"""
BEFORE kernel: Redundant load + sigmoid computation inside tile loop.

Optimization target: Category #9 — Hoist redundant operations out of loops.

This kernel applies a fixed sigmoid gate to every TILE_P-row block of x:
    y = x * sigmoid(w)
where w [TILE_P, N] is a loop-invariant weight tensor. The before version
reloads w from HBM AND recomputes sigmoid(w) on every iteration of the
M // TILE_P tile loop.

With M=1024, TILE_P=128 there are 8 iterations. Each iteration redundantly:
  - Issues 1 DMA load of w  (TILE_P * N * 4 bytes = 256 KB per load → 7 wasted)
  - Runs 1 ScalarE sigmoid  (TILE_P * N = 65536 elements → 7 wasted)

The sigmoid result is identical every iteration since w never changes.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128  # NKI partition size limit


@nki.jit(platform_target="trn1")
def apply_gate_before(x, w):
    """
    Compute y = x * sigmoid(w), where w is broadcast over M // TILE_P tiles.

    BEFORE: loads w and computes sigmoid(w) inside the loop —
    (M // TILE_P) = 8 redundant loads + 8 redundant sigmoid computations.

    x: [M, N]      — input activation (HBM)
    w: [TILE_P, N] — fixed gate weight (HBM), loop-invariant
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.affine_range(M // TILE_P):
        # Load input tile
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        # REDUNDANT: w does not change across iterations — reloaded M//TILE_P times
        w_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tile, src=w[0:TILE_P, 0:N])

        # REDUNDANT: sigmoid(w) is the same every iteration — recomputed wastefully
        gate = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gate, op=nl.sigmoid, data=w_tile)

        # Apply gate: y_tile = x_tile * sigmoid(w)
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

    y = apply_gate_before(x, w)
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
