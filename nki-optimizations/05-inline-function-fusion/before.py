"""
BEFORE kernel: Scale-then-shift via two separate tile loops with HBM intermediate.

Optimization target: Category #5 — Inline a function so it can be fused.

This kernel computes Y = (X * 2.0) + 1.0 but implements the two operations as
separate tile loops, each fully iterating over M. After the scale pass, the
intermediate result is stored tile-by-tile to nl.private_hbm. The shift pass then
reloads each tile from HBM. This produces an unnecessary HBM round-trip for every
tile: 8 extra stores (scale → HBM) + 8 extra loads (HBM → shift) = 16 extra DMA
transactions and 4 MB of avoidable HBM traffic.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def scale_then_shift_hbm(x):
    """
    Compute y = (x * 2.0) + 1.0 using two separate loops with HBM intermediate.

    BEFORE: scale loop stores to private_hbm; shift loop reloads from private_hbm.
    HBM traffic = 2*M*N reads + 2*M*N writes (intermediate doubles the traffic).

    x: [M, N] in HBM
    Returns: y [M, N] in HBM
    """
    # Intermediate buffer in HBM — forces round-trip between scale and shift
    intermediate = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Pass 1: scale by 2.0 — stores results to HBM
    for i in nl.affine_range(M // TILE_P):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        scaled = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=x_tile, op0=nl.multiply, operand0=2.0)

        # UNNECESSARY STORE: writes intermediate result to HBM
        nisa.dma_copy(dst=intermediate[i * TILE_P:(i + 1) * TILE_P, 0:N], src=scaled)

    # Pass 2: shift by 1.0 — reloads intermediate from HBM
    for i in nl.affine_range(M // TILE_P):
        # UNNECESSARY LOAD: reloads what was just computed from HBM
        inter_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=inter_tile, src=intermediate[i * TILE_P:(i + 1) * TILE_P, 0:N])

        shifted = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=shifted, data=inter_tile, op0=nl.add, operand0=1.0)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=shifted)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = scale_then_shift_hbm(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = x_cpu * 2.0 + 1.0
    assert torch.allclose(y_cpu, ref, atol=1e-5), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
