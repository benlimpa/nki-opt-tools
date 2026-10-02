"""
BEFORE kernel: Three-step pipeline using three separate tile loops with HBM intermediates.

Optimization target: Category #7 — Fuse loops (reordering if necessary).

This kernel computes Y = (X * 2.0 + 1.0) * 0.5 = X + 0.5 in three separate passes.
Each pass iterates over all M // TILE_P = 8 tiles and stores its result to a
nl.private_hbm intermediate buffer. The next pass reloads from that buffer.

Three separate loops × 8 tiles × 2 DMAs (load + store) per tile = 48 DMA transactions.
- inter1 costs: 8 stores (scale→HBM) + 8 loads (HBM→add) = 16 DMA ops, 4 MB HBM traffic
- inter2 costs: 8 stores (add→HBM) + 8 loads (HBM→scale) = 16 DMA ops, 4 MB HBM traffic
Total avoidable HBM traffic: 8 MB (4 MB reads + 4 MB writes for two intermediates).
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def pipeline_three_loops(x):
    """
    Compute Y = (X * 2.0 + 1.0) * 0.5 via three separate tile loops.

    BEFORE: each operation is a standalone pass over all M//TILE_P tiles.
    Inter1 and inter2 are HBM buffers that act as unnecessary hand-offs between
    passes — the compiler marks them as spill_save/spill_reload traffic.

    x: [M, N] in HBM
    Returns: y [M, N] in HBM
    """
    inter1 = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)
    inter2 = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Pass 1: scale by 2.0 — stores to inter1 in HBM
    for i in nl.affine_range(M // TILE_P):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])
        scaled = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=x_tile, op0=nl.multiply, operand0=2.0)
        # UNNECESSARY STORE: scaled result written to HBM
        nisa.dma_copy(dst=inter1[i * TILE_P:(i + 1) * TILE_P, 0:N], src=scaled)

    # Pass 2: add 1.0 — reloads inter1 from HBM, stores to inter2
    for i in nl.affine_range(M // TILE_P):
        # UNNECESSARY LOAD: reload what was just computed
        t1 = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t1, src=inter1[i * TILE_P:(i + 1) * TILE_P, 0:N])
        added = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=added, data=t1, op0=nl.add, operand0=1.0)
        # UNNECESSARY STORE: added result written to HBM
        nisa.dma_copy(dst=inter2[i * TILE_P:(i + 1) * TILE_P, 0:N], src=added)

    # Pass 3: scale by 0.5 — reloads inter2 from HBM, stores final output
    for i in nl.affine_range(M // TILE_P):
        # UNNECESSARY LOAD: reload what was just computed
        t2 = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t2, src=inter2[i * TILE_P:(i + 1) * TILE_P, 0:N])
        result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result, data=t2, op0=nl.multiply, operand0=0.5)
        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    y = pipeline_three_loops(x)
    xm.mark_step()

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    ref = (x_cpu * 2.0 + 1.0) * 0.5
    assert torch.allclose(y_cpu, ref, atol=1e-5), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
