"""
AFTER kernel: Bias-add with bias loaded once into SBUF before the loop.

Optimization target: Category #2 — Minimize data movement.

What changed vs before.py:
- The bias tile is loaded ONCE into SBUF before the sequential_range loop, rather
  than being reloaded from HBM on every iteration.
- This eliminates 7 of the 8 DMA loads of the 256 KB bias tile (1.75 MB of saved
  HBM reads across the 8-iteration loop).
- The bias_sbuf tensor persists in SBUF for all iterations, being reused 8 times
  without any additional HBM traffic.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024      # total rows (must be multiple of TILE_M)
N = 512       # columns
TILE_M = 128  # partition size


@nki.jit(platform_target="trn1")
def bias_add_load_once(x, bias):
    """
    Compute y[i*TILE_M:(i+1)*TILE_M, :] = x[i*TILE_M:(i+1)*TILE_M, :] + bias[0:TILE_M, :]
    for each tile i in [0, M//TILE_M).

    AFTER: bias is loaded once into SBUF before the loop — only 1 DMA load of 256 KB.

    x:    [M, N] in HBM
    bias: [TILE_M, N] in HBM (same bias applied to every tile)
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # SINGLE LOAD: load bias once into SBUF before the loop
    bias_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sbuf, src=bias[0:TILE_M, 0:N])

    for i in nl.sequential_range(M // TILE_M):
        # Load input tile from HBM
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        # Add bias (already in SBUF — no HBM load needed)
        result = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=bias_sbuf, op=nl.add)

        # Store result tile to HBM
        nisa.dma_copy(dst=y[i * TILE_M:(i + 1) * TILE_M, 0:N], src=result)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).to(device)
    bias = torch.randn(TILE_M, N, dtype=torch.float32).to(device)
    y = bias_add_load_once(x, bias)
    xm.mark_step()

    # Verify correctness
    x_cpu = x.cpu()
    bias_cpu = bias.cpu()
    y_cpu = y.cpu()
    ref = torch.zeros(M, N, dtype=torch.float32)
    for i in range(M // TILE_M):
        ref[i * TILE_M:(i + 1) * TILE_M, :] = x_cpu[i * TILE_M:(i + 1) * TILE_M, :] + bias_cpu
    assert torch.allclose(y_cpu, ref, atol=1e-4), f"Max diff: {(y_cpu - ref).abs().max()}"
    print("Correctness check passed.")
