"""
BEFORE kernel: Bias-add with redundant HBM loads of the bias tile each iteration.

Optimization target: Category #2 — Minimize data movement.

This kernel adds a fixed bias[TILE_M, N] to every TILE_M-row block of X[M, N].
The BEFORE version reloads the 128×512 bias tile from HBM inside every iteration
of the sequential_range loop — generating M//TILE_M = 8 redundant DMA loads of
the same 256 KB tile (2 MB of unnecessary traffic).
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024      # total rows (must be multiple of TILE_M)
N = 512       # columns
TILE_M = 128  # partition size


@nki.jit(platform_target="trn1")
def bias_add_redundant_loads(x, bias):
    """
    Compute y[i*TILE_M:(i+1)*TILE_M, :] = x[i*TILE_M:(i+1)*TILE_M, :] + bias[0:TILE_M, :]
    for each tile i in [0, M//TILE_M).

    BEFORE: bias is reloaded from HBM on every iteration — 8 unnecessary loads of 256 KB.

    x:    [M, N] in HBM
    bias: [TILE_M, N] in HBM (same bias applied to every tile)
    Returns: y [M, N] in HBM
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(M // TILE_M):
        # Load input tile from HBM
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        # REDUNDANT LOAD: reload the same bias tile from HBM every iteration
        bias_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_tile, src=bias[0:TILE_M, 0:N])

        # Add bias to input tile
        result = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=bias_tile, op=nl.add)

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
    y = bias_add_redundant_loads(x, bias)
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
