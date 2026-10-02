"""
BEFORE kernel: Element-wise multiply with nl.sequential_range.

Optimization target: Category #4 — Loop reordering and restructuring.

This kernel multiplies A[M, N] * B[M, N] using sequential_range to tile over M.
sequential_range forces iterations to execute one at a time — the compiler cannot
pipeline DMA loads for the next tile while computing the current tile. Each of the
8 iterations runs fully before the next begins, leaving the DMA engine idle during
compute and the VectorE idle during DMA.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def mul_sequential(a, b):
    """
    Compute c = a * b, tiled over M with TILE_P=128.

    BEFORE: nl.sequential_range — 8 iterations run strictly in order;
    compiler cannot overlap DMA and compute across iterations.

    a: [M, N] in HBM
    b: [M, N] in HBM
    Returns: c [M, N] in HBM
    """
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(M // TILE_P):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.multiply)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    a = torch.randn(M, N, dtype=torch.float32).to(device)
    b = torch.randn(M, N, dtype=torch.float32).to(device)
    c = mul_sequential(a, b)
    xm.mark_step()

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_cpu = c.cpu()
    ref = a_cpu * b_cpu
    assert torch.allclose(c_cpu, ref, atol=1e-5), f"Max diff: {(c_cpu - ref).abs().max()}"
    print("Correctness check passed.")
