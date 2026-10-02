"""
AFTER kernel: Transpose using the DMA Engine (dma_transpose).
Optimization: Category #37 — Other (DMA Transpose).

KEY CHANGE: Use nisa.dma_transpose to perform the transpose during 
the memory transfer from HBM to SBUF.
To satisfy dma_transpose quirks:
1. Input must be 4D. We use [128, 1, 1, 128].
2. Destination must be SBUF.
3. Transpose permutation is (3, 1, 2, 0), which swaps axis 0 and 3.
   By putting rows on axis 0 and columns on axis 3, we get a 2D transpose.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

# Shapes must be 4D for dma_transpose quirks
# We treat D0 as rows and D3 as columns
D0, D1, D2, D3 = 128, 1, 1, 128
NUM_ITERS = 100

@nki.jit(platform_target="trn1")
def transpose_specialized(X):
    """
    Transpose a [128, 1, 1, 128] tensor using DMA Engine.
    Result will be [128, 1, 1, 128] where 0 and 3 are swapped.
    """
    Y = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Repeat computation
    y_tile = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.sbuf)
    
    for _ in nl.affine_range(NUM_ITERS):
        # Optimized: Transpose DURING load from HBM
        # Swaps axis 0 and 3
        nisa.dma_transpose(dst=y_tile, src=X)

    # Store result
    nisa.dma_copy(dst=Y, src=y_tile)

    return Y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    # Input: rows=128 (dim 0), cols=128 (dim 3)
    X_cpu = torch.randn(D0, D1, D2, D3, dtype=torch.float32)
    X_dev = X_cpu.to(device)

    Y_dev = transpose_specialized(X_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    # Ref: swap 0 and 3
    ref = X_cpu.transpose(0, 3)
    
    assert torch.allclose(Y_cpu, ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
