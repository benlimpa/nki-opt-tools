"""
BEFORE kernel: Transpose using the Tensor Engine (nc_transpose).
Baseline: Loads a [128, 1, 1, 128] tensor from HBM to SBUF, 
then uses the Tensor Engine to perform the transpose in SBUF.

Category: #37 — Other (DMA Transpose)
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

# Shapes must be 4D for consistency with after.py
# Treat D0 as rows, D3 as columns
D0, D1, D2, D3 = 128, 1, 1, 128
NUM_ITERS = 100

@nki.jit(platform_target="trn1")
def transpose_baseline(X):
    """
    Transpose a [128, 1, 1, 128] tensor.
    """
    Y = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Load input into a 2D tile [128, 128]
    x_tile = nl.ndarray((D0, D3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X[:, 0, 0, :])
    
    # Repeat to highlight overhead
    y_psum = nl.zeros((D3, D0), dtype=nl.float32, buffer=nl.psum)
    
    for _ in nl.affine_range(NUM_ITERS):
        # Must zero psum because nc_transpose (Tensor Engine) accumulates
        nisa.tensor_scalar(dst=y_psum, data=y_psum, op0=nl.multiply, operand0=0.0)
        
        # Transpose using Tensor Engine
        # Tensor Engine nc_transpose writes to PSUM
        nisa.nc_transpose(dst=y_psum, data=x_tile)

    # Store result (move from PSUM to SBUF first)
    y_sbuf = nl.ndarray((D3, D0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=y_sbuf, op=nl.copy, data=y_psum)
    nisa.dma_copy(dst=Y[:, 0, 0, :], src=y_sbuf)

    return Y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    X_cpu = torch.randn(D0, D1, D2, D3, dtype=torch.float32)
    X_dev = X_cpu.to(device)

    Y_dev = transpose_baseline(X_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    ref = X_cpu.transpose(0, 3)
    
    assert torch.allclose(Y_cpu, ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
