"""
BEFORE kernel: tanh(x * scale + bias) using separate instructions.
Baseline: Uses separate instructions for scaling, bias, and activation.
Since nisa.tensor_tensor does not support broadcasting [128, 1] to [128, 512],
we must loop over columns for the scale and bias steps.

Category: #35 — Replace general code with specialized instructions
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

P = 128
N = 512
NUM_ITERS = 100

@nki.jit(platform_target="trn1")
def tanh_baseline(X, scale, bias):
    """
    Compute y = tanh(X * scale + bias)
    X: [P, N]
    scale: [P, 1]
    bias: [P, 1]
    """
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load inputs
    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    b_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=bias)

    # Repeat computation
    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        # 1. Scale (Multiplication) - Loop over columns to handle broadcasting
        tmp1 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp1[:, j:j+1], data1=x_tile[:, j:j+1], data2=s_tile, op=nl.multiply)

        # 2. Bias (Addition) - Loop over columns to handle broadcasting
        tmp2 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp2[:, j:j+1], data1=tmp1[:, j:j+1], data2=b_tile, op=nl.add)

        # 3. Tanh Activation
        nisa.activation(dst=res_tile, data=tmp2, op=nl.tanh)

    # Store result
    nisa.dma_copy(dst=Y, src=res_tile)

    return Y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    X_cpu = torch.randn(P, N, dtype=torch.float32)
    S_cpu = torch.randn(P, 1, dtype=torch.float32)
    B_cpu = torch.randn(P, 1, dtype=torch.float32)

    X_dev = X_cpu.to(device)
    S_dev = S_cpu.to(device)
    B_dev = B_cpu.to(device)

    Y_dev = tanh_baseline(X_dev, S_dev, B_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    ref = torch.tanh(X_cpu * S_cpu + B_cpu)

    assert torch.allclose(Y_cpu, ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
