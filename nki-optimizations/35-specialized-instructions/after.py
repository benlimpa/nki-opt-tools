"""
AFTER kernel: tanh(x * scale + bias) using fused specialized instruction.
Optimization: Category #35 — Replace general code with specialized instructions.

KEY CHANGE: Use nisa.activation with fused scale and bias.
The specialized nisa.activation instruction can perform data * scale + bias
natively before applying the activation function, all in a single 
Scalar Engine instruction cycle.

This reduces:
1. Total instruction count (3 -> 1 per loop iteration).
2. SBUF memory traffic (intermediates tmp1 and tmp2 are not written to SBUF).
3. Scalar Engine active time.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

P = 128
N = 512
NUM_ITERS = 100

@nki.jit(platform_target="trn1")
def tanh_specialized(X, scale, bias):
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
        # Optimized: Fused Scale + Bias + Activation
        # nisa.activation(dst, op, data, bias, scale)
        nisa.activation(
            dst=res_tile,
            op=nl.tanh,
            data=x_tile,
            scale=s_tile,
            bias=b_tile
        )

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

    Y_dev = tanh_specialized(X_dev, S_dev, B_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    ref = torch.tanh(X_cpu * S_cpu + B_cpu)
    
    assert torch.allclose(Y_cpu, ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
