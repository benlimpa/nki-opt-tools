"""
BEFORE kernel: Redundant loading of invariant data inside a loop.
Baseline: Loads the row-max vector from HBM inside every tile iteration of the sum-exp pass.

Category: #36 — Hoist nl.load() for reused data outside inner loops

Concretized Description:
In a [128, 2048] sum-exp kernel tiled by 128 elements (16 tiles),
the baseline find the global row-max first and stores it to HBM.
In the second pass (summing exp(x - max)), it reloads this max vector 
from HBM for every one of the 16 tiles.
Total HBM loads of max: 16 (instead of 1).
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

P = 128
N = 2048
TILE_N = 128


@nki.jit(platform_target="trn1")
def sum_exp_baseline(X):
    """
    Compute row-wise sum(exp(X - max(X))).
    """
    # 1. Find max (Pass 1)
    # Using private_hbm for intermediate results on trn1
    row_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.private_hbm)
    
    # Initialize max with small value
    max_sb = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="max_sb")
    nisa.tensor_scalar(dst=max_sb, data=max_sb, op0=nl.add, operand0=-1e9)
    
    for i in nl.affine_range(N // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=X[0:P, i * TILE_N : (i + 1) * TILE_N])
        
        # Local max
        cur_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cur_max, data=x_tile, op=nl.maximum, axis=1)
        
        # Update global max
        nisa.tensor_tensor(dst=max_sb, data1=max_sb, data2=cur_max, op=nl.maximum)
    
    nisa.dma_copy(dst=row_max, src=max_sb)
    
    # 2. Sum exp(x - max) (Pass 2)
    row_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    sum_sb = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="sum_sb")
    
    for i in nl.affine_range(N // TILE_N):
        # Load data tile
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=X[0:P, i * TILE_N : (i + 1) * TILE_N])
        
        # REDUNDANT: Load row_max from HBM for every tile!
        max_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=max_tile, src=row_max)
        
        # exp(x - max)
        minus_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=minus_max, data=max_tile, op0=nl.multiply, operand0=-1.0)
        
        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=minus_max)
        
        # Sum into sum_sb
        cur_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cur_sum, data=exp_tile, op=nl.add, axis=1)
        nisa.tensor_tensor(dst=sum_sb, data1=sum_sb, data2=cur_sum, op=nl.add)
        
    nisa.dma_copy(dst=row_sum, src=sum_sb)
    
    return row_sum


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    X_cpu = torch.randn(P, N, dtype=torch.float32)
    X_dev = X_cpu.to(device)

    Y_dev = sum_exp_baseline(X_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    # Reference
    max_val = torch.max(X_cpu, dim=1, keepdim=True)[0]
    ref = torch.sum(torch.exp(X_cpu - max_val), dim=1, keepdim=True)
    
    assert torch.allclose(Y_cpu, ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
