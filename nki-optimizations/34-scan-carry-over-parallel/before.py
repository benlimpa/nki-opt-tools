"""
BEFORE kernel: Sequential 1D scan using row-by-row processing.
Baseline: Processes a 16384 element array as 128 rows of 128 elements.
It uses nl.sequential_range to scan each row one by one, carrying the 
last element of row i as the initial value for row i+1.
This is sequential and slow.

Category: #34 — Scan carry-over to parallelize scan
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

N_TOTAL = 16384
P = 128
TILE_N = 128


@nki.jit(platform_target="trn1")
def scan_sequential_rows(X):
    """
    Compute 1D cumulative sum of X by scanning rows sequentially.
    X: [N_TOTAL] (mapped to [128, 128])
    Returns: Y: [N_TOTAL]
    """
    X_2d = X.reshape((P, TILE_N))
    Y_2d = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Carry-over between rows (scalar)
    carry_sb = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="carry_sb")

    # Ones for scan
    ones_sb = nl.zeros((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="ones_sb")
    nisa.tensor_scalar(dst=ones_sb, data=ones_sb, op0=nl.add, operand0=1.0)

    # Use a fixed SBUF tile for all operations to ensure partition alignment (Partition 0)
    data_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="data_sb")
    result_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="result_sb")

    for i in nl.sequential_range(P):
        # Load row i into partition 0 of data_sb
        nisa.dma_copy(dst=data_sb[0:1, :], src=X_2d[i : i + 1, :])

        # Scan partition 0
        nisa.tensor_tensor_scan(
            dst=result_sb[0:1, :],
            data0=ones_sb[0:1, :],
            data1=data_sb[0:1, :],
            initial=carry_sb[0:1, :],
            op0=nl.multiply,
            op1=nl.add,
        )

        # Store partition 0 back to row i
        nisa.dma_copy(dst=Y_2d[i : i + 1, :], src=result_sb[0:1, :])

        # Update carry for next row: last element of current result
        nisa.tensor_copy(dst=carry_sb[0:1, :], src=result_sb[0:1, TILE_N - 1 : TILE_N])

    return Y_2d.reshape((N_TOTAL,))


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    X_cpu = torch.randn(N_TOTAL, dtype=torch.float32)
    X_dev = X_cpu.to(device)

    Y_dev = scan_sequential_rows(X_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    ref = torch.cumsum(X_cpu, dim=0)
    
    assert torch.allclose(Y_cpu, ref, atol=1e-2, rtol=1e-2), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
