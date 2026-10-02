"""
AFTER kernel: parallelized 1D scan using multi-partition layout and MatMul.
Optimization: Category #34 — Scan carry-over to parallelize scan.

KEY CHANGE: Reshape the 1D array into a [128, 128] 2D tile to utilize all
128 partitions of the Vector Engine in parallel.
1. Local Scan: Perform nisa.tensor_tensor_scan on the [128, 128] tile.
   This scans each of the 128 rows in parallel.
2. Carry-over Across Partitions: The last element of row i must be added
   to all elements of row i+1, row i+2, etc.
3. MatMul Trick: Use a single nc_matmul with a strictly upper-triangular
   matrix of ones to compute the prefix sum of the 128 last elements (offsets)
   in a single hardware instruction.
4. Global Correctness: Add these offsets back to the respective rows.

This achieves massive speedup over the single-partition baseline.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

N_TOTAL = 16384
P = 128
TILE_N = 128


@nki.jit(platform_target="trn1")
def scan_1d_parallel(X, L_tri):
    """
    Compute 1D cumulative sum of X using parallelized hardware.
    X: [N_TOTAL]
    L_tri: [P, P] strictly upper triangular matrix of ones
    Returns: Y: [N_TOTAL]
    """
    # Reshape HBM to 2D to match SBUF layout [128, 128]
    X_2d = X.reshape((P, TILE_N))
    Y_2d = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.shared_hbm)

    # 1. Load 16384 elements into a [128, 128] tile
    data_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=data_sb, src=X_2d)

    # 2. Local scan on each of the 128 partitions in parallel
    ones_sb = nl.zeros((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="ones_sb")
    nisa.tensor_scalar(dst=ones_sb, data=ones_sb, op0=nl.add, operand0=1.0)

    res_local_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=res_local_sb,
        data0=ones_sb,
        data1=data_sb,
        initial=0.0,
        op0=nl.multiply,
        op1=nl.add,
    )

    # 3. Extract last element of each row (carry-overs)
    carries_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=carries_sb, src=res_local_sb[:, TILE_N-1 : TILE_N])

    # 4. Compute partition offsets using MatMul trick
    L_tri_sb = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=L_tri_sb, src=L_tri)

    offsets_psum = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.psum, name="offsets_psum")
    nisa.nc_matmul(dst=offsets_psum, stationary=L_tri_sb, moving=carries_sb)

    offsets_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=offsets_sb, op=nl.copy, data=offsets_psum)

    # 5. Add offsets back to the local results
    # Since nisa.tensor_tensor doesn't support broadcasting (128,1) to (128,128),
    # we loop over columns. This is still parallel across partitions.
    final_res_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    for j in nl.affine_range(TILE_N):
        nisa.tensor_tensor(dst=final_res_sb[:, j : j + 1],
                           data1=res_local_sb[:, j : j + 1],
                           data2=offsets_sb,
                           op=nl.add)

    # 6. Store back
    nisa.dma_copy(dst=Y_2d, src=final_res_sb)

    return Y_2d.reshape((N_TOTAL,))


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    # Strictly upper triangular matrix of ones
    L_tri_cpu = torch.triu(torch.ones(P, P), diagonal=1)
    L_tri_dev = L_tri_cpu.to(device)

    X_cpu = torch.randn(N_TOTAL, dtype=torch.float32)
    X_dev = X_cpu.to(device)

    Y_dev = scan_1d_parallel(X_dev, L_tri_dev)
    xm.mark_step()

    Y_cpu = Y_dev.cpu()
    ref = torch.cumsum(X_cpu, dim=0)

    assert torch.allclose(Y_cpu, ref, atol=1e-2, rtol=1e-2), (
        f"Max diff: {(Y_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
