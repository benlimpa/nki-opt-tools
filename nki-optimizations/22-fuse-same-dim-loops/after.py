"""
AFTER kernel: Dual reduction (sum and sum of squares) over x [P=128, K=8192]
fused into a single sequential_range loop.

Optimization: Category #22 — Fuse loops over the same dimension to
improve intermediate data reuse.

KEY CHANGE:
  Before: Loop 1 reads x, computes sum. Loop 2 reads x AGAIN, computes sq sum.
          (2x HBM reads for x)
  After:  Fused loop reads x ONCE, computes both sum and sum of squares
          while the tile is live in SBUF.
          (1x HBM read for x)

Expected savings:
  - Halve total HBM read bytes (x loaded once)
  - DMA active time halved
  - Eliminate one entire loop overhead
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
K = 8192
TILE_K = 512

@nki.jit(platform_target="trn1")
def dual_reduction_fused(x):
    """
    Compute sum and sum of squares along the K dimension.

    AFTER: Single fused sequential_range loop. x is loaded once.

    x: [P=128, K=8192] float32
    Returns: sum_res, sq_sum_res (both [128, 1])
    """
    sum_res = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    sq_sum_res = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    sum_acc = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="sum_acc")
    sq_sum_acc = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="sq_sum_acc")

    # ---------- Single Fused Loop ----------
    for k in nl.sequential_range(K // TILE_K):
        # Read once
        x_tile = nl.ndarray((P, TILE_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, k * TILE_K:(k + 1) * TILE_K])

        # Compute sum
        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)
        nisa.tensor_scalar(dst=sum_acc, data=sum_acc, op0=nl.add, operand0=tile_sum)

        # Compute square sum USING THE SAME x_tile
        x_sq = nl.ndarray((P, TILE_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x_sq, data1=x_tile, data2=x_tile, op=nl.multiply)

        tile_sq_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sq_sum, data=x_sq, op=nl.add, axis=1, keepdims=True)

        nisa.tensor_scalar(dst=sq_sum_acc, data=sq_sum_acc, op0=nl.add, operand0=tile_sq_sum)

    # Write results
    nisa.dma_copy(dst=sum_res[0:P, 0:1], src=sum_acc)
    nisa.dma_copy(dst=sq_sum_res[0:P, 0:1], src=sq_sum_acc)

    return sum_res, sq_sum_res

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, K, dtype=torch.float32)
    x_dev = x.to(device)

    sum_dev, sq_sum_dev = dual_reduction_fused(x_dev)
    xm.mark_step()

    ref_sum = x.sum(dim=1, keepdim=True)
    ref_sq_sum = (x * x).sum(dim=1, keepdim=True)

    assert torch.allclose(sum_dev.cpu(), ref_sum, atol=1e-4), "Sum mismatch"
    assert torch.allclose(sq_sum_dev.cpu(), ref_sq_sum, atol=1e-3), "Sq sum mismatch"
    print("Correctness check passed.")
