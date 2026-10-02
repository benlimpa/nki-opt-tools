"""
BEFORE kernel: Dual reduction (sum and sum of squares) over x [P=128, K=8192]
using two separate sequential_range loops over the K dimension.

Optimization target: Category #22 — Fuse loops over the same dimension to
improve intermediate data reuse.

Pattern:
  Loop 1 (over K): load x tile, compute tile sum, accumulate to running sum
  Loop 2 (over K): load x tile AGAIN, compute tile sq_sum, accumulate to sq_sum

The compiler enforces strict ordering for sequential_range, so it cannot
auto-fuse these two loops. This means the input tensor x is loaded from HBM
twice, doubling the memory traffic for a purely memory-bound kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
K = 8192
TILE_K = 512

@nki.jit(platform_target="trn1")
def dual_reduction_separate(x):
    """
    Compute sum and sum of squares along the K dimension.

    BEFORE: Two separate sequential_range loops. x is loaded twice.

    x: [P=128, K=8192] float32
    Returns: sum_res, sq_sum_res (both [128, 1])
    """
    sum_res = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    sq_sum_res = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---------- Loop 1: sum ----------
    sum_acc = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="sum_acc")
    for k in nl.sequential_range(K // TILE_K):
        x_tile = nl.ndarray((P, TILE_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, k * TILE_K:(k + 1) * TILE_K])

        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        nisa.tensor_scalar(dst=sum_acc, data=sum_acc, op0=nl.add, operand0=tile_sum)

    nisa.dma_copy(dst=sum_res[0:P, 0:1], src=sum_acc)

    # ---------- Loop 2: square sum ----------
    sq_sum_acc = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="sq_sum_acc")
    for k in nl.sequential_range(K // TILE_K):
        x_tile2 = nl.ndarray((P, TILE_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile2, src=x[0:P, k * TILE_K:(k + 1) * TILE_K])

        x_sq = nl.ndarray((P, TILE_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x_sq, data1=x_tile2, data2=x_tile2, op=nl.multiply)

        tile_sq_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sq_sum, data=x_sq, op=nl.add, axis=1, keepdims=True)

        nisa.tensor_scalar(dst=sq_sum_acc, data=sq_sum_acc, op0=nl.add, operand0=tile_sq_sum)

    nisa.dma_copy(dst=sq_sum_res[0:P, 0:1], src=sq_sum_acc)

    return sum_res, sq_sum_res

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, K, dtype=torch.float32)
    x_dev = x.to(device)

    sum_dev, sq_sum_dev = dual_reduction_separate(x_dev)
    xm.mark_step()

    ref_sum = x.sum(dim=1, keepdim=True)
    ref_sq_sum = (x * x).sum(dim=1, keepdim=True)

    assert torch.allclose(sum_dev.cpu(), ref_sum, atol=1e-4), "Sum mismatch"
    assert torch.allclose(sq_sum_dev.cpu(), ref_sq_sum, atol=1e-3), "Sq sum mismatch"
    print("Correctness check passed.")
