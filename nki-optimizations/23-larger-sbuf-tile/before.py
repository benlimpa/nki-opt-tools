"""
BEFORE kernel: Multi-pass Softmax on x [P=128, N=4096].
Uses small SBUF tiles (512 elements) and iterates over N three times.
Because SBUF only holds 512 elements at a time, x must be paged in from HBM
on EVERY pass.

Optimization target: Category #23 — Allocate a larger tile in SBUF to keep
data instead of HBM round-trips.

Pattern:
  Pass 1 (Max): Load x from HBM in tiles.
  Pass 2 (Sum): Load x from HBM AGAIN in tiles.
  Pass 3 (Out): Load x from HBM A THIRD TIME in tiles.

Total HBM reads: 3 * (128 * 4096 * 4) bytes = 6 MB.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N = 4096
TILE_N = 512

@nki.jit(platform_target="trn1")
def softmax_before(x):
    """
    BEFORE: 3 separate passes reading x from HBM in 512-element tiles.
    x is loaded 3 times.

    x: [P=128, N=4096] float32
    """
    y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---------- Pass 1: Row Max ----------
    max_val = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="max_val")
    nisa.tensor_scalar(dst=max_val, data=max_val, op0=nl.add, operand0=-10000.0)

    for n in nl.sequential_range(N // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, data=x_tile, op=nl.maximum, axis=1, keepdims=True)

        nisa.tensor_tensor(dst=max_val, data1=max_val, data2=tile_max, op=nl.maximum)

    # ---------- Pass 2: Row Sum of Exps ----------
    sum_val = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="sum_val")

    for n in nl.sequential_range(N // TILE_N):
        # READ FROM HBM AGAIN
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        x_norm = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_norm, data=x_tile, op0=nl.subtract, operand0=max_val)

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, data=x_norm, op=nl.exp)

        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, data=exp_tile, op=nl.add, axis=1, keepdims=True)

        nisa.tensor_tensor(dst=sum_val, data1=sum_val, data2=tile_sum, op=nl.add)

    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=sum_val)

    # ---------- Pass 3: Exps / Sum ----------
    for n in nl.sequential_range(N // TILE_N):
        # READ FROM HBM YET AGAIN
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        x_norm = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_norm, data=x_tile, op0=nl.subtract, operand0=max_val)

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, data=x_norm, op=nl.exp)

        out_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out_tile, data=exp_tile, op0=nl.multiply, operand0=inv_sum)

        nisa.dma_copy(dst=y[0:P, n * TILE_N:(n + 1) * TILE_N], src=out_tile)

    return y

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, N, dtype=torch.float32)
    x_dev = x.to(device)

    y_dev = softmax_before(x_dev)
    xm.mark_step()

    ref_y = torch.softmax(x, dim=1)

    assert torch.allclose(y_dev.cpu(), ref_y, atol=1e-3), "Softmax mismatch"
    print("Correctness check passed.")
