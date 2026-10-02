"""
BEFORE kernel: Heavy element-wise math using float32 intermediates.
Input and Output lie in HBM as bfloat16, but all computations are implicitly
upcasted and executed in full float32 SBUF tiles.

Optimization target: Category #24 — Allocate buffers in lower precision
to reduce SRAM footprint and increase vector throughput.

Pattern:
Polynomial `y = 0.79788 * (x + 0.044715 * x^3)`
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N = 131072
TILE_N = 512

@nki.jit(platform_target="trn1")
def poly_float32(x):
    """
    BEFORE: Allocating intermediates in float32.

    x: [128, 131072] bfloat16
    """
    y = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for n in nl.sequential_range(N // TILE_N):
        # 1. Load: Converts from BF16 HBM to FP32 SBUF
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n*TILE_N:(n+1)*TILE_N])

        # 2. x^2
        x2 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x2, data1=x_tile, data2=x_tile, op=nl.multiply)

        # 3. x^3
        x3 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x3, data1=x2, data2=x_tile, op=nl.multiply)

        # 4. 0.044715 * x^3
        term1 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=term1, data=x3, op0=nl.multiply, operand0=0.044715)

        # 5. x + term1
        res = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res, data1=x_tile, data2=term1, op=nl.add)

        # 6. final multiply: 0.79788 * res
        res2 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=res2, data=res, op0=nl.multiply, operand0=0.79788)

        # 7. Store: Converts from FP32 SBUF to BF16 HBM
        nisa.dma_copy(dst=y[0:P, n*TILE_N:(n+1)*TILE_N], src=res2)

    return y

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, N, dtype=torch.bfloat16)
    x_dev = x.to(device)

    y_dev = poly_float32(x_dev)
    xm.mark_step()

    # Correctness assert logic (approximate since we compare fp32 vs bf16 precision)
    ref_x = x.float()
    ref_y = 0.79788 * (ref_x + 0.044715 * (ref_x ** 3))
    ref_y_bf16 = ref_y.bfloat16()

    print("Max diff:", torch.max(torch.abs(y_dev.cpu().float() - ref_y_bf16.float())))
    # Using a looser tolerance due to downcasting
    assert torch.allclose(y_dev.cpu().float(), ref_y_bf16.float(), atol=1e-1, rtol=1e-2), "Mismatch"
    print("Correctness check passed.")
