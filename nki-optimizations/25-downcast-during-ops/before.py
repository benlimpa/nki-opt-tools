"""
BEFORE kernel: Downcasting via DMA copy.
The kernel computes in `float32`. The final result remains in `float32` in SBUF.
The final `nisa.dma_copy` writes the `float32` SBUF tile into the `bfloat16` HBM tensor.
This forces the DMA engine to handle the casting (a software/hardware bottleneck).
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
N = 131072
TILE_N = 512

@nki.jit(platform_target="trn1")
def downcast_before(x):
    # Input x is float32. Output y is bfloat16.
    y = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for n in nl.sequential_range(N // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n*TILE_N:(n+1)*TILE_N])

        # Heavy math in fp32
        res1 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res1, data1=x_tile, data2=x_tile, op=nl.multiply)

        res2 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res2, data1=res1, data2=x_tile, op=nl.add)

        # Store: DMA engine downcasts `float32` (SBUF) to `bfloat16` (HBM)
        nisa.dma_copy(dst=y[0:P, n*TILE_N:(n+1)*TILE_N], src=res2)

    return y

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, N, dtype=torch.float32).to(device)

    y_dev = downcast_before(x)
    xm.mark_step()

    # Correctness assert
    ref_y = (x * x + x).bfloat16()
    print("Max diff:", torch.max(torch.abs(y_dev.cpu().float() - ref_y.cpu().float())))
    assert torch.allclose(y_dev.cpu().float(), ref_y.cpu().float(), atol=1e-1, rtol=1e-2), "Mismatch"
    print("Correctness check passed.")
