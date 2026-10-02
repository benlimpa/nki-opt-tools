"""
BEFORE kernel: y = x * exp(x) computed in two passes with an HBM intermediate.

Optimization target: Category #17 — Keep data in SBUF/PSUM instead of HBM

This kernel computes y = x * exp(x) over x [M=1024, N=512] using two separate
affine_range loops separated by a private_hbm intermediate buffer:

  Pass 1 (8 tiles): load x[m] → compute exp(x[m]) → store to exp_hbm[m]
  Pass 2 (8 tiles): load x[m] + load exp_hbm[m] → multiply → store result[m]

HBM traffic per element of x:
  - Pass 1: read x     (2 MB) + write exp_hbm (2 MB)  = 4 MB
  - Pass 2: read x     (2 MB) + read  exp_hbm (2 MB)  = 4 MB
            write result (2 MB)                        = 2 MB
  Total:                                               = 10 MB

The two loops cannot be fused by the compiler because the private_hbm buffer
creates a hard dependency boundary: all of Pass 1 must complete (and exp_hbm
must be fully written to HBM) before Pass 2 can begin reading it.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_M = 128


@nki.jit(platform_target="trn1")
def exp_mul_hbm(x):
    """
    Compute y = x * exp(x) using a two-pass approach with HBM intermediate.

    BEFORE: exp(x) written to private_hbm, then reloaded for multiply.
    HBM traffic: 5 × 2 MB = 10 MB (x read twice, exp_hbm written+read, result written).

    x: [M=1024, N=512] float32
    Returns: y [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # HBM intermediate: exp(x) must be fully materialized before Pass 2
    exp_hbm = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)

    # ── Pass 1: compute exp(x), store to HBM ──────────────────────────────────
    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        exp_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, data=x_tile, op=nl.exp)

        # Write to HBM intermediate — forces all tiles out to HBM
        nisa.dma_copy(dst=exp_hbm[m * TILE_M:(m + 1) * TILE_M, 0:N],
                      src=exp_tile)

    # ── Pass 2: reload x and exp(x) from HBM, compute product ─────────────────
    for m in nl.affine_range(M // TILE_M):
        # Reload x from HBM (second read)
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # Reload exp(x) from HBM
        exp_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_tile,
                      src=exp_hbm[m * TILE_M:(m + 1) * TILE_M, 0:N])

        out_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out_tile, data1=x_tile, data2=exp_tile,
                           op=nl.multiply)

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:N],
                      src=out_tile)

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32).clamp(-5, 5)
    x_dev = x.to(device)

    result = exp_mul_hbm(x_dev)
    xm.mark_step()

    ref = x * torch.exp(x)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-2), (
        f"Max diff: {(result_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
