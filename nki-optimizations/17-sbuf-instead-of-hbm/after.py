"""
AFTER kernel: y = x * exp(x) in a single pass — exp(x) stays in SBUF.

Optimization: Category #17 — Keep data in SBUF/PSUM instead of HBM

KEY CHANGE: Replace the two-pass HBM-intermediate approach with a single
affine_range loop. exp(x) is computed into SBUF and immediately used for the
multiply in the same tile iteration — it never touches HBM.

HBM traffic:
  - Single pass: read x (2 MB) + write result (2 MB)  = 4 MB
  (vs 10 MB before — 2.5× reduction)

Before: exp(x) → private_hbm (2 MB write) → reload (2 MB read)  [+4 MB overhead]
        x read twice (2 MB × 2 = 4 MB)
After:  exp(x) stays in SBUF (0 HBM traffic)
        x read once (2 MB)
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_M = 128


@nki.jit(platform_target="trn1")
def exp_mul_sbuf(x):
    """
    Compute y = x * exp(x) in a single pass, keeping exp(x) in SBUF.

    AFTER: no HBM intermediate; exp(x) computed and consumed within one tile iter.
    HBM traffic: 2 × 2 MB = 4 MB (x read once, result written once).

    x: [M=1024, N=512] float32
    Returns: y [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Single pass: load x, compute exp(x) in SBUF, multiply, store
    for m in nl.affine_range(M // TILE_M):
        # Load x once
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile,
                      src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        # exp(x) stays in SBUF — never written to HBM
        exp_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, data=x_tile, op=nl.exp)

        # Multiply immediately from SBUF — no reload needed
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

    result = exp_mul_sbuf(x_dev)
    xm.mark_step()

    ref = x * torch.exp(x)
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, ref, atol=1e-2), (
        f"Max diff: {(result_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
