"""
BEFORE kernel: RMS normalization over x [M=512, N=512] using two separate
affine_range loops with an HBM intermediate for per-row inverse-RMS values.

Optimization target: Category #30 — Fuse tile-wise reductions with
transformation passes.

Pattern:
  Loop 1 (reduction pass): load x[t], compute per-row inv_rms = 1/sqrt(mean(x²))
    via activation_reduce, store inv_rms to private_hbm[M, 1]
  Loop 2 (transformation pass): reload x[t] from HBM, reload inv_rms[t] from HBM,
    multiply x * inv_rms → store y[t]

The inv_rms intermediate crosses the loop boundary, forcing both an extra HBM
write (loop 1) and an extra HBM read (loop 2). x must also be re-read from HBM
in loop 2 since SBUF from loop 1 is not live across the affine_range boundary.

HBM traffic per kernel call (M=512, N=512, float32):
  - Loop 1: read x        (512×512×4 =  1 MB)
            write inv_rms (512×1×4   =  2 KB)
  - Loop 2: read x        (512×512×4 =  1 MB)  ← redundant reload
            read inv_rms  (512×1×4   =  2 KB)  ← redundant reload
            write y       (512×512×4 =  1 MB)
  Total:                                         ~3 MB + 4 KB
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 512
N = 512
TILE_M = 128    # partition tile size (NKI max = 128)
EPS = 1e-5


@nki.jit(platform_target="trn1")
def rms_norm_two_pass(x):
    """
    Compute y = x / rms(x, dim=-1) per row, using two separate loops.

    BEFORE: reduction (inv_rms computation) and transformation (normalization)
    are in separate affine_range loops. inv_rms must be written to HBM to cross
    the loop boundary; x must be reloaded from HBM in the second loop.

    x: [M=512, N=512] float32 in HBM
    Returns: y [M=512, N=512] float32
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # HBM intermediate: per-row inv_rms must survive across loop boundaries
    inv_rms_hbm = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # ── Loop 1: reduction pass — compute per-row inv_rms ──────────────────────
    for t in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[t * TILE_M:(t + 1) * TILE_M, 0:N])

        # Compute x^2 element-wise AND sum per row in one instruction:
        # sq_tile = x^2 [TILE_M, N], sq_sum = sum(x^2) per row [TILE_M, 1]
        sq_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        sq_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sq_tile, op=nl.square, data=x_tile,
                               reduce_op=nl.add, reduce_res=sq_sum)

        # mean(x^2) per row: scale sum by 1/N
        mean_sq = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean_sq, data=sq_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        # add eps for numerical stability
        mean_sq_eps = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean_sq_eps, data=mean_sq,
                           op0=nl.add, operand0=EPS)

        # inv_rms = 1 / sqrt(mean(x^2) + eps)
        inv_rms_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_rms_tile, data=mean_sq_eps, op=nl.rsqrt)

        # Store inv_rms to HBM — forces loop 2 to reload from HBM
        nisa.dma_copy(dst=inv_rms_hbm[t * TILE_M:(t + 1) * TILE_M, 0:1],
                      src=inv_rms_tile)

    # ── Loop 2: transformation pass — normalize x using inv_rms ───────────────
    for t in nl.affine_range(M // TILE_M):
        # Reload x from HBM (SBUF tile from loop 1 is no longer live)
        x_tile2 = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile2, src=x[t * TILE_M:(t + 1) * TILE_M, 0:N])

        # Reload inv_rms from HBM
        inv_rms_tile2 = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=inv_rms_tile2,
                      src=inv_rms_hbm[t * TILE_M:(t + 1) * TILE_M, 0:1])

        # y = x * inv_rms (broadcast [TILE_M, 1] over N free dimension)
        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile2,
                           op0=nl.multiply, operand0=inv_rms_tile2)

        nisa.dma_copy(dst=y[t * TILE_M:(t + 1) * TILE_M, 0:N], src=y_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(42)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    y_dev = rms_norm_two_pass(x_dev)
    xm.mark_step()

    # Reference: y = x / sqrt(mean(x^2) + eps) per row
    rms = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + EPS)
    ref = x / rms

    y_cpu = y_dev.cpu()
    max_err = (y_cpu - ref).abs().max().item()
    assert max_err < 5e-4, f"Max diff: {max_err}"
    print(f"Correctness check passed. Max diff: {max_err:.2e}")
