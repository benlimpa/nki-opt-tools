"""
AFTER kernel: RMS normalization over x [M=512, N=512] using a single fused
affine_range loop — reduction and transformation in the same iteration.

Optimization: Category #30 — Fuse tile-wise reductions with transformation
passes.

KEY CHANGES vs. before:
  - Single affine_range loop replaces two separate loops.
  - x is loaded ONCE per tile; it stays in SBUF across the reduction and
    transformation within the same iteration.
  - inv_rms is computed in SBUF and used immediately — never written to HBM.
  - Eliminates private_hbm intermediate entirely.

HBM traffic per kernel call (M=512, N=512, float32):
  - Single loop: read x  (512×512×4 = 1 MB)  ← one read, not two
                 write y (512×512×4 = 1 MB)
  Total:                                        2 MB  (vs ~3 MB before)

Savings: ~1 MB redundant x read + ~4 KB inv_rms intermediate eliminated.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 512
N = 512
TILE_M = 128
EPS = 1e-5


@nki.jit(platform_target="trn1")
def rms_norm_fused(x):
    """
    Compute y = x / rms(x, dim=-1) per row using a single fused loop.

    AFTER: reduction (inv_rms) and transformation (x * inv_rms) are fused into
    one affine_range loop. x stays in SBUF — loaded once, used for both the
    sum-of-squares reduction and the final multiply. inv_rms never touches HBM.

    x: [M=512, N=512] float32 in HBM
    Returns: y [M=512, N=512] float32
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # ── Fused loop: reduce then transform within the same tile iteration ───────
    for t in nl.affine_range(M // TILE_M):
        # Load x tile ONCE — reused for both reduction and transformation
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[t * TILE_M:(t + 1) * TILE_M, 0:N])

        # ── Reduction: per-row inv_rms (all in SBUF, no HBM) ──────────────────
        # Compute x^2 AND sum(x^2) per row in one instruction
        sq_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        sq_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sq_tile, op=nl.square, data=x_tile,
                               reduce_op=nl.add, reduce_res=sq_sum)

        # mean(x^2) per row
        mean_sq = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean_sq, data=sq_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        # add eps for numerical stability
        mean_sq_eps = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean_sq_eps, data=mean_sq,
                           op0=nl.add, operand0=EPS)

        # inv_rms = 1 / sqrt(mean(x^2) + eps) — stays in SBUF
        inv_rms_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_rms_tile, data=mean_sq_eps, op=nl.rsqrt)

        # ── Transformation: normalize using inv_rms (both in SBUF) ────────────
        # x_tile still in SBUF — no reload needed
        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile,
                           op0=nl.multiply, operand0=inv_rms_tile)

        nisa.dma_copy(dst=y[t * TILE_M:(t + 1) * TILE_M, 0:N], src=y_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(42)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    y_dev = rms_norm_fused(x_dev)
    xm.mark_step()

    # Reference: y = x / sqrt(mean(x^2) + eps) per row
    rms = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + EPS)
    ref = x / rms

    y_cpu = y_dev.cpu()
    max_err = (y_cpu - ref).abs().max().item()
    assert max_err < 5e-4, f"Max diff: {max_err}"
    print(f"Correctness check passed. Max diff: {max_err:.2e}")
