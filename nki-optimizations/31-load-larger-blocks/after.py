"""
AFTER kernel: Row-wise min-max normalization over x [M=1024, N=512].

Optimization: Category #31 — Load larger blocks to increase SBUF reuse.

KEY CHANGE:
  Before: TILE_N=32 — three inner loops, each reloading x from HBM.
          48 tile loads per M-tile × 8 M-tiles = 384 tile reads total.

  After:  TILE_N=512 (full free dimension) — one DMA per M-tile loads the
          entire [TILE_P=128, N=512] row block into SBUF. All three
          computations (min, max, normalize) operate directly on the
          stationary SBUF tile. Zero redundant HBM reads of x.
          8 tile loads per M-tile iteration = 8 total.

Expected savings:
  - hbm_read_bytes: ~3× reduction (6 MB → 2 MB) if compiler doesn't cache
  - software_dynamic_dma_packet_count: 48× fewer packets per M-tile
  - dma_active_time: proportional reduction
  - Wall-clock speedup depends on DMA bottleneck fraction
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128   # partition tile size (NKI max = 128)
TILE_N = N     # large free-dim tile: entire N in one load = 512 elements
EPS = 1e-8


@nki.jit(platform_target="trn1")
def minmax_norm_large_tile(x):
    """
    Compute y[m, n] = (x[m, n] - row_min[m]) / (row_max[m] - row_min[m] + eps).

    AFTER: large TILE_N=512. One DMA per M-tile loads the full row block
    [TILE_P=128, N=512] into SBUF. Min, max, and normalize all run on the
    stationary SBUF tile — no HBM reload for x.

    x: [M=1024, N=512] float32 in HBM
    Returns: y [M=1024, N=512] float32
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):

        # ── Load full row block into SBUF — single DMA per M-tile ─────────────
        x_block = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_block,
                      src=x[m * TILE_P:(m + 1) * TILE_P, 0:N])

        # ── Compute per-row min from SBUF (no HBM reload) ─────────────────────
        min_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=min_val, data=x_block,
                           op=nl.minimum, axis=1, keepdims=True)

        # ── Compute per-row max from SBUF (no HBM reload) ─────────────────────
        max_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x_block,
                           op=nl.maximum, axis=1, keepdims=True)

        # ── Compute per-row range and its inverse ─────────────────────────────
        range_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_val, data=max_val,
                           op0=nl.subtract, operand0=min_val)

        range_eps = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_eps, data=range_val,
                           op0=nl.add, operand0=EPS)

        inv_range = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_range, data=range_eps, op=nl.reciprocal)

        # ── Normalize from SBUF — no HBM reload ──────────────────────────────
        x_shifted = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_shifted, data=x_block,
                           op0=nl.subtract, operand0=min_val)

        y_block = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_block, data=x_shifted,
                           op0=nl.multiply, operand0=inv_range)

        # ── Single store per M-tile ────────────────────────────────────────────
        nisa.dma_copy(dst=y[m * TILE_P:(m + 1) * TILE_P, 0:N],
                      src=y_block)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(42)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    y_dev = minmax_norm_large_tile(x_dev)
    xm.mark_step()

    # Reference: per-row min-max normalization
    row_min = x.min(dim=1, keepdim=True).values
    row_max = x.max(dim=1, keepdim=True).values
    ref = (x - row_min) / (row_max - row_min + EPS)

    y_cpu = y_dev.cpu()
    max_err = (y_cpu - ref).abs().max().item()
    assert max_err < 1e-4, f"Max diff: {max_err:.2e}"
    print(f"Correctness check passed. Max diff: {max_err:.2e}")
