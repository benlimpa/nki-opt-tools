"""
BEFORE kernel: Row-wise min-max normalization over x [M=1024, N=512].
Uses TILE_N=32 (small free-dimension tiles) — 16 tiles cover N per pass.

Optimization target: Category #31 — Load larger blocks to increase SBUF
reuse and reduce memory traffic.

Pattern: two inner loops per M-tile (128 rows):
  Pass 1 (min+max combined): load x[m, n:n+32] for n = 0..15 → 16 DMA reads.
                             Initial min/max seeded from the first tile; remaining
                             15 tiles (sequential_range 15) accumulate via running
                             tensor_tensor min/max.
  Pass 2 (normalize):        reload x[m, n:n+32] for n = 0..15 → 16 DMA reads.

Total per M-tile: 32 tile reads × 128×32×4 = 32 × 16 KB = 512 KB
Total (8 M-tiles):   8 × 512 KB = 4 MB of HBM reads for x
(vs x = 1024×512×4 = 2 MB — potential 2× redundant reads).
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
N = 512
TILE_P = 128   # partition tile size (NKI max = 128)
TILE_N = 32    # small free-dim tile: N // TILE_N = 16 tiles per pass
EPS = 1e-8


@nki.jit(platform_target="trn1")
def minmax_norm_small_tiles(x):
    """
    Compute y[m, n] = (x[m, n] - row_min[m]) / (row_max[m] - row_min[m] + eps).

    BEFORE: small TILE_N=32. Two inner loop passes per M-tile:
      - Pass 1: combined min+max scan over 16 tiles (16 HBM reads of x)
      - Pass 2: normalize over 16 tiles (16 more HBM reads of x — reload)
    Total: 32 tile reads per M-tile from HBM.

    x: [M=1024, N=512] float32 in HBM
    Returns: y [M=1024, N=512] float32
    """
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):

        # ── Load first tile to seed min_val and max_val ───────────────────────
        x0 = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x0,
                      src=x[m * TILE_P:(m + 1) * TILE_P, 0:TILE_N])

        min_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=min_val, data=x0,
                           op=nl.minimum, axis=1, keepdims=True)

        max_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x0,
                           op=nl.maximum, axis=1, keepdims=True)

        # ── Pass 1: update min/max with tiles 1..15 (sequential_range 15) ─────
        for n in nl.sequential_range(N // TILE_N - 1):
            x_tile = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile,
                          src=x[m * TILE_P:(m + 1) * TILE_P,
                                (n + 1) * TILE_N:(n + 2) * TILE_N])

            tile_min = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_min, data=x_tile,
                               op=nl.minimum, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=min_val, data1=min_val, data2=tile_min,
                               op=nl.minimum)

            tile_max = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=x_tile,
                               op=nl.maximum, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=max_val, data1=max_val, data2=tile_max,
                               op=nl.maximum)

        # ── Compute per-row range and its inverse ─────────────────────────────
        range_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_val, data=max_val,
                           op0=nl.subtract, operand0=min_val)

        range_eps = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_eps, data=range_val,
                           op0=nl.add, operand0=EPS)

        inv_range = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_range, data=range_eps, op=nl.reciprocal)

        # ── Pass 2: normalize — reload all 16 tiles from HBM ─────────────────
        for n in nl.affine_range(N // TILE_N):
            x_tile2 = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile2,
                          src=x[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_N:(n + 1) * TILE_N])

            x_shifted = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_shifted, data=x_tile2,
                               op0=nl.subtract, operand0=min_val)

            y_tile = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y_tile, data=x_shifted,
                               op0=nl.multiply, operand0=inv_range)

            nisa.dma_copy(dst=y[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_N:(n + 1) * TILE_N],
                          src=y_tile)

    return y


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(42)

    x = torch.randn(M, N, dtype=torch.float32)
    x_dev = x.to(device)

    y_dev = minmax_norm_small_tiles(x_dev)
    xm.mark_step()

    # Reference: per-row min-max normalization
    row_min = x.min(dim=1, keepdim=True).values
    row_max = x.max(dim=1, keepdim=True).values
    ref = (x - row_min) / (row_max - row_min + EPS)

    y_cpu = y_dev.cpu()
    max_err = (y_cpu - ref).abs().max().item()
    assert max_err < 1e-4, f"Max diff: {max_err:.2e}"
    print(f"Correctness check passed. Max diff: {max_err:.2e}")
