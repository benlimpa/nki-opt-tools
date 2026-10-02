"""
BEFORE kernel: Attention score scaling — PSUM eviction then separate scalar multiply.

Optimization target: Category #29 — Optimize accumulation patterns in PSUM.

This kernel computes scores = (Q @ K.T) * scale for Q:[M,d], K:[N,d], scale=1/sqrt(d).
This is the core of self-attention score computation.
M=512, d=128 (1 head dimension tile), N=512, TILE_M=128, TILE_K=128, TILE_N=512.

The BEFORE version:
  1. Accumulates Q @ K.T K-tiles into PSUM (nc_matmul PSUM group)
  2. Evicts PSUM → SBUF with a plain activation copy (scale=1.0)
  3. SEPARATELY multiplies the SBUF result by the 1/sqrt(d) scale factor

The separate scalar multiply is the inefficiency: nisa.activation already supports
a `scale` parameter that fuses a scalar multiplication into the PSUM eviction step.
The before version ignores this, paying an extra activation instruction per M-tile.

Cost per M-tile (before):
  - 1 × nc_matmul          (TensorE)
  - 1 × PSUM→SBUF eviction with scale=1.0    (ScalarE activation)  ← unnecessary
  - 1 × scale mul: activation(copy, data=sbuf, scale=1/sqrt(d)) (ScalarE activation) ← AVOIDABLE
  - 1 × DMA store to scores
"""

import math
import nki
import nki.isa as nisa
import nki.language as nl

M = 512
d = 128    # head dimension, one nc_matmul TILE_K tile
N = 512

TILE_M = 128
TILE_K = 128   # single K tile (d == TILE_K, so no K-loop needed)
TILE_N = 512

SCALE = 1.0 / math.sqrt(d)


@nki.jit(platform_target="trn1")
def attention_score_separate_scale(q_t, k):
    """
    Compute scores = (Q @ K.T) * scale, scale = 1/sqrt(d).

    BEFORE: PSUM eviction with scale=1.0, then separate activation to apply scale.

    q_t:  [d, M]  Q transposed, stationary  (one head, d=TILE_K)
    k:    [d, N]  K^T = K transposed for nc_matmul moving operand
    Returns: scores [M, N] in HBM
    """
    scores = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        # PSUM slot for accumulation (single K-tile: d == TILE_K)
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        q_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile,
                      src=q_t[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        k_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_tile, src=k[0:TILE_K, 0:TILE_N])

        nisa.nc_matmul(dst=res_psum, stationary=q_tile, moving=k_tile)

        # Step 1: Evict PSUM to SBUF without scale (plain copy)
        raw_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=raw_sbuf, op=nl.copy, data=res_psum)

        # Step 2: SEPARATE scale multiply — avoidable!
        scaled_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scaled_sbuf, op=nl.copy, data=raw_sbuf, scale=SCALE)

        nisa.dma_copy(dst=scores[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=scaled_sbuf)

    return scores


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    Q = torch.randn(M, d, dtype=torch.float32)
    K = torch.randn(N, d, dtype=torch.float32)

    q_t = Q.T.contiguous().to(device)   # [d, M]
    k_t = K.T.contiguous().to(device)   # [d, N] — K transposed for nc_matmul

    S = attention_score_separate_scale(q_t, k_t)
    xm.mark_step()

    S_cpu = S.cpu()
    ref = (Q @ K.T) * SCALE   # [M, N]

    assert torch.allclose(S_cpu, ref, atol=1e-3), (
        f"Max diff: {(S_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
