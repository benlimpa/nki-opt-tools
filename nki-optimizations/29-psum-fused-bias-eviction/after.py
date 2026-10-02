"""
AFTER kernel: Attention score scaling — scale fused into PSUM eviction.

Optimization: Category #29 — Optimize accumulation patterns in PSUM.

KEY CHANGE: The 1/sqrt(d) scale factor is passed directly as the `scale`
parameter to the PSUM-eviction activation call. The Neuron compiler fuses the
PSUM read, the scalar multiply, and the SBUF write into a SINGLE ScalarE
instruction, completely eliminating the separate scale-multiply pass.

  nisa.activation(dst=out, op=nl.copy, data=res_psum, scale=SCALE)
  ↑ reads PSUM ──────────────────────── multiplies by 1/sqrt(d) in-flight ↑

Before: 2 activation calls per M-tile (evict plain + scale separately)
After:  1 activation call per M-tile (evict-with-scale)

Per-M-tile savings: 1 fewer activation instruction, 1 fewer SBUF intermediate.
Total across 4 M-tiles: 4 fewer activate instructions.
"""

import math
import nki
import nki.isa as nisa
import nki.language as nl

M = 512
d = 128
N = 512

TILE_M = 128
TILE_K = 128
TILE_N = 512

SCALE = 1.0 / math.sqrt(d)


@nki.jit(platform_target="trn1")
def attention_score_fused_scale(q_t, k):
    """
    Compute scores = (Q @ K.T) * scale, scale = 1/sqrt(d).

    AFTER: scale fused into the PSUM eviction activation — single instruction.

    q_t:  [d, M]  Q transposed, stationary
    k:    [d, N]  K transposed for nc_matmul moving operand
    Returns: scores [M, N] in HBM
    """
    scores = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        q_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile,
                      src=q_t[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        k_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_tile, src=k[0:TILE_K, 0:TILE_N])

        nisa.nc_matmul(dst=res_psum, stationary=q_tile, moving=k_tile)

        # FUSED: evict PSUM → SBUF AND apply scale in a SINGLE activation instruction.
        out_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, op=nl.copy, data=res_psum, scale=SCALE)

        nisa.dma_copy(dst=scores[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=out_tile)

    return scores


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    Q = torch.randn(M, d, dtype=torch.float32)
    K = torch.randn(N, d, dtype=torch.float32)

    q_t = Q.T.contiguous().to(device)
    k_t = K.T.contiguous().to(device)

    S = attention_score_fused_scale(q_t, k_t)
    xm.mark_step()

    S_cpu = S.cpu()
    ref = (Q @ K.T) * SCALE

    assert torch.allclose(S_cpu, ref, atol=1e-3), (
        f"Max diff: {(S_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
