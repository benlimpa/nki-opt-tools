"""
AFTER kernel: matmul with 3-stage pipeline — load, compute, and store overlap.

Optimization: Category #16 — Pipeline compute and data movement (sequential_range)

KEY CHANGE: Input ping-pong (from #14) PLUS output ping-pong. Each sequential_range
iteration simultaneously executes all three pipeline stages on different tiles:

  Iter m:
    DMA: load  input[m+1]   → input_bufs[:, (m+1)%2, :]   ← stage A (next input)
    DMA: store output[m-1]  from output_bufs[:, (m-1)%2, :] ← stage B (prev output)
    TensorE: compute input[m] from input_bufs[:, m%2, :]
             → output_bufs[:, m%2, :]                        ← stage C (current)

All three target different SBUF addresses → DMA and TensorE execute concurrently.
The two DMA ops (load + store) are queued and can also pipeline within the DMA engine.

Pipeline stages:
  input_bufs  (TILE_K=128, 2, TILE_M=128) — 128 KB, two input slots
  output_bufs (TILE_M=128, 2, TILE_N=512) — 512 KB, two output slots

Before: T_total ≈ (T_load + T_compute + T_store) × 8  [serial]
After:  T_total ≈ max(T_load + T_store, T_compute) × 8 + fill/drain  [pipelined]
        Since output store (256 KB) >> input load (64 KB), DMA bottleneck ≈ T_store × 8
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 1024
K = 128
N = 512

TILE_M = 128
TILE_K = 128
TILE_N = 512


@nki.jit(platform_target="trn1")
def matmul_pipelined(lhsT, rhs):
    """
    C = lhsT.T @ rhs — 3-stage pipeline overlapping load, compute, and store.

    AFTER: input prefetch + output double-buffering run concurrently with TensorE.

    lhsT: [K=128, M=1024] float32
    rhs:  [K=128, N=512]  float32
    Returns: C [M=1024, N=512] float32
    """
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load rhs once
    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    NUM_TILES = M // TILE_M  # 8

    # Two input slots: load into (m+1)%2, compute from m%2 — no conflict
    # Shape: (TILE_K, 2, TILE_M) → partition=128, slot=2, free=128
    input_bufs = nl.ndarray((TILE_K, 2, TILE_M), dtype=nl.float32, buffer=nl.sbuf)

    # Two output slots: write into m%2, store from (m-1)%2 — no conflict
    # Shape: (TILE_M, 2, TILE_N) → partition=128, slot=2, free=512
    output_bufs = nl.ndarray((TILE_M, 2, TILE_N), dtype=nl.float32, buffer=nl.sbuf)

    # ─── Prologue: fill pipeline ───────────────────────────────────────────────

    # Stage P1: load tile 0 → slot 0
    nisa.dma_copy(dst=input_bufs[0:TILE_K, 0, 0:TILE_M],
                  src=lhsT[0:TILE_K, 0:TILE_M])

    # Stage P2: load tile 1 → slot 1, compute tile 0 (from slot 0) → output slot 0
    # DMA and TensorE target different SBUF regions → overlap
    nisa.dma_copy(dst=input_bufs[0:TILE_K, 1, 0:TILE_M],
                  src=lhsT[0:TILE_K, TILE_M:2 * TILE_M])
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=input_bufs[0:TILE_K, 0, 0:TILE_M],
                   moving=rhs_tile)
    nisa.activation(dst=output_bufs[0:TILE_M, 0, 0:TILE_N],
                    op=nl.copy, data=res_psum)

    # ─── Steady state: m = 1 to NUM_TILES-2 ──────────────────────────────────
    # Each iteration simultaneously:
    #   DMA: load tile m+1   → input_bufs[:, (m+1)%2, :]  (next input)
    #   DMA: store tile m-1  from output_bufs[:, (m-1)%2, :] (prev output)
    #   TensorE: compute tile m  input_bufs[:, m%2, :] → output_bufs[:, m%2, :]
    for m in nl.sequential_range(1, NUM_TILES - 1):
        # Issue both DMA ops before TensorE — queued together, overlap with each other
        # and with TensorE (all target different SBUF addresses)
        nisa.dma_copy(dst=input_bufs[0:TILE_K, (m + 1) % 2, 0:TILE_M],
                      src=lhsT[0:TILE_K, (m + 1) * TILE_M:(m + 2) * TILE_M])
        nisa.dma_copy(dst=result[(m - 1) * TILE_M:m * TILE_M, 0:TILE_N],
                      src=output_bufs[0:TILE_M, (m - 1) % 2, 0:TILE_N])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum,
                       stationary=input_bufs[0:TILE_K, m % 2, 0:TILE_M],
                       moving=rhs_tile)
        nisa.activation(dst=output_bufs[0:TILE_M, m % 2, 0:TILE_N],
                        op=nl.copy, data=res_psum)

    # ─── Epilogue: drain pipeline ──────────────────────────────────────────────

    # Stage E1: store tile NUM_TILES-2, compute tile NUM_TILES-1
    # DMA store and TensorE target different output slots → overlap
    last = NUM_TILES - 1
    nisa.dma_copy(dst=result[(last - 1) * TILE_M:last * TILE_M, 0:TILE_N],
                  src=output_bufs[0:TILE_M, (last - 1) % 2, 0:TILE_N])
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=input_bufs[0:TILE_K, last % 2, 0:TILE_M],
                   moving=rhs_tile)
    nisa.activation(dst=output_bufs[0:TILE_M, last % 2, 0:TILE_N],
                    op=nl.copy, data=res_psum)

    # Stage E2: store final tile NUM_TILES-1
    nisa.dma_copy(dst=result[last * TILE_M:(last + 1) * TILE_M, 0:TILE_N],
                  src=output_bufs[0:TILE_M, last % 2, 0:TILE_N])

    return result


if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    lhsT = A.T.contiguous().to(device)
    rhs = B.to(device)

    C = matmul_pipelined(lhsT, rhs)
    xm.mark_step()

    ref = A @ B
    C_cpu = C.cpu()
    assert torch.allclose(C_cpu, ref, atol=1e-3), (
        f"Max diff: {(C_cpu - ref).abs().max()}"
    )
    print("Correctness check passed.")
