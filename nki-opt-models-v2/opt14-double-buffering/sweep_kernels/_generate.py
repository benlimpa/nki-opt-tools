#!/usr/bin/env python3
"""
Generate parameterized matmul before/after kernels for the opt14 sweep.

Usage:  python _generate.py

VARIANTS list: (M, K, N) tuples. TILE_K=K (single K-tile) so the
M-loop is the only sequential dimension. TILE_M=128, TILE_N=N.

Before kernel: sequential_range over M-tiles with single SBUF buffer.
After kernel: sequential_range over M-tiles with two-slot SBUF
double-buffering (prologue + main loop + epilogue).
"""

from __future__ import annotations

import pathlib

TILE_M = 128

VARIANTS = [
    # (M, K, N) — K=128 single K-tile, TILE_M=128, TILE_N=N
    (1024, 128, 512),  # v1 anchor: n_tiles=8
    (512,  128, 512),  # n_tiles=4
    (2048, 128, 512),  # n_tiles=16
    (1024, 128, 256),  # smaller N (less compute per tile)
    (1024, 128, 128),  # smallest N (TensorE-light → DMA-bound)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt14 BEFORE: single-buffer matmul, sequential_range over M.
M={M}, K={K}, N={N}, TILE_M={TILE_M}.
DMA load of the next lhsT M-tile blocks the next nc_matmul (single buffer).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K = {K}
TILE_N = {N}


@nki.jit(platform_target="trn1")
def matmul_single_buffer(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)

    for m in nl.sequential_range(M // TILE_M):
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=res_sbuf)

    return result
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt14 AFTER: double-buffered matmul.
M={M}, K={K}, N={N}, TILE_M={TILE_M}.
Two alternating SBUF slots; DMA(tile m+1) overlaps nc_matmul(tile m).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K = {K}
TILE_N = {N}


@nki.jit(platform_target="trn1")
def matmul_double_buffer(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    NUM_TILES = M // TILE_M

    lhsT_bufs = nl.ndarray((TILE_K, 2, TILE_M), dtype=nl.float32, buffer=nl.sbuf)

    nisa.dma_copy(dst=lhsT_bufs[0:TILE_K, 0, 0:TILE_M],
                  src=lhsT[0:TILE_K, 0:TILE_M])

    for m in nl.sequential_range(NUM_TILES - 1):
        nisa.dma_copy(dst=lhsT_bufs[0:TILE_K, (m + 1) % 2, 0:TILE_M],
                      src=lhsT[0:TILE_K, (m + 1) * TILE_M:(m + 2) * TILE_M])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum,
                       stationary=lhsT_bufs[0:TILE_K, m % 2, 0:TILE_M],
                       moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=res_sbuf)

    last = NUM_TILES - 1
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=lhsT_bufs[0:TILE_K, last % 2, 0:TILE_M],
                   moving=rhs_tile)
    res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
    nisa.dma_copy(dst=result[last * TILE_M:(last + 1) * TILE_M, 0:TILE_N],
                  src=res_sbuf)

    return result
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, K, N in VARIANTS:
        if M % TILE_M != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_M={TILE_M}")
        ctx = dict(M=M, K=K, N=N, TILE_M=TILE_M)
        base = f"mm_M{M}_K{K}_N{N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
