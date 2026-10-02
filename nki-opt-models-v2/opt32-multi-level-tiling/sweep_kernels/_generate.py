#!/usr/bin/env python3
"""
Generate parameterized matmul before/after kernels for the opt32 sweep.

Usage:  python _generate.py

Both kernels compute C = lhsT.T @ rhs with M=128, TILE_M=TILE_N=TILE_K=128.

Before kernel: N outer, K inner; A reloaded per N-tile.
After kernel: K outer, N inner; A loaded once per K-tile and reused
across all N-tiles via a wide PSUM accumulator.
"""

from __future__ import annotations

import pathlib

TILE_M = 128
TILE_N = 128
TILE_K = 128

VARIANTS = [
    # (M, K, N) — M=128 fixed (single M-tile keeps PSUM size manageable).
    (128, 2048, 2048),  # v1 anchor: n_n=16
    (128, 1024, 2048),  # smaller K
    (128, 2048, 1024),  # n_n=8
    (128, 1024, 1024),  # both smaller
    (128, 2048, 512),   # n_n=4 (smaller savings)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt32 BEFORE: N-outer, K-inner. A reloaded per N-tile.
M={M}, K={K}, N={N}, TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_N = {TILE_N}
TILE_K = {TILE_K}


@nki.jit(platform_target="trn1")
def matmul_baseline(lhsT, rhs):
    C = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    zero_sbuf = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

    for n_idx in nl.affine_range(N // TILE_N):
        nisa.activation(dst=res_psum, data=zero_sbuf, op=nl.copy)

        for k_idx in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k_idx * TILE_K:(k_idx + 1) * TILE_K, 0:TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k_idx * TILE_K:(k_idx + 1) * TILE_K,
                                  n_idx * TILE_N:(n_idx + 1) * TILE_N])

            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=C[0:TILE_M, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                      src=res_sbuf)

    return C
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt32 AFTER: K-outer, N-inner. A loaded once per K-tile.
M={M}, K={K}, N={N}, TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_N = {TILE_N}
TILE_K = {TILE_K}


@nki.jit(platform_target="trn1")
def matmul_multi_level_tiling(lhsT, rhs):
    C = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    zero_sbuf = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

    for n_idx in nl.affine_range(N // TILE_N):
        nisa.activation(dst=res_psum[:, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                        data=zero_sbuf, op=nl.copy)

    for k_idx in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k_idx * TILE_K:(k_idx + 1) * TILE_K, 0:TILE_M])

        for n_idx in nl.affine_range(N // TILE_N):
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k_idx * TILE_K:(k_idx + 1) * TILE_K,
                                  n_idx * TILE_N:(n_idx + 1) * TILE_N])

            nisa.nc_matmul(dst=res_psum[:, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                           stationary=lhsT_tile, moving=rhs_tile)

    for n_idx in nl.affine_range(N // TILE_N):
        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy,
                        data=res_psum[:, n_idx * TILE_N:(n_idx + 1) * TILE_N])
        nisa.dma_copy(dst=C[0:TILE_M, n_idx * TILE_N:(n_idx + 1) * TILE_N],
                      src=res_sbuf)

    return C
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, K, N in VARIANTS:
        if M != TILE_M:
            raise SystemExit(f"M={M} must equal TILE_M={TILE_M} (single M-tile)")
        ctx = dict(M=M, K=K, N=N, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K)
        base = f"mm_M{M}_K{K}_N{N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
