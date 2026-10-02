#!/usr/bin/env python3
"""
Generate parameterized matmul before/after kernels for the opt11 sweep.

Usage:  python _generate.py

VARIANTS list: (M, K, N, TILE_N_BEFORE) tuples. The before kernel uses
TILE_N=TILE_N_BEFORE; the after kernel uses TILE_N=min(N, 512).
"""

from __future__ import annotations

import pathlib

TILE_N_MAX = 512
TILE_M = 128
TILE_K = 128

VARIANTS = [
    # (M, K, N, TILE_N_BEFORE) — N divisible by TILE_N_BEFORE and by min(N, 512)
    (1024, 128,  512, 128),  # v1 anchor (4× shrink)
    (1024, 128,  512, 256),  # 2× shrink
    ( 512, 128,  512, 128),  # off-axis: smaller M
    (2048, 128,  512, 128),  # off-axis: larger M
    (1024, 128,  256, 128),  # off-axis: smaller N (target=256)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt11 BEFORE: nc_matmul with small TILE_N={TILE_N_BEFORE}.
M={M}, K={K}, N={N} → {N_BEFORE} nc_matmul calls per M-tile.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = 128
TILE_K = 128
TILE_N = {TILE_N_BEFORE}


@nki.jit(platform_target="trn1")
def matmul_small_n_tiles(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        for n in nl.affine_range(N // TILE_N):
            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[0:TILE_K, n * TILE_N:(n + 1) * TILE_N])

            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
            nisa.dma_copy(
                dst=result[m * TILE_M:(m + 1) * TILE_M,
                            n * TILE_N:(n + 1) * TILE_N],
                src=res_sbuf)

    return result
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt11 AFTER: nc_matmul with maximum TILE_N={TILE_N_AFTER}.
M={M}, K={K}, N={N} → 1 nc_matmul call per M-tile (rhs hoisted).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = 128
TILE_K = 128
TILE_N = {TILE_N_AFTER}


@nki.jit(platform_target="trn1")
def matmul_large_n_tile(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    for m in nl.affine_range(M // TILE_M):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
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


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, K, N, TILE_N_BEFORE in VARIANTS:
        if M % TILE_M != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_M={TILE_M}")
        if N % TILE_N_BEFORE != 0:
            raise SystemExit(f"N={N} must be divisible by TILE_N_BEFORE={TILE_N_BEFORE}")
        TILE_N_AFTER = min(N, TILE_N_MAX)
        if N % TILE_N_AFTER != 0:
            raise SystemExit(f"N={N} must be divisible by TILE_N_AFTER={TILE_N_AFTER}")
        n_before = N // TILE_N_BEFORE
        ctx_b = dict(M=M, K=K, N=N, TILE_N_BEFORE=TILE_N_BEFORE, N_BEFORE=n_before)
        ctx_a = dict(M=M, K=K, N=N, TILE_N_AFTER=TILE_N_AFTER)
        base = f"mm_M{M}_K{K}_N{N}_T{TILE_N_BEFORE}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx_b))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx_a))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
