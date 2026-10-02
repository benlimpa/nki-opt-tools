#!/usr/bin/env python3
"""
Generate parameterized matmul before/after kernels for the opt18 sweep.

Usage:  python _generate.py

Each variant uses TILE_M=128 (partition max) and TILE_K_BEFORE for the
fine K-tiling baseline; the after kernel always uses TILE_K=128.
Constraints: K ≥ 128, K % TILE_K_BEFORE == 0, K % 128 == 0.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (M, K, N, TILE_K_BEFORE) — TILE_M=128 fixed; TILE_K_AFTER=128 always.
    (1024, 512, 512, 16),   # ratio=8, large shape
    (1024, 512, 512, 32),   # ratio=4 (v1 anchor)
    (1024, 512, 512, 64),   # ratio=2, large shape
    (512,  256, 256, 32),   # ratio=4, smaller shape (off-axis)
    (512,  256, 256, 64),   # ratio=2, smaller shape (off-axis)
]

TILE_M = 128
TILE_K_AFTER = 128


_TEMPLATE = '''\
"""
Sweep variant — opt18 {WHICH}: matmul with TILE_K={TILE_K}.
M={M}, K={K}, N={N}, TILE_M={TILE_M}, TILE_K={TILE_K}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K_BEFORE = {TILE_K_BEFORE}
TILE_K_AFTER = {TILE_K_AFTER}
TILE_K = {TILE_K}


@nki.jit(platform_target="trn1")
def matmul(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                   m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:N])

            nisa.nc_matmul(dst=res_psum,
                           stationary=lhsT_tile,
                           moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:N],
                      src=res_sbuf)

    return result
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, K, N, TILE_K_BEFORE in VARIANTS:
        base = f"mm_M{M}_K{K}_N{N}_T{TILE_K_BEFORE}"
        before_ctx = dict(WHICH="BEFORE", M=M, K=K, N=N, TILE_M=TILE_M,
                          TILE_K_BEFORE=TILE_K_BEFORE, TILE_K_AFTER=TILE_K_AFTER,
                          TILE_K=TILE_K_BEFORE)
        after_ctx = dict(WHICH="AFTER", M=M, K=K, N=N, TILE_M=TILE_M,
                         TILE_K_BEFORE=TILE_K_BEFORE, TILE_K_AFTER=TILE_K_AFTER,
                         TILE_K=TILE_K_AFTER)
        (here / f"{base}_before.py").write_text(_TEMPLATE.format(**before_ctx))
        (here / f"{base}_after.py").write_text(_TEMPLATE.format(**after_ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
