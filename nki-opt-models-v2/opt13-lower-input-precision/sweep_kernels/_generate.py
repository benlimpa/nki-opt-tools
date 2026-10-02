#!/usr/bin/env python3
"""
Generate parameterized matmul before (f32 inputs) / after (bf16 inputs)
kernels for the opt13 sweep.

Usage:  python _generate.py

VARIANTS list: (M, K, N) tuples. TILE_M=TILE_K=128 fixed; TILE_N is
the moving width (= min(N, 512)). Both kernels use the PSUM-accumulation
pattern (affine_range over K-tiles) so the comparison isolates only
the precision effect.
"""

from __future__ import annotations

import pathlib

TILE_M = 128
TILE_K = 128

VARIANTS = [
    # (M, K, N) — all multiples of TILE_M / TILE_K; N ≤ 512.
    (512,  512, 512),   # v1 anchor (read=2MB, write=1MB, read fraction 2/3)
    (512,  512, 256),   # write-light (read=1.5MB, write=0.5MB, read fraction 0.75)
    (1024, 512, 512),   # write-heavy (read=3MB, write=2MB, read fraction 3/5)
    (512, 1024, 512),   # read-heavy (read=3MB, write=1MB, read fraction 0.75)
    (1024, 1024, 256),  # large overall (read=5MB, write=1MB, read fraction 5/6)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt13 BEFORE: float32 input matmul.
M={M}, K={K}, N={N}, TILE_M={TILE_M}, TILE_K={TILE_K}.
Inputs lhsT, rhs are f32 (4 B/elem). Output is f32.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K = {TILE_K}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def matmul_fp32(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N], src=res_sbuf)

    return result
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt13 AFTER: bfloat16 input matmul.
M={M}, K={K}, N={N}, TILE_M={TILE_M}, TILE_K={TILE_K}.
Inputs lhsT, rhs are bf16 (2 B/elem). Output is f32 (PSUM accumulates in f32).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K = {TILE_K}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def matmul_bf16(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile,
                          src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                    m * TILE_M:(m + 1) * TILE_M])

            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=rhs_tile,
                          src=rhs[k * TILE_K:(k + 1) * TILE_K, 0:TILE_N])

            nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

        res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)
        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N], src=res_sbuf)

    return result
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, K, N in VARIANTS:
        if M % TILE_M != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_M={TILE_M}")
        if K % TILE_K != 0:
            raise SystemExit(f"K={K} must be divisible by TILE_K={TILE_K}")
        TILE_N = min(N, 512)
        if N % TILE_N != 0:
            raise SystemExit(f"N={N} must be divisible by TILE_N={TILE_N}")
        ctx = dict(M=M, K=K, N=N, TILE_M=TILE_M, TILE_K=TILE_K, TILE_N=TILE_N)
        base = f"mm_M{M}_K{K}_N{N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
